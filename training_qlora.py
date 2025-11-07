#!/usr/bin/env python3
# InternVL2.5-4B QLoRA fine-tuning with YAML-driven config (fixed sequencing & gradient checks)
import os
import sys
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image

# Environment
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface/transformers")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Init] Device: {device} (GPUs: {torch.cuda.device_count()})")

# YAML args

parser = argparse.ArgumentParser(description="InternVL2.5-4B QLoRA fine-tuning")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
args = parser.parse_args()

if not os.path.exists(args.config):
    sys.exit(f"[Error] Config not found: {args.config}")

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Core settings
MODEL_PATH = cfg.get("model_name_or_path", "/workspace/models/InternVL4B")
DATASET_NAME = cfg.get("dataset", "maica_internvl_sft")
DATASET_DIR = cfg.get("dataset_dir", "/workspace/data")
MEDIA_DIR = cfg.get("media_dir", "/workspace/data/images")
DATA_PATH = os.path.join(DATASET_DIR, f"{DATASET_NAME}.jsonl")
OUTPUT_DIR = cfg.get("output_dir", "/workspace/output")

# Training hparams
QBIT = int(cfg.get("quantization_bit", 8))            # 4 or 8 for QLoRA
LR = float(cfg.get("learning_rate", 2e-4))
EPOCHS = int(cfg.get("num_train_epochs", 3))
BATCH_SIZE = int(cfg.get("per_device_train_batch_size", 1))
ACC_STEPS = int(cfg.get("gradient_accumulation_steps", 8))

# LoRA config
LORA_TARGET = cfg.get("lora_target", "q_proj,k_proj,v_proj,o_proj").split(",")
LORA_RANK = int(cfg.get("lora_rank", 8))
LORA_ALPHA = int(cfg.get("lora_alpha", 32))
LORA_DROPOUT = float(cfg.get("lora_dropout", 0.05))

# Freeze flags
FREEZE_VISION = bool(cfg.get("freeze_vision_tower", False))
FREEZE_PROJECTOR = bool(cfg.get("freeze_multi_modal_projector", False))

bf16 = bool(cfg.get("bf16", True))
train_dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float16

print(f"[Config] YAML: {args.config}")
print(f"[Config] Quant={QBIT}-bit | LR={LR} | Epochs={EPOCHS} | BZ={BATCH_SIZE} | Accum={ACC_STEPS}")
print(f"[Config] LoRA: r={LORA_RANK}, alpha={LORA_ALPHA}, drop={LORA_DROPOUT}, target={LORA_TARGET}")
print(f"[Config] Freeze vision={FREEZE_VISION} | Freeze projector={FREEZE_PROJECTOR}")
print(f"[Paths] Model={MODEL_PATH} | Data={DATA_PATH} | MediaDir={MEDIA_DIR} | Out={OUTPUT_DIR}")

# Load base model under QLoRA
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=(QBIT == 4),
    load_in_8bit=(QBIT == 8),
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("[Model] Loading base model (quantized) ...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_cfg,
    trust_remote_code=True,
)

# Tokenizer / processor
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    image_processor = getattr(processor, "image_processor", AutoImageProcessor.from_pretrained(MODEL_PATH))
except Exception:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Apply freezing (base modules)
if hasattr(model, "vision_model"):
    for n, p in model.vision_model.named_parameters():
        p.requires_grad = not FREEZE_VISION
    print(f"[Freeze] Vision tower frozen: {FREEZE_VISION}")

if hasattr(model, "multi_modal_projector"):
    for n, p in model.multi_modal_projector.named_parameters():
        p.requires_grad = not FREEZE_PROJECTOR
    print(f"[Freeze] Projector frozen: {FREEZE_PROJECTOR}")


# Inject LoRA FIRST , then prepare model for k-bit training
lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET,
)
model = get_peft_model(model, lora_cfg)

# IMPORTANT: prepare AFTER LoRA so adapters remain trainable
model = prepare_model_for_kbit_training(model)

# Enable grad checkpointing and move to device
model.gradient_checkpointing_enable(use_reentrant=False)
model.to(device)
model.train()

# Verify trainable parameters 
total_params = sum(p.numel() for p in model.parameters())
trainable_params = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
num_trainable = sum(n for _, n in trainable_params)

print(f"[Params] Total params: {total_params:,}")
print(f"[Params] Trainable params: {num_trainable:,} "
      f"({100.0 * num_trainable / max(total_params,1):.2f}%)")
if num_trainable == 0:
    raise RuntimeError("[Fatal] No trainable parameters detected. Check LoRA injection and freeze flags.")


# Load Json Dataset
if not os.path.exists(DATA_PATH):
    sys.exit(f"[Error] Dataset JSONL not found: {DATA_PATH}")

print(f"[Dataset] Loading JSONL: {DATA_PATH}")
dataset = load_dataset("json", data_files=DATA_PATH, keep_in_memory=False)["train"]

def preprocess_fn(example):
    # Expecting: {"image": "rel_path.jpg", "conversations":[{"from":"human","value":"..."} , ...]}
    text = example["conversations"][0]["value"]
    img_path = os.path.join(MEDIA_DIR, example["image"])
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            pix = image_processor(im, return_tensors="pt")["pixel_values"].squeeze(0)
    except Exception as e:
        print(f"[Warn] Bad image {img_path}: {e}")
        pix = torch.zeros((3, 224, 224))
    toks = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    return {
        "input_ids": toks["input_ids"][0].tolist(),
        "attention_mask": toks["attention_mask"][0].tolist(),
        "pixel_values": pix.tolist(),
    }

dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)

def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([torch.tensor(b[k]) for b in batch], dim=0)
    return out

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


# Optimizer / AMP
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

print(f"[Train] Start training: epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}, accum={ACC_STEPS}")


# Training Loop
global_step = 0
model.train()
for epoch in range(EPOCHS):
    running = 0.0
    for step, batch in enumerate(train_loader):
        # Cast inputs to device; inputs don't need requires_grad=True
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        image_flags = torch.ones((input_ids.shape[0], 1), device=device, dtype=torch.long)

        with torch.autocast(device_type="cuda", dtype=train_dtype, enabled=(device=="cuda")):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_flags=image_flags,
                labels=input_ids,
                return_loss=True,  # crucial for InternVL training loss path
            )
            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise RuntimeError("[Fatal] Model.forward did not return a loss. Check inputs/labels.")
            loss = loss / max(ACC_STEPS, 1)

        scaler.scale(loss).backward()
        if (step + 1) % ACC_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        running += loss.item() * ACC_STEPS
        if global_step % 10 == 0:
            print(f"[Epoch {epoch+1}] GlobalStep {global_step} | Loss: {running / 10:.4f}")
            running = 0.0

print("[Train] Finished training.")

# Save model 
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[Save] Adapter + tokenizer saved to: {OUTPUT_DIR}")


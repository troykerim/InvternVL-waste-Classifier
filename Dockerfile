FROM nvcr.io/nvidia/pytorch:24.03-py3

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Python packages 
RUN pip install -U pip \
    && pip install \
        numpy==1.26.4 \
        pandas==2.2.2 \
        transformers==4.37.2 \
        datasets==2.18.0 \
        peft==0.10.0 \
        accelerate==0.27.2 \
        timm==0.9.12 \
        bitsandbytes==0.43.1 \
        sentencepiece==0.1.99 \
        safetensors==0.4.2 \
        Pillow==10.2.0 \
        einops==0.7.0 \
        huggingface_hub==0.20.2 \
        protobuf==4.25.8 \
        opencv-python-headless==4.9.0.80

# Hugging Face directories
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    TMPDIR=/workspace/tmp

# Working directory
WORKDIR /workspace

# Copy your project files into the image
COPY . /workspace

# Default command (overridden by YAML)
CMD ["bash"]


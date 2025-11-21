# Base Docker Image of verl, with CUDA/Torch/FlashAttn/Apex/TransformerEngine, without other frameworks
# Target: verlai/verl:base-v2-cu124-cudnn9.8-torch2.6-fa2.8.0-te2.3
# Start from the NVIDIA official image (ubuntu-24.04 + cuda-12.6 + python-3.12)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-11.html
FROM nvcr.io/nvidia/pytorch:24.11-py3

# Define environments
ENV MAX_JOBS=16
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Define installation arguments
ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Set apt source
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    { \
    echo "deb ${APT_SOURCE} jammy main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-updates main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-backports main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list

# Install systemctl
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd && \
    apt-get clean

# Install tini
RUN apt-get update && \
    apt-get install -y tini aria2 && \
    apt-get clean

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip

# Uninstall nv-pytorch fork
RUN pip uninstall -y torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    xgboost transformer_engine flash_attn apex megatron-core grpcio

# Reinstall CUDA 12.4
RUN aria2c https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin && \
    mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN aria2c --always-resume=true --max-tries=99999 https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4 && \
    rm cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    update-alternatives --set cuda /usr/local/cuda-12.4 && \
    rm -rf /usr/local/cuda-12.6

# Install torch
RUN pip install torch==2.6.0

# install other dependencies
RUN pip install --no-build-isolation \
    vllm==0.8.5.post1 \
    accelerate==1.7.0 \
    "Tapeagents[finetune]==0.1.16" \
    transformers==4.51.1 \
    flash-attn==2.7.4.post1 \
    ring-flash-attn==0.1.6 \
    "math-verify[antlr4_9_3]==0.7.0" \
    orjson==3.10.16 \
    redis==5.2.1 \
    hf_transfer==0.1.9 \
    hydra-core>=1.3.2

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

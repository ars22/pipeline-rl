#!/bin/bash

# PipelineRL Installation Script
# # Use default version
# ./install.sh

# # Use custom version
# VLLM_VERSION=0.11.0 ./install.sh

# Configuration
VLLM_VERSION=${VLLM_VERSION:-0.8.5.post1}

set -e  # Exit on any error

echo "🚀 Starting PipelineRL installation..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   Follow the UV Installation Guide: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "📦 Updating uv..."
uv self update

echo "🐍 Creating Python virtual environment..."
uv venv prl --python 3.11

echo "🔧 Activating virtual environment and upgrading pip..."
source prl/bin/activate
uv pip install --upgrade pip
uv pip install setuptools

echo "⚡ Installing vLLM ${VLLM_VERSION}..."
uv pip install vllm==${VLLM_VERSION}

echo "📚 Installing PipelineRL dependencies..."
uv pip install -e .

echo "🧪 Running installation tests..."
python -c "
import sys
try:
    from vllm import LLM
    print('✅ vLLM import successful')
except ImportError as e:
    print(f'❌ vLLM import failed: {e}')
    sys.exit(1)

try:
    import torch
    if torch.cuda.is_available():
        tensor = torch.tensor([1, 2, 3]).cuda()
        print(f'✅ PyTorch CUDA tensor creation successful: {tensor}')
        print(f'✅ CUDA device: {torch.cuda.get_device_name()}')
    else:
        tensor = torch.tensor([1, 2, 3])
        print(f'⚠️  CUDA not available, created CPU tensor: {tensor}')
except Exception as e:
    print(f'❌ PyTorch tensor creation failed: {e}')
    sys.exit(1)

print('🎉 All tests passed!')
"

echo "✅ Installation complete!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "   source prl/bin/activate"
echo ""
echo "💡 Tips for Hugging Face cluster users:"
echo "   - Add 'export UV_LINK_MODE=copy' to your ~/.bashrc"
echo "   - Add 'module load cuda/12.9' to your ~/.bashrc"

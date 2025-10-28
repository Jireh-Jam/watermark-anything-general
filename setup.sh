#!/bin/bash
# Setup script for Watermark Anything Tool

echo "🐤 Watermark Anything Tool Setup"
echo "================================"
echo ""

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python version: $python_version"

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. It's recommended to use one:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
    read -p "Continue without virtual environment? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Please create a virtual environment first."
        exit 1
    fi
fi

# Install PyTorch
echo ""
echo "Installing PyTorch..."
echo "Choose your setup:"
echo "1. CUDA 12.4 (GPU support)"
echo "2. CPU only"
read -p "Enter choice (1 or 2): " cuda_choice

if [ "$cuda_choice" = "1" ]; then
    echo "Installing PyTorch with CUDA 12.4..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo ""
echo "Installing other requirements..."
pip install -r requirements.txt

# Download model weights
echo ""
echo "Downloading model weights..."
python3 download_model.py

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python3 -c "import gradio; print(f'✓ Gradio {gradio.__version__} installed')"
python3 -c "from watermark_utils import WatermarkProcessor; print('✓ Watermark modules working')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Quick start:"
echo "  - Embed watermark: python3 watermark_tool.py embed -i image.jpg -o watermarked.jpg"
echo "  - Detect watermark: python3 watermark_tool.py detect -i watermarked.jpg"
echo "  - Launch GUI: python3 watermark_tool.py gui"
echo "  - Run tests: python3 test_watermark_tool.py"
echo ""
echo "See WATERMARK_TOOL_README.md for full documentation."
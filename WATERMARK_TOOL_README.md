# 🐤 Watermark Anything Tool - Complete Implementation

A comprehensive command-line and GUI tool for embedding and detecting invisible watermarks in images using the Watermark Anything model.

## Features

- **Single Watermark Embedding**: Embed invisible watermarks with custom messages
- **Multiple Watermarks**: Embed multiple watermarks in different regions of an image
- **Watermark Detection**: Detect and decode embedded watermarks
- **Batch Processing**: Process multiple images at once
- **Custom Masks**: Define specific regions for watermark placement
- **GUI Interface**: User-friendly Gradio web interface
- **Robustness Testing**: Test watermark resilience against transformations
- **Quality Metrics**: PSNR, SSIM, and MSE calculations

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
pip install gradio  # For GUI interface
```

2. **Download Model Weights**:
```bash
python download_model.py
```
Or manually download from:
- MIT License: https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth
- Original (CC-BY-NC): https://dl.fbaipublicfiles.com/watermark_anything/wam_coco.pth

Save to: `checkpoints/checkpoint.pth`

## Usage

### Command Line Interface

#### Basic Watermark Embedding
```bash
# Embed random watermark
python watermark_tool.py embed -i input.jpg -o watermarked.jpg

# Embed text message
python watermark_tool.py embed -i input.jpg -o watermarked.jpg -m "Hello"

# Embed binary message
python watermark_tool.py embed -i input.jpg -o watermarked.jpg -m "01010101010101010101010101010101"

# Embed with custom strength (higher = more robust but less invisible)
python watermark_tool.py embed -i input.jpg -o watermarked.jpg --strength 3.0

# Embed in specific region (50% of image)
python watermark_tool.py embed -i input.jpg -o watermarked.jpg -p 0.5
```

#### Watermark Detection
```bash
# Detect watermark
python watermark_tool.py detect -i watermarked.jpg

# Detect and save results
python watermark_tool.py detect -i watermarked.jpg -o results/

# Try to decode as text
python watermark_tool.py detect -i watermarked.jpg --decode-text
```

#### Batch Processing
```bash
# Batch embed watermarks
python watermark_tool.py batch-embed -i input_dir/ -o output_dir/ -m "Copyright 2024"

# Batch detect watermarks
python watermark_tool.py batch-detect -i input_dir/ -o results_dir/

# Process recursively
python watermark_tool.py batch-embed -i input_dir/ -o output_dir/ -m "Watermark" --recursive
```

#### Multiple Watermarks
```bash
# Embed multiple messages in one image
python watermark_tool.py multi-embed -i input.jpg -o multi_watermarked.jpg \
    -m "First" "Second" "01010101010101010101010101010101"
```

#### Verify Watermark
```bash
# Compare original and watermarked images
python watermark_tool.py verify -o original.jpg -w watermarked.jpg

# Verify specific message
python watermark_tool.py verify -o original.jpg -w watermarked.jpg -m "Expected Message"
```

### GUI Interface

Launch the web interface:
```bash
python watermark_tool.py gui

# Custom port
python watermark_tool.py gui --port 8080

# Create public link
python watermark_tool.py gui --share
```

Or run directly:
```bash
python watermark_gui.py
```

### Python API

```python
from watermark_utils import WatermarkProcessor, encode_text_to_binary

# Initialize processor
processor = WatermarkProcessor()

# Embed watermark
from PIL import Image
image = Image.open("input.jpg")
message = encode_text_to_binary("Hello World")

result = processor.embed_watermark(
    image,
    message,
    mask_proportion=0.8,
    watermark_strength=2.5
)

# Save watermarked image
result['watermarked_image'].save("watermarked.jpg")
print(f"PSNR: {result['psnr']:.2f} dB")
print(f"SSIM: {result['ssim']:.4f}")

# Detect watermark
detection = processor.detect_watermark("watermarked.jpg")
if detection['detected']:
    print(f"Confidence: {detection['confidence']:.2%}")
    print(f"Message: {detection['message_binary']}")
```

## Testing

Run the comprehensive test suite:
```bash
python test_watermark_tool.py
```

This will test:
- Message encoding/decoding
- Single and multiple watermark embedding
- Detection accuracy
- Robustness to transformations
- Batch processing
- Edge cases

## File Structure

```
watermark_tool.py          # Main CLI tool
watermark_utils.py         # Core watermarking utilities
watermark_gui.py           # Gradio GUI interface
test_watermark_tool.py     # Comprehensive test suite
download_model.py          # Model weight downloader
WATERMARK_TOOL_README.md   # This file
```

## Message Format

- **Binary**: 32-bit string of 0s and 1s
- **Text**: Automatically encoded to binary (limited to ~4 characters due to 32-bit constraint)
- **Random**: Randomly generated 32-bit message

## Watermark Strength

The `strength` parameter (default: 2.0) controls the trade-off between:
- **Lower values (0.5-1.5)**: More invisible but less robust
- **Higher values (2.5-5.0)**: More robust but potentially visible

## Custom Masks

Create custom watermark regions:
```python
from watermark_utils import create_custom_mask

# Center region
mask = create_custom_mask((512, 512), 'center', size=0.5)

# Corner region
mask = create_custom_mask((512, 512), 'corner', corner='top_left', size=0.25)

# Ring pattern
mask = create_custom_mask((512, 512), 'ring', outer_radius=0.4, inner_radius=0.2)
```

## Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better (typically >30 dB)
- **SSIM (Structural Similarity Index)**: Closer to 1 is better (typically >0.9)
- **MSE (Mean Squared Error)**: Lower is better

## Limitations

- Messages limited to 32 bits (~4 ASCII characters for text)
- Detection accuracy depends on image quality and transformations
- Very small images (<64x64) may have reduced accuracy
- Heavy compression or modifications may affect detection

## Troubleshooting

1. **Model not found**: Run `python download_model.py` first
2. **CUDA out of memory**: The tool automatically falls back to CPU
3. **Import errors**: Ensure all requirements are installed
4. **Detection failures**: Try increasing watermark strength during embedding

## Examples

See the `assets/images/` directory for sample images to test with.

## License

The code follows the license of the Watermark Anything repository (MIT for code, model-specific licenses for weights).
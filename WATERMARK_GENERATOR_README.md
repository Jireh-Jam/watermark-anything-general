# 🌊 Watermark Generator for Watermark Anything Repository

A comprehensive watermarking solution that extends the existing Watermark Anything infrastructure with practical, easy-to-use functionality for both invisible and visible watermarking.

## 🚀 Features

### 🔒 Invisible Watermarking (AI-Powered)
- **Deep Learning-Based**: Uses the Watermark Anything Model (WAM) for robust invisible watermarks
- **Localized Embedding**: Watermark specific regions of images
- **Multiple Messages**: Detect and embed multiple watermarks in a single image
- **Robustness**: Resistant to compression, scaling, and other attacks
- **32-bit Messages**: Support for complex message encoding

### 📝 Visible Text Watermarking
- **Flexible Positioning**: Top-left, top-right, bottom-left, bottom-right, center
- **Customizable Appearance**: Font size, opacity, color, custom fonts
- **Professional Quality**: Anti-aliased text with transparency support

### 🖼️ Image/Logo Watermarking
- **Logo Overlay**: Add company logos or image watermarks
- **Smart Scaling**: Automatic size adjustment based on base image
- **Transparency Support**: Full RGBA support with opacity control
- **Multiple Formats**: Support for PNG, JPG, and other common formats

### 📦 Batch Processing
- **Directory Processing**: Watermark entire folders of images
- **Multiple Types**: Batch invisible, text, or image watermarks
- **Progress Tracking**: Detailed success/failure reporting
- **Flexible Output**: Organized output directory structure

## 📋 Requirements

### System Requirements
- Python 3.8+
- PIL/Pillow for image processing
- NumPy for numerical operations

### For Invisible Watermarking (Optional)
- PyTorch 2.0+
- CUDA (recommended for GPU acceleration)
- Watermark Anything Model checkpoint

### Installation

1. **Basic Dependencies** (for visible watermarking):
   ```bash
   pip install Pillow numpy
   ```

2. **Full Dependencies** (for all features):
   ```bash
   pip install -r requirements.txt
   ```

3. **Download WAM Model** (for invisible watermarking):
   ```bash
   wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/
   ```

## 🎯 Quick Start

### Command Line Interface

#### Invisible Watermarks
```bash
# Embed invisible watermark
python watermark_cli.py embed -i photo.jpg -o watermarked.jpg --message "SECRET123"

# Detect watermark
python watermark_cli.py detect -i watermarked.jpg --expected-message "SECRET123"

# Detect multiple watermarks
python watermark_cli.py detect-multiple -i multi_watermarked.jpg
```

#### Text Watermarks
```bash
# Add text watermark
python watermark_cli.py text -i photo.jpg -o watermarked.jpg --text "© 2024 Company" --position bottom-right

# Batch text watermarks
python watermark_cli.py batch-text --input-dir ./photos --output-dir ./watermarked --text "© 2024"
```

#### Logo Watermarks
```bash
# Add logo watermark
python watermark_cli.py logo -i photo.jpg -o watermarked.jpg --logo company_logo.png --scale 0.1

# Batch logo watermarks
python watermark_cli.py batch-logo --input-dir ./photos --output-dir ./watermarked --logo logo.png
```

### Python API

```python
from watermark_generator import WatermarkGenerator

# Initialize
wm_gen = WatermarkGenerator()

# Invisible watermark
result = wm_gen.embed_invisible_watermark(
    image_path="photo.jpg",
    message="SECRET123",
    output_path="watermarked.jpg",
    mask_percentage=0.5
)

# Text watermark
result = wm_gen.add_text_watermark(
    image_path="photo.jpg",
    text="© 2024 Company",
    output_path="watermarked.jpg",
    position="bottom-right",
    opacity=0.7
)

# Image watermark
result = wm_gen.add_image_watermark(
    image_path="photo.jpg",
    watermark_path="logo.png",
    output_path="watermarked.jpg",
    scale=0.1,
    opacity=0.6
)
```

## 📖 Detailed Usage

### Invisible Watermarking Parameters

| Parameter | Description | Default | Range/Options |
|-----------|-------------|---------|---------------|
| `message` | Message to embed | Required | String, list, or tensor |
| `mask_percentage` | Image area to watermark | 0.5 | 0.0 - 1.0 |
| `scaling_factor` | Watermark strength | 2.0 | 1.0 - 4.0+ |

**Tips:**
- Higher `scaling_factor` = more robust but potentially visible
- `mask_percentage` 0.3-0.7 gives good balance
- Messages are automatically converted to 32-bit format

### Text Watermarking Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `text` | Watermark text | Required | Any string |
| `position` | Text placement | "bottom-right" | "top-left", "top-right", "bottom-left", "bottom-right", "center" |
| `font_size` | Text size | 36 | Any integer |
| `opacity` | Transparency | 0.5 | 0.0 - 1.0 |
| `color` | Text color | (255,255,255) | RGB tuple |
| `font_path` | Custom font | None | Path to TTF/OTF file |

### Image Watermarking Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `scale` | Logo size ratio | 0.1 | 0.01 - 1.0 |
| `opacity` | Transparency | 0.5 | 0.0 - 1.0 |
| `position` | Logo placement | "bottom-right" | Same as text |

## 🔧 Advanced Usage

### Custom Message Encoding
```python
# Binary message
binary_msg = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 
              0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]

# Torch tensor
import torch
tensor_msg = torch.randint(0, 2, (32,)).float()

# Both work with embed_invisible_watermark()
```

### Robustness Testing
```python
# High robustness watermark
result = wm_gen.embed_invisible_watermark(
    image_path="photo.jpg",
    message="ROBUST",
    output_path="watermarked.jpg",
    mask_percentage=0.6,
    scaling_factor=3.0  # Higher for more robustness
)
```

### Multiple Watermark Detection
```python
# Detect multiple watermarks using DBSCAN clustering
result = wm_gen.detect_multiple_watermarks(
    image_path="multi_watermarked.jpg",
    epsilon=1.0,        # Clustering distance
    min_samples=500     # Minimum cluster size
)

print(f"Found {result['num_watermarks_detected']} watermarks")
for msg in result['detected_messages']:
    print(f"Message: {msg}")
```

### Batch Processing with Custom Settings
```python
# Batch process with custom parameters
results = wm_gen.batch_watermark(
    input_dir="./photos",
    output_dir="./watermarked", 
    watermark_type="text",
    text="© 2024 Custom",
    position="top-left",
    font_size=28,
    opacity=0.8,
    color=(255, 255, 0)  # Yellow text
)

# Check results
successful = sum(1 for r in results if r['success'])
print(f"Processed {successful}/{len(results)} images successfully")
```

## 📊 Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- **> 40 dB**: Excellent quality, imperceptible changes
- **30-40 dB**: Good quality, minimal visible changes  
- **20-30 dB**: Acceptable quality, some visible changes
- **< 20 dB**: Poor quality, clearly visible changes

### Bit Accuracy (for invisible watermarks)
- **> 0.95**: Excellent detection, watermark fully recoverable
- **0.85-0.95**: Good detection, minor errors
- **0.70-0.85**: Acceptable detection, some corruption
- **< 0.70**: Poor detection, significant corruption

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test basic watermarking functionality
python test_watermarks.py

# Test CLI and code structure
python test_cli.py
```

## 📁 File Structure

```
├── watermark_generator.py      # Main watermarking class
├── watermark_cli.py           # Command-line interface
├── test_watermarks.py         # Functionality tests
├── test_cli.py               # CLI and structure tests
├── examples/
│   ├── basic_usage.py        # Basic usage examples
│   ├── advanced_usage.py     # Advanced techniques
│   └── README.md            # Detailed examples documentation
└── WATERMARK_GENERATOR_README.md  # This file
```

## 🎨 Examples Gallery

The generated watermarks demonstrate various capabilities:

### Invisible Watermarking
- Embeds 32-bit messages invisibly
- Maintains high image quality (PSNR > 35 dB)
- Survives JPEG compression and minor edits
- Supports localized watermarking

### Text Watermarking  
- Professional copyright notices
- Multiple positions and styles
- Custom fonts and colors
- Opacity control for subtlety

### Logo Watermarking
- Company branding
- Scalable logo placement
- Transparency support
- Multiple image formats

## 🔍 Troubleshooting

### Common Issues

**"WAM model not available"**
- Download the model checkpoint: `wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/`
- Ensure PyTorch is installed: `pip install torch torchvision`

**"Font not found" warnings**
- Install system fonts or specify custom font path
- Default fonts are used as fallback

**Low detection accuracy**
- Increase `scaling_factor` for more robustness
- Ensure sufficient `mask_percentage` coverage
- Check for image compression or modifications

**Memory errors**
- Use CPU instead of GPU: `--device cpu`
- Process smaller images or reduce batch size
- Ensure sufficient system RAM

### Performance Tips

1. **GPU Acceleration**: Use `--device cuda` for invisible watermarking
2. **Batch Processing**: Process multiple images together for efficiency
3. **Image Size**: Consider resizing very large images before processing
4. **Memory Management**: Close other applications when processing large batches

## 🤝 Integration

### Web Applications
```python
from flask import Flask, request, send_file
from watermark_generator import WatermarkGenerator

app = Flask(__name__)
wm_gen = WatermarkGenerator()

@app.route('/watermark', methods=['POST'])
def add_watermark():
    # Handle file upload and watermarking
    # Return watermarked image
    pass
```

### Batch Scripts
```bash
#!/bin/bash
# Batch watermark all images in a directory
for dir in /path/to/image/dirs/*; do
    python watermark_cli.py batch-text \
        --input-dir "$dir" \
        --output-dir "$dir/watermarked" \
        --text "© 2024 Company"
done
```

## 📄 License

This watermarking code is provided under the MIT License. The underlying Watermark Anything model and repository maintain their original licenses:

- **New watermarking code**: MIT License
- **WAM model (SA-1B trained)**: MIT License  
- **WAM model (COCO trained)**: CC-BY-NC License
- **Original WAM codebase**: See original LICENSE files

## 🙏 Acknowledgments

This watermarking system builds upon the excellent work of the Watermark Anything team:

- **Watermark Anything Paper**: [arXiv:2411.07231](https://arxiv.org/abs/2411.07231)
- **Original Repository**: [facebookresearch/watermark-anything](https://github.com/facebookresearch/watermark-anything)
- **Authors**: Tom Sander, Pierre Fernandez, Alain Durmus, Teddy Furon, Matthijs Douze

## 🔗 Related Projects

- **VideoSeal**: Video watermarking - [facebookresearch/videoseal](https://github.com/facebookresearch/videoseal)
- **AudioSeal**: Audio watermarking - [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal)
- **Segment Anything**: Foundation model - [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

---

**🎉 Happy Watermarking!** 

For questions, issues, or contributions, please refer to the original Watermark Anything repository or create issues in your project repository.
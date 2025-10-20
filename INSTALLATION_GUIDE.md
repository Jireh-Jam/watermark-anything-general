# 🚀 Watermark Generator Installation Guide

This guide will help you set up and use the comprehensive watermarking system for the Watermark Anything repository.

## 📦 What's Included

Your watermarking system includes:

### Core Files
- **`watermark_generator.py`** - Main WatermarkGenerator class with all functionality
- **`watermark_cli.py`** - User-friendly command-line interface
- **`WATERMARK_GENERATOR_README.md`** - Complete documentation

### Testing & Validation
- **`test_watermarks.py`** - Comprehensive functionality tests
- **`test_cli.py`** - CLI and code structure validation

### Examples & Documentation
- **`examples/basic_usage.py`** - Simple usage examples
- **`examples/advanced_usage.py`** - Advanced techniques and parameters
- **`examples/README.md`** - Detailed examples documentation
- **`INSTALLATION_GUIDE.md`** - This installation guide

## 🔧 Installation Options

### Option 1: Basic Installation (Text & Image Watermarks Only)

Perfect for visible watermarking without AI features:

```bash
# Install basic dependencies
pip install Pillow numpy

# Test basic functionality
python test_watermarks.py
```

**Features Available:**
- ✅ Text watermarking with custom fonts, colors, positions
- ✅ Image/logo watermarking with transparency
- ✅ Batch processing for multiple images
- ✅ Command-line interface
- ❌ Invisible AI-powered watermarks (requires full installation)

### Option 2: Full Installation (All Features)

For complete functionality including invisible watermarks:

```bash
# Install all dependencies
pip install -r requirements.txt

# Download the WAM model (MIT license version)
wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/

# Test all functionality
python test_watermarks.py
python test_cli.py
```

**Features Available:**
- ✅ All basic features from Option 1
- ✅ Invisible AI-powered watermarks
- ✅ Multiple watermark detection
- ✅ Robustness testing against attacks
- ✅ Custom message encoding (binary, tensor)

## 🎯 Quick Verification

### Test Basic Functionality
```bash
# Create a test image and add text watermark
python -c "
from watermark_generator import WatermarkGenerator
from PIL import Image, ImageDraw

# Create test image
img = Image.new('RGB', (400, 300), 'lightblue')
draw = ImageDraw.Draw(img)
draw.text((200, 150), 'TEST', fill='black', anchor='mm')
img.save('test.jpg')

# Add watermark
wm = WatermarkGenerator()
result = wm.add_text_watermark('test.jpg', '© 2024', 'watermarked.jpg')
print('✅ Success!' if result['success'] else '❌ Failed')
"
```

### Test CLI Interface
```bash
# Show help
python watermark_cli.py --help

# Test text watermark command
python watermark_cli.py text --help
```

### Test Invisible Watermarks (Full Installation Only)
```bash
# Test with sample image
python -c "
from watermark_generator import WatermarkGenerator
wm = WatermarkGenerator()
if wm.wam_model:
    print('✅ Invisible watermarking available')
else:
    print('ℹ️ Invisible watermarking not available (model not loaded)')
"
```

## 🎨 First Usage Examples

### 1. Add Text Watermark (CLI)
```bash
# Create a test image first
python -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (600, 400), 'skyblue')
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 550, 350], fill='lightgreen', outline='darkgreen', width=3)
draw.text((300, 200), 'My Photo', fill='darkblue', anchor='mm')
img.save('my_photo.jpg')
print('✅ Test image created: my_photo.jpg')
"

# Add watermark
python watermark_cli.py text \
    -i my_photo.jpg \
    -o watermarked_photo.jpg \
    --text "© 2024 My Company" \
    --position bottom-right \
    --opacity 0.7
```

### 2. Add Logo Watermark (CLI)
```bash
# Create a test logo
python -c "
from PIL import Image, ImageDraw
logo = Image.new('RGBA', (120, 60), (0, 0, 0, 0))
draw = ImageDraw.Draw(logo)
draw.rectangle([5, 5, 115, 55], fill=(255, 0, 0, 180), outline=(255, 255, 255, 255), width=2)
draw.text((60, 30), 'LOGO', fill=(255, 255, 255, 255), anchor='mm')
logo.save('my_logo.png')
print('✅ Test logo created: my_logo.png')
"

# Add logo watermark
python watermark_cli.py logo \
    -i my_photo.jpg \
    -o logo_watermarked.jpg \
    --logo my_logo.png \
    --scale 0.15 \
    --position top-left \
    --opacity 0.8
```

### 3. Invisible Watermark (Full Installation)
```bash
# Only works with full installation
python watermark_cli.py embed \
    -i my_photo.jpg \
    -o invisible_watermarked.jpg \
    --message "SECRET123" \
    --mask-percentage 0.5

# Detect the watermark
python watermark_cli.py detect \
    -i invisible_watermarked.jpg \
    --expected-message "SECRET123"
```

### 4. Batch Processing
```bash
# Create a directory with test images
mkdir test_images
python -c "
from PIL import Image, ImageDraw
import os

for i in range(3):
    img = Image.new('RGB', (400, 300), f'C{i*50}')
    draw = ImageDraw.Draw(img)
    draw.text((200, 150), f'Image {i+1}', fill='white', anchor='mm')
    img.save(f'test_images/image_{i+1}.jpg')

print('✅ Created 3 test images in test_images/')
"

# Batch watermark all images
python watermark_cli.py batch-text \
    --input-dir test_images \
    --output-dir watermarked_batch \
    --text "© 2024 Batch Processed" \
    --position bottom-right
```

## 🎓 Learning Path

### Beginner (Start Here)
1. **Run the tests**: `python test_watermarks.py`
2. **Try basic CLI**: `python watermark_cli.py text --help`
3. **Read examples**: `examples/README.md`
4. **Run basic examples**: `python examples/basic_usage.py`

### Intermediate
1. **Explore API**: Study `watermark_generator.py`
2. **Try advanced examples**: `python examples/advanced_usage.py`
3. **Experiment with parameters**: Different positions, opacities, scales
4. **Batch processing**: Process your own image collections

### Advanced (Full Installation)
1. **Invisible watermarks**: Embed and detect secret messages
2. **Multiple watermarks**: Use DBSCAN clustering for detection
3. **Robustness testing**: Test against compression and attacks
4. **Custom integration**: Build into your own applications

## 🔧 Customization

### Add Your Own Fonts
```python
# Use custom fonts for text watermarks
result = wm_gen.add_text_watermark(
    image_path="photo.jpg",
    text="Custom Font Text",
    output_path="custom_font.jpg",
    font_path="/path/to/your/font.ttf",
    font_size=32
)
```

### Custom Colors and Styles
```python
# Colorful watermarks
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
]

for i, color in enumerate(colors):
    wm_gen.add_text_watermark(
        image_path="photo.jpg",
        text=f"Color {i+1}",
        output_path=f"colored_{i+1}.jpg",
        color=color,
        opacity=0.8
    )
```

### Integration with Your Code
```python
# Example: Web application integration
from flask import Flask, request, send_file
from watermark_generator import WatermarkGenerator
import tempfile
import os

app = Flask(__name__)
wm_gen = WatermarkGenerator()

@app.route('/add_watermark', methods=['POST'])
def add_watermark():
    # Get uploaded file
    file = request.files['image']
    text = request.form.get('text', '© 2024')
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
        file.save(tmp_input.name)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_output:
            # Add watermark
            result = wm_gen.add_text_watermark(
                tmp_input.name, text, tmp_output.name
            )
            
            if result['success']:
                return send_file(tmp_output.name, as_attachment=True)
            else:
                return "Watermarking failed", 500
```

## 🚨 Troubleshooting

### Common Issues and Solutions

**Issue**: "ModuleNotFoundError: No module named 'torch'"
```bash
# Solution: Install PyTorch
pip install torch torchvision
# Or use basic installation for text/image watermarks only
```

**Issue**: "WAM model not available"
```bash
# Solution: Download the model
wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/
```

**Issue**: Font warnings or default fonts
```bash
# Solution: Install system fonts or specify font path
sudo apt-get install fonts-dejavu-core  # Ubuntu/Debian
# Or specify custom font in your code
```

**Issue**: Low image quality after watermarking
```python
# Solution: Adjust parameters
result = wm_gen.add_text_watermark(
    # ... other parameters ...
    opacity=0.5,  # Lower opacity for subtlety
    font_size=24  # Smaller font size
)
```

**Issue**: Invisible watermark detection fails
```python
# Solution: Increase robustness
result = wm_gen.embed_invisible_watermark(
    # ... other parameters ...
    scaling_factor=3.0,      # Higher for more robustness
    mask_percentage=0.6      # More coverage area
)
```

## 🎉 You're Ready!

Congratulations! You now have a fully functional watermarking system. Here's what you can do:

### ✅ Immediate Actions
- [ ] Run `python test_watermarks.py` to verify installation
- [ ] Try the CLI examples above
- [ ] Experiment with your own images
- [ ] Read the full documentation in `WATERMARK_GENERATOR_README.md`

### 🚀 Next Steps
- [ ] Integrate into your workflow or application
- [ ] Explore advanced features like invisible watermarks
- [ ] Set up batch processing for your image collections
- [ ] Customize for your specific use case

### 📚 Resources
- **Full Documentation**: `WATERMARK_GENERATOR_README.md`
- **Examples**: `examples/` directory
- **Original Paper**: [Watermark Anything (arXiv:2411.07231)](https://arxiv.org/abs/2411.07231)
- **Original Repository**: [facebookresearch/watermark-anything](https://github.com/facebookresearch/watermark-anything)

**Happy Watermarking! 🌊**
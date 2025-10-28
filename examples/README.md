# Watermark Generator Examples

This directory contains comprehensive examples demonstrating how to use the Watermark Generator for various watermarking tasks.

## Files Overview

- **`basic_usage.py`** - Simple examples covering all main features
- **`advanced_usage.py`** - Advanced techniques and parameter tuning
- **`README.md`** - This documentation file

## Quick Start

### Prerequisites

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -r ../requirements.txt
   ```

2. For invisible watermarking features, download the WAM model:
   ```bash
   wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P ../checkpoints/
   ```

### Running Examples

#### Basic Examples
```bash
cd examples
python basic_usage.py
```

#### Advanced Examples
```bash
cd examples
python advanced_usage.py
```

## Example Categories

### 1. Invisible Watermarking (Requires WAM Model)

**Embedding:**
```python
from watermark_generator import WatermarkGenerator

wm_gen = WatermarkGenerator()
result = wm_gen.embed_invisible_watermark(
    image_path="input.jpg",
    message="SECRET123",
    output_path="watermarked.jpg",
    mask_percentage=0.5
)
```

**Detection:**
```python
result = wm_gen.detect_invisible_watermark(
    image_path="watermarked.jpg",
    expected_message="SECRET123"
)
print(f"Detected: {result['detected_message']}")
print(f"Accuracy: {result['bit_accuracy']:.3f}")
```

### 2. Text Watermarking

```python
result = wm_gen.add_text_watermark(
    image_path="input.jpg",
    text="© 2024 Company",
    output_path="watermarked.jpg",
    position="bottom-right",
    font_size=24,
    opacity=0.7,
    color=(255, 255, 255)
)
```

### 3. Image/Logo Watermarking

```python
result = wm_gen.add_image_watermark(
    image_path="input.jpg",
    watermark_path="logo.png",
    output_path="watermarked.jpg",
    position="top-left",
    scale=0.1,
    opacity=0.6
)
```

### 4. Batch Processing

```python
results = wm_gen.batch_watermark(
    input_dir="./images",
    output_dir="./watermarked",
    watermark_type="text",
    text="© 2024 Batch Processed"
)
```

## Command Line Usage

The examples also demonstrate CLI usage through `watermark_cli.py`:

### Invisible Watermarks
```bash
# Embed
python ../watermark_cli.py embed -i input.jpg -o output.jpg --message "SECRET"

# Detect
python ../watermark_cli.py detect -i watermarked.jpg --expected-message "SECRET"

# Detect multiple
python ../watermark_cli.py detect-multiple -i multi_watermarked.jpg
```

### Text Watermarks
```bash
python ../watermark_cli.py text -i input.jpg -o output.jpg --text "© 2024" --position bottom-right
```

### Logo Watermarks
```bash
python ../watermark_cli.py logo -i input.jpg -o output.jpg --logo company_logo.png --scale 0.1
```

### Batch Processing
```bash
# Batch text watermarks
python ../watermark_cli.py batch-text --input-dir ./images --output-dir ./output --text "© 2024"

# Batch logo watermarks
python ../watermark_cli.py batch-logo --input-dir ./images --output-dir ./output --logo logo.png
```

## Parameter Explanations

### Invisible Watermarking Parameters

- **`message`**: Text, binary list, or tensor to embed
- **`mask_percentage`**: Portion of image to watermark (0.0-1.0)
- **`scaling_factor`**: Watermark strength (higher = more robust but visible)

### Text Watermarking Parameters

- **`position`**: "top-left", "top-right", "bottom-left", "bottom-right", "center"
- **`font_size`**: Size of the text font
- **`opacity`**: Transparency level (0.0-1.0)
- **`color`**: RGB color tuple (e.g., (255, 255, 255) for white)

### Image Watermarking Parameters

- **`scale`**: Size relative to base image (0.0-1.0)
- **`opacity`**: Transparency level (0.0-1.0)
- **`position`**: Same as text watermarking

## Advanced Features Demonstrated

### 1. Robustness Testing
- Testing watermark survival under compression
- Different scaling factors for robustness vs. quality trade-offs

### 2. Multiple Watermark Detection
- DBSCAN clustering to find multiple embedded messages
- Parameter tuning for detection sensitivity

### 3. Custom Message Encoding
- Binary arrays
- Torch tensors
- String conversion and padding

### 4. Error Handling
- Graceful handling of missing files
- Invalid parameter validation
- Comprehensive error reporting

## Output Files

After running the examples, you'll find various output files:

- `watermarked_*.jpg` - Images with different types of watermarks
- `batch_watermarked/` - Directory with batch-processed images
- Various test files demonstrating different parameters

## Tips for Best Results

### Invisible Watermarking
- Use `mask_percentage` 0.3-0.7 for good balance
- Higher `scaling_factor` (2.0-4.0) for robustness
- Lower `scaling_factor` (1.0-2.0) for imperceptibility

### Text Watermarking
- Use opacity 0.5-0.8 for visibility without being intrusive
- Choose contrasting colors for better readability
- Consider image content when choosing position

### Image Watermarking
- Scale 0.05-0.15 works well for most logos
- Use semi-transparent logos (opacity 0.4-0.7)
- Position in corners to avoid covering main content

## Troubleshooting

### WAM Model Issues
If invisible watermarking doesn't work:
1. Check if model files exist in `checkpoints/`
2. Verify CUDA availability for GPU acceleration
3. Ensure sufficient memory for model loading

### Font Issues
If text watermarking shows default fonts:
1. Install system fonts or specify custom font path
2. Use the `--font-path` parameter in CLI
3. Fallback fonts are automatically used if custom fonts fail

### Performance Tips
- Use GPU (`--device cuda`) for invisible watermarking
- Process smaller batches if memory is limited
- Consider resizing very large images before processing

## Next Steps

After running these examples:
1. Experiment with different parameters
2. Test on your own images
3. Integrate the code into your applications
4. Explore the advanced robustness testing features
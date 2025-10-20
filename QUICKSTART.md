# 🚀 Quick Start Guide - Watermark Generation

This is a quick guide to get you started with watermarking images in this repository.

## Prerequisites

1. **Python 3.10+** (tested with 3.10.14)
2. **PyTorch 2.0+** with CUDA support (optional, but recommended)
3. **Dependencies** from requirements.txt

## Installation

### Step 1: Set up Python environment

```bash
# Create conda environment (recommended)
conda create -n watermark_anything python=3.10.14
conda activate watermark_anything

# Install PyTorch with CUDA (for GPU acceleration)
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# OR install PyTorch CPU-only version
# pip install torch torchvision
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download model weights

The script will automatically download the MIT-licensed model on first use, or you can download manually:

```bash
python3 generate_watermark.py download
```

## 5-Minute Tutorial

### 1️⃣ Embed a Watermark (Single Image)

```bash
python3 generate_watermark.py embed \
    assets/images/alpaca.jpg \
    outputs/alpaca_watermarked.png
```

**Output:**
```
✓ Watermark embedded successfully!
  Output: outputs/alpaca_watermarked.png
  Message: 10110100101010010111000110101010
  PSNR: 42.35 dB
```

### 2️⃣ Detect a Watermark

```bash
python3 generate_watermark.py detect \
    outputs/alpaca_watermarked.png \
    --output-dir outputs/
```

**Output:**
```
✓ Watermark detected!
  Detected message: 10110100101010010111000110101010
  Mask coverage: 98.45%
```

### 3️⃣ Batch Process Multiple Images

```bash
python3 generate_watermark.py embed \
    assets/images/ \
    outputs/batch/ \
    --batch
```

**Output:**
```
Processing 5 images...
[1/5] Processing alpaca.jpg...
  ✓ Saved to outputs/batch/watermarked_alpaca.jpg
  Message: 11001100110011001100110011001100
  PSNR: 41.23 dB
...
✓ Successfully processed 5 images
```

### 4️⃣ Use a Custom Message

```bash
python3 generate_watermark.py embed \
    assets/images/ducks.jpg \
    outputs/ducks_custom.png \
    --message "10101010101010101010101010101010"
```

### 5️⃣ Partial Watermarking (50% of image)

```bash
python3 generate_watermark.py embed \
    assets/images/trex_bike.jpg \
    outputs/trex_partial.png \
    --mask-percentage 0.5
```

## Python API - Quick Examples

### Example 1: Basic Usage

```python
from generate_watermark import WatermarkGenerator

# Initialize
generator = WatermarkGenerator()

# Embed watermark
result = generator.embed_watermark(
    image_path="assets/images/alpaca.jpg",
    output_path="outputs/alpaca_wm.png"
)

print(f"Message: {result['message_str']}")
print(f"PSNR: {result['psnr']:.2f} dB")
```

### Example 2: Embed and Verify

```python
import torch
from generate_watermark import WatermarkGenerator

generator = WatermarkGenerator()

# Create custom message
message = torch.tensor([1, 0, 1, 0] * 8).float()

# Embed
generator.embed_watermark(
    "assets/images/alpaca.jpg",
    "outputs/alpaca_wm.png",
    message=message
)

# Verify
result = generator.verify_watermark(
    "outputs/alpaca_wm.png",
    message
)

print(f"Match: {result['match']}")
print(f"Bit accuracy: {result['bit_accuracy']*100:.1f}%")
```

### Example 3: Batch Processing

```python
from generate_watermark import WatermarkGenerator

generator = WatermarkGenerator()

results = generator.batch_embed(
    input_dir="assets/images",
    output_dir="outputs/batch",
    mask_percentage=1.0
)

print(f"Processed {len(results)} images")
```

## Run All Examples

```bash
python3 watermark_examples.py
```

This will run 7 comprehensive examples covering:
- Basic embedding
- Custom messages
- Partial watermarking
- Detection
- Verification
- Batch processing
- Advanced usage

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
# Install PyTorch
conda install pytorch torchvision -c pytorch
# OR
pip install torch torchvision
```

### "CUDA out of memory"
```bash
# Use CPU instead
python3 generate_watermark.py embed input.jpg output.jpg --device cpu
```

### "Model weights not found"
```bash
# Download manually
python3 generate_watermark.py download
```

## What's Next?

1. **Read the full documentation**: See [WATERMARK_USAGE.md](WATERMARK_USAGE.md)
2. **Adjust watermark strength**: Modify `model.scaling_w` parameter
3. **Explore multiple watermarks**: See `notebooks/inference.ipynb`
4. **Train your own model**: See training section in README.md

## Performance Tips

- **Use GPU**: 10-20x faster than CPU
- **Batch processing**: More efficient for multiple images
- **Adjust mask_percentage**: Lower percentage = faster processing
- **PSNR > 40 dB**: Imperceptible watermarks
- **Bit accuracy > 90%**: Successful detection

## Support

For issues or questions:
- Check [WATERMARK_USAGE.md](WATERMARK_USAGE.md) for detailed documentation
- See [README.md](README.md) for project overview
- Review example code in `watermark_examples.py`

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Ready to watermark!** 🐤

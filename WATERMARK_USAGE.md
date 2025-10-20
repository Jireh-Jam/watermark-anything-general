# Watermark Generation - Usage Guide

This guide explains how to use the watermark generation code in this repository.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model Weights

The script will automatically download the MIT-licensed model weights on first use, or you can download manually:

```bash
python generate_watermark.py download
```

### 3. Basic Usage

#### Command Line Interface

**Embed a watermark in a single image:**
```bash
python generate_watermark.py embed assets/images/alpaca.jpg outputs/alpaca_watermarked.png
```

**Embed with a custom message (32 bits):**
```bash
python generate_watermark.py embed assets/images/alpaca.jpg outputs/alpaca_watermarked.png \
    --message "10101010110011001111000011110000"
```

**Watermark only part of the image (50%):**
```bash
python generate_watermark.py embed assets/images/alpaca.jpg outputs/alpaca_partial.png \
    --mask-percentage 0.5
```

**Batch process all images in a directory:**
```bash
python generate_watermark.py embed assets/images/ outputs/ --batch
```

**Batch with different messages per image:**
```bash
python generate_watermark.py embed assets/images/ outputs/ --batch --different-messages
```

**Detect a watermark:**
```bash
python generate_watermark.py detect outputs/alpaca_watermarked.png --output-dir outputs/
```

**Verify a specific watermark:**
```bash
python generate_watermark.py verify outputs/alpaca_watermarked.png \
    "10101010110011001111000011110000"
```

---

## Programmatic Usage

### Basic Example

```python
from generate_watermark import WatermarkGenerator

# Initialize the generator
generator = WatermarkGenerator()

# Embed a watermark
result = generator.embed_watermark(
    image_path="assets/images/alpaca.jpg",
    output_path="outputs/alpaca_watermarked.png",
    mask_percentage=1.0  # Watermark entire image
)

print(f"Embedded message: {result['message_str']}")
print(f"PSNR: {result['psnr']:.2f} dB")
```

### Custom Message

```python
import torch
from generate_watermark import WatermarkGenerator

generator = WatermarkGenerator()

# Create a custom 32-bit message
custom_message = torch.tensor([
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1, 0, 0, 0, 0
]).float()

result = generator.embed_watermark(
    image_path="assets/images/ducks.jpg",
    output_path="outputs/ducks_watermarked.png",
    message=custom_message
)
```

### Detection and Verification

```python
from generate_watermark import WatermarkGenerator
import torch

generator = WatermarkGenerator()

# Embed with a known message
message = torch.tensor([1, 0] * 16).float()
embed_result = generator.embed_watermark(
    image_path="assets/images/alpaca.jpg",
    output_path="outputs/alpaca_wm.png",
    message=message
)

# Detect the watermark
detect_result = generator.detect_watermark(
    image_path="outputs/alpaca_wm.png",
    output_dir="outputs"
)

print(f"Detected: {detect_result['message_str']}")

# Verify it matches the original
verify_result = generator.verify_watermark(
    image_path="outputs/alpaca_wm.png",
    original_message=message
)

print(f"Bit accuracy: {verify_result['bit_accuracy']*100:.1f}%")
print(f"Match: {verify_result['match']}")
```

### Batch Processing

```python
from generate_watermark import WatermarkGenerator

generator = WatermarkGenerator()

# Process all images in a directory with the same message
results = generator.batch_embed(
    input_dir="assets/images",
    output_dir="outputs/batch",
    mask_percentage=1.0,
    use_same_message=True
)

print(f"Processed {len(results)} images")
for result in results:
    print(f"  {result['input_path']}: PSNR={result['psnr']:.2f}dB")
```

---

## Features

### ✅ Fully Functional Features

1. **Watermark Embedding**
   - Single image watermarking
   - Batch processing
   - Custom or auto-generated messages
   - Partial or full image watermarking
   - Configurable mask percentage

2. **Watermark Detection**
   - Automatic message extraction
   - Detection mask visualization
   - Confidence metrics

3. **Watermark Verification**
   - Bit accuracy calculation
   - Hamming distance measurement
   - Match verification

4. **Model Management**
   - Automatic weight download
   - MIT-licensed model support
   - GPU/CPU auto-detection

5. **Output Options**
   - Watermarked images
   - Detection masks
   - PSNR metrics
   - Binary message strings

---

## API Reference

### WatermarkGenerator Class

#### Constructor
```python
WatermarkGenerator(checkpoint_dir="checkpoints", device=None)
```
- `checkpoint_dir`: Directory containing model files
- `device`: 'cuda' or 'cpu' (auto-detect if None)

#### Methods

**embed_watermark()**
```python
embed_watermark(image_path, output_path, message=None, mask_percentage=1.0, save_mask=True)
```
Embed a watermark into an image.

**detect_watermark()**
```python
detect_watermark(image_path, output_dir=None, save_detection_mask=True)
```
Detect and extract watermark from an image.

**verify_watermark()**
```python
verify_watermark(image_path, original_message)
```
Verify if a specific watermark exists in an image.

**batch_embed()**
```python
batch_embed(input_dir, output_dir, message=None, mask_percentage=1.0, use_same_message=True)
```
Embed watermarks in all images in a directory.

**generate_random_message()**
```python
generate_random_message(num_bits=32)
```
Generate a random binary message.

---

## Examples

Run the provided examples:

```bash
python watermark_examples.py
```

This will run 7 different examples demonstrating:
1. Basic embedding
2. Custom messages
3. Partial watermarking
4. Detection
5. Verification
6. Batch processing
7. Advanced programmatic usage

---

## Performance Notes

- **PSNR**: Higher is better (typically 35-45 dB for imperceptible watermarks)
- **Bit Accuracy**: >90% is considered a successful detection
- **Mask Coverage**: Percentage of image detected as watermarked
- **Processing Speed**: ~1-2 seconds per image on GPU, ~5-10 seconds on CPU

---

## Troubleshooting

### Model not found
```bash
python generate_watermark.py download
```

### CUDA out of memory
Use CPU instead:
```bash
python generate_watermark.py embed input.jpg output.jpg --device cpu
```

### Import errors
```bash
pip install -r requirements.txt
```

---

## Advanced Configuration

### Adjusting Watermark Strength

The watermark strength is controlled by `model.scaling_w` (default: 2.0):

```python
generator = WatermarkGenerator()
generator.load_model()
generator.model.scaling_w = 3.0  # Stronger watermark (lower PSNR, more robust)
# or
generator.model.scaling_w = 1.5  # Weaker watermark (higher PSNR, less robust)
```

### Using Different Model Weights

```python
# Use COCO-trained model (non-commercial license)
generator = WatermarkGenerator(checkpoint_dir="checkpoints")
# Download COCO weights manually:
# wget https://dl.fbaipublicfiles.com/watermark_anything/wam_coco.pth -O checkpoints/checkpoint.pth
```

---

## License

The code and MIT-trained model are under MIT License. See [LICENSE](LICENSE) for details.

The COCO-trained model weights are under CC-BY-NC License. See [LICENSE-COCO](LICENSE-COCO) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{sander2025watermark,
  title={Watermark Anything with Localized Messages},
  author={Sander, Tom and Fernandez, Pierre and Durmus, Alain and Furon, Teddy and Douze, Matthijs},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

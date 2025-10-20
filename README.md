# Watermark Anything (Localized Image Watermarks)

A practical, ready-to-run implementation for embedding and detecting localized image watermarks. This repository provides:

- A pretrained model and configuration (params provided; checkpoint can auto-download)
- A simple CLI to embed watermarks in images (`watermark_anything/cli.py`)
- A detection-only CLI that decodes watermarks, including multi-watermark clustering (`watermark_anything/detect_cli.py`)

If you use this work, please consider citing the paper and referencing the project links below.

- Paper: [Watermark Anything with Localized Messages (ICLR 2025)](https://arxiv.org/abs/2411.07231)
- Demo: [Hugging Face Spaces](https://huggingface.co/spaces/xiaoyao9184/watermark-anything)


## Requirements and Installation

Tested with Python 3.10.14, PyTorch 2.5.1, CUDA 12.4, Torchvision 0.20.1.

```bash
# Create and activate an environment (example with conda)
conda create -n watermark_anything python=3.10.14 -y
conda activate watermark_anything

# Install a matching PyTorch build for your system (examples)
# CUDA 12.4 build:
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
# or CPU-only build:
# conda install pytorch torchvision cpuonly -c pytorch -y

# Install Python dependencies
pip install -r requirements.txt
```


## Weights

- `checkpoints/params.json` (provided in this repo) describes the model and config paths.
- The checkpoint file can be auto-downloaded with `--auto-download` flags in the CLIs. By default we try Hugging Face Hub first and fall back to the MIT-licensed checkpoint hosted by Meta.

Manual download options:

- MIT-licensed checkpoint: [`wam_mit.pth` (Meta public files)](https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth)
  - Save as `checkpoints/checkpoint.pth`
- Or via Hugging Face Hub:

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="facebook/watermark-anything", filename="checkpoint.pth")
# Then copy/link to checkpoints/checkpoint.pth
```


## Quickstart

### Embed Watermarks (CLI)

Embed a watermark into all images in a folder and save outputs to `outputs/`:

```bash
python -m watermark_anything.cli assets/images --auto-download --output outputs
```

Useful options:

- `--msg` binary string or `random` (default). Length must match `nbits` (default from `params.json`, e.g., 32).
- `--mask-ratio` `[0..1]` portion of the image to watermark (1.0 means full image). Example: `--mask-ratio 0.5`.
- `--mask-file` path to a binary mask image (white=watermark region).
- `--scaling-w` adjust robustness/imperceptibility trade-off (larger -> more robust, more visible).
- `--device` `auto|cpu|cuda`.
- `--detect` also runs detection and saves predicted masks and decoded message.

Outputs per image:

- `<name>_wm.png`: watermarked image
- `<name>_mask.png`: mask used for embedding
- `<name>_pred.png`: predicted detection mask (only if `--detect`)

Examples:

```bash
# Embed to full image, detect after embedding
python -m watermark_anything.cli assets/images \
  --auto-download --detect --output outputs

# Embed with a custom 32-bit message and 50% mask
python -m watermark_anything.cli assets/images \
  --auto-download --msg 01010110011001010100101100110101 \
  --mask-ratio 0.5 --output outputs
```


### Detect Watermarks (CLI)

Run detection-only on a folder of images. Saves predicted masks and decoded messages to `outputs_detect/`.

```bash
python -m watermark_anything.detect_cli assets/images --auto-download --mode single --output outputs_detect
```

- `--mode single`: decodes a single watermark, writes `<name>_message.txt`
- `--mode multi`: detects and clusters multiple localized watermarks (DBSCAN), writes `<name>_clusters.png` and `<name>_messages.txt`
- `--eps`, `--min-samples`: DBSCAN params (multi)
- `--threshold`: bit threshold before clustering (multi)

Example (multi-watermark):

```bash
python -m watermark_anything.detect_cli assets/images \
  --auto-download --mode multi --eps 1.0 --min-samples 500 \
  --output outputs_detect
```


## Programmatic Usage (Python)

```python
import torch
from PIL import Image
from watermark_anything.cli import load_model_from_checkpoint
from watermark_anything.data.transforms import default_transform, unnormalize_img

# Device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wam = load_model_from_checkpoint(
    "checkpoints/params.json", "checkpoints/checkpoint.pth"
).to(device).eval()

# Prepare input
img = Image.open("assets/images/ducks.jpg").convert("RGB")
img_pt = default_transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
msg = torch.randint(0, 2, (32,), dtype=torch.float32, device=device)

# Embed
outputs = wam.embed(img_pt, msg)
img_w = outputs["imgs_w"]  # [1,3,H,W]

# Detect
preds = wam.detect(img_w)["preds"]           # [1,1+K,256,256]
mask_pred = torch.sigmoid(preds[:, 0:1])      # [1,1,256,256]
bit_preds = preds[:, 1:, :, :]                # [1,K,256,256]
```


## Notes

- `nbits` is read from `checkpoints/params.json` (commonly 32). You can override via the CLI `--nbits` flag when embedding.
- `--scaling-w` controls visibility/robustness; start with the default and adjust for your use case.
- Images are normalized internally; saved outputs are denormalized for visualization.


## Troubleshooting

- "No images found": verify your input path and file extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
- Checkpoint not found: add `--auto-download` or manually place `checkpoints/checkpoint.pth`.
- GPU/CPU: set `--device` explicitly if automatic selection is not desired.
- PyTorch install: ensure your PyTorch build matches your CUDA driver, or use CPU-only if needed.
- Network errors during auto-download: re-run with a stable connection or download manually using the links above.


## License

The code in this repository and the model trained on the SA-1B dataset are released under the [MIT License](LICENSE).

For reproducibility, the COCO-based model weights from the publication are also available but under [CC-BY-NC](LICENSE-COCO).


## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{sander2025watermark,
  title={Watermark Anything with Localized Messages},
  author={Sander, Tom and Fernandez, Pierre and Durmus, Alain and Furon, Teddy and Douze, Matthijs},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

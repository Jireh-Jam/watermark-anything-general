"""
Command-line interface for embedding localized watermarks into images using
the Watermark Anything (WAM) model contained in this repository.

Examples:
  - Embed a random message into all images of a directory (full image):
      python -m watermark_anything.cli embed --input assets/images --output-dir outputs

  - Embed a specific 32-bit message (as a binary string) into a single image:
      python -m watermark_anything.cli embed --input assets/images/ducks.jpg \
          --message 01010101010101010101010101010101

  - Embed into a random region covering 40% of the image and verify detection:
      python -m watermark_anything.cli embed --input assets/images --mask random \
          --mask-percent 0.4 --detect

Weights:
  This CLI expects model configuration at checkpoints/params.json.
  If --checkpoint is not provided, it looks for one of:
    - checkpoints/wam_mit.pth (MIT-licensed recommended weights)
    - checkpoints/checkpoint.pth
  If neither exists and --auto-download is set, it downloads wam_mit.pth
  from the official URL into checkpoints/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request
from typing import Iterable, List, Optional, Tuple

from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import omegaconf

from watermark_anything.models import Wam, build_embedder, build_extractor
from watermark_anything.augmentation.augmenter import Augmenter
from watermark_anything.modules.jnd import JND
from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img
from watermark_anything.data.metrics import msg_predict_inference


DEFAULT_PARAMS = os.path.join("checkpoints", "params.json")
DEFAULT_CKPT_CANDIDATES = [
    os.path.join("checkpoints", "wam_mit.pth"),
    os.path.join("checkpoints", "checkpoint.pth"),
]
OFFICIAL_WAM_MIT_URL = (
    "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth"
)


def _is_image_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"))


def _gather_image_paths(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        paths = [
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if _is_image_file(os.path.join(input_path, name))
        ]
        return sorted(paths)
    if os.path.isfile(input_path) and _is_image_file(input_path):
        return [input_path]
    # Treat as glob
    return sorted([p for p in glob.glob(input_path) if _is_image_file(p)])


def _ensure_checkpoint(params_path: str, checkpoint_path: Optional[str], auto_download: bool) -> Tuple[str, str]:
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Params file not found: {params_path}. Expected checkpoints/params.json to exist."
        )

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        return params_path, checkpoint_path

    # Otherwise, search default candidates
    for candidate in DEFAULT_CKPT_CANDIDATES:
        if os.path.exists(candidate):
            return params_path, candidate

    # Auto-download MIT weights if requested
    if auto_download:
        os.makedirs(os.path.dirname(DEFAULT_CKPT_CANDIDATES[0]), exist_ok=True)
        target = DEFAULT_CKPT_CANDIDATES[0]
        print(f"Checkpoint not found. Downloading MIT weights to {target} ...")
        urllib.request.urlretrieve(OFFICIAL_WAM_MIT_URL, target)
        print("Download complete.")
        return params_path, target

    raise FileNotFoundError(
        "No checkpoint found. Provide --checkpoint or place one of "
        f"{DEFAULT_CKPT_CANDIDATES} and ensure {params_path} exists."
    )


def load_model_from_checkpoint(json_path: str, ckpt_path: str) -> Wam:
    """
    Load a model from a checkpoint file and a JSON file containing the parameters.
    Mirrors notebooks.inference_utils.load_model_from_checkpoint to avoid import issues.
    """
    with open(json_path, "r") as file:
        params = json.load(file)
    args = argparse.Namespace(**params)

    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    embedder_params = embedder_cfg[args.embedder_model]
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    extractor_params = extractor_cfg[args.extractor_model]
    augmenter_cfg = omegaconf.OmegaConf.load(args.augmentation_config)
    attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)

    embedder = build_embedder(args.embedder_model, embedder_params, args.nbits)
    extractor = build_extractor(extractor_cfg.model, extractor_params, args.img_size, args.nbits)
    augmenter = Augmenter(**augmenter_cfg)
    try:
        attenuation = JND(**attenuation_cfg[args.attenuation], preprocess=unnormalize_img, postprocess=normalize_img)
    except Exception:
        attenuation = None

    wam = Wam(embedder, extractor, augmenter, attenuation, args.scaling_w, args.scaling_i)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        wam.load_state_dict(checkpoint)
        print("Model loaded successfully from", ckpt_path)
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    return wam


def create_random_mask(img_pt: torch.Tensor, num_masks: int = 1, mask_percentage: float = 0.1, max_attempts: int = 100) -> torch.Tensor:
    """Create a random rectangular mask or set of masks over the image tensor.
    Returns shape [num_masks, 1, H, W].
    """
    _, _, height, width = img_pt.shape
    mask_area = int(height * width * mask_percentage)
    masks = torch.zeros((num_masks, 1, height, width), dtype=img_pt.dtype, device=img_pt.device)

    if mask_percentage >= 0.999:
        return torch.ones((num_masks, 1, height, width), dtype=img_pt.dtype, device=img_pt.device)

    for ii in range(num_masks):
        placed = False
        attempts = 0
        while not placed and attempts < max_attempts:
            attempts += 1
            max_dim = int(mask_area ** 0.5)
            # ensure strictly positive sizes
            mask_width = max(1, min(width, int(torch.randint(1, max_dim + 1, ()).item())))
            mask_height = max(1, mask_area // mask_width)

            aspect_ratio = mask_width / mask_height if mask_height != 0 else 0
            if 0.25 <= aspect_ratio <= 4 and mask_height <= height and mask_width <= width:
                x_start = int(torch.randint(0, width - mask_width + 1, ()).item())
                y_start = int(torch.randint(0, height - mask_height + 1, ()).item())
                overlap = False
                for jj in range(ii):
                    if torch.sum(masks[jj, :, y_start:y_start + mask_height, x_start:x_start + mask_width]) > 0:
                        overlap = True
                        break
                if not overlap:
                    masks[ii, :, y_start:y_start + mask_height, x_start:x_start + mask_width] = 1
                    placed = True

        if not placed:
            center_h = height // 2
            center_w = width // 2
            half_area = int((mask_area // 2) ** 0.5)
            h_half = min(center_h, half_area)
            w_half = min(center_w, half_area)
            masks[ii, :, center_h - h_half:center_h + h_half, center_w - w_half:center_w + w_half] = 1

    return masks


def _parse_message(message_str: Optional[str], nbits: int, device: torch.device) -> Optional[torch.Tensor]:
    if message_str is None:
        return None
    msg = message_str.strip()
    if len(msg) != nbits or any(ch not in ("0", "1") for ch in msg):
        raise ValueError(
            f"--message must be a {nbits}-character string of 0/1. Got: '{message_str}'."
        )
    bits = [1.0 if ch == "1" else 0.0 for ch in msg]
    return torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _load_mask_for_image(mask_arg: str, img_tensor: torch.Tensor, mask_percent: float) -> torch.Tensor:
    """
    Returns a mask tensor of shape [1, 1, H, W] on the same device/dtype as img_tensor.
    mask_arg can be:
      - "full": full mask of ones
      - "random": a random rectangle mask covering mask_percent of the image
      - a path to an image mask file; non-zero pixels are considered in-mask
    """
    b, c, h, w = img_tensor.shape
    assert b == 1, "img_tensor must be a batch of size 1"
    device = img_tensor.device
    dtype = img_tensor.dtype

    if mask_arg == "full":
        return torch.ones((1, 1, h, w), dtype=dtype, device=device)

    if mask_arg == "random":
        masks = create_random_mask(img_tensor, num_masks=1, mask_percentage=float(mask_percent))
        return masks[:1]

    if os.path.exists(mask_arg):
        with Image.open(mask_arg) as im_mask:
            im_mask = im_mask.convert("L").resize((w, h))
            mask_np = (torch.tensor(list(im_mask.getdata())).view(h, w) > 0).float()
            return mask_np.view(1, 1, h, w).to(device=device, dtype=dtype)

    raise ValueError(
        f"--mask must be 'full', 'random', or a valid mask image path. Got: {mask_arg}"
    )


def _save_tensor_image(img_tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(unnormalize_img(img_tensor).clamp(0, 1), path)


def cmd_embed(args: argparse.Namespace) -> None:
    params_path, ckpt_path = _ensure_checkpoint(
        params_path=args.params,
        checkpoint_path=args.checkpoint,
        auto_download=args.auto_download,
    )

    device_str = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    device = torch.device(device_str)

    # Build model
    wam = load_model_from_checkpoint(params_path, ckpt_path).to(device).eval()

    # Determine number of bits from model
    inferred_nbits = int(wam.get_random_msg(1).shape[-1])

    # Prepare message tensor if provided
    msg_tensor = _parse_message(args.message, inferred_nbits, device)

    # Iterate images
    inputs: List[str] = []
    for in_arg in args.input:
        inputs.extend(_gather_image_paths(in_arg))
    if not inputs:
        raise FileNotFoundError("No input images found for the provided --input arguments.")

    for img_path in inputs:
        image = _load_image(img_path)
        img_pt = default_transform(image).unsqueeze(0).to(device)

        outputs = wam.embed(img_pt, msg_tensor)
        used_msg = outputs["msgs"].detach()

        # Build target mask and localize watermark
        mask = _load_mask_for_image(args.mask, img_pt, args.mask_percent)
        img_w = outputs["imgs_w"] * mask + img_pt * (1 - mask)

        # Save watermarked image
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_img = os.path.join(args.output_dir, f"{base}_wm.png")
        _save_tensor_image(img_w, out_img)

        # Optionally save the target mask for reference
        if args.save_masks:
            target_path = os.path.join(args.output_dir, f"{base}_target.png")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            save_image(mask, target_path)

        # Optional detection/verification
        if args.detect:
            preds = wam.detect(img_w)["preds"]  # [1, 1+K, 256, 256]
            mask_pred = torch.sigmoid(preds[:, 0, :, :]).unsqueeze(1)  # [1,1,256,256]
            bit_preds = preds[:, 1:, :, :]  # [1,K,256,256]

            pred_message = msg_predict_inference(bit_preds, mask_pred)  # [1,K]
            if msg_tensor is not None:
                bit_acc = (pred_message.float() == used_msg.float()).float().mean().item()
                print(f"[{base}] Bit accuracy: {bit_acc:.3f}")
            else:
                bits = pred_message[0].to(torch.bool).tolist()
                pred_str = "".join("1" if b else "0" for b in bits)
                print(f"[{base}] Predicted message: {pred_str}")

            # Save predicted mask resized to original image size
            mask_pred_res = F.interpolate(
                mask_pred, size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False
            )
            pred_path = os.path.join(args.output_dir, f"{base}_pred.png")
            save_image(mask_pred_res, pred_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watermark_anything",
        description="Embed localized watermarks into images using Watermark Anything.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_embed = sub.add_parser("embed", help="Embed a watermark into image(s)")
    p_embed.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input image file, directory, or glob pattern (can pass multiple)",
    )
    p_embed.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write watermarked images (default: outputs)",
    )
    p_embed.add_argument(
        "--params",
        default=DEFAULT_PARAMS,
        help=f"Path to params.json (default: {DEFAULT_PARAMS})",
    )
    p_embed.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (.pth). If not provided, tries defaults and auto-download.",
    )
    p_embed.add_argument(
        "--auto-download",
        action="store_true",
        help="If set, automatically download MIT weights if checkpoint is missing.",
    )
    p_embed.add_argument(
        "--message",
        default=None,
        help="Binary string of length K (e.g., 32) to embed. If omitted, a random message is used.",
    )
    p_embed.add_argument(
        "--mask",
        default="full",
        help="Mask to use: 'full', 'random', or a mask image path",
    )
    p_embed.add_argument(
        "--mask-percent",
        type=float,
        default=1.0,
        help="If --mask=random, fraction of pixels to watermark (0-1). Default: 1.0",
    )
    p_embed.add_argument(
        "--detect",
        action="store_true",
        help="Run detection on the output and save predicted mask (and bit accuracy if message provided).",
    )
    p_embed.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save the target mask used for embedding.",
    )
    p_embed.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device selection (default: auto)",
    )

    p_embed.set_defaults(func=cmd_embed)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

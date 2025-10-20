import argparse
import os
import sys
import urllib.request
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from omegaconf import OmegaConf

from .models import Wam, build_embedder, build_extractor
from .augmentation.augmenter import Augmenter
from .data.transforms import (
    default_transform,
    normalize_img,
    unnormalize_img,
)
from .modules.jnd import JND
from .data.metrics import msg_predict_inference


HF_REPO_ID = "facebook/watermark-anything"
HF_FILENAME = "checkpoint.pth"
FB_PUBLIC_WAM_MIT = (
    "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth"
)


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def try_download_checkpoint(checkpoints_dir: str) -> Optional[str]:
    """
    Try downloading the checkpoint using huggingface_hub, fallback to public URL.
    Returns the local path if successful, otherwise None.
    """
    ensure_dir(checkpoints_dir)

    # 1) Try Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=checkpoints_dir,
        )
        # Normalize to a predictable path inside checkpoints_dir
        dst_path = os.path.join(checkpoints_dir, "checkpoint.pth")
        if os.path.abspath(local_path) != os.path.abspath(dst_path):
            # If the hub created nested dirs, just copy/rename
            try:
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                os.replace(local_path, dst_path)
                local_path = dst_path
            except Exception:
                # If atomic replace fails, fallback to copy
                import shutil

                shutil.copy(local_path, dst_path)
                local_path = dst_path
        return local_path
    except Exception:
        pass

    # 2) Fallback to direct download of the MIT checkpoint
    try:
        dst_path = os.path.join(checkpoints_dir, "checkpoint.pth")
        with urllib.request.urlopen(FB_PUBLIC_WAM_MIT) as response:
            with open(dst_path, "wb") as f:
                f.write(response.read())
        return dst_path
    except Exception:
        return None


def load_model_from_checkpoint(params_json: str, ckpt_path: str) -> Wam:
    """
    Load a WAM model from params.json and a checkpoint path.
    This is a streamlined version that avoids importing notebook utilities.
    """
    if not os.path.isfile(params_json):
        raise FileNotFoundError(
            f"Params JSON not found at {params_json}. Expected checkpoints/params.json."
        )

    params = OmegaConf.load(params_json)

    # Load configurations
    embedder_cfg = OmegaConf.load(params.embedder_config)
    embedder_params = embedder_cfg[params.embedder_model]
    extractor_cfg = OmegaConf.load(params.extractor_config)
    extractor_params = extractor_cfg[params.extractor_model]
    augmenter_cfg = OmegaConf.load(params.augmentation_config)
    attenuation_cfg = OmegaConf.load(params.attenuation_config)

    # Build models
    embedder = build_embedder(params.embedder_model, embedder_params, params.nbits)
    extractor = build_extractor(
        extractor_cfg.model,
        extractor_params,
        params.img_size,
        params.nbits,
    )
    augmenter = Augmenter(**augmenter_cfg)

    try:
        attenuation = JND(
            **attenuation_cfg[params.attenuation],
            preprocess=unnormalize_img,
            postprocess=normalize_img,
        )
    except Exception:
        attenuation = None

    wam = Wam(
        embedder,
        extractor,
        augmenter,
        attenuation,
        params.scaling_w,
        params.scaling_i,
        roll_probability=getattr(params, "roll_probability", 0.0),
        img_size_extractor=getattr(params, "img_size_extractor", params.img_size),
    )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run with --auto-download to fetch it."
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    wam.load_state_dict(checkpoint)
    return wam


def parse_msg(msg_str: Optional[str], nbits: int, device: torch.device) -> torch.Tensor:
    if msg_str is None or msg_str.lower() == "random":
        return torch.randint(0, 2, (nbits,), dtype=torch.float32, device=device)
    bits = msg_str.strip()
    if not all(c in {"0", "1"} for c in bits):
        raise ValueError("--msg must be a binary string or 'random'")
    if len(bits) != nbits:
        raise ValueError(f"--msg length ({len(bits)}) must equal nbits ({nbits})")
    return torch.tensor([float(c) for c in bits], dtype=torch.float32, device=device)


def create_full_mask(img_pt: torch.Tensor) -> torch.Tensor:
    return torch.ones((1, 1, img_pt.shape[-2], img_pt.shape[-1]), dtype=img_pt.dtype, device=img_pt.device)


def create_random_mask(
    img_pt: torch.Tensor, num_masks: int = 1, mask_percentage: float = 0.1
) -> torch.Tensor:
    """Lightweight random rectangular mask generator (single mask used here)."""
    _ = num_masks  # not used but kept for signature compatibility
    _, _, height, width = img_pt.shape
    mask_area = int(height * width * mask_percentage)
    mask = torch.zeros((1, 1, height, width), dtype=img_pt.dtype, device=img_pt.device)
    if mask_percentage >= 0.999:
        return create_full_mask(img_pt)
    # constrain rectangle size
    side = max(1, int(mask_area ** 0.5))
    side_h = min(height, side)
    side_w = min(width, side)
    # center placement
    y0 = max(0, (height - side_h) // 2)
    x0 = max(0, (width - side_w) // 2)
    mask[:, :, y0 : y0 + side_h, x0 : x0 + side_w] = 1
    return mask


def load_mask_from_file(mask_path: str, target_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    img = Image.open(mask_path).convert("L")
    img = img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    mask_np = (np.array(img).astype("float32") / 255.0)
    mask_np = (mask_np > 0.5).astype("float32")
    mask_pt = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    return mask_pt


def find_images(path: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.isdir(path):
        files = []
        for root, _, filenames in os.walk(path):
            for fn in filenames:
                if os.path.splitext(fn.lower())[1] in exts:
                    files.append(os.path.join(root, fn))
        return sorted(files)
    if os.path.isfile(path) and os.path.splitext(path.lower())[1] in exts:
        return [path]
    return []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed and optionally detect localized image watermarks"
    )
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory to write outputs (default: outputs)",
    )
    parser.add_argument(
        "--params",
        default=os.path.join("checkpoints", "params.json"),
        help="Path to params.json (default: checkpoints/params.json)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "checkpoint.pth"),
        help="Path to checkpoint file (default: checkpoints/checkpoint.pth)",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Automatically download checkpoint if missing",
    )
    parser.add_argument(
        "--msg",
        default="random",
        help="Binary string of length nbits or 'random' (default)",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=None,
        help="Number of bits (default: taken from params.json)",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=1.0,
        help="Proportion of image to watermark (1.0 = full image)",
    )
    parser.add_argument(
        "--mask-file",
        default=None,
        help="Optional binary mask image path (white=watermark region)",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run detection and decoding after embedding",
    )
    parser.add_argument(
        "--scaling-w",
        type=float,
        default=None,
        help="Override watermark scaling factor (trade-off imperceptibility/robustness)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )

    args = parser.parse_args(argv)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Ensure checkpoint availability
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path) and args.auto_download:
        downloaded = try_download_checkpoint(os.path.dirname(ckpt_path) or ".")
        if downloaded is not None:
            ckpt_path = downloaded

    # Load params to know expected nbits and scalings
    params = OmegaConf.load(args.params)
    nbits = args.nbits if args.nbits is not None else int(params.nbits)

    # Build and load model
    wam = load_model_from_checkpoint(args.params, ckpt_path).to(device).eval()
    if args.scaling_w is not None:
        wam.scaling_w = float(args.scaling_w)

    # IO
    ensure_dir(args.output)
    image_paths = find_images(args.input)
    if not image_paths:
        print("No images found. Supported: .jpg .jpeg .png .bmp .webp")
        return 1

    # Prepare message
    wm_msg = parse_msg(args.msg, nbits, device)

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_pt = default_transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

            # Embed
            outputs = wam.embed(img_pt, wm_msg)

            # Build mask
            if args.mask_file:
                mask = load_mask_from_file(
                    args.mask_file, (img_pt.shape[-2], img_pt.shape[-1]), device
                )
            elif args.mask_ratio >= 0.999:
                mask = create_full_mask(img_pt)
            else:
                mask = create_random_mask(img_pt, num_masks=1, mask_percentage=args.mask_ratio)

            img_w = outputs["imgs_w"] * mask + img_pt * (1 - mask)

            # Write outputs
            base = os.path.basename(img_path)
            root, _ = os.path.splitext(base)
            save_image(unnormalize_img(img_w), os.path.join(args.output, f"{root}_wm.png"))
            save_image(mask, os.path.join(args.output, f"{root}_mask.png"))

            # Optionally detect and decode
            if args.detect:
                preds = wam.detect(img_w)["preds"]  # [1, 1+K, 256, 256]
                mask_pred = torch.sigmoid(preds[:, 0:1, :, :])
                bit_preds = preds[:, 1:, :, :]
                # Upscale predicted mask to original image size
                mask_pred_res = F.interpolate(
                    mask_pred, size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False
                )
                save_image(mask_pred_res, os.path.join(args.output, f"{root}_pred.png"))

                pred_message = msg_predict_inference(bit_preds, mask_pred).to(torch.float32)
                pred_bits = pred_message[0].cpu().numpy().astype(int).tolist()
                print(f"{base}: predicted message = {''.join(map(str, pred_bits))}")

                bit_acc = (pred_message[0] == wm_msg).float().mean().item()
                print(f"{base}: bit accuracy = {bit_acc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

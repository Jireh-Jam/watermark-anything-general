import argparse
import os
from typing import Optional, List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from sklearn.cluster import DBSCAN
from omegaconf import OmegaConf

from .cli import ensure_dir, try_download_checkpoint, load_model_from_checkpoint
from .data.transforms import default_transform, unnormalize_img
from .data.metrics import msg_predict_inference


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


def decode_single(bit_preds: torch.Tensor, mask_pred: torch.Tensor) -> torch.Tensor:
    """Return predicted message vector [1,K] using msg_predict_inference."""
    return msg_predict_inference(bit_preds, mask_pred).to(torch.float32)


def colorize_labels(label_map: torch.Tensor, num_labels: int) -> torch.Tensor:
    """
    Convert a HxW label map in [-1..num_labels-1] to an RGB visualization tensor [3,H,W].
    """
    h, w = label_map.shape
    rgb = torch.zeros(3, h, w, dtype=torch.float32)
    # simple color palette (repeating if many labels)
    palette = torch.tensor([
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 1.0, 1.0],  # cyan
        [1.0, 0.5, 0.0],  # orange
        [0.5, 0.0, 1.0],  # purple
        [0.3, 0.7, 0.9],  # light blue
        [0.9, 0.3, 0.7],  # pink
    ], dtype=torch.float32)
    # noise/unassigned as black
    rgb[:, label_map == -1] = 0.0
    if num_labels > 0:
        for idx in range(num_labels):
            color = palette[idx % len(palette)].view(3, 1, 1)
            rgb[:, label_map == idx] = color
    return rgb.clamp(0, 1)


def multiwm_dbscan(bit_preds: torch.Tensor, mask_pred: torch.Tensor, *, threshold: float, eps: float, min_samples: int) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Cluster per-pixel bit predictions to detect multiple watermarks.

    Args:
      bit_preds: [1, K, 256, 256] raw logits for bits
      mask_pred: [1, 1, 256, 256] sigmoid mask in [0,1]
      threshold: threshold applied to bit logit (0.0 matches training usage)
      eps: DBSCAN epsilon (distance in bit-space)
      min_samples: DBSCAN min_samples

    Returns:
      (centroids_dict, full_labels):
        - centroids_dict maps cluster_id -> message tensor [K] (0/1 ints)
        - full_labels is HxW label map with -1 for noise
    """
    with torch.no_grad():
        # boolean per pixel
        bin_bits = (bit_preds > threshold)  # [1, K, H, W]
        union_mask = (mask_pred > 0.5).squeeze(1)  # [1, H, W]
        h, w = union_mask.shape[-2:]
        k = bin_bits.shape[1]

        preds_hwk = bin_bits[0].permute(1, 2, 0).contiguous().view(-1, k).to(torch.float32)  # [H*W, K]
        valid = union_mask[0].reshape(-1) > 0
        valid_points = preds_hwk[valid]

        if valid_points.numel() == 0:
            return {}, torch.full((h, w), -1, dtype=torch.long)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels_np = db.fit_predict(valid_points.cpu().numpy())  # [-1..C-1]
        labels = torch.from_numpy(labels_np).to(torch.long)

        # reconstruct full label map
        full_labels = torch.full((h * w,), -1, dtype=torch.long)
        full_labels[valid] = labels
        full_labels = full_labels.view(h, w)

        unique = torch.unique(labels)
        unique = unique[unique >= 0]
        centroids: Dict[int, torch.Tensor] = {}
        for lab in unique.tolist():
            pts = valid_points[labels == lab]
            if pts.shape[0] == 0:
                continue
            centroid = (pts.mean(dim=0) > 0.5).to(torch.int32)  # [K]
            centroids[lab] = centroid

        return centroids, full_labels


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect localized watermarks and decode messages")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--output", default="outputs_detect", help="Directory to write outputs")
    parser.add_argument("--params", default=os.path.join("checkpoints", "params.json"), help="Path to params.json")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "checkpoint.pth"),
        help="Path to checkpoint file (default: checkpoints/checkpoint.pth)",
    )
    parser.add_argument("--auto-download", action="store_true", help="Download checkpoint if missing")
    parser.add_argument("--mode", choices=["single", "multi"], default="single", help="Detection mode")
    parser.add_argument("--eps", type=float, default=1.0, help="DBSCAN epsilon (multi mode)")
    parser.add_argument("--min-samples", type=int, default=500, help="DBSCAN min samples (multi mode)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Bit threshold before clustering")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    args = parser.parse_args(argv)

    # device
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))

    # ensure checkpoint
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path) and args.auto_download:
        downloaded = try_download_checkpoint(os.path.dirname(ckpt_path) or ".")
        if downloaded is not None:
            ckpt_path = downloaded

    # load params and model
    params = OmegaConf.load(args.params)
    wam = load_model_from_checkpoint(args.params, ckpt_path).to(device).eval()

    ensure_dir(args.output)
    image_paths = find_images(args.input)
    if not image_paths:
        print("No images found. Supported: .jpg .jpeg .png .bmp .webp")
        return 1

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_pt = default_transform(img).unsqueeze(0).to(device)

            preds = wam.detect(img_pt)["preds"]  # [1, 1+K, 256, 256]
            mask_pred = torch.sigmoid(preds[:, 0:1, :, :])
            bit_preds = preds[:, 1:, :, :]

            base = os.path.basename(img_path)
            stem, _ = os.path.splitext(base)

            # save predicted mask at input size
            mask_pred_res = F.interpolate(mask_pred, size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)
            save_image(mask_pred_res, os.path.join(args.output, f"{stem}_pred.png"))

            if args.mode == "single":
                pred_message = decode_single(bit_preds, mask_pred)
                bits = pred_message[0].cpu().numpy().astype(int).tolist()
                with open(os.path.join(args.output, f"{stem}_message.txt"), "w") as f:
                    f.write("".join(map(str, bits)))
                print(f"{base}: message={''.join(map(str, bits))}")
            else:
                centroids, labels_hw = multiwm_dbscan(bit_preds, mask_pred, threshold=args.threshold, eps=args.eps, min_samples=args.min_samples)
                num_labels = len(centroids)
                # colorize on detection resolution
                colored = colorize_labels(labels_hw, num_labels)
                # resize to input image size
                colored = colored.unsqueeze(0)
                colored = F.interpolate(colored, size=(img_pt.shape[-2], img_pt.shape[-1]), mode="nearest")[0]
                save_image(colored, os.path.join(args.output, f"{stem}_clusters.png"))
                # save messages
                with open(os.path.join(args.output, f"{stem}_messages.txt"), "w") as f:
                    for lab, msg in centroids.items():
                        f.write(f"cluster {lab}: {''.join(map(str, msg.cpu().numpy().astype(int).tolist()))}\n")
                print(f"{base}: clusters={num_labels}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

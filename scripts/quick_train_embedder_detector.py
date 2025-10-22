"""
Quick training loop: optimize embedder + detector only (no attacker, no discriminator).
Run small experiments to verify msg_loss decreases and BER improves.
"""
import os, sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import yaml
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from watermark_anything.models.embedder import build_embedder
from watermark_anything.models.extractor import build_extractor
from watermark_anything.models.wam import Wam

# Optional ECC
try:
    from watermark_anything.modules.ecc import RepetitionECC, ber_from_bits
except Exception:
    RepetitionECC = None


    def ber_from_bits(a, b):
        return float((a != b).float().mean().item())


def build_dataloader(data_root, image_size, batch_size, num_workers=4):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(data_root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def majority_decode_from_logits(mean_logits, rep):
    probs = torch.sigmoid(mean_logits)
    b, kr = probs.shape
    k_raw = kr // rep
    probs = probs.view(b, k_raw, rep)
    mean_probs = probs.mean(dim=2)
    return (mean_probs >= 0.5).float()


def compute_message_loss_from_raw(preds_enc_logits, raw_msgs, rep=1):
    b, kenc, H, W = preds_enc_logits.shape
    preds_enc_logits = preds_enc_logits.view(b, kenc, -1)
    mean_logits = preds_enc_logits.mean(dim=2)  # B x kenc
    k_raw = raw_msgs.shape[1]
    probs = torch.sigmoid(mean_logits)
    probs = probs.view(b, k_raw, rep)
    mean_probs = probs.mean(dim=2)
    logits_raw = torch.log(mean_probs.clamp(1e-6, 1 - 1e-6) / (1 - mean_probs.clamp(1e-6, 1 - 1e-6)))
    loss = nn.BCEWithLogitsLoss()(logits_raw, raw_msgs.float())
    pred_bits = (mean_probs >= 0.5).float()
    return loss, logits_raw, pred_bits


def run_quick(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = build_dataloader(args.data, args.img_size, args.batch_size, num_workers=args.num_workers)
    it = iter(loader)
    imgs, _ = next(it)
    imgs = imgs.to(device)

    cfg_raw = OmegaConf.create(yaml.safe_load(open(args.config, 'r')))
    embedder_cfg_raw = cfg_raw.get('embedder', {})
    extractor_cfg_raw = cfg_raw.get('extractor', {})
    embedder_name = embedder_cfg_raw.get('name', 'vae')
    extractor_name = extractor_cfg_raw.get('name', 'sam')

    embedder_params_raw = embedder_cfg_raw.get('cfg', embedder_cfg_raw.get('params', embedder_cfg_raw))
    extractor_params_raw = extractor_cfg_raw.get('cfg', extractor_cfg_raw.get('params', extractor_cfg_raw))
    embedder_params = OmegaConf.create(embedder_params_raw)
    extractor_params = OmegaConf.create(extractor_params_raw)

    rep = args.rep
    nbits = args.nbits
    k_enc = nbits * rep
    ecc = RepetitionECC(rep=rep) if RepetitionECC is not None else None

    embedder = build_embedder(embedder_name, embedder_params, nbits=k_enc).to(device)
    extractor = build_extractor(extractor_name, extractor_params, img_size=args.img_size, nbits=k_enc).to(device)
    wam = Wam(embedder=embedder, detector=extractor, augmenter=None, attenuation=None,
              scaling_w=args.scaling_w, scaling_i=args.scaling_i, roll_probability=0.0,
              img_size_extractor=args.img_size).to(device)

    # optimizer: embedder + detector
    opt = torch.optim.AdamW(list(wam.embedder.parameters()) + list(wam.detector.parameters()), lr=args.lr,
                            weight_decay=1e-2)

    # quick loop
    wam.train()
    for step in range(1, args.steps + 1):
        raw_msgs = torch.randint(0, 2, (imgs.shape[0], nbits), device=device).float()
        encoded_msgs = ecc.encode(raw_msgs).to(device) if ecc is not None else raw_msgs
        outputs = wam.embed(imgs, msgs=encoded_msgs)
        imgs_w = outputs['imgs_w']
        preds = wam.detect(imgs_w)['preds']
        pred_enc_logits = preds[:, 1:, :, :] if preds.shape[1] > 1 else preds[:, :1, :, :]
        loss, logits_raw, pred_bits = compute_message_loss_from_raw(pred_enc_logits, raw_msgs, rep=rep)
        ber = float((pred_bits != raw_msgs).float().mean().item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.print_every == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={loss.item():.4f} BER={ber:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--rep', type=int, default=1)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--scaling_w', type=float, default=1.0)
    parser.add_argument('--scaling_i', type=float, default=1.0)
    args = parser.parse_args()
    run_quick(args)

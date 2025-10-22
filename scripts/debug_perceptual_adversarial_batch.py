# """
# Quick debug runner: run one batch through the WAM + attacker + ECC pipeline and
# print detailed diagnostics (clean vs attacked loss, bit error, shapes, grad norms).
#
# Usage (example):
#     python scripts/debug_perceptual_adversarial_batch.py \
#       --config configs/wam_config.yaml \
#       --data "C:\Users\User\Desktop\stable_signature\data" \
#       --img_size 128 --batch_size 4 --nbits 8 --rep 3 --use_learned_decoder
#
# This script does NOT perform optimizer steps; it only runs forward/backward
# to show gradients and diagnostic numbers (it zeroes gradients afterwards).
# """
import os
import sys
import argparse
import time
from collections import defaultdict

# ensure repo root is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import yaml
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# repo imports
from watermark_anything.models.embedder import build_embedder
from watermark_anything.models.extractor import build_extractor
from watermark_anything.models.wam import Wam
from watermark_anything.attacks.learned_remover import LearnedRemover
from watermark_anything.losses.detperceptual import LPIPSWithDiscriminator
from watermark_anything.attacks.learned_remover import LearnedRemover
from watermark_anything.modules.ecc import RepetitionECC, LearnedECCDecoder, ber_from_bits
from omegaconf import OmegaConf

# Optional ECC learned decoder if present in branch
try:
    from watermark_anything.modules.ecc import RepetitionECC, LearnedECCDecoder, ber_from_bits
except Exception:
    RepetitionECC = None
    LearnedECCDecoder = None


    def ber_from_bits(a, b):
        return float((a != b).float().mean().item())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_dataloader(data_root, image_size, batch_size, num_workers=0):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(data_root, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader


def majority_decode_from_logits(mean_logits, rep):
    # mean_logits: B x (k_raw*rep) (raw logits averaged spatially)
    probs = torch.sigmoid(mean_logits)
    b, kr = probs.shape
    assert kr % rep == 0
    k_raw = kr // rep
    probs = probs.view(b, k_raw, rep)
    mean_probs = probs.mean(dim=2)
    decoded = (mean_probs >= 0.5).float()
    return decoded


def compute_message_loss_from_raw(preds_enc_logits, raw_msgs, ecc_decoder=None, rep=3, use_learned_decoder=False):
    # preds_enc_logits: B x k_enc x H x W
    b, kenc, H, W = preds_enc_logits.shape
    preds_enc_logits = preds_enc_logits.view(b, kenc, -1)
    mean_logits = preds_enc_logits.mean(dim=2)  # B x kenc
    k_raw = raw_msgs.shape[1]
    if use_learned_decoder and (ecc_decoder is not None):
        logits_raw = ecc_decoder(mean_logits)  # B x k_raw
    else:
        # convert grouped probs into raw logits approximation
        probs = torch.sigmoid(mean_logits)
        probs = probs.view(b, k_raw, rep)
        mean_probs = probs.mean(dim=2)
        logits_raw = torch.log(mean_probs.clamp(1e-6, 1 - 1e-6) / (1 - mean_probs.clamp(1e-6, 1 - 1e-6)))
    loss = nn.BCEWithLogitsLoss()(logits_raw, raw_msgs.float())
    # decoded bits for evaluation:
    if use_learned_decoder and (ecc_decoder is not None):
        pred_bits = (torch.sigmoid(ecc_decoder(mean_logits)) > 0.5).float()
    else:
        pred_bits = majority_decode_from_logits(mean_logits, rep)
    return loss, logits_raw, pred_bits


def param_grad_stats(params):
    norms = []
    for p in params:
        if p.grad is not None:
            norms.append(p.grad.detach().norm().item())
    if len(norms) == 0:
        return {"max_grad": 0.0, "mean_grad": 0.0, "count": 0}
    import math
    return {"max_grad": max(norms), "mean_grad": sum(norms) / len(norms), "count": len(norms)}


def run_debug(args):
    print("Device:", device)
    loader = build_dataloader(args.data, args.img_size, args.batch_size, num_workers=args.num_workers)
    it = iter(loader)
    imgs, _ = next(it)
    imgs = imgs.to(device)

    # load config and build models (convert dict to OmegaConf if needed)
    if args.config is None:
        raise RuntimeError("Please pass --config")
    cfg_raw = OmegaConf.create(yaml.safe_load(open(args.config, 'r')))

    embedder_cfg_raw = cfg_raw.get('embedder', {})
    extractor_cfg_raw = cfg_raw.get('extractor', {})
    embedder_name = embedder_cfg_raw.get('name', 'vae')
    extractor_name = extractor_cfg_raw.get('name', 'sam')

    embedder_params_raw = embedder_cfg_raw.get('cfg', embedder_cfg_raw.get('params', embedder_cfg_raw))
    extractor_params_raw = extractor_cfg_raw.get('cfg', extractor_cfg_raw.get('params', extractor_cfg_raw))

    embedder_params = OmegaConf.create(embedder_params_raw)
    extractor_params = OmegaConf.create(extractor_params_raw)

    # ECC
    rep = args.rep
    nbits = args.nbits
    k_enc = nbits * rep
    ecc = RepetitionECC(rep=rep) if RepetitionECC is not None else None
    learned_decoder = None
    if args.use_learned_decoder and LearnedECCDecoder is not None:
        learned_decoder = LearnedECCDecoder(k_raw=nbits, rep=rep).to(device)

    # make sure embedder/extractor get the encoded bit length
    print(f"Building embedder={embedder_name} extractor={extractor_name} with k_enc={k_enc}")
    embedder = build_embedder(embedder_name, embedder_params, nbits=k_enc).to(device)
    extractor = build_extractor(extractor_name, extractor_params, img_size=args.img_size, nbits=k_enc).to(device)

    # augmenter/jnd optional: reuse original script logic minimal fallback None
    from watermark_anything.augmentation.augmenter import Augmenter  # if present in repo
    augmenter = None
    attenuation = None

    wam = Wam(embedder=embedder, detector=extractor, augmenter=augmenter, attenuation=attenuation,
              scaling_w=args.scaling_w, scaling_i=args.scaling_i, roll_probability=args.roll_probability,
              img_size_extractor=args.img_size)
    wam.to(device).eval()  # evaluation behavior ok for diagnostics

    attack_net = LearnedRemover(in_channels=3, base_ch=args.attack_base_ch, depth=args.attack_depth,
                                tanh_scale=args.attack_scale).to(device)
    loss_module = LPIPSWithDiscriminator(
        balanced=True,
        percep_weight=args.percep_weight,
        disc_weight=args.disc_weight,
        detect_weight=args.detect_weight,
        decode_weight=args.decode_weight,
        disc_start=args.disc_start,
        disc_num_layers=args.disc_num_layers,
        disc_in_channels=3,
        percep_loss=args.percep_loss
    ).to(device)

    # Create a fake message batch
    b = imgs.shape[0]
    raw_msgs = torch.randint(0, 2, (b, nbits), device=device).float()
    encoded_msgs = ecc.encode(raw_msgs).to(device) if ecc is not None else raw_msgs

    # with torch.no_grad():
    #     wam_outputs = wam.embed(imgs, msgs=encoded_msgs)
    #     imgs_w = wam_outputs['imgs_w']
    # allow gradients through embedder for diagnostics
    wam.train()  # ensure module is in train mode so parameters require grad behavior is normal
    wam_outputs = wam.embed(imgs, msgs=encoded_msgs)
    imgs_w = wam_outputs['imgs_w']
    print("imgs.shape", imgs.shape, "imgs_w.shape", imgs_w.shape)

    # DETECTION ON CLEAN
    preds_clean = wam.detect(imgs_w)['preds']  # B x (1 + k_enc) x H x W
    print("preds_clean.shape", preds_clean.shape)
    pred_enc_logits_clean = preds_clean[:, 1:, :, :] if preds_clean.shape[1] > 1 else preds_clean[:, :1, :, :]

    loss_clean, logits_raw_clean, pred_bits_clean = compute_message_loss_from_raw(pred_enc_logits_clean, raw_msgs,
                                                                                  learned_decoder, rep,
                                                                                  use_learned_decoder=bool(
                                                                                      learned_decoder))
    print(f"clean_loss={loss_clean.item():.6f}")

    # ATTACK
    attack_net.eval()
    attacked = (imgs_w + attack_net(imgs_w)).clamp(0., 1.)
    print("attacked.shape", attacked.shape)
    preds_attacked = wam.detect(attacked)['preds']
    print("preds_attacked.shape", preds_attacked.shape)
    pred_enc_logits_att = preds_attacked[:, 1:, :, :] if preds_attacked.shape[1] > 1 else preds_attacked[:, :1, :, :]

    loss_attacked, logits_raw_att, pred_bits_att = compute_message_loss_from_raw(pred_enc_logits_att, raw_msgs,
                                                                                 learned_decoder, rep,
                                                                                 use_learned_decoder=bool(
                                                                                     learned_decoder))
    print(f"attacked_loss={loss_attacked.item():.6f}")

    # BER / bit-accuracy
    # pred_bits_* are B x nbits (0/1 float)
    ber_clean = float((pred_bits_clean != raw_msgs).float().mean().item())
    ber_att = float((pred_bits_att != raw_msgs).float().mean().item())
    print(f"BER_clean={ber_clean:.4f} BER_attacked={ber_att:.4f}")

    # compute gradients for diagnostics WITHOUT updating weights
    #  - check embedder+detector gradients from clean loss
    for p in wam.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss_clean.backward(retain_graph=True)
    embedder_params_list = list(wam.embedder.parameters())
    detector_params_list = list(wam.detector.parameters())
    print("After clean backward: embedder grad stats:", param_grad_stats(embedder_params_list))
    print("After clean backward: detector grad stats:", param_grad_stats(detector_params_list))
    # zero
    for p in wam.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # gradients for attack network from attack objective (maximize msg loss -> minimize -loss)
    for p in attack_net.parameters():
        if p.grad is not None:
            p.grad.zero_()
    attack_loss_for_grads = -loss_attacked
    attack_loss_for_grads.backward()
    print("After attack backward: attack grad stats:", param_grad_stats(list(attack_net.parameters())))
    # zero again
    for p in attack_net.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Print a small summary table
    print("SUMMARY:")
    print(
        f"  raw_msgs.shape: {raw_msgs.shape}, encoded length: {encoded_msgs.shape if isinstance(encoded_msgs, torch.Tensor) else 'N/A'}")
    print(f"  clean_loss: {loss_clean.item():.6f}, attacked_loss: {loss_attacked.item():.6f}")
    print(f"  BER_clean: {ber_clean:.4f}, BER_attacked: {ber_att:.4f}")
    print("  preds_clean channels:", preds_clean.shape[1], "preds_attacked channels:", preds_attacked.shape[1])
    print("  sample logits_raw_clean (first example):",
          logits_raw_clean[0].detach().cpu().numpy()[:min(10, logits_raw_clean.shape[1])])
    print("  sample logits_raw_att  (first example):",
          logits_raw_att[0].detach().cpu().numpy()[:min(10, logits_raw_att.shape[1])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--nbits', type=int, default=16)
    parser.add_argument('--rep', type=int, default=3)
    parser.add_argument('--use_learned_decoder', action='store_true')
    parser.add_argument('--attack_base_ch', type=int, default=32)
    parser.add_argument('--attack_depth', type=int, default=3)
    parser.add_argument('--attack_scale', type=float, default=0.05)
    parser.add_argument('--percep_weight', type=float, default=1.0)
    parser.add_argument('--disc_weight', type=float, default=1.0)
    parser.add_argument('--detect_weight', type=float, default=1.0)
    parser.add_argument('--decode_weight', type=float, default=0.0)
    parser.add_argument('--disc_start', type=int, default=0)
    parser.add_argument('--disc_num_layers', type=int, default=3)
    parser.add_argument('--percep_loss', type=str, default='lpips')
    parser.add_argument('--scaling_w', type=float, default=1.0)
    parser.add_argument('--scaling_i', type=float, default=1.0)
    parser.add_argument('--roll_probability', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    run_debug(args)

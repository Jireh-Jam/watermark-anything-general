"""
Full training script to run perceptual + adversarial training using the repo's
LPIPSWithDiscriminator and a learned remover attack. This script supports:
 - building the WAM model automatically from a YAML config (embedder/extractor names + params),
 - integrating a repetition ECC (or learned ECC decoder) end-to-end,
 - alternating adversarial training with a learned remover.

Usage (example):
  python scripts/train_perceptual_adversarial_full.py --config configs/wam_config.yaml \
      --data /path/to/images --out_dir ./runs/percep_adv --epochs 30 --batch_size 8

Important:
 - Provide a config YAML that contains 'embedder' and 'extractor' sub-configs
   following the repo's builder arg format. The script will call the repo's
   build_embedder and build_extractor helpers.
"""
import os
import argparse
import time
import yaml
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os, sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo/scripts -> repo
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# repo imports (builders)
from watermark_anything.models.embedder import build_embedder
from watermark_anything.models.extractor import build_extractor
from watermark_anything.models.wam import Wam
from watermark_anything.losses.detperceptual import LPIPSWithDiscriminator
from watermark_anything.attacks.learned_remover import LearnedRemover
from watermark_anything.modules.ecc import RepetitionECC, LearnedECCDecoder, ber_from_bits
from omegaconf import OmegaConf

# optional imports (augmenter / JND)
try:
    from watermark_anything.augmentation.augmenter import Augmenter
except Exception:
    Augmenter = None

try:
    from watermark_anything.modules.jnd import JND
except Exception:
    JND = None


def build_dataloader(data_root, image_size, batch_size, num_workers=4):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(data_root, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader


# def compute_message_loss_from_raw(preds_enc_logits, masks, raw_msgs, ecc_decoder, rep, use_learned_decoder=False):
#     """
#     preds_enc_logits: B x (k_enc) x H x W  (detector output for encoded bits as logits)
#     masks: B x z x H x W  (mask channels)
#     raw_msgs: B x k_raw  (0/1)
#     ecc_decoder: LearnedECCDecoder instance (if use_learned_decoder)
#     rep: repetition factor used in encoding
#     Returns BCE loss between decoded raw bits and raw_msgs (averaged).
#     """
#     b, kenc, H, W = preds_enc_logits.shape
#     device = preds_enc_logits.device
#     # average logits across mask region for each encoded bit per image
#     # flatten spatial dims and average within masked pixels
#     k_raw = raw_msgs.shape[1]
#     preds_enc_logits = preds_enc_logits.view(b, kenc, -1)  # b x kenc x (H*W)
#     # pick only pixels within mask (we assume masks include a channel per message index or single full mask)
#     # We'll average over full image as fallback
#     mean_logits = preds_enc_logits.mean(dim=2)  # b x kenc
#
#     if use_learned_decoder and ecc_decoder is not None:
#         # Learned decoder expects shape b x (k_raw * rep)
#         logits_raw = ecc_decoder(mean_logits)
#     else:
#         # majority decode from logits: use sigmoid then average rep groups
#         rep_factor = rep
#         assert kenc == k_raw * rep_factor
#         probs = torch.sigmoid(mean_logits)
#         probs = probs.view(b, k_raw, rep_factor)
#         mean_probs = probs.mean(dim=2)
#         logits_raw = torch.log(mean_probs.clamp(1e-6, 1 - 1e-6) / (1 - mean_probs.clamp(1e-6, 1 - 1e-6)))
#         # logits_raw are approximate logits from averaged probabilities
#
#     # BCE loss per raw bit
#     bce = nn.BCEWithLogitsLoss()
#     loss = bce(logits_raw, raw_msgs.float())
#     return loss, logits_raw

def compute_message_loss_from_raw(preds_enc_logits, masks, raw_msgs, ecc_decoder, rep, use_learned_decoder=False):
    """
    Mask-aware message loss.

    preds_enc_logits: B x k_enc x H x W (logits for encoded bits)
    masks: None OR tensor of shape:
           - B x 1 x H x W : same mask used for all encoded bits (1==watermarked)
           - B x k_enc x H x W : per-encoded-bit mask
    raw_msgs: B x k_raw (0/1)
    ecc_decoder: learned decoder instance (optional)
    rep: repetition factor (k_enc == k_raw * rep)
    Returns: (loss, logits_raw) where logits_raw is B x k_raw
    """
    b, kenc, H, W = preds_enc_logits.shape
    device = preds_enc_logits.device
    k_raw = raw_msgs.shape[1]

    # reshape for per-pixel operations
    preds_flat = preds_enc_logits.view(b, kenc, -1)  # B x kenc x (H*W)

    if masks is None:
        # fallback: mean over all spatial positions
        mean_logits = preds_flat.mean(dim=2)  # B x kenc
    else:
        # prepare mask: allow Bx1xHxW or BxkencxHxW
        if masks.dim() == 4 and masks.shape[1] == 1:
            mask = masks.repeat(1, kenc, 1, 1)  # B x kenc x H x W
        elif masks.dim() == 4 and masks.shape[1] == kenc:
            mask = masks
        else:
            # incompatible mask shape; fall back to full-image mean with a warning
            print(f"Warning: masks shape {tuple(masks.shape)} not compatible with preds kenc={kenc}. Using full-image average.")
            mean_logits = preds_flat.mean(dim=2)
            mask = None

        if mask is not None:
            mask_flat = mask.view(b, kenc, -1).to(device).float()  # B x kenc x (H*W)
            # Avoid zero division: compute sum and replace zeros with 1.0
            mask_sum = mask_flat.sum(dim=2)  # B x kenc
            # Weighted sum of logits within mask
            masked_sum = (preds_flat * mask_flat).sum(dim=2)  # B x kenc
            # If mask_sum is zero, we fall back to full mean for that image/bit
            fallback_mean = preds_flat.mean(dim=2)
            # compute safe mean
            mask_sum_safe = mask_sum.clone()
            zero_mask = mask_sum_safe == 0
            mask_sum_safe[zero_mask] = 1.0
            mean_logits = masked_sum / mask_sum_safe  # B x kenc
            # restore fallback where mask was empty
            if zero_mask.any():
                mean_logits[zero_mask] = fallback_mean[zero_mask]

    # At this point mean_logits is B x kenc
    if use_learned_decoder and (ecc_decoder is not None):
        logits_raw = ecc_decoder(mean_logits)  # B x k_raw
    else:
        # majority-style grouping of repeated encoded bits
        assert kenc == k_raw * rep, f"kenc ({kenc}) != k_raw ({k_raw}) * rep ({rep})"
        probs = torch.sigmoid(mean_logits)
        probs = probs.view(b, k_raw, rep)
        mean_probs = probs.mean(dim=2)  # B x k_raw
        # approximate logits from averaged probs
        logits_raw = torch.log(mean_probs.clamp(1e-6, 1 - 1e-6) / (1 - mean_probs.clamp(1e-6, 1 - 1e-6)))

    loss = nn.BCEWithLogitsLoss()(logits_raw, raw_msgs.float())
    return loss, logits_raw
# def maybe_make_augmenter(cfg):
#     if Augmenter is None:
#         return None
#     try:
#         return Augmenter(**(cfg or {}))
#     except Exception:
#         return Augmenter()
def maybe_make_augmenter(cfg):
    """
    Build an Augmenter from:
      - a path to a YAML config,
      - an OmegaConf/ dict config,
      - or None (in which case we try to load configs/all_augs.yaml from repo root).
    Returns:
      - an Augmenter instance, or None if Augmenter is unavailable or construction fails.
    """
    if Augmenter is None:
        print("Augmenter class not available in repo; continuing without augmenter.")
        return None

    # resolve repo root and default config path (scripts/ -> repo root)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_augmenter_path = os.path.join(repo_root, "configs", "all_augs.yaml")

    try:
        # If cfg is a string, treat as path to yaml
        if isinstance(cfg, str):
            if os.path.exists(cfg):
                aug_conf = OmegaConf.load(cfg)
            elif os.path.exists(default_augmenter_path):
                print(f"Provided augmenter path '{cfg}' not found; loading default {default_augmenter_path}")
                aug_conf = OmegaConf.load(default_augmenter_path)
            else:
                print(
                    f"Provided augmenter path '{cfg}' not found and default {default_augmenter_path} missing. No augmenter will be used.")
                return None
        elif cfg is None:
            # try loading default augmenter config
            if os.path.exists(default_augmenter_path):
                aug_conf = OmegaConf.load(default_augmenter_path)
            else:
                print(
                    f"No augmenter config provided and default config not found at {default_augmenter_path}. No augmenter will be used.")
                return None
        else:
            # cfg is dict-like or OmegaConf; convert to OmegaConf if needed
            if OmegaConf.is_config(cfg):
                aug_conf = cfg
            else:
                aug_conf = OmegaConf.create(cfg)
            # aug_conf = OmegaConf.create(cfg) if not isinstance(cfg, OmegaConf.__class__) else cfg

        # Convert to plain dict resolving references
        aug_kwargs = OmegaConf.to_container(aug_conf, resolve=True)

        # Augmenter expects specific keys like 'masks', 'augs', 'augs_params' (see repo configs)
        augmenter = Augmenter(**aug_kwargs)
        print(f"Augmenter built from config, using keys: {list(aug_kwargs.keys())}")
        return augmenter
    except Exception as e:
        print(f"Warning: failed to build Augmenter from config: {e}\nContinuing without augmenter.")
        return None


def maybe_make_jnd(cfg):
    if JND is None:
        return None
    try:
        return JND(**(cfg or {}))
    except Exception:
        return None


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare dataloader
    loader = build_dataloader(args.data, args.img_size, args.batch_size, num_workers=args.num_workers)

    # parse config and build models
    cfg = {}
    if args.config is None:
        raise RuntimeError("Please pass --config that describes embedder and extractor builder configurations.")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # ECC parameters
    rep = args.rep
    nbits = args.nbits  # raw bits
    k_enc = nbits * rep

    # Build embedder (name and cfg expected)
    # in scripts/train_perceptual_adversarial_full.py, near where cfg is loaded

    embedder_cfg = cfg.get('embedder', {})
    extractor_cfg = cfg.get('extractor', {})
    embedder_name = embedder_cfg.get('name', 'vae')
    extractor_name = extractor_cfg.get('name', 'sam')

    # Accept multiple possible shapes in the YAML:
    # embedder: { name: vae, cfg: { ... } }  OR embedder: { name: vae, ...params... }
    embedder_params_raw = embedder_cfg.get('cfg', embedder_cfg.get('params', embedder_cfg))
    extractor_params_raw = extractor_cfg.get('cfg', extractor_cfg.get('params', extractor_cfg))

    # Convert raw dicts into OmegaConf DictConfig so build_embedder can use attribute access
    embedder_params = OmegaConf.create(embedder_params_raw)
    extractor_params = OmegaConf.create(extractor_params_raw)

    # Ensure there's a msg_processor section (build_embedder mutates it)
    if 'msg_processor' not in embedder_params:
        embedder_params.msg_processor = OmegaConf.create(
            {'nbits': 0, 'hidden_size': 0, 'msg_processor_type': 'binary+concat'})

    # Build the models (k_enc = nbits * rep calculated earlier)
    embedder = build_embedder(embedder_name, embedder_params, nbits=k_enc)
    extractor = build_extractor(extractor_name, extractor_params, img_size=args.img_size, nbits=k_enc)
    augmenter = maybe_make_augmenter(cfg.get('augmenter'))
    attenuation = maybe_make_jnd(cfg.get('jnd'))

    wam = Wam(embedder=embedder, detector=extractor, augmenter=augmenter, attenuation=attenuation,
              scaling_w=args.scaling_w, scaling_i=args.scaling_i, roll_probability=args.roll_probability,
              img_size_extractor=args.img_size)

    wam.to(device).train()

    # ECC modules
    ecc = RepetitionECC(rep=rep)
    learned_decoder = None
    if args.use_learned_decoder:
        learned_decoder = LearnedECCDecoder(k_raw=nbits, rep=rep)
        learned_decoder.to(device)

    # attack, loss modules, optimizers
    attack_net = LearnedRemover(in_channels=3, base_ch=args.attack_base_ch, depth=args.attack_depth,
                                tanh_scale=args.attack_scale)
    attack_net.to(device)

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
    )
    loss_module.to(device)

    opt_attack = torch.optim.AdamW(attack_net.parameters(), lr=args.lr_attack, weight_decay=args.weight_decay)
    params_mod = list(wam.embedder.parameters()) + list(wam.detector.parameters())
    if learned_decoder is not None:
        params_mod += list(learned_decoder.parameters())
    # opt_wam = torch.optim.AdamW(params_mod, lr=args.lr_wam, weight_decay=args.weight_decay)
    opt_wam = torch.optim.AdamW(list(wam.embedder.parameters()) + list(wam.detector.parameters()), lr=args.lr_wam,
                                weight_decay=1e-2)
    # Replace the unconditional opt_disc creation with this conditional block:
    if hasattr(loss_module, 'discriminator') and getattr(loss_module, 'discriminator') is not None:
        opt_disc = torch.optim.AdamW(loss_module.discriminator.parameters(), lr=args.lr_disc,
                                     weight_decay=args.weight_decay)
    else:
        opt_disc = None
    opt_disc = torch.optim.AdamW(loss_module.discriminator.parameters(), lr=args.lr_disc,
                                 weight_decay=args.weight_decay)
    opt_decoder = None
    if learned_decoder is not None:
        opt_decoder = torch.optim.AdamW(learned_decoder.parameters(), lr=args.lr_wam, weight_decay=args.weight_decay)

    global_step = 0
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_logs = defaultdict(list)
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            b = imgs.shape[0]

            # generate raw messages and encoded messages
            raw_msgs = torch.randint(0, 2, (b, nbits), device=device).float()
            encoded_msgs = ecc.encode(raw_msgs).to(device)  # b x k_enc

            # Train attack (maximize message loss)
            for _ in range(args.attack_steps):
                opt_attack.zero_grad()
                wam_outputs = wam.embed(imgs, msgs=encoded_msgs)
                imgs_w = wam_outputs['imgs_w']
                print("Embedder requires_grad (sample):", any(p.requires_grad for p in wam.embedder.parameters()))
                # loss.backward(retain_graph=True)  # or the actual backward call you already use
                attacked = imgs_w + attack_net(imgs_w)
                attacked = attacked.clamp(0., 1.)
                preds = wam.detect(attacked)['preds']  # logits: b x (1 + k_enc) x H x W or similar
                # detector outputs first channel mask; encoded bits start at index 1
                pred_enc_logits = preds[:, 1:, :, :] if preds.shape[1] > 1 else preds[:, :1, :, :]
                msg_loss, _ = compute_message_loss_from_raw(pred_enc_logits, None, raw_msgs, learned_decoder, rep,
                                                            use_learned_decoder=args.use_learned_decoder)
                attack_loss = -msg_loss
                attack_loss.backward()

                opt_attack.step()

            # Discriminator update
            opt_disc.zero_grad()
            # with torch.no_grad():
            #     wam_outputs = wam.embed(imgs, msgs=encoded_msgs)
            #     imgs_w = wam_outputs['imgs_w']
            wam.train()  # ensure module is in train mode so parameters require grad behavior is normal
            wam_outputs = wam.embed(imgs, msgs=encoded_msgs)
            imgs_w = wam_outputs['imgs_w']
            # print("imgs.shape", imgs.shape, "imgs_w.shape", imgs_w.shape)
            preds_clean = wam.detect(imgs_w)['preds']
            d_loss, d_log = loss_module(inputs=imgs, reconstructions=imgs_w,
                                        masks=torch.ones((b, 1, args.img_size, args.img_size), device=device),
                                        msgs=raw_msgs, preds=preds_clean,
                                        optimizer_idx=1, global_step=global_step,
                                        last_layer=wam.embedder.get_last_layer())
            d_loss.backward()
            opt_disc.step()

            # Train wam + learned decoder (if any) to survive
            for _ in range(args.wam_steps):
                opt_wam.zero_grad()
                if opt_decoder is not None:
                    opt_decoder.zero_grad()
                wam_outputs = wam.embed(imgs, msgs=encoded_msgs)
                imgs_w = wam_outputs['imgs_w']
                attacked = imgs_w + attack_net(imgs_w)
                attacked = attacked.clamp(0., 1.)
                preds_attacked = wam.detect(attacked)['preds']
                pred_enc_logits_attacked = preds_attacked[:, 1:, :, :] if preds_attacked.shape[
                                                                              1] > 1 else preds_attacked[:, :1, :, :]

                msg_loss, logits_raw = compute_message_loss_from_raw(pred_enc_logits_attacked, None, raw_msgs,
                                                                     learned_decoder, rep,
                                                                     use_learned_decoder=args.use_learned_decoder)
                total_percep_loss, log = loss_module(inputs=imgs, reconstructions=imgs_w,
                                                     masks=torch.ones((b, 1, args.img_size, args.img_size),
                                                                      device=device),
                                                     msgs=raw_msgs, preds=wam.detect(imgs_w)['preds'],
                                                     optimizer_idx=0, global_step=global_step,
                                                     last_layer=wam.embedder.get_last_layer())

                total_loss = args.lambda_msg * msg_loss + total_percep_loss
                total_loss.backward()
                opt_wam.step()
                if opt_decoder is not None:
                    opt_decoder.step()

            epoch_logs['msg_loss'].append(msg_loss.item() if torch.is_tensor(msg_loss) else float(msg_loss))
            epoch_logs['percep_loss'].append(log['percep_loss'].item() if 'percep_loss' in log else 0.0)
            epoch_logs['disc_loss'].append(d_log['disc_loss'].item() if 'disc_loss' in d_log else 0.0)

            global_step += 1

            if (i + 1) % args.log_interval == 0:
                print(
                    f"Epoch {epoch} step {i + 1}/{len(loader)} | msg_loss={epoch_logs['msg_loss'][-1]:.4f} percep={epoch_logs['percep_loss'][-1]:.4f} disc={epoch_logs['disc_loss'][-1]:.4f}")

        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch} finished in {epoch_time:.1f}s. Mean msg_loss={sum(epoch_logs['msg_loss']) / len(epoch_logs['msg_loss']):.4f}")
        # Replace the existing torch.save({...}) with the following
        ckpt = {
            'attack': attack_net.state_dict(),
            'wam': wam.state_dict(),
            'loss_module': loss_module.state_dict(),
            'opt_attack': opt_attack.state_dict(),
            'opt_wam': opt_wam.state_dict(),
        }
        if opt_disc is not None:
            ckpt['opt_disc'] = opt_disc.state_dict()
        if opt_decoder is not None:
            ckpt['opt_decoder'] = opt_decoder.state_dict()

        torch.save(ckpt, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt"))
        # torch.save({
        #     'attack': attack_net.state_dict(),
        #     'wam': wam.state_dict(),
        #     'loss_module': loss_module.state_dict(),
        #     'opt_attack': opt_attack.state_dict(),
        #     'opt_wam': opt_wam.state_dict(),
        # }, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt"))

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='YAML config with embedder/extractor params')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./outputs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--attack_steps', type=int, default=1)
    parser.add_argument('--wam_steps', type=int, default=1)
    parser.add_argument('--attack_base_ch', type=int, default=32)
    parser.add_argument('--attack_depth', type=int, default=3)
    parser.add_argument('--attack_scale', type=float, default=0.05)
    parser.add_argument('--lr_attack', type=float, default=2e-4)
    parser.add_argument('--lr_wam', type=float, default=2e-4)
    parser.add_argument('--lr_disc', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--percep_weight', type=float, default=1.0)
    parser.add_argument('--disc_weight', type=float, default=1.0)
    parser.add_argument('--detect_weight', type=float, default=1.0)
    parser.add_argument('--decode_weight', type=float, default=0.0)
    parser.add_argument('--disc_start', type=int, default=0)
    parser.add_argument('--disc_num_layers', type=int, default=3)
    parser.add_argument('--percep_loss', type=str, default='lpips')
    parser.add_argument('--lambda_msg', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--nbits', type=int, default=16, help='raw message bits (before ECC)')
    parser.add_argument('--rep', type=int, default=3, help='repetition factor for ECC; encoded bits = nbits * rep')
    parser.add_argument('--use_learned_decoder', action='store_true', help='train learned ECC decoder')
    parser.add_argument('--scaling_w', type=float, default=1.0)
    parser.add_argument('--scaling_i', type=float, default=1.0)
    parser.add_argument('--roll_probability', type=float, default=0.0)
    args = parser.parse_args()
    train(args)

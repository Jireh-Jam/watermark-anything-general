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

# repo imports (builders)
from watermark_anything.models.embedder import build_embedder
from watermark_anything.models.extractor import build_extractor
from watermark_anything.models.wam import Wam
from watermark_anything.losses.detperceptual import LPIPSWithDiscriminator
from watermark_anything.attacks.learned_remover import LearnedRemover
from watermark_anything.modules.ecc import RepetitionECC, LearnedECCDecoder, ber_from_bits

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


def compute_message_loss_from_raw(preds_enc_logits, masks, raw_msgs, ecc_decoder, rep, use_learned_decoder=False):
    """
    preds_enc_logits: B x (k_enc) x H x W  (detector output for encoded bits as logits)
    masks: B x z x H x W  (mask channels)
    raw_msgs: B x k_raw  (0/1)
    ecc_decoder: LearnedECCDecoder instance (if use_learned_decoder)
    rep: repetition factor used in encoding
    Returns BCE loss between decoded raw bits and raw_msgs (averaged).
    """
    b, kenc, H, W = preds_enc_logits.shape
    device = preds_enc_logits.device
    # average logits across mask region for each encoded bit per image
    # flatten spatial dims and average within masked pixels
    k_raw = raw_msgs.shape[1]
    preds_enc_logits = preds_enc_logits.view(b, kenc, -1)  # b x kenc x (H*W)
    # pick only pixels within mask (we assume masks include a channel per message index or single full mask)
    # We'll average over full image as fallback
    mean_logits = preds_enc_logits.mean(dim=2)  # b x kenc

    if use_learned_decoder and ecc_decoder is not None:
        # Learned decoder expects shape b x (k_raw * rep)
        logits_raw = ecc_decoder(mean_logits)
    else:
        # majority decode from logits: use sigmoid then average rep groups
        rep_factor = rep
        assert kenc == k_raw * rep_factor
        probs = torch.sigmoid(mean_logits)
        probs = probs.view(b, k_raw, rep_factor)
        mean_probs = probs.mean(dim=2)
        logits_raw = torch.log(mean_probs.clamp(1e-6, 1 - 1e-6) / (1 - mean_probs.clamp(1e-6, 1 - 1e-6)))
        # logits_raw are approximate logits from averaged probabilities

    # BCE loss per raw bit
    bce = nn.BCEWithLogitsLoss()
    loss = bce(logits_raw, raw_msgs.float())
    return loss, logits_raw


def maybe_make_augmenter(cfg):
    if Augmenter is None:
        return None
    try:
        return Augmenter(**(cfg or {}))
    except Exception:
        return Augmenter()


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
    embedder_cfg = cfg.get('embedder', {})
    extractor_cfg = cfg.get('extractor', {})
    embedder_name = embedder_cfg.get('name', 'vae')
    extractor_name = extractor_cfg.get('name', 'sam')

    # call build helpers: build_embedder(name, cfg, nbits)
    embedder = build_embedder(embedder_name, embedder_cfg.get('cfg', embedder_cfg.get('params', {})), nbits=k_enc)
    extractor = build_extractor(extractor_name, extractor_cfg.get('cfg', extractor_cfg.get('params', {})), img_size=args.img_size, nbits=k_enc)

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
    attack_net = LearnedRemover(in_channels=3, base_ch=args.attack_base_ch, depth=args.attack_depth, tanh_scale=args.attack_scale)
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
    opt_wam = torch.optim.AdamW(params_mod, lr=args.lr_wam, weight_decay=args.weight_decay)
    opt_disc = torch.optim.AdamW(loss_module.discriminator.parameters(), lr=args.lr_disc, weight_decay=args.weight_decay)
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
                attacked = imgs_w + attack_net(imgs_w)
                attacked = attacked.clamp(0., 1.)
                preds = wam.detect(attacked)['preds']  # logits: b x (1 + k_enc) x H x W or similar
                # detector outputs first channel mask; encoded bits start at index 1
                pred_enc_logits = preds[:, 1:, :, :] if preds.shape[1] > 1 else preds[:, :1, :, :]
                msg_loss, _ = compute_message_loss_from_raw(pred_enc_logits, None, raw_msgs, learned_decoder, rep, use_learned_decoder=args.use_learned_decoder)
                attack_loss = -msg_loss
                attack_loss.backward()
                opt_attack.step()

            # Discriminator update
            opt_disc.zero_grad()
            with torch.no_grad():
                wam_outputs = wam.embed(imgs, msgs=encoded_msgs)
                imgs_w = wam_outputs['imgs_w']
            preds_clean = wam.detect(imgs_w)['preds']
            d_loss, d_log = loss_module(inputs=imgs, reconstructions=imgs_w, masks=torch.ones((b,1,args.img_size,args.img_size),device=device), msgs=raw_msgs, preds=preds_clean,
                                        optimizer_idx=1, global_step=global_step, last_layer=wam.embedder.get_last_layer())
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
                pred_enc_logits_attacked = preds_attacked[:, 1:, :, :] if preds_attacked.shape[1] > 1 else preds_attacked[:, :1, :, :]

                msg_loss, logits_raw = compute_message_loss_from_raw(pred_enc_logits_attacked, None, raw_msgs, learned_decoder, rep, use_learned_decoder=args.use_learned_decoder)
                total_percep_loss, log = loss_module(inputs=imgs, reconstructions=imgs_w, masks=torch.ones((b,1,args.img_size,args.img_size),device=device),
                                                     msgs=raw_msgs, preds=wam.detect(imgs_w)['preds'],
                                                     optimizer_idx=0, global_step=global_step, last_layer=wam.embedder.get_last_layer())

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
                print(f"Epoch {epoch} step {i+1}/{len(loader)} | msg_loss={epoch_logs['msg_loss'][-1]:.4f} percep={epoch_logs['percep_loss'][-1]:.4f} disc={epoch_logs['disc_loss'][-1]:.4f}")

        epoch_time = time.time() - t0
        print(f"Epoch {epoch} finished in {epoch_time:.1f}s. Mean msg_loss={sum(epoch_logs['msg_loss'])/len(epoch_logs['msg_loss']):.4f}")

        torch.save({
            'attack': attack_net.state_dict(),
            'wam': wam.state_dict(),
            'loss_module': loss_module.state_dict(),
            'opt_attack': opt_attack.state_dict(),
            'opt_wam': opt_wam.state_dict(),
        }, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt"))

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
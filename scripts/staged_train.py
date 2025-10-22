#!/usr/bin/env python3
# """
# staged_train.py
#
# Simple wrapper that runs your existing training script in two stages:
#   1) Pretrain: embedder+detector only (no attacker, no discriminator)
#   2) Adversarial fine-tune: enable attacker and optionally discriminator
#
# This wrapper calls 'python scripts/train_perceptual_adversarial_full.py' with
# modified arguments. It does not modify your training script on disk.
# Usage example:
#   python scripts/staged_train.py \
#     --config configs/wam_config.yaml \
#     --data "C:/Users/User/Desktop/stable_signature/data" \
#     --out_dir ./runs/staged_run \
#     --pretrain_epochs 10 \
#     --adv_epochs 20 \
#     --nbits 8 --rep 1 --img_size 128 --batch_size 8
# """
import argparse
import shlex
import subprocess
import os
import sys

TRAIN_SCRIPT = os.path.join("scripts", "train_perceptual_adversarial_full.py")
PYTHON = sys.executable


def build_cmd(base_args, overrides: dict):
    args = base_args.copy()
    for k, v in overrides.items():
        # convert arg names to CLI form
        if isinstance(v, bool):
            if v:
                args += [f"--{k}"]
        else:
            args += [f"--{k}", str(v)]
    return [PYTHON, TRAIN_SCRIPT] + args


def run_stage(cmd_list):
    print("Running:", " ".join(shlex.quote(x) for x in cmd_list))
    proc = subprocess.run(cmd_list)
    if proc.returncode != 0:
        raise SystemExit(f"Stage failed with exit code {proc.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--adv_epochs", type=int, default=20)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attack_scale", type=float, default=0.01)
    parser.add_argument("--attack_steps", type=int, default=1)
    parser.add_argument("--lr_wam", type=float, default=2e-4)
    parser.add_argument("--lr_attack", type=float, default=1e-4)
    parser.add_argument("--disc_weight", type=float, default=0.0)
    parser.add_argument("--extra", type=str, default="", help="extra CLI args to append to both stages")
    args = parser.parse_args()

    # build base args common to both stages
    base_args = [
        "--config", args.config,
        "--data", args.data,
        "--nbits", str(args.nbits),
        "--rep", str(args.rep),
        "--img_size", str(args.img_size),
        "--batch_size", str(args.batch_size),
        "--lr_wam", str(args.lr_wam),
    ]
    if args.extra:
        base_args += shlex.split(args.extra)

    # Stage 1: Pretrain embedder+detector only (no attacker, no discriminator)
    pretrain_out = os.path.join(args.out_dir, "pretrain")
    os.makedirs(pretrain_out, exist_ok=True)
    pretrain_overrides = {
        "out_dir": pretrain_out,
        "epochs": args.pretrain_epochs,
        "attack_steps": 0,
        "disc_weight": 0,
        # keep wam_steps default in your script; optionally expose here
    }
    cmd_pretrain = build_cmd(base_args, pretrain_overrides)
    run_stage(cmd_pretrain)

    # Stage 2: Adversarial fine-tune (resume from latest pretrain checkpoint if your train script supports)
    adv_out = os.path.join(args.out_dir, "adv")
    os.makedirs(adv_out, exist_ok=True)
    adv_overrides = {
        "out_dir": adv_out,
        "epochs": args.adv_epochs,
        "attack_steps": args.attack_steps,
        "attack_scale": args.attack_scale,
        "lr_attack": args.lr_attack,
        "disc_weight": args.disc_weight,
        # You may add --resume or --checkpoint handling if your train script supports it.
    }
    cmd_adv = build_cmd(base_args, adv_overrides)
    run_stage(cmd_adv)

    print("Staged training finished successfully.")


if __name__ == "__main__":
    main()

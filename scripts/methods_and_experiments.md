# Methods and Experiments

## Overview
We build on the Watermark Anything Model (WAM) and propose a staged training procedure and mask-aware message loss designed to recover short messages from multiple small, localized watermarked regions. Our implementation follows the same high-level architecture: an embedder that imperceptibly modifies an image and an extractor that simultaneously segments watermarked areas and decodes messages from detected regions. We augment the training pipeline with (1) a staged schedule — pretraining the embedder/extractor for reliable decoding, then adversarial fine-tuning; and (2) a mask-aware message-loss which averages detector logits only over the watermarked pixels to prevent signal dilution when watermark regions are small.

## Architecture
- Embedder: same encoder/decoder backbone used in WAM; the embedder accepts an image and a bit-stream and produces a perturbed image of the same resolution. We preserve the repository's message-processing component (msg_processor) and ensure its nbits parameter equals encoded length (k_enc = nbits × rep).
- Extractor: segmentation head that outputs (1) a segmentation mask channel (watermarked vs not) and (2) k_enc logits (one per encoded bit) spatially. The detector's encoded-bit channels are averaged over mask pixels during decoding.
- ECC: a repetition encoder (RepetitionECC) is used in core experiments for robustness. Optionally, a learned ECC decoder can be trained jointly in later stages.
- Adversary: a learned remover (U-Net variant) trained to remove the watermark residual. To stabilize adversarial stages this attacker is introduced gradually and initialized with a small output scale.

## Training schedule
We use a three-stage training schedule:
1. Low-resolution and robustness pretraining (pretrain):
   - Objective: make the embedder+extractor co-adapt to reliably encode and decode messages without perceptual constraints.
   - Loss: message BCE on decoded raw bits (mask-aware), no perceptual loss, attacker disabled.
   - Purpose: confirm capacity for reliable decoding on clean watermarked images.

2. Perceptual fine-tuning:
   - Objective: recover imperceptibility lost during pretraining.
   - Loss: LPIPS (or combined LPIPS + adversarial discriminator) applied to the watermarked image vs original, plus the message loss.
   - Purpose: reduce visible artifacts while preserving decodability.

3. Adversarial fine-tuning:
   - Objective: robustness to removal attacks (inpainting, learned remover).
   - Training alternates updates:
     a. Attacker update(s): maximize message loss by producing residuals added to watermarked images.
     b. Embedder+extractor update(s): minimize weighted sum of message loss and perceptual loss on attacked examples (mask-aware message loss).
   - The attacker is incrementally strengthened: start with small tanh_scale (0.005–0.02) and attack_steps=1, then grow.

Staged training is implemented via a wrapper that runs the pretrain and adversarial stages with separate flags; the training script itself accepts flags to toggle attack/discriminator and to set per-stage hyperparameters.

## Mask-aware message-loss
When watermark regions are small relative to the image, averaging detector logits over the whole image dilutes the signal and impedes learning. We therefore compute the mean encoded-bit logits per image restricted to the pixels inside the mask:
- If a single-channel mask is provided (B×1×H×W), replicate it across the k_enc bit channels;
- If per-bit masks are provided (B×k_enc×H×W) use them directly;
- Compute per-image, per-bit masked average logit = sum(logit * mask) / sum(mask) (fallback to full-image mean when mask is empty).
- Group repeated encoded bits and convert to raw logits either via a learned ECC decoder or by averaging probabilities within repetition groups and converting back to logits for BCE.

## Datasets and data synthesis
- Base images: use the same datasets as prior WAM experiments (COCO, Places or the repository's benchmarks) resized to 256×256 for primary experiments.
- Synthetic splices: paste small patches (2.5%, 5%, 10% of image area) from donor images into hosts to simulate splicing. Place patches at random positions and optionally blend edges.
- Inpainting: remove and fill small regions using a classical or learned inpainter to evaluate inpainting robustness.
- Standard augmentations (resize, crop, color jitter) are applied consistently during training; the augmenter module is optional and used in final fine-tune.

## Baselines and ablations
- Baselines:
  - Original WAM (re-implemented or from published weights).
  - Global watermarking method (e.g., StegaStamp) as a non-localized baseline.
- Ablations:
  - no mask-aware loss (full-image average)
  - no staged training (single-phase with attacker from start)
  - no message-adaptive embedding
  - no hierarchical extractor (single-scale extractor)
Each ablation isolates the contribution of the corresponding design choice.

## Metrics
- Message recovery: Bit Error Rate (BER) and % messages with ≤1 bit error over test images and attacked variants.
- Localization: IoU for predicted watermark masks vs ground-truth watermark region(s); precision/recall curves and mAP across IoU thresholds.
- Imperceptibility: LPIPS, PSNR, SSIM between original and watermarked image (and between original and attacked outputs when applicable).
- Robustness: BER under inpainting, splicing, compression (JPEG), blur, and learned remover attacks.
- Compute: parameter counts, inference time per 256×256 image, and training GPU-hours.

## Visualization and reporting
- Qualitative examples of spliced images with predicted masks and recovered bit vectors and bit-error annotations.
- Plots:
  - BER vs watermark area fraction (2.5%, 5%, 10%)
  - BER vs attack strength (attacker scale)
  - IoU distribution and mean IoU for mask predictions
  - Tradeoff curves showing LPIPS vs BER to illustrate imperceptibility vs robustness.
- Tables: numerical BERs and LPIPS across baselines and ablation variants.

## Expected outcomes and criteria
- X-WAM should significantly reduce BER on small, localized watermarks compared to baselines (target: <1 bit error on average for 32-bit payloads on regions ≤10% area).
- Localization IoU should be meaningfully higher than the original WAM on spliced examples, reflecting better segmentation of small watermarked areas.
- Imperceptibility (LPIPS) should remain competitive with WAM after perceptual fine-tuning.
- Ablations should show that mask-aware loss and staged training are the primary contributors to improvements on small-region tasks.

## Practical notes
- For small watermarked regions, masks are sparse: ensure balanced gradient contributions (mask-aware averaging) and consider upweighting message loss in the small-region regime.
- Pretraining without adversary is crucial: do not introduce a strong attacker before the embedder+extractor have converged on reliable decoding.
- Use a repetition ECC or a learned decoder to tradeoff payload capacity vs robustness; experiment with rep ∈ {1, 3} and nbits ∈ {8, 16, 32}.
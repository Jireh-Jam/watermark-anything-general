"""
Simple ECC utilities: repetition encoder + majority-vote decoder, and a small
learned decoder (MLP) that maps encoded-bit logits to decoded bits. This file
is intentionally small and self-contained so that it is easy to run unit tests.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RepetitionECC:
    """Simple repetition code encoder/decoder.

    Encodes a B x K binary tensor into B x (K * rep) by repeating each bit rep times.
    Decoding can be done with majority vote (on binaries) or majority on logits.
    """
    def __init__(self, rep: int = 3):
        assert rep >= 1 and isinstance(rep, int)
        self.rep = rep

    def encode(self, msgs: torch.Tensor) -> torch.Tensor:
        """Encode raw messages by repeating bits.

        Args:
            msgs: tensor B x K with 0/1 values (or floats)
        Returns:
            encoded: B x (K*rep)
        """
        if msgs is None:
            return None
        b, k = msgs.shape
        # repeat along a new axis
        msgs = msgs.view(b, k, 1).float()
        encoded = msgs.repeat(1, 1, self.rep).view(b, k * self.rep)
        return encoded

    def decode_majority_from_logits(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Decode using majority on probabilities/logits.

        Args:
            logits: B x (K*rep) either raw logits or probabilities (0..1)
        Returns:
            decoded: B x K of 0/1 (float tensor)
        """
        if logits is None:
            return None
        probs = torch.sigmoid(logits)
        b, kr = probs.shape
        assert kr % self.rep == 0
        k = kr // self.rep
        probs = probs.view(b, k, self.rep)
        mean_probs = probs.mean(dim=2)
        decoded = (mean_probs >= threshold).float()
        return decoded

    def decode_majority_from_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Decode by simple majority on binary bits.

        Args:
            bits: B x (K*rep) of 0/1
        Returns:
            decoded: B x K of 0/1
        """
        if bits is None:
            return None
        b, kr = bits.shape
        assert kr % self.rep == 0
        k = kr // self.rep
        bits = bits.view(b, k, self.rep)
        s = bits.sum(dim=2)
        decoded = (s >= (self.rep / 2.0)).float()
        return decoded


class LearnedECCDecoder(nn.Module):
    """Simple learned decoder: MLP that maps encoded-bit logits (or probs)
    to decoded raw bits. This is used optionally to improve decoding over
    majority voting.

    Input: B x (K * rep) logits or probabilities.
    Output: B x K logits for decoded bits (before sigmoid).
    """
    def __init__(self, k_raw: int, rep: int = 3, hidden: Optional[int] = None):
        super().__init__()
        self.k_raw = k_raw
        self.rep = rep
        self.k_enc = k_raw * rep
        if hidden is None:
            hidden = max(128, self.k_enc // 2)
        self.net = nn.Sequential(
            nn.Linear(self.k_enc, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.k_raw),
        )

    def forward(self, logits_enc: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        logits_enc can be raw logits or probabilities; the decoder expects real
        values and outputs raw logits for k_raw bits.
        """
        b, kr = logits_enc.shape
        assert kr == self.k_enc, f"Expected input width {self.k_enc}, got {kr}"
        out = self.net(logits_enc)
        return out


# convenience functions

def ber_from_bits(pred_bits: torch.Tensor, target_bits: torch.Tensor) -> float:
    """Compute bit error rate between prediction and target (both 0/1 tensors).
    Returns scalar float in [0,1]."""
    assert pred_bits.shape == target_bits.shape
    b = pred_bits.numel()
    if b == 0:
        return 0.0
    return float((pred_bits != target_bits).float().sum().item() / b)


__all__ = ["RepetitionECC", "LearnedECCDecoder", "ber_from_bits"]
"""
Small learned remover (lightweight U-Net style) that produces a residual image
which is added to a watermarked image to produce an attacked (watermark-removed)
image. The network is intentionally small so that training is fast for experiments.

This version fixes a channel mismatch by letting Down blocks do ONLY pooling,
while encoder ConvBlocks handle channel growth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, norm: bool = True):
        super().__init__()
        padding = kernel // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downsample only (no channel change)."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Up(nn.Module):
    """Upsample, concat skip, then conv to out_ch."""
    def __init__(self, in_ch: int, out_ch: int):
        # in_ch should equal (prev_decoder_channels + skip_channels) when skip is provided
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LearnedRemover(nn.Module):
    def __init__(self, in_channels: int = 3, base_ch: int = 32, depth: int = 4, tanh_scale: float = 0.05):
        """
        in_channels: input image channels (usually 3)
        base_ch: base number of channels
        depth: number of encoder levels (>=1)
        tanh_scale: scales the residual so magnitude stays small
        """
        super().__init__()
        assert depth >= 1
        self.depth = depth
        self.tanh_scale = tanh_scale

        # Channels per encoder level, e.g. [32, 64, 128, 256]
        channels = [base_ch * (2 ** i) for i in range(depth)]

        # Encoder: conv blocks that grow channels; pooling handled by Down
        encs: list[nn.Module] = []
        prev_ch = in_channels
        for ch in channels:
            encs.append(ConvBlock(prev_ch, ch))
            prev_ch = ch
        self.encs = nn.ModuleList(encs)

        # Downsamplers: depth-1 pool-only modules
        self.downs = nn.ModuleList([Down() for _ in range(depth - 1)])

        # Bottleneck: double channels at bottom
        bottleneck_ch = channels[-1] * 2
        self.bottleneck = ConvBlock(channels[-1], bottleneck_ch)

        # Decoder: each Up takes (prev_decoder_channels + matching_skip_channels) -> skip_ch
        ups: list[nn.Module] = []
        prev_dec_ch = bottleneck_ch
        for i in reversed(range(depth)):
            skip_ch = channels[i]
            in_ch = prev_dec_ch + skip_ch
            out_ch = skip_ch
            ups.append(Up(in_ch, out_ch))
            prev_dec_ch = out_ch
        self.ups = nn.ModuleList(ups)

        # Final conv to residual perturbation: lowest-level channels -> in_channels
        self.final = nn.Conv2d(channels[0], in_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skips
        skips = []
        out = x
        for i, enc in enumerate(self.encs):
            out = enc(out)            # channel change happens here
            skips.append(out)         # store skip BEFORE spatial downsampling
            if i < len(self.downs):   # downsample except after last level
                out = self.downs[i](out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder mirrors encoder
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 1)]
            out = up(out, skip=skip)

        # Residual perturbation
        delta = torch.tanh(self.final(out)) * self.tanh_scale
        return delta

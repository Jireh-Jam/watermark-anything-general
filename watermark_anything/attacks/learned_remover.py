"""
Small learned remover (lightweight U-Net style) that produces a residual image
which is added to a watermarked image to produce an attacked (watermark-removed)
image. The network is intentionally small so that training is fast for experiments.

Usage:
    from watermark_anything.attacks.learned_remover import LearnedRemover
    attack = LearnedRemover(in_channels=3, base_ch=32)
    attacked_imgs = imgs + attack(imgs)
    attacked_imgs = attacked_imgs.clamp(0., 1.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, norm=True):
        super().__init__()
        padding = kernel // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # pad if needed
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LearnedRemover(nn.Module):
    def __init__(self, in_channels=3, base_ch=32, depth=4, tanh_scale=0.05):
        """
        in_channels: input image channels (usually 3)
        base_ch: base number of channels
        depth: depth of the U-Net (4 recommended)
        tanh_scale: scale the tanh output so the perturbation magnitude is small
        """
        super().__init__()
        self.depth = depth
        self.tanh_scale = tanh_scale

        # encoder
        encs = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            encs.append(ConvBlock(ch, out_ch))
            ch = out_ch
        self.encs = nn.ModuleList(encs)
        self.downs = nn.ModuleList([Down(ch, ch) for ch in [base_ch * (2 ** i) for i in range(depth - 1)]])

        # bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2)

        # decoder
        decs = []
        ups = []
        for i in reversed(range(depth)):
            in_ch = ch * 2 if i == depth - 1 else (base_ch * (2 ** i)) * 2
            out_ch = base_ch * (2 ** i)
            ups.append(Up(in_ch, out_ch))
            ch = out_ch
        self.ups = nn.ModuleList(ups)

        # final conv to perturbation
        self.final = nn.Conv2d(base_ch, in_channels, 3, padding=1)

        # initialize weights
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

    def forward(self, x):
        # encoder with skip connections
        skips = []
        out = x
        for enc in self.encs:
            out = enc(out)
            skips.append(out)
            out = F.avg_pool2d(out, 2)

        out = self.bottleneck(out)

        # decode
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 1)]
            out = up(out, skip=skip)

        delta = torch.tanh(self.final(out)) * self.tanh_scale
        # return residual perturbation; caller should add to image and clamp
        return delta
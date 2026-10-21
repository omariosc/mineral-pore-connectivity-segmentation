"""Shared U-Net implementation for the registered revision ablations.

The four factorial arms share ordinary convolutions, interpolation, widths,
normalization and initialization. A fifth arm recovers the original boundary
modules, which are a separate mechanism from multiscale features and attention.
"""

from __future__ import annotations
import torch.nn as nn
from src.models.multiscale_attention_unet import (
    MultiScaleAttentionUNet,
    DoubleConv,
    Down,
    Up,
)


class RevisionUNet(MultiScaleAttentionUNet):
    def __init__(
        self,
        n_channels: int,
        *,
        multiscale: bool,
        attention: bool,
        boundary: bool,
        base_features: int = 32,
    ):
        nn.Module.__init__(self)
        if n_channels not in (1, 2):
            raise ValueError("Revision inputs are grayscale or grayscale plus gate")
        if base_features < 4 or base_features % 4:
            raise ValueError("Base width must be a positive multiple of four")
        self.architecture_name = "revision_factorial_unet"
        self.n_channels = n_channels
        self.n_classes = 2
        self.bilinear = True
        self.base_features = base_features
        self.deep_supervision = False
        b = base_features
        self.inc = DoubleConv(n_channels, b)
        self.down1 = Down(b, b * 2, use_multiscale=False)
        self.down2 = Down(b * 2, b * 4, use_multiscale=multiscale)
        self.down3 = Down(b * 4, b * 8, use_multiscale=multiscale)
        self.down4 = Down(b * 8, b * 8, use_multiscale=multiscale)
        self.up1 = Up(b * 16, b * 4, True, use_attention=attention, use_boundary=False)
        self.up2 = Up(
            b * 8, b * 2, True, use_attention=attention, use_boundary=boundary
        )
        self.up3 = Up(b * 4, b, True, use_attention=attention, use_boundary=boundary)
        self.up4 = Up(b * 2, b, True, use_attention=attention, use_boundary=boundary)
        self.outc = nn.Conv2d(b, 2, kernel_size=1)
        self._init_weights()

"""Lightweight pyramid-pooling context for a U-Net bottleneck.

The block is intentionally independent of the existing model factory.  It can
be inserted prospectively at a 256-channel bottleneck without changing the
reference architecture, which keeps an ablation between the two models
explicit.  Group normalization is used throughout because confirmatory
full-tile evaluation may use a batch size of one.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_POOL_GRIDS: Tuple[int, ...] = (1, 2, 4, 8)

__all__ = ["DEFAULT_POOL_GRIDS", "PyramidContextBlock"]


class _PooledContextBranch(nn.Module):
    """Pool to one grid, project channels, normalize, and regularize."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.normalize = nn.GroupNorm(norm_groups, out_channels)
        self.activate = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.activate(self.normalize(self.project(self.pool(x))))
        )


class PyramidContextBlock(nn.Module):
    """Residual pyramid-pooling block for the 256-channel U-Net bottleneck.

    By default, four adaptive-average-pooled branches use grids 1, 2, 4, and
    8.  Each branch produces 32 channels.  Their bilinearly resized features
    are concatenated with the unmodified input and fused back to 256 channels
    before residual addition.

    Dropout is stochastic while the module is in training mode, as expected.
    Evaluation is deterministic for fixed weights and inputs; training is
    reproducible when the caller restores the PyTorch random seed/state.
    """

    def __init__(
        self,
        in_channels: int = 256,
        branch_channels: int = 32,
        pool_grids: Sequence[int] = DEFAULT_POOL_GRIDS,
        norm_groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        resolved_grids = tuple(int(grid) for grid in pool_grids)
        self._validate_config(
            in_channels=in_channels,
            branch_channels=branch_channels,
            pool_grids=resolved_grids,
            norm_groups=norm_groups,
            dropout=dropout,
        )

        self.in_channels = int(in_channels)
        self.branch_channels = int(branch_channels)
        self.pool_grids = resolved_grids
        self.norm_groups = int(norm_groups)
        self.dropout_probability = float(dropout)
        self.concat_channels = self.in_channels + (
            len(self.pool_grids) * self.branch_channels
        )

        self.branches = nn.ModuleList(
            [
                _PooledContextBranch(
                    in_channels=self.in_channels,
                    out_channels=self.branch_channels,
                    grid_size=grid,
                    norm_groups=self.norm_groups,
                    dropout=self.dropout_probability,
                )
                for grid in self.pool_grids
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.concat_channels,
                self.in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(self.norm_groups, self.in_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=self.dropout_probability),
        )
        self._initialize_weights()

    @staticmethod
    def _validate_config(
        *,
        in_channels: int,
        branch_channels: int,
        pool_grids: Tuple[int, ...],
        norm_groups: int,
        dropout: float,
    ) -> None:
        if int(in_channels) <= 0 or int(branch_channels) <= 0:
            raise ValueError("in_channels and branch_channels must be positive")
        if not pool_grids or any(grid <= 0 for grid in pool_grids):
            raise ValueError("pool_grids must contain positive integers")
        if len(set(pool_grids)) != len(pool_grids):
            raise ValueError("pool_grids must not contain duplicates")
        if tuple(sorted(pool_grids)) != pool_grids:
            raise ValueError("pool_grids must be in increasing order")
        if int(norm_groups) <= 0:
            raise ValueError("norm_groups must be positive")
        if int(in_channels) % int(norm_groups) != 0:
            raise ValueError("in_channels must be divisible by norm_groups")
        if int(branch_channels) % int(norm_groups) != 0:
            raise ValueError("branch_channels must be divisible by norm_groups")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in the half-open interval [0, 1)")

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        """Return the number of parameters owned by this optional block."""

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def resolved_config(self) -> Dict[str, object]:
        """Return a JSON-serializable architecture/provenance record."""

        return {
            "schema_version": 1,
            "implementation_class": type(self).__name__,
            "operation": "adaptive_average_pyramid_pooling",
            "in_channels": self.in_channels,
            "out_channels": self.in_channels,
            "pool_grids": list(self.pool_grids),
            "branch_channels": self.branch_channels,
            "number_of_branches": len(self.pool_grids),
            "concat_channels": self.concat_channels,
            "normalization": {
                "type": "GroupNorm",
                "groups": self.norm_groups,
            },
            "activation": "ReLU",
            "dropout": {
                "type": "Dropout2d",
                "probability": self.dropout_probability,
            },
            "resize": {
                "mode": "bilinear",
                "align_corners": False,
                "target": "input_spatial_shape",
            },
            "fusion": "concat_1x1_conv_groupnorm_relu_dropout",
            "residual": True,
            "convolution_bias": False,
            "weight_initialization": "kaiming_normal_fan_out_relu",
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("input must have shape [batch, channels, height, width]")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, received {x.shape[1]}"
            )
        if x.shape[-2] <= 0 or x.shape[-1] <= 0:
            raise ValueError("input spatial dimensions must be positive")

        spatial_shape = x.shape[-2:]
        context_features = [x]
        for branch in self.branches:
            pooled = branch(x)
            context_features.append(
                F.interpolate(
                    pooled,
                    size=spatial_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fused = self.fuse(torch.cat(context_features, dim=1))
        return x + fused

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"branch_channels={self.branch_channels}, "
            f"pool_grids={self.pool_grids}, "
            f"norm_groups={self.norm_groups}, "
            f"dropout={self.dropout_probability}"
        )

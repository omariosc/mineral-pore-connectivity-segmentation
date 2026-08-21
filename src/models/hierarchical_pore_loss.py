"""Differentiable hierarchical loss for three-class pore segmentation.

The objective separates three questions that are otherwise entangled in a
single three-class loss: region labelling, pore-versus-mineral segmentation,
and disconnected-versus-connected classification conditional on a pixel being
a pore.  Every term operates on soft probabilities; there is no argmax,
NumPy conversion, connected-component labelling, or other detached prediction
path.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPoreConnectivityLoss(nn.Module):
    """Normalized soft-Dice objective for C0, C1, and mineral (C2).

    ``training_class_counts`` must contain pixel counts from the training
    masks only, in canonical ``[C0, C1, C2]`` order.  The counts are used only
    to derive an inverse-square-root balance between the two conditional pore
    classes.  Supplying them explicitly makes the data dependency auditable
    and prevents validation or test targets from influencing the objective.

    The three component weights are normalized to sum to one.  Each component
    is itself a weighted mean of soft-Dice losses and is therefore bounded
    (up to floating-point round-off) between zero and one.
    """

    architecture_name = "hierarchical_pore_connectivity"

    def __init__(
        self,
        training_class_counts: Sequence[int],
        component_weights: Sequence[float] = (1.0, 1.0, 1.0),
        smooth: float = 1e-6,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(training_class_counts, dtype=torch.float64)
        if counts.ndim != 1 or counts.numel() != 3:
            raise ValueError(
                "training_class_counts must contain exactly [C0, C1, C2]"
            )
        if not torch.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError("training_class_counts must be finite positive values")

        weights = torch.as_tensor(component_weights, dtype=torch.float64)
        if weights.ndim != 1 or weights.numel() != 3:
            raise ValueError(
                "component_weights must contain region, pore_union, and conditional"
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("component_weights must be finite and non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("at least one component weight must be positive")
        if not 0 < float(smooth) < 1:
            raise ValueError("smooth must be finite and between zero and one")

        pore_counts = counts[:2]
        pore_frequencies = pore_counts / pore_counts.sum()
        conditional_weights = torch.rsqrt(pore_frequencies)
        conditional_weights = conditional_weights / conditional_weights.sum()

        self.register_buffer("training_class_counts", counts)
        self.register_buffer(
            "training_class_frequencies", counts / counts.sum()
        )
        self.register_buffer("conditional_class_weights", conditional_weights)
        self.register_buffer("component_weights", weights / weights.sum())
        self.smooth = float(smooth)
        self.ignore_index = int(ignore_index)

    def _weighted_soft_dice_loss(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return a normalized class-weighted soft-Dice loss."""
        reduce_dims = (0, 2, 3)
        intersection = (probabilities * targets).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + targets.sum(
            dim=reduce_dims
        )
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        normalized_weights = class_weights / class_weights.sum()
        return ((1.0 - dice) * normalized_weights).sum()

    def component_losses(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute the three differentiable, normalized component losses."""
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("logits must have shape [batch, 3, height, width]")
        if targets.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
            raise ValueError("targets must have shape [batch, height, width]")

        valid = targets != self.ignore_index
        if not bool(valid.any()):
            raise ValueError("hierarchical loss received no valid target pixels")
        invalid_labels = valid & ((targets < 0) | (targets > 2))
        if bool(invalid_labels.any()):
            values = torch.unique(targets[invalid_labels]).detach().cpu().tolist()
            raise ValueError(f"targets contain labels outside 0, 1, 2: {values}")

        clean_targets = targets.masked_fill(~valid, 0).long()
        # Keep softmax and all Dice reductions in float32 under AMP. This
        # preserves gradients for extreme rare-class logits that fp16 epsilon
        # would otherwise clamp away.
        target_one_hot = F.one_hot(clean_targets, num_classes=3).permute(
            0, 3, 1, 2
        ).to(dtype=torch.float32)
        valid_pixels = valid.unsqueeze(1).to(dtype=torch.float32)
        float_logits = logits.float()
        probabilities = F.softmax(float_logits, dim=1) * valid_pixels
        target_one_hot = target_one_hot * valid_pixels

        equal_region_weights = torch.ones(3, device=logits.device, dtype=torch.float32)
        region = self._weighted_soft_dice_loss(
            probabilities, target_one_hot, equal_region_weights
        )

        pore_probability = probabilities[:, :2].sum(dim=1, keepdim=True)
        pore_target = target_one_hot[:, :2].sum(dim=1, keepdim=True)
        pore_union = self._weighted_soft_dice_loss(
            pore_probability,
            pore_target,
            torch.ones(1, device=logits.device, dtype=torch.float32),
        )

        # Conditional probabilities are evaluated only at ground-truth pore
        # pixels. Region and pore-union terms handle false pore predictions on
        # mineral pixels, while this term isolates the C0-versus-C1 decision.
        conditional_probability = (
            F.softmax(float_logits[:, :2], dim=1) * pore_target
        )
        conditional_target = target_one_hot[:, :2]
        conditional = self._weighted_soft_dice_loss(
            conditional_probability,
            conditional_target,
            self.conditional_class_weights.to(
                device=logits.device, dtype=torch.float32
            ),
        )

        return {
            "region": region,
            "pore_union": pore_union,
            "conditional_c0_c1": conditional,
        }

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        components = self.component_losses(logits, targets)
        ordered = torch.stack(
            (
                components["region"],
                components["pore_union"],
                components["conditional_c0_c1"],
            )
        )
        return (
            ordered
            * self.component_weights.to(device=logits.device, dtype=torch.float32)
        ).sum()

    def resolved_config(self) -> Dict[str, object]:
        """Return public-safe constants required to reproduce the objective."""
        return {
            "implementation_class": type(self).__name__,
            "num_classes": 3,
            "component_names": [
                "three_class_region_soft_dice",
                "pore_union_soft_dice",
                "conditional_c0_c1_soft_dice",
            ],
            "component_weights_normalized": [
                float(value) for value in self.component_weights.detach().cpu().tolist()
            ],
            "conditional_balance": "inverse_sqrt_training_pore_frequency",
            "conditional_class_weights_normalized": [
                float(value)
                for value in self.conditional_class_weights.detach().cpu().tolist()
            ],
            "training_class_counts": [
                int(value)
                for value in self.training_class_counts.detach().cpu().tolist()
            ],
            "training_class_frequencies": [
                float(value)
                for value in self.training_class_frequencies.detach().cpu().tolist()
            ],
            "class_count_source": "authoritative_training_masks_only",
            "smooth": self.smooth,
            "ignore_index": self.ignore_index,
            "prediction_path": "softmax_probabilities_only_no_argmax_or_numpy",
        }


def create_hierarchical_pore_connectivity_loss(
    training_class_counts: Iterable[int],
    component_weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> HierarchicalPoreConnectivityLoss:
    """Construct the candidate loss from prospectively recorded constants."""
    return HierarchicalPoreConnectivityLoss(
        training_class_counts=list(training_class_counts),
        component_weights=component_weights,
    )

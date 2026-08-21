"""Prospective loss for C0-versus-C1 classification on pore pixels only."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalPoreFocalDiceLoss(nn.Module):
    """Normalized focal plus macro-Dice loss with mineral pixels ignored.

    Pixel counts must be derived from the authoritative training masks in
    canonical ``[C0, C1, C2]`` order.  Inverse-square-root C0/C1 frequencies
    balance the focal term; the Dice term is an equal macro-average over C0
    and C1.  There is deliberately no prevalence regularizer or heuristic
    weight softening.
    """

    def __init__(
        self,
        training_class_counts: Sequence[int],
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0,
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

        component_weights = torch.tensor(
            [focal_weight, dice_weight], dtype=torch.float64
        )
        if (
            not torch.isfinite(component_weights).all()
            or (component_weights < 0).any()
            or float(component_weights.sum()) <= 0
        ):
            raise ValueError("focal and Dice weights must be finite and non-negative")
        if not torch.isfinite(torch.tensor(float(gamma))) or float(gamma) < 0:
            raise ValueError("gamma must be finite and non-negative")
        if not 0 < float(smooth) < 1:
            raise ValueError("smooth must be finite and between zero and one")

        pore_frequencies = counts[:2] / counts[:2].sum()
        class_weights = torch.rsqrt(pore_frequencies)
        class_weights = class_weights / class_weights.sum()
        self.register_buffer("training_class_counts", counts)
        self.register_buffer(
            "training_class_frequencies", counts / counts.sum()
        )
        self.register_buffer("class_weights", class_weights)
        self.register_buffer(
            "component_weights", component_weights / component_weights.sum()
        )
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.ignore_index = int(ignore_index)

    def component_losses(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError("logits must have shape [batch, 2, height, width]")
        if targets.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
            raise ValueError("targets must have shape [batch, height, width]")

        valid = targets != self.ignore_index
        if not bool(valid.any()):
            raise ValueError("conditional pore loss received no C0/C1 target pixels")
        invalid = valid & ((targets < 0) | (targets > 1))
        if bool(invalid.any()):
            values = torch.unique(targets[invalid]).detach().cpu().tolist()
            raise ValueError(f"targets contain non-pore labels: {values}")

        # Compute focal probabilities/log-probabilities and Dice reductions in
        # float32 even under AMP. Half-precision epsilon is far too coarse for
        # an extremely unlikely rare C0 target and can flatten its gradient.
        valid_logits = logits.permute(0, 2, 3, 1)[valid].float()
        valid_targets = targets[valid].long()
        log_probabilities = F.log_softmax(valid_logits, dim=1)
        probabilities = log_probabilities.exp()
        target_probability = probabilities.gather(
            1, valid_targets.unsqueeze(1)
        ).squeeze(1)
        target_log_probability = log_probabilities.gather(
            1, valid_targets.unsqueeze(1)
        ).squeeze(1)
        focal = -(
            (1.0 - target_probability).pow(self.gamma)
            * target_log_probability
        )
        sample_weights = self.class_weights.to(
            device=logits.device, dtype=torch.float32
        )[valid_targets]
        focal = (focal * sample_weights).sum() / sample_weights.sum()

        targets_one_hot = F.one_hot(valid_targets, num_classes=2).to(
            dtype=torch.float32
        )
        intersection = (probabilities * targets_one_hot).sum(dim=0)
        denominator = probabilities.sum(dim=0) + targets_one_hot.sum(dim=0)
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        macro_dice = dice_per_class.mean()
        return {"focal": focal, "macro_dice": macro_dice}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        components = self.component_losses(logits, targets)
        values = torch.stack((components["focal"], components["macro_dice"]))
        return (
            values
            * self.component_weights.to(device=logits.device, dtype=torch.float32)
        ).sum()

    def resolved_config(self) -> Dict[str, object]:
        return {
            "implementation_class": type(self).__name__,
            "num_output_classes": 2,
            "target_classes": ["C0", "C1"],
            "ignored_source_class": "C2_mineral",
            "ignore_index": self.ignore_index,
            "components": ["conditional_focal", "conditional_macro_soft_dice"],
            "component_weights_normalized": [
                float(value) for value in self.component_weights.detach().cpu().tolist()
            ],
            "focal_gamma": self.gamma,
            "conditional_balance": "inverse_sqrt_training_pore_frequency",
            "conditional_class_weights_normalized": [
                float(value) for value in self.class_weights.detach().cpu().tolist()
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
            "prediction_path": "softmax_probabilities_only_no_prevalence_regularizer",
        }


def create_conditional_pore_focal_dice_loss(
    training_class_counts: Iterable[int],
) -> ConditionalPoreFocalDiceLoss:
    """Create the locked conditional candidate objective."""
    return ConditionalPoreFocalDiceLoss(
        training_class_counts=list(training_class_counts),
        focal_weight=0.5,
        dice_weight=0.5,
        gamma=2.0,
    )

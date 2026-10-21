"""Numerical controls for the revision architecture and input ablations."""

import pytest
import torch
from scripts.revision_models import RevisionUNet
from src.models.multiscale_attention_unet import MultiScaleAttentionUNet


@pytest.mark.parametrize(
    "multiscale,attention,boundary",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_architecture_arms_preserve_odd_resolution_and_finite_gradients(
    multiscale, attention, boundary
):
    torch.set_num_threads(2)
    model = RevisionUNet(
        2,
        multiscale=multiscale,
        attention=attention,
        boundary=boundary,
        base_features=8,
    )
    value = torch.randn(2, 2, 33, 35)
    logits = model(value)
    assert logits.shape == (2, 2, 33, 35)
    loss = logits.square().mean()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_full_boundary_arm_exactly_reconstructs_original_model():
    torch.set_num_threads(2)
    original = MultiScaleAttentionUNet(
        n_channels=2, n_classes=2, base_features=8, deep_supervision=False
    ).eval()
    control = RevisionUNet(
        2, multiscale=True, attention=True, boundary=True, base_features=8
    ).eval()
    control.load_state_dict(original.state_dict(), strict=True)
    value = torch.randn(1, 2, 33, 35)
    with torch.inference_mode():
        assert torch.equal(control(value), original(value))


def test_plain_architecture_has_no_multiscale_attention_or_boundary_modules():
    from src.models.multiscale_attention_unet import (
        MultiScaleFeatureExtractor,
        SpatialAttentionGate,
        BoundaryRefinementModule,
    )

    model = RevisionUNet(
        1, multiscale=False, attention=False, boundary=False, base_features=8
    )
    assert not any(
        isinstance(
            m,
            (
                MultiScaleFeatureExtractor,
                SpatialAttentionGate,
                BoundaryRefinementModule,
            ),
        )
        for m in model.modules()
    )

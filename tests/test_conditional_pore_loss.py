import pytest
import torch

from src.models.conditional_pore_loss import ConditionalPoreFocalDiceLoss


TRAINING_COUNTS = [1_816_734, 41_571_134, 266_990_628]


def test_conditional_loss_has_finite_gradients_only_on_nonignored_pixels():
    criterion = ConditionalPoreFocalDiceLoss(TRAINING_COUNTS)
    logits = torch.randn(
        1, 2, 2, 3, generator=torch.Generator().manual_seed(17)
    ).requires_grad_(True)
    targets = torch.tensor([[[0, 1, -100], [1, 0, -100]]])

    loss = criterion(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[:, :, :, :2].abs().sum()) > 0
    ignored = (targets == -100).unsqueeze(1).expand_as(logits)
    assert torch.count_nonzero(logits.grad[ignored]) == 0


def test_conditional_loss_is_invariant_to_ignored_pixel_logits():
    criterion = ConditionalPoreFocalDiceLoss(TRAINING_COUNTS)
    logits = torch.tensor(
        [[[[3.0, -2.0, 0.0]], [[-3.0, 2.0, 0.0]]]], requires_grad=True
    )
    targets = torch.tensor([[[0, 1, -100]]])
    changed = logits.detach().clone()
    changed[0, :, 0, 2] = torch.tensor([100.0, -100.0])

    assert float(criterion(changed, targets)) == pytest.approx(
        float(criterion(logits, targets).detach()), abs=1e-8
    )


def test_conditional_loss_records_train_only_balance_without_legacy_regularizer():
    criterion = ConditionalPoreFocalDiceLoss(TRAINING_COUNTS)
    resolved = criterion.resolved_config()
    weights = resolved["conditional_class_weights_normalized"]

    assert weights[0] / weights[1] == pytest.approx(
        (TRAINING_COUNTS[1] / TRAINING_COUNTS[0]) ** 0.5
    )
    assert sum(weights) == pytest.approx(1.0)
    assert resolved["component_weights_normalized"] == pytest.approx([0.5, 0.5])
    assert resolved["training_class_counts"] == TRAINING_COUNTS
    assert resolved["prediction_path"] == (
        "softmax_probabilities_only_no_prevalence_regularizer"
    )
    assert "prevalence_target" not in resolved
    assert "minimum_mean_probability" not in resolved


def test_conditional_loss_fails_closed_without_pore_targets_or_with_c2_label():
    criterion = ConditionalPoreFocalDiceLoss(TRAINING_COUNTS)
    logits = torch.zeros(1, 2, 1, 2)

    with pytest.raises(ValueError, match="no C0/C1"):
        criterion(logits, torch.full((1, 1, 2), -100))
    with pytest.raises(ValueError, match="non-pore labels"):
        criterion(logits, torch.tensor([[[0, 2]]]))


def test_conditional_loss_keeps_extreme_fp16_rare_class_gradient_finite():
    criterion = ConditionalPoreFocalDiceLoss(TRAINING_COUNTS)
    logits = torch.tensor(
        [[[[ -30.0, 30.0]], [[30.0, -30.0]]]],
        dtype=torch.float16,
        requires_grad=True,
    )
    targets = torch.tensor([[[0, 1]]])

    loss = criterion(logits, targets)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == logits.numel()

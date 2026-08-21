import ast
import inspect

import pytest
import torch

from src.models.hierarchical_pore_loss import HierarchicalPoreConnectivityLoss


TRAINING_COUNTS = [1_816_734, 41_571_134, 266_990_628]


def _example():
    targets = torch.tensor(
        [
            [
                [0, 0, 1, 1],
                [0, 2, 1, 2],
                [2, 2, 1, 2],
            ]
        ],
        dtype=torch.long,
    )
    logits = torch.randn(1, 3, 3, 4, generator=torch.Generator().manual_seed(7))
    return logits.requires_grad_(True), targets


def test_hierarchical_loss_has_finite_prediction_gradients_and_bounded_components():
    criterion = HierarchicalPoreConnectivityLoss(TRAINING_COUNTS)
    logits, targets = _example()

    components = criterion.component_losses(logits, targets)
    loss = criterion(logits, targets)
    loss.backward()

    assert set(components) == {"region", "pore_union", "conditional_c0_c1"}
    assert torch.isfinite(loss)
    for component in components.values():
        assert torch.isfinite(component)
        assert 0.0 <= float(component) <= 1.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0
    assert all(float(logits.grad[:, class_id].abs().sum()) > 0.0 for class_id in range(3))


def test_hierarchical_loss_is_invariant_to_per_pixel_logit_shift():
    criterion = HierarchicalPoreConnectivityLoss(TRAINING_COUNTS)
    logits, targets = _example()
    shift = torch.randn(
        logits.shape[0],
        1,
        logits.shape[2],
        logits.shape[3],
        generator=torch.Generator().manual_seed(11),
    )

    original = criterion(logits, targets)
    shifted = criterion(logits + shift, targets)

    assert float(shifted.detach()) == pytest.approx(
        float(original.detach()), abs=1e-7
    )


def test_pore_union_is_invariant_to_c0_c1_swap_but_conditional_term_is_not():
    criterion = HierarchicalPoreConnectivityLoss(TRAINING_COUNTS)
    logits, targets = _example()
    swapped = logits[:, [1, 0, 2]]

    original = criterion.component_losses(logits, targets)
    changed = criterion.component_losses(swapped, targets)

    assert float(changed["pore_union"].detach()) == pytest.approx(
        float(original["pore_union"].detach()), abs=1e-7
    )
    assert float(changed["conditional_c0_c1"]) != pytest.approx(
        float(original["conditional_c0_c1"]), abs=1e-5
    )


def test_training_counts_are_recorded_and_only_set_conditional_balance():
    criterion = HierarchicalPoreConnectivityLoss(TRAINING_COUNTS)
    resolved = criterion.resolved_config()

    expected_ratio = (TRAINING_COUNTS[1] / TRAINING_COUNTS[0]) ** 0.5
    actual = resolved["conditional_class_weights_normalized"]
    assert actual[0] / actual[1] == pytest.approx(expected_ratio)
    assert sum(actual) == pytest.approx(1.0)
    assert resolved["training_class_counts"] == TRAINING_COUNTS
    assert resolved["class_count_source"] == "authoritative_training_masks_only"
    assert resolved["component_weights_normalized"] == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )


def test_hierarchical_candidate_has_no_detached_prediction_operator():
    tree = ast.parse(inspect.getsource(__import__(
        "src.models.hierarchical_pore_loss", fromlist=["*"]
    )))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "numpy" not in imported_names
    assert "argmax" not in called_attributes


@pytest.mark.parametrize(
    "counts",
    ([1, 2], [1, 2, 0], [1, float("nan"), 3]),
)
def test_hierarchical_loss_rejects_unusable_training_counts(counts):
    with pytest.raises(ValueError, match="training_class_counts"):
        HierarchicalPoreConnectivityLoss(counts)


def test_hierarchical_loss_fails_closed_on_invalid_or_empty_targets():
    criterion = HierarchicalPoreConnectivityLoss(TRAINING_COUNTS)
    logits = torch.zeros(1, 3, 2, 2)

    with pytest.raises(ValueError, match="outside 0, 1, 2"):
        criterion(logits, torch.tensor([[[0, 1], [2, 9]]]))
    with pytest.raises(ValueError, match="no valid target pixels"):
        criterion(logits, torch.full((1, 2, 2), -100))


def test_hierarchical_loss_keeps_extreme_fp16_rare_class_gradient_finite():
    criterion = HierarchicalPoreConnectivityLoss(TRAINING_COUNTS)
    logits = torch.tensor(
        [
                [
                    [[-6.0, 6.0, 6.0]],
                    [[6.0, -6.0, 0.0]],
                    [[0.0, 0.0, -6.0]],
            ]
        ],
        dtype=torch.float16,
        requires_grad=True,
    )
    targets = torch.tensor([[[0, 1, 2]]])

    loss = criterion(logits, targets)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0, 0, 0, 0]) != 0.0
    assert float(logits.grad[0, 1, 0, 1]) != 0.0

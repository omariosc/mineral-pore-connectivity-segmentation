"""Hand-counted controls for the post-review validation analyses."""

import numpy as np
import pytest
from scripts.revision_metrics import (
    compose_gate_logits,
    confusion,
    summarize_confusion,
    gate_confusion,
    component_recovery,
)


def test_gate_ignores_background_logit_and_threshold_is_strict():
    raw = np.array([[99, 100, 0]], dtype=np.uint8)
    logits = np.array([[[2.0, 9.0, 0.0]], [[1.0, 0.0, 3.0]], [[99.0, 0.0, 99.0]]])
    assert compose_gate_logits(raw, logits, 100).tolist() == [[0, 2, 1]]


def test_gate_rejects_invalid_or_misaligned_logits():
    with pytest.raises(ValueError):
        compose_gate_logits(np.zeros((1, 2), dtype=np.uint8), np.zeros((3, 2, 2)), 100)
    with pytest.raises(ValueError):
        compose_gate_logits(
            np.zeros((1, 2), dtype=np.uint8), np.full((3, 1, 2), np.nan), 100
        )


def test_confusion_retains_both_pore_errors_and_gate_reference_errors():
    target = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.uint8)
    pred = np.array([[0, 1, 0, 1, 0, 2]], dtype=np.uint8)
    cm = confusion(target, pred)
    assert cm.tolist() == [[1, 1, 0], [1, 1, 0], [1, 0, 1]]
    m = summarize_confusion(cm)
    assert m["c0_iou"] == pytest.approx(0.25)
    assert m["c0_precision"] == pytest.approx(1 / 3)
    assert m["c0_recall"] == pytest.approx(0.5)
    assert m["pore_union_iou"] == pytest.approx(4 / 5)


def test_within_gate_includes_mineral_reference_row():
    raw = np.array([[0, 0, 0, 100]], dtype=np.uint8)
    target = np.array([[0, 1, 2, 0]], dtype=np.uint8)
    pred = np.array([[1, 1, 0, 2]], dtype=np.uint8)
    assert gate_confusion(raw, target, pred, 100).tolist() == [[0, 1], [0, 1], [1, 0]]


def test_empty_class_is_undefined_instead_of_perfect():
    m = summarize_confusion(np.diag([0, 3, 4]))
    assert m["c0_iou"] is None
    assert m["c0_precision"] is None
    assert m["c0_recall"] is None


def test_component_recall_uses_coverage_and_flags_edge_censoring():
    target = np.full((7, 7), 2, dtype=np.uint8)
    target[0, 0:2] = 0
    target[3, 3:5] = 0
    pred = np.full((7, 7), 2, dtype=np.uint8)
    pred[0, 0] = 0
    r = component_recovery(target, pred)
    assert r["reference_objects"] == 2
    assert r["detected_objects"] == 1
    assert r["recall_at_coverage_0_5"] == 0.5
    assert r["edge_objects"] == 1
    assert r["edge_detected_objects"] == 1
    assert r["interior_detected_objects"] == 0
    assert sorted(x["covered_pixels"] for x in r["objects"]) == [0, 1]


def test_diagonal_pixels_form_one_8_connected_object():
    target = np.full((4, 4), 2, dtype=np.uint8)
    target[1, 1] = 0
    target[2, 2] = 0
    r = component_recovery(target, target)
    assert r["reference_objects"] == 1
    assert r["recall_at_coverage_0_5"] == 1.0

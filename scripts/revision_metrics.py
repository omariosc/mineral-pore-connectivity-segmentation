"""Descriptive post-review metrics; no fitting or model selection."""

from __future__ import annotations
import numpy as np
from scipy import ndimage


def compose_gate_logits(
    image: np.ndarray, logits: np.ndarray, threshold: int = 100
) -> np.ndarray:
    image = np.asarray(image)
    logits = np.asarray(logits)
    if (
        image.dtype != np.uint8
        or image.ndim != 2
        or type(threshold) is not int
        or not 1 <= threshold <= 255
    ):
        raise ValueError(
            "Expected raw uint8 grayscale and an integer threshold in 1..255"
        )
    if (
        logits.shape not in [(2, *image.shape), (3, *image.shape)]
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Expected finite aligned two- or three-class logits")
    return np.where(image < threshold, logits[:2].argmax(axis=0), 2).astype(np.uint8)


def confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Expected matching two-dimensional hard masks")
    if not np.isin(target, [0, 1, 2]).all() or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError("Masks must use canonical integer classes 0,1,2")
    return np.bincount(
        (target.astype(np.int64) * 3 + prediction).ravel(), minlength=9
    ).reshape(3, 3)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def summarize_confusion(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or not (matrix == matrix.astype(np.int64)).all()
    ):
        raise ValueError("Expected nonnegative integer 3x3 confusion counts")
    values = {}
    for i in range(3):
        tp = int(matrix[i, i])
        truth = int(matrix[i].sum())
        pred = int(matrix[:, i].sum())
        values.update(
            {
                f"c{i}_iou": _ratio(tp, truth + pred - tp),
                f"c{i}_precision": _ratio(tp, pred),
                f"c{i}_recall": _ratio(tp, truth),
                f"c{i}_dice": _ratio(2 * tp, truth + pred),
            }
        )
    pore_tp = int(matrix[:2, :2].sum())
    values["pore_union_iou"] = _ratio(
        pore_tp, pore_tp + int(matrix[:2, 2].sum()) + int(matrix[2, :2].sum())
    )
    a, b = values["c0_iou"], values["c1_iou"]
    values["harmonic_pore_iou"] = (
        None if a is None or b is None else (2 * a * b / (a + b + 1e-8))
    )
    return values


def gate_confusion(
    image: np.ndarray, target: np.ndarray, prediction: np.ndarray, threshold: int = 100
) -> np.ndarray:
    confusion(target, prediction)
    image = np.asarray(image)
    if image.shape != target.shape or image.dtype != np.uint8:
        raise ValueError("Expected aligned uint8 image")
    gate = image < threshold
    if (prediction[gate] > 1).any():
        raise ValueError("Within-gate predictions must be C0 or C1")
    return np.bincount(
        (target[gate].astype(np.int64) * 2 + prediction[gate]).ravel(), minlength=6
    ).reshape(3, 2)


def component_recovery(target: np.ndarray, prediction: np.ndarray) -> dict:
    """C0 component coverage recall, with no size exclusion or matching claim.

    A reference 8-connected component is detected if at least half its pixels
    are predicted C0. A merged prediction can detect several reference objects;
    this is coverage recall, not one-to-one instance recognition.
    """
    confusion(target, prediction)
    labels, count = ndimage.label(
        target == 0, structure=np.ones((3, 3), dtype=np.uint8)
    )
    area = np.bincount(labels.ravel(), minlength=count + 1)
    covered = np.bincount(labels[prediction == 0], minlength=count + 1)
    edge = set(
        np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        ).tolist()
    )
    objects = [
        {
            "object_id": i,
            "area_pixels": int(area[i]),
            "covered_pixels": int(covered[i]),
            "coverage": float(covered[i] / area[i]),
            "detected": bool(2 * covered[i] >= area[i]),
            "touches_tile_edge": i in edge,
        }
        for i in range(1, count + 1)
    ]
    detected = sum(x["detected"] for x in objects)
    edge_objects = [x for x in objects if x["touches_tile_edge"]]
    return {
        "reference_objects": count,
        "detected_objects": detected,
        "missed_objects": count - detected,
        "recall_at_coverage_0_5": _ratio(detected, count),
        "edge_objects": len(edge_objects),
        "edge_detected_objects": sum(x["detected"] for x in edge_objects),
        "interior_objects": count - len(edge_objects),
        "interior_detected_objects": detected
        - sum(x["detected"] for x in edge_objects),
        "objects": objects,
    }

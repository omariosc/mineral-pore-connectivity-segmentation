"""Synthetic-only tests for the downstream publication-results assembler."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from scripts import assemble_publication_results as assembler
from scripts.evaluate_confirmatory_checkpoint import (
    LOCKED_BOOTSTRAP_REPLICATES,
    LOCKED_BOOTSTRAP_SEED,
    LOCKED_CONFIDENCE,
    LOCKED_CURVE_BINS as EVALUATOR_LOCKED_CURVE_BINS,
    EXPECTED_TILE_SHAPE as NEURAL_EXPECTED_TILE_SHAPE,
    qualitative_figure_contract,
)
from scripts.evaluate_locked_classical_comparators import (
    CLASSICAL_COMPARATOR_IDS,
    EVALUATOR_SCHEMA_VERSION as CLASSICAL_EVALUATOR_SCHEMA_VERSION,
    EXPECTED_TILE_SHAPE as CLASSICAL_EXPECTED_TILE_SHAPE,
)
from src.training.screen_selection import SCREEN_CANDIDATE_ORDER, SCREEN_SEEDS


FREEZE_ID = "neural-freeze-" + "1" * 16
FREEZE_SHA = "2" * 64
FIT_ID = "classical-fit-" + "3" * 16
LOCK_SHA = "5" * 64
PAIR_SHA = "5d640e178cbf476c3e13b6ee7cf59e9146fa70749fdab385ce824c4054995160"
FREEZE_MANIFEST_PATH = (
    f"results/neural_freeze/locked/{FREEZE_ID}/neural_freeze_manifest.json"
)
FREEZE_FILE_SHA = "6" * 64
SELECTED_LOCK_PATH = "config/selected_method_lock.json"
SELECTED_LOCK_SHA = "7" * 64
EVALUATOR_SHA = "8" * 64
TRAINING_SOURCE_MAP = {
    "src/training/patch_trainer.py": "9" * 64,
    "src/training/screen_selection.py": "a" * 64,
}
TRAINING_SOURCE_ATTESTATION = {
    "verification_status": "matched_screen_checkpoint_and_live_sources",
    "file_count": len(TRAINING_SOURCE_MAP),
    "files": TRAINING_SOURCE_MAP,
}
CLASSICAL_LOCK_PATH = f"results/classical/locked/{FIT_ID}/classical_lock.json"
CLASSICAL_LOCK_RAW_SHA = "b" * 64
CLASSICAL_SOURCE_MAP = {
    "scripts/fit_classical_comparators.py": "c" * 64,
    "src/classical/comparators.py": "d" * 64,
}
B2_MODEL_PATH = f"results/classical/locked/{FIT_ID}/b2_extra_trees.npz"
B2_MODEL_SHA = "e" * 64
B2_MODEL_SEMANTIC_SHA = "f" * 64
TILE_NAMES = ("tile_01.png", "tile_02.png", "tile_03.png")
ROW_TOTALS = (1_000_000, 500_000, 2_694_304)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _artifact(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _file_sha(path),
    }


def _input_hash(name: str) -> str:
    return _sha(f"input:{name}".encode())


def _target_hash(name: str) -> str:
    return _sha(f"target:{name}".encode())


def _checkpoint_map(selected_method: str) -> dict[str, list[str]]:
    return {
        "primary_multiscale": [
            _sha(f"checkpoint:{selected_method}:{seed}".encode())
            for seed in assembler.NEURAL_SEEDS
        ],
        "plain_unet_comparator": [
            _sha(f"checkpoint:plain_unet:{seed}".encode())
            for seed in assembler.NEURAL_SEEDS
        ],
    }


def _matrix(method_id: str, seed: int | None, tile_index: int) -> np.ndarray:
    method_shift = {
        "R3": 0,
        "C2-FP": 45_000,
        "plain_unet": 20_000,
        "B0_small_components": -90_000,
        "B1_marker_watershed": -55_000,
        "B2_extra_trees": -30_000,
    }[method_id]
    seed_shift = {42: -10_000, 123: 0, 2025: 10_000, None: 0}[seed]
    tile_shift = (-5_000, 0, 5_000)[tile_index]
    if method_id == "B0_small_components":
        c0_as_c1 = 850_000 + tile_shift
        c1_as_c1 = 350_000 + tile_shift
        c2_as_c1 = 100_000 + tile_shift
        return np.asarray(
            [
                [0, c0_as_c1, ROW_TOTALS[0] - c0_as_c1],
                [0, c1_as_c1, ROW_TOTALS[1] - c1_as_c1],
                [0, c2_as_c1, ROW_TOTALS[2] - c2_as_c1],
            ],
            dtype=np.int64,
        )
    shift = method_shift + seed_shift + tile_shift
    c0_tp = 800_000 + shift
    c1_tp = 390_000 + shift // 2
    c2_tp = 2_600_000 + shift // 3
    return np.asarray(
        [
            [c0_tp, 110_000 - shift // 2, ROW_TOTALS[0] - c0_tp - (110_000 - shift // 2)],
            [55_000 - shift // 4, c1_tp, ROW_TOTALS[1] - c1_tp - (55_000 - shift // 4)],
            [55_000 - shift // 6, ROW_TOTALS[2] - c2_tp - (55_000 - shift // 6), c2_tp],
        ],
        dtype=np.int64,
    )


def _metric_block(matrix: np.ndarray, *, classical: bool) -> dict[str, Any]:
    flat = assembler.metrics_from_confusion(matrix)
    per_class = []
    for class_id in range(3):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": assembler.CLASS_NAMES[class_id],
                "support_pixels": int(matrix[class_id].sum()),
                **{
                    metric: flat[f"c{class_id}_{metric}"]
                    for metric in assembler.METRIC_NAMES
                },
            }
        )
    result: dict[str, Any] = {
        "total_pixels": int(matrix.sum()),
        "per_class": per_class,
    }
    if classical:
        result.update(
            {
                "balanced_pore_iou": flat["c0_c1_harmonic_iou"],
                "pore_union_iou": 0.8,
                "overall_accuracy_reported": False,
                "ranking_or_selection_role": (
                    "none_external_comparator_description_only"
                ),
            }
        )
    else:
        result["selection_metrics"] = {
            "c0_c1_harmonic_iou": flat["c0_c1_harmonic_iou"],
            "pore_union_iou": 0.8,
            "pore_union_agreement": 0.9,
        }
    return result


def _aggregate_csv_rows(
    method_matrices: Mapping[str, np.ndarray], *, classical: bool
) -> list[dict[str, Any]]:
    rows = []
    for method_id, matrix in method_matrices.items():
        flat = assembler.metrics_from_confusion(matrix)
        for class_id in range(3):
            for metric in assembler.METRIC_NAMES:
                row = {
                    "scope": "class",
                    "class_id": class_id,
                    "class_name": assembler.CLASS_NAMES[class_id],
                    "metric": metric,
                    "value": flat[f"c{class_id}_{metric}"],
                    "ci_lower": "",
                    "ci_upper": "",
                }
                if classical:
                    row = {"comparator": method_id, **row}
                else:
                    row.update(
                        {
                            "bootstrap_unit": "synthetic whole tile",
                            "bootstrap_replicates": 10,
                            "bootstrap_seed": 1,
                            "confidence": 0.95,
                        }
                    )
                rows.append(row)
        harmonic_row = {
            "scope": (
                "pore_focus" if classical else "selection_and_pore_union"
            ),
            "class_id": "",
            "class_name": "",
            "metric": (
                "balanced_pore_iou"
                if classical
                else "c0_c1_harmonic_iou"
            ),
            "value": flat["c0_c1_harmonic_iou"],
            "ci_lower": "",
            "ci_upper": "",
        }
        if classical:
            harmonic_row = {"comparator": method_id, **harmonic_row}
        else:
            harmonic_row.update(
                {
                    "bootstrap_unit": "synthetic whole tile",
                    "bootstrap_replicates": 10,
                    "bootstrap_seed": 1,
                    "confidence": 0.95,
                }
            )
        rows.append(harmonic_row)
    return rows


def _neural_per_tile_csv_rows(
    matrices: Sequence[np.ndarray],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    required, _ = assembler._neural_per_tile_allowed_fields()
    fields = tuple(sorted(required))
    rows = []
    for index, (name, matrix) in enumerate(zip(TILE_NAMES, matrices, strict=True)):
        flat = assembler.metrics_from_confusion(matrix)
        row: dict[str, Any] = {field: 0.5 for field in fields}
        row.update(
            {
                "image_id": index + 1,
                "file_name": name,
                "height": 2048,
                "width": 2048,
                "pixels": int(matrix.sum()),
                "accuracy": float(np.trace(matrix) / matrix.sum()),
                "selection_c0_c1_harmonic_iou": flat[
                    "c0_c1_harmonic_iou"
                ],
                "selection_pore_union_iou": 0.8,
                "selection_pore_union_agreement": 0.9,
            }
        )
        for class_id in range(3):
            row[f"class_{class_id}_support_pixels"] = int(
                matrix[class_id].sum()
            )
            for metric in assembler.METRIC_NAMES:
                row[f"class_{class_id}_{metric}"] = flat[
                    f"c{class_id}_{metric}"
                ]
            row[f"class_{class_id}_f1"] = flat[f"c{class_id}_dice"]
        rows.append(row)
    return fields, rows


def _classical_per_tile_csv_rows(
    matrices: Mapping[str, Sequence[np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for method_id in assembler.CLASSICAL_METHOD_IDS:
        for ordinal, (name, matrix) in enumerate(
            zip(TILE_NAMES, matrices[method_id], strict=True), start=1
        ):
            flat = assembler.metrics_from_confusion(matrix)
            row = {
                "comparator": method_id,
                "evaluation_ordinal": ordinal,
                "file_name": name,
                "input_sha256": _input_hash(name),
                "target_sha256": _target_hash(name),
                "balanced_pore_iou": flat["c0_c1_harmonic_iou"],
                "pore_union_iou": 0.8,
            }
            for class_id in range(3):
                for metric in assembler.METRIC_NAMES:
                    row[f"c{class_id}_{metric}"] = flat[
                        f"c{class_id}_{metric}"
                    ]
            rows.append(row)
    return rows


def _confusion_csv_rows(
    matrices: Mapping[str, Sequence[np.ndarray]],
    *,
    classical: bool,
) -> list[dict[str, Any]]:
    rows = []
    for method_id, values in matrices.items():
        for ordinal, (name, matrix) in enumerate(
            zip(TILE_NAMES, values, strict=True), start=1
        ):
            for true_id in range(3):
                for predicted_id in range(3):
                    if classical:
                        rows.append(
                            {
                                "comparator": method_id,
                                "evaluation_ordinal": ordinal,
                                "file_name": name,
                                "true_class_id": true_id,
                                "predicted_class_id": predicted_id,
                                "pixel_count": int(matrix[true_id, predicted_id]),
                            }
                        )
                    else:
                        rows.append(
                            {
                                "image_id": ordinal,
                                "file_name": name,
                                "true_class_id": true_id,
                                "true_class_name": assembler.CLASS_NAMES[true_id],
                                "predicted_class_id": predicted_id,
                                "predicted_class_name": assembler.CLASS_NAMES[
                                    predicted_id
                                ],
                                "pixel_count": int(matrix[true_id, predicted_id]),
                            }
                        )
    return rows


def _distribute(total: int, weights: Sequence[int]) -> np.ndarray:
    result = np.asarray(
        [total * weight // sum(weights) for weight in weights], dtype=np.int64
    )
    result[-1] += total - int(result.sum())
    return result


def _curve_files(
    directory: Path,
    aggregate: np.ndarray,
    *,
    method_id: str,
) -> tuple[Path, Path, dict[str, Any]]:
    bins = assembler.LOCKED_CURVE_BINS
    positive = np.zeros((3, bins), dtype=np.int64)
    negative = np.zeros((3, bins), dtype=np.int64)
    quality = 1 if method_id == "R3" else 2
    positive_weights = tuple(
        (bin_id + 1) ** quality for bin_id in range(bins)
    )
    negative_weights = tuple(reversed(positive_weights))
    for class_id in range(3):
        support = int(aggregate[class_id].sum())
        total = int(aggregate.sum())
        positive[class_id] = _distribute(support, positive_weights)
        negative[class_id] = _distribute(total - support, negative_weights)
    histogram_rows = []
    pr_rows = []
    summary = {}
    for class_id in range(3):
        for bin_id in range(bins):
            histogram_rows.append(
                {
                    "class_id": class_id,
                    "class_name": assembler.CLASS_NAMES[class_id],
                    "bin_id": bin_id,
                    "score_lower": bin_id / bins,
                    "score_upper": (bin_id + 1) / bins,
                    "positive_pixels": int(positive[class_id, bin_id]),
                    "negative_pixels": int(negative[class_id, bin_id]),
                }
            )
        curve = assembler._curves_from_histograms(
            positive[class_id], negative[class_id]
        )
        summary[str(class_id)] = {
            "class_name": assembler.CLASS_NAMES[class_id],
            "positive_pixels": curve["positive_pixels"],
            "negative_pixels": curve["negative_pixels"],
            "average_precision_histogram_approximation": curve[
                "average_precision"
            ],
        }
        for index, threshold in enumerate(curve["thresholds"]):
            pr_rows.append(
                {
                    "class_id": class_id,
                    "class_name": assembler.CLASS_NAMES[class_id],
                    "threshold_lower_edge": threshold,
                    "recall": curve["recall"][index],
                    "precision": curve["precision"][index],
                    "cumulative_true_positive": int(
                        curve["cumulative_true_positive"][index]
                    ),
                    "cumulative_false_positive": int(
                        curve["cumulative_false_positive"][index]
                    ),
                    "positive_pixels": curve["positive_pixels"],
                    "negative_pixels": curve["negative_pixels"],
                }
            )
    histogram_path = directory / "probability_histograms.csv"
    pr_path = directory / "precision_recall_curve.csv"
    _write_csv(
        histogram_path, assembler.HISTOGRAM_FIELDS, histogram_rows
    )
    _write_csv(pr_path, assembler.PRECISION_RECALL_FIELDS, pr_rows)
    return (
        histogram_path,
        pr_path,
        {
            "method": "one-vs-rest fixed-width probability histograms",
            "bins": bins,
            "score_bin_width": 1.0 / bins,
            "raw_probabilities_persisted": False,
            "summary": summary,
        },
    )


def _neural_record(
    root: Path,
    *,
    method_id: str,
    display_name: str,
    selected_method: str,
    role: str,
    seed: int,
    cell_index: int,
    with_curves: bool,
) -> dict[str, Any]:
    directory = root / "artifacts" / method_id / f"seed_{seed}"
    directory.mkdir(parents=True)
    matrices = [_matrix(method_id, seed, index) for index in range(len(TILE_NAMES))]
    aggregate = np.stack(matrices).sum(axis=0)
    checkpoint_sha = _sha(f"checkpoint:{method_id}:{seed}".encode())
    checkpoint_semantic_sha = _sha(
        f"checkpoint-state:{method_id}:{seed}".encode()
    )
    checkpoint_map = _checkpoint_map(selected_method)
    per_tile = []
    for ordinal, (name, matrix) in enumerate(
        zip(TILE_NAMES, matrices, strict=True), start=1
    ):
        per_tile.append(
            {
                "evaluation_ordinal": ordinal,
                "image_id": ordinal,
                "file_name": name,
                "input_image_sha256": _input_hash(name),
                "target_mask_sha256": _target_hash(name),
                "height": 2048,
                "width": 2048,
                "confusion_matrix": matrix.tolist(),
                "metrics": _metric_block(matrix, classical=False),
            }
        )
    aggregate_path = directory / "aggregate_metrics.csv"
    _write_csv(
        aggregate_path,
        assembler.NEURAL_AGGREGATE_FIELDS,
        _aggregate_csv_rows({method_id: aggregate}, classical=False),
    )
    tile_fields, tile_rows = _neural_per_tile_csv_rows(matrices)
    tile_path = directory / "per_tile_metrics.csv"
    _write_csv(tile_path, tile_fields, tile_rows)
    confusion_path = directory / "per_tile_confusion.csv"
    _write_csv(
        confusion_path,
        assembler.NEURAL_CONFUSION_FIELDS,
        _confusion_csv_rows({method_id: matrices}, classical=False),
    )
    histogram_path: Path | None = None
    pr_path: Path | None = None
    curves: Any = None
    if with_curves:
        histogram_path, pr_path, curves = _curve_files(
            directory, aggregate, method_id=method_id
        )
    outputs = [
        "evaluation_summary.json",
        "aggregate_metrics.csv",
        "per_tile_metrics.csv",
        "per_tile_confusion.csv",
        "probability_histograms.csv",
        "precision_recall_curve.csv",
    ]
    report = {
        "schema_version": assembler.NEURAL_REPORT_SCHEMA_VERSION,
        "evaluation_kind": assembler.NEURAL_EVALUATION_KIND,
        "status": "complete",
        "checkpoint": {
            "sha256": checkpoint_sha,
            "state_dict_semantic_sha256": checkpoint_semantic_sha,
        },
        "code": {
            "evaluator_path": assembler.NEURAL_EVALUATOR_PATH,
            "evaluator_sha256": EVALUATOR_SHA,
            "training_execution_source_attestation": TRAINING_SOURCE_ATTESTATION,
        },
        "selected_method_lock": {
            "path": SELECTED_LOCK_PATH,
            "sha256": SELECTED_LOCK_SHA,
            "selected_method": selected_method,
            "selected_architecture_role": role,
        },
        "neural_freeze": {
            "manifest_id": FREEZE_ID,
            "manifest_path": FREEZE_MANIFEST_PATH,
            "manifest_file_sha256": FREEZE_FILE_SHA,
            "scientific_identity_sha256": FREEZE_SHA,
            "selected_retraining_checkpoint_sha256": checkpoint_map,
        },
        "locked_evaluation_identity": {
            "verification_status": (
                "matched_content_addressed_neural_freeze_cell"
            ),
            "neural_freeze_manifest_id": FREEZE_ID,
            "neural_freeze_scientific_identity_sha256": FREEZE_SHA,
            "architecture_role": role,
            "cell_index": cell_index,
            "training_seed": seed,
            "selected_method": selected_method,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_state_dict_semantic_sha256": checkpoint_semantic_sha,
            "reservation_key": f"{FREEZE_ID}:{role}:cell_{cell_index:02d}",
        },
        "data": {
            "evaluation_split": "test",
            "test_image_files": list(TILE_NAMES),
            "test_tile_count": len(TILE_NAMES),
            "native_tile_shape": [2048, 2048],
        },
        "inference": {
            "test_passes": 1,
            "post_processing": "none",
            "output_overwrite_allowed": False,
        },
        "aggregate_confusion_matrix": aggregate.tolist(),
        "aggregate_metrics": _metric_block(aggregate, classical=False),
        "per_tile": per_tile,
        "curves": curves,
        "outputs": outputs,
    }
    report_path = directory / "evaluation_summary.json"
    _write_json(report_path, report)
    return {
        "method_id": method_id,
        "display_name": display_name,
        "report": _artifact(root, report_path),
        "aggregate_metrics_csv": _artifact(root, aggregate_path),
        "per_tile_metrics_csv": _artifact(root, tile_path),
        "per_tile_confusion_csv": _artifact(root, confusion_path),
        "probability_histograms_csv": (
            _artifact(root, histogram_path) if histogram_path else None
        ),
        "precision_recall_curve_csv": (
            _artifact(root, pr_path) if pr_path else None
        ),
        "expected_identity": {
            "neural_freeze_manifest_id": FREEZE_ID,
            "neural_freeze_scientific_identity_sha256": FREEZE_SHA,
            "architecture_role": role,
            "cell_index": cell_index,
            "training_seed": seed,
            "selected_method": selected_method,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_state_dict_semantic_sha256": checkpoint_semantic_sha,
            "evaluator_path": assembler.NEURAL_EVALUATOR_PATH,
            "evaluator_sha256": EVALUATOR_SHA,
            "training_execution_source_attestation": TRAINING_SOURCE_ATTESTATION,
            "selected_method_lock_path": SELECTED_LOCK_PATH,
            "selected_method_lock_sha256": SELECTED_LOCK_SHA,
            "neural_freeze_manifest_path": FREEZE_MANIFEST_PATH,
            "neural_freeze_manifest_file_sha256": FREEZE_FILE_SHA,
            "selected_retraining_checkpoint_sha256": checkpoint_map,
        },
    }


def _classical_record(root: Path, *, selected_method: str) -> dict[str, Any]:
    directory = root / "artifacts" / "classical"
    directory.mkdir(parents=True)
    matrices = {
        method_id: [
            _matrix(method_id, None, index) for index in range(len(TILE_NAMES))
        ]
        for method_id in assembler.CLASSICAL_METHOD_IDS
    }
    comparators = {}
    for method_id, method_matrices in matrices.items():
        aggregate = np.stack(method_matrices).sum(axis=0)
        per_tile = []
        for ordinal, (name, matrix) in enumerate(
            zip(TILE_NAMES, method_matrices, strict=True), start=1
        ):
            per_tile.append(
                {
                    "evaluation_ordinal": ordinal,
                    "image_id": "",
                    "file_name": name,
                    "input_image_sha256": _input_hash(name),
                    "target_mask_sha256": _target_hash(name),
                    "height": 2048,
                    "width": 2048,
                    "confusion_matrix": matrix.tolist(),
                    "metrics": _metric_block(matrix, classical=True),
                }
            )
        comparators[method_id] = {
            "aggregate_confusion_matrix": aggregate.tolist(),
            "aggregate_metrics": _metric_block(aggregate, classical=True),
            "per_tile": per_tile,
        }
    aggregate_path = directory / "aggregate_metrics.csv"
    aggregate_matrices = {
        method_id: np.stack(values).sum(axis=0)
        for method_id, values in matrices.items()
    }
    _write_csv(
        aggregate_path,
        assembler.CLASSICAL_AGGREGATE_FIELDS,
        _aggregate_csv_rows(aggregate_matrices, classical=True),
    )
    tile_path = directory / "per_tile_metrics.csv"
    _write_csv(
        tile_path,
        assembler._classical_per_tile_fields(),
        _classical_per_tile_csv_rows(matrices),
    )
    confusion_path = directory / "per_tile_confusion.csv"
    _write_csv(
        confusion_path,
        assembler.CLASSICAL_CONFUSION_FIELDS,
        _confusion_csv_rows(matrices, classical=True),
    )
    checkpoint_map = _checkpoint_map(selected_method)
    report = {
        "schema_version": assembler.CLASSICAL_REPORT_SCHEMA_VERSION,
        "evaluation_kind": assembler.CLASSICAL_EVALUATION_KIND,
        "status": "complete",
        "fit_id": FIT_ID,
        "evaluation_pair_identity_sha256": PAIR_SHA,
        "selection_relationship": "external_only_no_neural_selection_effect",
        "ranking_policy": "no_overall_accuracy_or_winner_ranking",
        "neural_freeze": {
            "manifest_id": FREEZE_ID,
            "manifest_path": FREEZE_MANIFEST_PATH,
            "manifest_file_sha256": FREEZE_FILE_SHA,
            "scientific_identity_sha256": FREEZE_SHA,
            "selected_method": selected_method,
            "selected_method_lock": {
                "repo_relative_identifier": SELECTED_LOCK_PATH,
                "raw_file_sha256": SELECTED_LOCK_SHA,
            },
            "selected_retraining_checkpoint_sha256": checkpoint_map,
        },
        "classical_lock": {
            "path": CLASSICAL_LOCK_PATH,
            "raw_file_sha256": CLASSICAL_LOCK_RAW_SHA,
            "canonical_identity_sha256": LOCK_SHA,
            "source_code_sha256": CLASSICAL_SOURCE_MAP,
        },
        "b2_model": {
            "path": B2_MODEL_PATH,
            "sha256": B2_MODEL_SHA,
            "semantic_sha256": B2_MODEL_SEMANTIC_SHA,
        },
        "runtime": {
            "overall_accuracy_reported": False,
            "native_tile_shape": list(assembler.EXPECTED_NATIVE_TILE_SHAPE),
        },
        "data": {"test_filenames": list(TILE_NAMES)},
        "comparators": comparators,
        "curves": {
            "reported": False,
            "reason": "synthetic hard-label fixture",
        },
        "outputs": [
            "evaluation_summary.json",
            "aggregate_metrics.csv",
            "per_tile_metrics.csv",
            "per_tile_confusion.csv",
        ],
    }
    report_path = directory / "evaluation_summary.json"
    _write_json(report_path, report)
    return {
        "report": _artifact(root, report_path),
        "aggregate_metrics_csv": _artifact(root, aggregate_path),
        "per_tile_metrics_csv": _artifact(root, tile_path),
        "per_tile_confusion_csv": _artifact(root, confusion_path),
        "expected_identity": {
            "fit_id": FIT_ID,
            "evaluation_pair_identity_sha256": PAIR_SHA,
            "neural_freeze_manifest_id": FREEZE_ID,
            "neural_freeze_scientific_identity_sha256": FREEZE_SHA,
            "classical_lock_identity_sha256": LOCK_SHA,
            "classical_lock_path": CLASSICAL_LOCK_PATH,
            "classical_lock_raw_file_sha256": CLASSICAL_LOCK_RAW_SHA,
            "classical_lock_source_code_sha256": CLASSICAL_SOURCE_MAP,
            "b2_model_path": B2_MODEL_PATH,
            "b2_model_sha256": B2_MODEL_SHA,
            "b2_model_semantic_sha256": B2_MODEL_SEMANTIC_SHA,
            "selected_method_lock_path": SELECTED_LOCK_PATH,
            "selected_method_lock_sha256": SELECTED_LOCK_SHA,
            "selected_retraining_checkpoint_sha256": checkpoint_map,
            "neural_freeze_manifest_path": FREEZE_MANIFEST_PATH,
            "neural_freeze_manifest_file_sha256": FREEZE_FILE_SHA,
        },
    }


def _build_fixture(
    root: Path,
    *,
    selected_method: str = "R3",
    with_curves: bool = False,
    plain_with_curves: bool | None = None,
    replicates: int = assembler.PAIRED_BOOTSTRAP_REPLICATES,
) -> Path:
    selected_display = {
        "R3": "R3",
        "C2-FP": "C2-FP",
    }[selected_method]
    neural = []
    for cell_index, seed in enumerate(assembler.NEURAL_SEEDS):
        neural.append(
            _neural_record(
                root,
                method_id=selected_method,
                display_name=selected_display,
                selected_method=selected_method,
                role="primary_multiscale",
                seed=seed,
                cell_index=cell_index,
                with_curves=with_curves,
            )
        )
    resolved_plain_curves = (
        with_curves if plain_with_curves is None else plain_with_curves
    )
    for cell_index, seed in enumerate(assembler.NEURAL_SEEDS):
        neural.append(
            _neural_record(
                root,
                method_id="plain_unet",
                display_name="Plain U-Net",
                selected_method=selected_method,
                role="plain_unet_comparator",
                seed=seed,
                cell_index=cell_index,
                with_curves=resolved_plain_curves,
            )
        )
    manifest = {
        "schema_version": assembler.INPUT_SCHEMA_VERSION,
        "generator": {
            "schema_version": assembler.GENERATOR_SCHEMA_VERSION,
            "version": assembler.GENERATOR_VERSION,
            "source_sha256": assembler.GENERATOR_SOURCE_SHA256,
        },
        "partition_label": assembler.PARTITION_LABEL,
        "bootstrap": {
            "replicates": replicates,
            "seed": assembler.PAIRED_BOOTSTRAP_SEED,
            "confidence": assembler.PAIRED_BOOTSTRAP_CONFIDENCE,
        },
        "neural_evaluations": neural,
        "classical_evaluation": _classical_record(
            root, selected_method=selected_method
        ),
        "qualitative_figures": [],
    }
    manifest_path = root / "publication_results_inputs.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _attach_qualitative_panel(
    root: Path,
    manifest_path: Path,
    *,
    seed: int = 42,
    outcome_used_for_selection: bool = False,
    declare_evaluator_outputs: bool = True,
) -> tuple[Path, Path]:
    manifest = _load_manifest(manifest_path)
    record = next(
        item
        for item in manifest["neural_evaluations"]
        if item["method_id"] != "plain_unet"
        and item["expected_identity"]["training_seed"] == seed
    )
    report_path = root / record["report"]["path"]
    publication_dir = report_path.parent / "publication"
    publication_dir.mkdir()
    pdf_path = publication_dir / "qualitative_triptych.pdf"
    png_path = publication_dir / "qualitative_triptych.png"
    figure_contract = qualitative_figure_contract(
        record["method_id"], image_id=1, file_name=TILE_NAMES[0]
    )

    plt = assembler._configure_matplotlib()
    figure, axes = plt.subplots(1, 6, figsize=(9.6, 1.8), constrained_layout=True)
    synthetic = np.ones((24, 24, 3), dtype=np.float64)
    palette = [
        tuple(
            int(assembler.CLASS_PALETTE[key][index : index + 2], 16) / 255.0
            for index in (1, 3, 5)
        )
        for key in ("C0", "C1", "C2")
    ]
    synthetic[:, :8] = palette[0]
    synthetic[:, 8:16] = palette[1]
    synthetic[:, 16:] = palette[2]
    grayscale = np.linspace(0.1, 0.9, 24 * 24).reshape(24, 24)
    c0_evidence = np.tile(np.linspace(0, 1, 24), (24, 1))
    c1_evidence = np.flipud(c0_evidence.T)
    error_overlay = np.ones_like(synthetic)
    error_overlay[5:12, 5:12] = palette[0]
    error_overlay[13:20, 13:20] = palette[1]
    rendered = (
        (grayscale, "gray", 0, 1),
        (synthetic, None, None, None),
        (c0_evidence, "Reds", 0, 1),
        (c1_evidence, "Greens", 0, 1),
        (synthetic, None, None, None),
        (error_overlay, None, None, None),
    )
    for axis, title, (image, cmap, lower, upper) in zip(
        axes, figure_contract["panel_titles"], rendered, strict=True
    ):
        axis.imshow(image, cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(
        pdf_path,
        metadata={
            "Creator": "synthetic assembler test fixture",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        png_path,
        dpi=120,
        metadata={"Software": "synthetic assembler test fixture"},
    )
    plt.close(figure)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["qualitative_example"] = {
        "selection_rule": (
            "selected after reviewing outcomes"
            if outcome_used_for_selection
            else assembler.QUALITATIVE_SELECTION_RULE
        ),
        "image_id": 1,
        "file_name": TILE_NAMES[0],
        "input_image_sha256": _input_hash(TILE_NAMES[0]),
        "target_mask_sha256": _target_hash(TILE_NAMES[0]),
        "post_processing": "none",
        "publication_figure_contract": figure_contract,
        "publication_files": [
            assembler.QUALITATIVE_PDF_PATH,
            assembler.QUALITATIVE_PNG_PATH,
        ],
    }
    if declare_evaluator_outputs:
        report["outputs"].extend(
            (assembler.QUALITATIVE_PDF_PATH, assembler.QUALITATIVE_PNG_PATH)
        )
    _write_json(report_path, report)
    _rehash(root, record["report"])
    manifest["qualitative_figures"].append(
        {
            "method_id": record["method_id"],
            "training_seed": seed,
            "pdf": _artifact(root, pdf_path),
            "png": _artifact(root, png_path),
        }
    )
    _save_manifest(manifest_path, manifest)
    return pdf_path, png_path


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    _write_json(path, manifest)


def _rehash(root: Path, specification: dict[str, str]) -> None:
    specification["sha256"] = _file_sha(root / specification["path"])


def _mutate_csv_cell(
    root: Path,
    specification: dict[str, str],
    *,
    row_index: int,
    column: str,
    value: str,
) -> None:
    path = root / specification["path"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[row_index][column] = value
    _write_csv(path, fields, rows)
    _rehash(root, specification)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_neural_seed_contract_matches_the_frozen_screen_source() -> None:
    assert assembler.NEURAL_SEEDS == SCREEN_SEEDS
    assert assembler.SELECTED_METHOD_IDS == SCREEN_CANDIDATE_ORDER


def test_paired_bootstrap_contract_matches_the_frozen_evaluator() -> None:
    assert assembler.PAIRED_BOOTSTRAP_REPLICATES == LOCKED_BOOTSTRAP_REPLICATES
    assert assembler.PAIRED_BOOTSTRAP_SEED == LOCKED_BOOTSTRAP_SEED
    assert assembler.PAIRED_BOOTSTRAP_CONFIDENCE == LOCKED_CONFIDENCE
    assert assembler.LOCKED_CURVE_BINS == EVALUATOR_LOCKED_CURVE_BINS
    assert assembler.EXPECTED_NATIVE_TILE_SHAPE == NEURAL_EXPECTED_TILE_SHAPE
    assert assembler.EXPECTED_NATIVE_TILE_SHAPE == CLASSICAL_EXPECTED_TILE_SHAPE
    assert assembler.CLASSICAL_METHOD_IDS == CLASSICAL_COMPARATOR_IDS
    assert (
        assembler.CLASSICAL_REPORT_SCHEMA_VERSION
        == CLASSICAL_EVALUATOR_SCHEMA_VERSION
    )


def test_qualitative_contract_matches_the_pinned_evaluator_schema() -> None:
    for method_id in assembler.SELECTED_METHOD_IDS:
        native = qualitative_figure_contract(
            method_id, image_id=1, file_name=TILE_NAMES[0]
        )
        json_round_trip = json.loads(json.dumps(native))
        assert assembler._qualitative_figure_contract(
            json_round_trip,
            method_id=method_id,
            image_id=1,
            file_name=TILE_NAMES[0],
            label="pinned evaluator contract",
        ) == json_round_trip
    assert assembler.QUALITATIVE_PDF_PATH == (
        "publication/qualitative_triptych.pdf"
    )
    assert assembler.QUALITATIVE_PNG_PATH == (
        "publication/qualitative_triptych.png"
    )


def test_schema_v2_requires_an_explicit_qualitative_figures_list(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    del manifest["qualitative_figures"]
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="keys mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_complete_assembly_is_deterministic_and_publication_ready(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(
        tmp_path, selected_method="C2-FP", with_curves=True
    )
    first = assembler.assemble_publication_results(manifest, tmp_path / "out_a")
    second = assembler.assemble_publication_results(manifest, tmp_path / "out_b")

    first_files = sorted(path.name for path in first.iterdir())
    second_files = sorted(path.name for path in second.iterdir())
    assert first_files == second_files
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    report = json.loads((first / "publication_results.json").read_text())
    assert report["partition"] == assembler.PARTITION_LABEL
    assert report["generator"]["schema_version"] == 1
    assert report["generator"]["version"] == assembler.GENERATOR_VERSION
    assert report["generator"]["source_sha256"] == (
        assembler.GENERATOR_SOURCE_SHA256
    )
    assert report["generator"]["runtime_versions"]["numpy"] == np.__version__
    assert report["class_palette"] == {
        "C0": "#B33A3A",
        "C1": "#2E8B57",
        "C2": "#4C78A8",
    }
    assert report["precision_recall"]["status"] == "created"
    assert report["aggregate_confusion_matrix"]["status"] == "created"
    assert report["prediction_error_summary"]["status"] == "created"
    assert report["c0_c1_recall_summary"]["status"] == "created"
    assert report["full_area_metrics"]["status"] == "created"
    assert report["prediction_error_summary"]["scope"] == (
        "non-spatial per-method confusion-derived summary"
    )
    assert report["spatial_prediction_figures"]["status"] == "not_reconstructed"
    assert report["qualitative_figures"]["status"] == "omitted"
    assert "does not embed the PDF/PNG SHA-256" in report[
        "qualitative_figures"
    ]["source_schema_limitation"]
    assert report["paired_bootstrap"]["interpretation"] == (
        "within-series whole-tile sensitivity interval"
    )
    assert report["selected_primary_method_id"] == "C2-FP"
    assert report["contrast_direction"] == (
        "selected_primary_minus_comparator"
    )
    assert report["precision_recall"]["methods"] == ["C2-FP", "plain_unet"]
    assert (
        report["precision_recall"]["average_precision_label"]
        == "AP (4096-bin approximation)"
    )
    assert (
        report["precision_recall"]["construction"]
        == "Counts are summed across seeds within a method and the "
        "precision-recall coordinates are reconstructed at the original "
        "bin thresholds without interpolation."
    )
    assert (first / "cross_method_c0_c1_precision_recall.pdf").is_file()
    assert (first / "cross_method_c0_c1_precision_recall.png").is_file()
    assert (first / "selected_primary_c0_c1_c2_confusion_matrix.pdf").is_file()
    assert (first / "selected_primary_c0_c1_c2_confusion_matrix.png").is_file()
    assert (first / "cross_method_c0_c1_recall_summary.pdf").is_file()
    assert (first / "cross_method_c0_c1_recall_summary.png").is_file()
    assert (first / assembler.FIGURE_TEX_FRAGMENT_NAME).is_file()
    pr_rows = _read_csv_rows(
        first / "cross_method_c0_c1_precision_recall.csv"
    )
    assert {row["method_id"] for row in pr_rows} == {"C2-FP", "plain_unet"}
    assert {row["class_id"] for row in pr_rows} == {"0", "1"}
    assert {row["histogram_bins"] for row in pr_rows} == {"4096"}
    assert all(
        row["average_precision_histogram_approximation"] for row in pr_rows
    )
    selected_confusion_rows = _read_csv_rows(
        first / "selected_primary_c0_c1_c2_confusion.csv"
    )
    assert len(selected_confusion_rows) == 9
    expected_selected_confusion = np.sum(
        [
            _matrix("C2-FP", seed, tile_index)
            for seed in assembler.NEURAL_SEEDS
            for tile_index in range(len(TILE_NAMES))
        ],
        axis=0,
    )
    for row in selected_confusion_rows:
        true_id = int(row["true_class_id"])
        predicted_id = int(row["predicted_class_id"])
        assert int(row["pixel_count"]) == int(
            expected_selected_confusion[true_id, predicted_id]
        )
        assert float(row["row_fraction"]) == pytest.approx(
            expected_selected_confusion[true_id, predicted_id]
            / expected_selected_confusion[true_id].sum()
        )
        assert row["partition"] == assembler.PARTITION_LABEL
    summary_rows = _read_csv_rows(
        first / "per_method_c0_c1_c2_prediction_error_summary.csv"
    )
    assert len(summary_rows) == 9 * 5
    assert {row["method_id"] for row in summary_rows} == {
        "C2-FP",
        "plain_unet",
        *assembler.CLASSICAL_METHOD_IDS,
    }
    for method_id in {row["method_id"] for row in summary_rows}:
        for true_id in range(3):
            fractions = [
                float(row["row_fraction"])
                for row in summary_rows
                if row["method_id"] == method_id
                and int(row["true_class_id"]) == true_id
            ]
            assert sum(fractions) == pytest.approx(1.0)

    full_area_rows = _read_csv_rows(
        first / "per_method_full_area_aggregate.csv"
    )
    assert len(full_area_rows) == 5
    assert {row["method_id"] for row in full_area_rows} == {
        "C2-FP",
        "plain_unet",
        *assembler.CLASSICAL_METHOD_IDS,
    }
    assert all(
        row["prediction_source"]
        == "fixed raw-intensity pore gate (rule-assisted C2)"
        for row in full_area_rows
    )
    for row in full_area_rows:
        for scope in ("c2", "pore_union"):
            for metric in assembler.METRIC_NAMES:
                value = row[f"{scope}_{metric}"]
                assert value
                assert 0.0 <= float(value) <= 1.0
    assert report["evidence_gated_omissions"]["parameter_count"]["status"] == (
        "omitted"
    )
    assert report["evidence_gated_omissions"]["training_time"]["status"] == (
        "omitted"
    )

    evidence = report["authenticated_evidence"]
    neural_evidence = [
        row for row in evidence if row["source_kind"] == "neural"
    ]
    classical_evidence = [
        row for row in evidence if row["source_kind"] == "classical"
    ]
    assert len(neural_evidence) == 6
    assert len(classical_evidence) == 3
    assert {
        (
            row["source_identity"]["architecture_role"],
            row["source_identity"]["cell_index"],
            row["source_identity"]["training_seed"],
        )
        for row in neural_evidence
    } == {
        (role, cell_index, seed)
        for role in ("primary_multiscale", "plain_unet_comparator")
        for cell_index, seed in enumerate(assembler.NEURAL_SEEDS)
    }
    assert all(
        row["neural_freeze_manifest_id"] == FREEZE_ID for row in evidence
    )
    assert all(
        row["neural_freeze_scientific_identity_sha256"] == FREEZE_SHA
        for row in evidence
    )
    assert all(
        "probability_histograms_csv" not in row["artifact_sha256"]
        for row in classical_evidence
    )
    assert {
        row["source_identity"]["comparator"] for row in classical_evidence
    } == set(assembler.CLASSICAL_METHOD_IDS)

    for path in first.iterdir():
        if path.suffix in {".json", ".csv", ".tex", ".sha256"}:
            text = path.read_text(encoding="utf-8").lower()
            assert "held-out" not in text
            assert "unseen" not in text
    tex = (first / assembler.TEX_FRAGMENT_NAME).read_text()
    assert assembler.PARTITION_LABEL in tex
    assert "sample standard deviations" in tex
    assert "selected primary minus comparator" in tex
    assert "95\\% equal-tail percentile interval" in tex
    assert "5,000 paired resamples of whole native tiles" in tex
    assert "within-series whole-tile sensitivity interval" in tex
    assert r"\begin{table*}[t]" not in tex
    assert r"\begin{table}[t]" not in tex
    assert r"\begin{table*}[H]" not in tex
    assert r"\begin{table}[H]" not in tex
    assert r"\begin{table*}[" not in tex
    assert r"\begin{table}[" not in tex
    assert r"\begin{table*}" in tex
    assert tex.count(r"\begin{table*}") == 5
    assert tex.count(r"\begin{table}") == 0
    assert "C0 disconnected pore" in tex
    assert "C1 connected pore" in tex
    assert "Pore union" in tex
    assert "Parameter counts and scheduler-recorded training times are omitted" in tex
    assert "R3" not in tex
    figure_tex = (first / assembler.FIGURE_TEX_FRAGMENT_NAME).read_text()
    assert "cross_method_c0_c1_precision_recall.pdf" in figure_tex
    assert "selected_primary_c0_c1_c2_confusion_matrix.pdf" in figure_tex
    assert "cross_method_c0_c1_recall_summary.pdf" in figure_tex
    assert figure_tex.index("cross_method_c0_c1_precision_recall.pdf") < (
        figure_tex.index("selected_primary_c0_c1_c2_confusion_matrix.pdf")
    )
    assert assembler.PARTITION_LABEL in figure_tex
    assert "correct prediction share" in figure_tex

    checksum_lines = (
        first / "checksums.sha256"
    ).read_text(encoding="ascii").splitlines()
    assert all("checksums.sha256" not in line for line in checksum_lines)
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        assert digest == _file_sha(first / name)


@pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="pdflatex is unavailable for the synthetic CAS-width fit check",
)
def test_generated_fragments_fit_a_synthetic_cas_width_without_overflow(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(
        tmp_path, selected_method="C2-FP", with_curves=True
    )
    assembler.assemble_publication_results(manifest, tmp_path / "out")
    driver = tmp_path / "cas_width_proxy.tex"
    driver.write_text(
        "\n".join(
            (
                r"\documentclass[twocolumn]{article}",
                r"\usepackage[a4paper,margin=10mm]{geometry}",
                r"\usepackage{graphicx}",
                r"\usepackage{booktabs}",
                r"\graphicspath{{out/}}",
                r"\begin{document}",
                r"\input{out/publication_results_tables.tex}",
                r"\input{out/publication_results_figures.tex}",
                r"\clearpage",
                r"\end{document}",
                "",
            )
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            str(shutil.which("pdflatex")),
            "-interaction=nonstopmode",
            "-halt-on-error",
            driver.name,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    log = completed.stdout + completed.stderr
    log_path = tmp_path / "cas_width_proxy.log"
    if log_path.is_file():
        log += log_path.read_text(encoding="utf-8", errors="replace")
    assert completed.returncode == 0, log
    assert "Overfull \\hbox" not in log
    assert "Overfull \\vbox" not in log


def test_full_area_rows_distinguish_learned_and_rule_assisted_sources(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path, selected_method="R3")
    output = assembler.assemble_publication_results(manifest, tmp_path / "out")
    rows = {
        row["method_id"]: row
        for row in _read_csv_rows(output / "per_method_full_area_aggregate.csv")
    }
    assert rows["R3"]["prediction_source"] == (
        "native learned three-class prediction"
    )
    assert rows["plain_unet"]["prediction_source"] == (
        "native learned three-class prediction"
    )
    for method_id in assembler.CLASSICAL_METHOD_IDS:
        assert rows[method_id]["prediction_source"] == (
            "fixed raw-intensity pore gate (rule-assisted C2)"
        )


def test_authenticated_qualitative_panel_is_copied_byte_for_byte(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path, selected_method="C2-FP")
    source_pdf, source_png = _attach_qualitative_panel(
        tmp_path, manifest
    )
    source_pdf_bytes = source_pdf.read_bytes()
    source_png_bytes = source_png.read_bytes()

    output = assembler.assemble_publication_results(manifest, tmp_path / "out")
    copied_pdf = (
        output / "selected_primary_outcome_independent_qualitative.pdf"
    )
    copied_png = (
        output / "selected_primary_outcome_independent_qualitative.png"
    )
    assert copied_pdf.read_bytes() == source_pdf_bytes
    assert copied_png.read_bytes() == source_png_bytes

    report = json.loads((output / "publication_results.json").read_text())
    assert report["qualitative_figures"]["status"] == "copied"
    panel = report["qualitative_figures"]["panels"][0]
    assert panel["method_id"] == "C2-FP"
    assert panel["training_seed"] == 42
    assert panel["source_pdf_sha256"] == _sha(source_pdf_bytes)
    assert panel["copied_pdf_sha256"] == _file_sha(copied_pdf)
    assert panel["source_png_sha256"] == _sha(source_png_bytes)
    assert panel["copied_png_sha256"] == _file_sha(copied_png)
    assert panel["copy_semantics"] == "byte-for-byte authenticated snapshot copy"
    scale_gate = panel["publication_figure_contract"][
        "physical_scale_author_evidence_gate"
    ]
    assert scale_gate["status"] == "open_author_evidence_required"
    assert scale_gate["scale_bar_shown"] is False
    figure_tex = (output / assembler.FIGURE_TEX_FRAGMENT_NAME).read_text()
    assert "selected_primary_outcome_independent_qualitative.pdf" in figure_tex
    assert "outcome-independent qualitative panel" in figure_tex
    assert "did not reread microscopy images, masks, or predictions" in figure_tex
    assert "No scale bar or pixel size is shown" in figure_tex


def test_qualitative_selection_that_used_outcomes_fails_closed(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path, selected_method="C2-FP")
    _attach_qualitative_panel(
        tmp_path,
        manifest,
        outcome_used_for_selection=True,
    )
    output = tmp_path / "out"
    with pytest.raises(
        assembler.ContractError,
        match="evaluator-authenticated and outcome-independent",
    ):
        assembler.assemble_publication_results(manifest, output)
    assert not output.exists()


def test_qualitative_source_absence_fails_closed(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    _attach_qualitative_panel(tmp_path, manifest_path)
    manifest = _load_manifest(manifest_path)
    panel_specification = manifest["qualitative_figures"][0]["png"]
    original_path = Path(panel_specification["path"])
    panel_specification["path"] = (
        original_path.parent / "missing" / original_path.name
    ).as_posix()
    _save_manifest(manifest_path, manifest)
    output = tmp_path / "out"
    with pytest.raises(assembler.ContractError, match="not a regular file"):
        assembler.assemble_publication_results(manifest_path, output)
    assert not output.exists()


def test_qualitative_seed_choice_is_prespecified_not_result_selected(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path, selected_method="C2-FP")
    _attach_qualitative_panel(tmp_path, manifest, seed=123)
    with pytest.raises(
        assembler.ContractError,
        match="prespecified seed 42",
    ):
        assembler.assemble_publication_results(manifest, tmp_path / "out")


def test_qualitative_symbolic_link_source_is_rejected(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    _, source_png = _attach_qualitative_panel(tmp_path, manifest_path)
    linked_dir = tmp_path / "linked_publication"
    linked_dir.mkdir()
    linked_png = linked_dir / source_png.name
    linked_png.symlink_to(source_png)
    manifest = _load_manifest(manifest_path)
    specification = manifest["qualitative_figures"][0]["png"]
    specification["path"] = linked_png.relative_to(tmp_path).as_posix()
    specification["sha256"] = _file_sha(source_png)
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="symbolic link"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_qualitative_artifacts_must_be_declared_evaluator_outputs(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path, selected_method="C2-FP")
    _attach_qualitative_panel(
        tmp_path,
        manifest,
        declare_evaluator_outputs=False,
    )
    with pytest.raises(
        assembler.ContractError,
        match="declared evaluator outputs",
    ):
        assembler.assemble_publication_results(manifest, tmp_path / "out")


def test_qualitative_copy_uses_one_authenticated_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _build_fixture(tmp_path, selected_method="C2-FP")
    _, source_png = _attach_qualitative_panel(tmp_path, manifest)
    authenticated_bytes = source_png.read_bytes()
    original_artifact_path = assembler._artifact_path
    mutation: dict[str, Any] = {}

    def mutate_panel_after_snapshot(*args: Any, **kwargs: Any) -> Any:
        artifact = original_artifact_path(*args, **kwargs)
        label = args[2]
        if label.endswith(".png") and not mutation:
            mutation["authenticated_sha256"] = artifact.sha256
            artifact.path.write_bytes(b"tampered after authentication\n")
        return artifact

    monkeypatch.setattr(assembler, "_artifact_path", mutate_panel_after_snapshot)
    output = assembler.assemble_publication_results(manifest, tmp_path / "out")
    copied = output / "selected_primary_outcome_independent_qualitative.png"
    assert copied.read_bytes() == authenticated_bytes
    assert _file_sha(copied) == mutation["authenticated_sha256"]
    assert _file_sha(source_png) != mutation["authenticated_sha256"]


def test_seed_means_sample_sd_harmonic_and_signed_contrasts_are_exact(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path, selected_method="C2-FP")
    manifest_payload = _load_manifest(manifest)
    classical_record = manifest_payload["classical_evaluation"]
    aggregate_rows = _read_csv_rows(
        tmp_path / classical_record["aggregate_metrics_csv"]["path"]
    )
    assert next(
        row["value"]
        for row in aggregate_rows
        if row["comparator"] == "B0_small_components"
        and row["scope"] == "class"
        and row["class_id"] == "0"
        and row["metric"] == "precision"
    ) == ""
    output = assembler.assemble_publication_results(manifest, tmp_path / "out")
    report = json.loads((output / "publication_results.json").read_text())
    method_rows = {
        row["method_id"]: row for row in report["per_method_aggregate"]
    }
    seed_rows = [
        row
        for row in report["per_seed_aggregate"]
        if row["method_id"] == "C2-FP"
    ]
    c0_values = [row["c0_iou"] for row in seed_rows]
    assert method_rows["C2-FP"]["c0_iou_mean"] == pytest.approx(
        statistics.fmean(c0_values)
    )
    assert method_rows["C2-FP"]["c0_iou_sample_sd"] == pytest.approx(
        statistics.stdev(c0_values)
    )
    assert all(
        row["selected_primary_minus_method_c0_iou"] is None
        for row in seed_rows
    )
    assert method_rows["C2-FP"][
        "selected_primary_minus_method_c0_c1_harmonic_iou"
    ] is None
    assert method_rows["B0_small_components"][
        "c0_iou_sample_sd"
    ] is None
    assert method_rows["B0_small_components"]["c0_precision_mean"] is None
    assert method_rows["B0_small_components"][
        "selected_primary_minus_method_c0_precision"
    ] is None
    assert method_rows["plain_unet"][
        "selected_primary_minus_method_c0_iou"
    ] > 0
    assert method_rows["plain_unet"][
        "selected_primary_minus_method_c1_iou"
    ] > 0
    assert method_rows["plain_unet"]["contrast_direction"] == (
        "selected_primary_minus_comparator"
    )

    for row in seed_rows:
        expected_harmonic = (
            2 * row["c0_iou"] * row["c1_iou"]
            / (row["c0_iou"] + row["c1_iou"] + 1e-8)
        )
        assert row["c0_c1_harmonic_iou"] == pytest.approx(expected_harmonic)


def test_non_r3_winner_requires_no_locked_r3_report(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    manifest = _load_manifest(manifest_path)
    assert {
        record["method_id"] for record in manifest["neural_evaluations"]
    } == {"C2-FP", "plain_unet"}
    output = assembler.assemble_publication_results(
        manifest_path, tmp_path / "out"
    )
    report = json.loads((output / "publication_results.json").read_text())
    assert report["selected_primary_method_id"] == "C2-FP"
    assert {
        row["method_id"]
        for row in report["authenticated_evidence"]
        if row["source_kind"] == "neural"
    } == {"C2-FP", "plain_unet"}
    assert report["validation_screen_r3_boundary"]["status"] == (
        "not_assembled_from_locked_evaluations"
    )


def test_extra_locked_r3_report_is_rejected_for_non_r3_winner(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    manifest = _load_manifest(manifest_path)
    manifest["neural_evaluations"].append(
        _neural_record(
            tmp_path,
            method_id="R3",
            display_name="R3",
            selected_method="R3",
            role="primary_multiscale",
            seed=assembler.NEURAL_SEEDS[0],
            cell_index=0,
            with_curves=False,
        )
    )
    _save_manifest(manifest_path, manifest)
    with pytest.raises(
        assembler.ContractError,
        match="Exactly three selected-primary and three plain-comparator",
    ):
        assembler.assemble_publication_results(
            manifest_path, tmp_path / "out"
        )


def test_publication_display_names_are_unique_canonical_method_labels(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    manifest = _load_manifest(manifest_path)
    for record in manifest["neural_evaluations"]:
        record["display_name"] = "Plain U-Net"
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="canonical publication labels"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_paired_bootstrap_point_estimates_match_signed_method_margins(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(
        tmp_path, selected_method="C2-FP"
    )
    output = assembler.assemble_publication_results(manifest, tmp_path / "out")
    report = json.loads((output / "publication_results.json").read_text())
    method_rows = {
        row["method_id"]: row for row in report["per_method_aggregate"]
    }
    for row in report["paired_bootstrap_differences"]:
        expected = method_rows[row["method_id"]][
            f"selected_primary_minus_method_{row['metric']}"
        ]
        if expected is None:
            assert row["difference"] is None
            assert row["ci_lower"] is None
            assert row["ci_upper"] is None
            assert row["finite_replicates"] == 0
        else:
            assert row["difference"] == pytest.approx(expected)
            assert row["ci_lower"] <= row["ci_upper"]
            assert row["finite_replicates"] > 0
        assert row["reference_method_id"] == "C2-FP"
        assert row["contrast_direction"] == (
            "selected_primary_minus_comparator"
        )
        assert "same resampled whole-tile indices" in row["pairing"]
        assert row["sampling_unit"].endswith(assembler.PARTITION_LABEL)


def test_paired_bootstrap_reuses_the_same_tile_indices_for_primary_and_comparator(
    tmp_path: Path,
) -> None:
    replicates = assembler.PAIRED_BOOTSTRAP_REPLICATES
    seed = assembler.PAIRED_BOOTSTRAP_SEED
    confidence = assembler.PAIRED_BOOTSTRAP_CONFIDENCE
    manifest = _build_fixture(
        tmp_path, selected_method="C2-FP"
    )
    output = assembler.assemble_publication_results(manifest, tmp_path / "out")
    report = json.loads((output / "publication_results.json").read_text())
    observed = next(
        row
        for row in report["paired_bootstrap_differences"]
        if row["method_id"] == "plain_unet" and row["metric"] == "c0_iou"
    )

    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(
        0,
        len(TILE_NAMES),
        size=(replicates, len(TILE_NAMES)),
        endpoint=False,
        dtype=np.int64,
    )
    differences = []
    for indices in draws:
        seed_differences = []
        for training_seed in assembler.NEURAL_SEEDS:
            target_tiles = np.stack(
                [
                    _matrix("plain_unet", training_seed, tile_index)
                    for tile_index in range(len(TILE_NAMES))
                ]
            )
            reference_tiles = np.stack(
                [
                    _matrix("C2-FP", training_seed, tile_index)
                    for tile_index in range(len(TILE_NAMES))
                ]
            )
            target = assembler.metrics_from_confusion(
                target_tiles[indices].sum(axis=0)
            )["c0_iou"]
            reference = assembler.metrics_from_confusion(
                reference_tiles[indices].sum(axis=0)
            )["c0_iou"]
            seed_differences.append(float(reference) - float(target))
        differences.append(statistics.fmean(seed_differences))
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        np.asarray(differences), [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    assert observed["ci_lower"] == pytest.approx(lower)
    assert observed["ci_upper"] == pytest.approx(upper)
    assert observed["difference"] > 0


def test_per_tile_diagnostics_keep_same_index_and_hash_provenance(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path, selected_method="C2-FP")
    output = assembler.assemble_publication_results(manifest, tmp_path / "out")
    rows = _read_csv_rows(output / "per_tile_diagnostics.csv")
    expected_rows = (3 + 3 + 3) * len(TILE_NAMES)
    assert len(rows) == expected_rows
    for row in rows:
        assert row["input_sha256"] == _input_hash(row["file_name"])
        assert row["target_sha256"] == _target_hash(row["file_name"])
        assert row["partition"] == assembler.PARTITION_LABEL
        assert row["reference_basis"] in {
            "selected primary self row; contrast intentionally undefined",
            "same-seed selected primary for the same tile index",
            "selected-primary seed mean for the same tile index",
        }
        assert row["worst_pore_iou_class"] in {"C0", "C1", "tie", "undefined"}


def test_cross_method_curve_is_omitted_with_only_one_evidenced_method(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(
        tmp_path,
        selected_method="C2-FP",
        with_curves=True,
        plain_with_curves=False,
    )
    output = assembler.assemble_publication_results(manifest, tmp_path / "out")
    report = json.loads((output / "publication_results.json").read_text())
    assert report["precision_recall"]["status"] == "omitted"
    assert report["precision_recall"]["methods"] == []
    assert not (output / "cross_method_c0_c1_precision_recall.csv").exists()
    assert not (output / "cross_method_c0_c1_precision_recall.pdf").exists()
    assert not (output / "cross_method_c0_c1_precision_recall.png").exists()
    figure_tex = (output / assembler.FIGURE_TEX_FRAGMENT_NAME).read_text()
    assert "cross_method_c0_c1_precision_recall.pdf" not in figure_tex
    assert "cross_method_c0_c1_recall_summary.pdf" in figure_tex


def test_report_schema_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["neural_evaluations"][0]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    report["schema_version"] = "2.1"
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="schema mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_file_digest_mismatch_fails_before_metric_use(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["neural_evaluations"][0]
    confusion_path = tmp_path / record["per_tile_confusion_csv"]["path"]
    confusion_path.write_text(
        confusion_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(assembler.ContractError, match="SHA-256 mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_clean_no_results_state_fails_without_creating_output(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    manifest["neural_evaluations"] = []
    _save_manifest(manifest_path, manifest)
    output = tmp_path / "out"
    with pytest.raises(assembler.ContractError, match="non-empty list"):
        assembler.assemble_publication_results(manifest_path, output)
    assert not output.exists()


def test_embedded_checkpoint_identity_mismatch_fails_even_when_rehashed(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["neural_evaluations"][0]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    report["locked_evaluation_identity"]["checkpoint_sha256"] = "f" * 64
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="identity mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


@pytest.mark.parametrize(
    "mutation",
    (
        "evaluator_sha256",
        "training_source_attestation",
        "selected_method_lock_sha256",
        "neural_freeze_manifest_file_sha256",
        "checkpoint_map",
        "locked_state_semantic_sha256",
        "loaded_state_semantic_sha256",
        "missing_loaded_state_semantic_sha256",
        "missing_expected_state_semantic_sha256",
        "missing_code",
    ),
)
def test_neural_source_and_freeze_provenance_mismatch_fails_when_rehashed(
    tmp_path: Path, mutation: str
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["neural_evaluations"][0]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    if mutation == "evaluator_sha256":
        report["code"]["evaluator_sha256"] = "0" * 64
    elif mutation == "training_source_attestation":
        report["code"]["training_execution_source_attestation"]["files"][
            "src/training/patch_trainer.py"
        ] = "0" * 64
    elif mutation == "selected_method_lock_sha256":
        report["selected_method_lock"]["sha256"] = "0" * 64
    elif mutation == "neural_freeze_manifest_file_sha256":
        report["neural_freeze"]["manifest_file_sha256"] = "0" * 64
    elif mutation == "checkpoint_map":
        report["neural_freeze"]["selected_retraining_checkpoint_sha256"][
            "primary_multiscale"
        ][0] = "0" * 64
    elif mutation == "locked_state_semantic_sha256":
        report["locked_evaluation_identity"][
            "checkpoint_state_dict_semantic_sha256"
        ] = "0" * 64
    elif mutation == "loaded_state_semantic_sha256":
        report["checkpoint"]["state_dict_semantic_sha256"] = "0" * 64
    elif mutation == "missing_loaded_state_semantic_sha256":
        del report["checkpoint"]["state_dict_semantic_sha256"]
    elif mutation == "missing_expected_state_semantic_sha256":
        del record["expected_identity"][
            "checkpoint_state_dict_semantic_sha256"
        ]
    else:
        del report["code"]
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


@pytest.mark.parametrize(
    "mutation",
    (
        "raw_lock_sha256",
        "lock_source_sha256",
        "b2_model_sha256",
        "b2_model_semantic_sha256",
        "selected_method_lock_sha256",
        "neural_freeze_manifest_file_sha256",
    ),
)
def test_classical_lock_model_and_freeze_provenance_mismatch_fails_when_rehashed(
    tmp_path: Path, mutation: str
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["classical_evaluation"]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    if mutation == "raw_lock_sha256":
        report["classical_lock"]["raw_file_sha256"] = "0" * 64
    elif mutation == "lock_source_sha256":
        report["classical_lock"]["source_code_sha256"][
            "src/classical/comparators.py"
        ] = "0" * 64
    elif mutation == "b2_model_sha256":
        report["b2_model"]["sha256"] = "0" * 64
    elif mutation == "b2_model_semantic_sha256":
        report["b2_model"]["semantic_sha256"] = "0" * 64
    elif mutation == "selected_method_lock_sha256":
        report["neural_freeze"]["selected_method_lock"][
            "raw_file_sha256"
        ] = "0" * 64
    else:
        report["neural_freeze"]["manifest_file_sha256"] = "0" * 64
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_classical_pair_identity_mismatch_fails_even_when_rehashed(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["classical_evaluation"]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    report["evaluation_pair_identity_sha256"] = "f" * 64
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="identity mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_hard_label_classical_report_cannot_claim_probability_curves(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["classical_evaluation"]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    report["curves"]["reported"] = True
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(
        assembler.ContractError,
        match="cannot supply PR evidence",
    ):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_json_csv_confusion_disagreement_fails_closed(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["neural_evaluations"][0]
    path = tmp_path / record["per_tile_confusion_csv"]["path"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["pixel_count"] = str(int(rows[0]["pixel_count"]) + 1)
    _write_csv(path, fields, rows)
    _rehash(tmp_path, record["per_tile_confusion_csv"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="differs between JSON and CSV"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_partition_hash_identity_disagreement_across_methods_fails(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    plain_record = next(
        item
        for item in manifest["neural_evaluations"]
        if item["method_id"] == "plain_unet"
    )
    path = tmp_path / plain_record["report"]["path"]
    report = json.loads(path.read_text())
    report["per_tile"][0]["target_mask_sha256"] = "a" * 64
    _write_json(path, report)
    _rehash(tmp_path, plain_record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(
        assembler.ContractError,
        match="authenticated partition identity",
    ):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_precision_recall_rows_must_reconstruct_from_histograms(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path, with_curves=True)
    manifest = _load_manifest(manifest_path)
    record = manifest["neural_evaluations"][0]
    path = tmp_path / record["precision_recall_curve_csv"]["path"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[2]["precision"] = "0.123456"
    _write_csv(path, fields, rows)
    _rehash(tmp_path, record["precision_recall_curve_csv"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="PR precision mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    manifest = _build_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        assembler.assemble_publication_results(manifest, output)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_symbolic_link_manifest_is_rejected_before_resolution(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path)
    linked_manifest = tmp_path / "manifest_link.json"
    linked_manifest.symlink_to(manifest.name)
    output = tmp_path / "out"
    with pytest.raises(assembler.ContractError, match="symbolic-link component"):
        assembler.assemble_publication_results(linked_manifest, output)
    assert not output.exists()


def test_dangling_symbolic_link_output_is_rejected_without_creating_target(
    tmp_path: Path,
) -> None:
    manifest = _build_fixture(tmp_path)
    missing_target = tmp_path / "missing-output-target"
    output = tmp_path / "out"
    output.symlink_to(missing_target, target_is_directory=True)
    with pytest.raises(assembler.ContractError, match="symbolic-link component"):
        assembler.assemble_publication_results(manifest, output)
    assert output.is_symlink()
    assert not missing_target.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("replicates", 2),
        ("seed", assembler.PAIRED_BOOTSTRAP_SEED + 1),
        ("confidence", 0.5),
    ),
)
def test_publication_bootstrap_settings_are_immutable(
    tmp_path: Path, field: str, value: int | float
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    manifest["bootstrap"][field] = value
    _save_manifest(manifest_path, manifest)
    output = tmp_path / "out"
    with pytest.raises(
        assembler.ContractError,
        match="prespecified locked publication contract",
    ):
        assembler.assemble_publication_results(manifest_path, output)
    assert not output.exists()


def test_manifest_rejects_wrong_partition_wording_and_extra_keys(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    manifest["partition_label"] = "test data"
    manifest["unsupported"] = True
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="keys mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_manifest_must_bind_the_running_generator_source(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    manifest["generator"]["source_sha256"] = "f" * 64
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="generator identity mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_manifest_digest_is_bound_to_the_validated_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    original_content = manifest_path.read_bytes()
    original_validate = assembler._validate_evaluation_collection

    def mutate_after_manifest_validation(evaluations: Any) -> str:
        manifest = _load_manifest(manifest_path)
        manifest["partition_label"] = "mutated after validation"
        _save_manifest(manifest_path, manifest)
        return original_validate(evaluations)

    monkeypatch.setattr(
        assembler,
        "_validate_evaluation_collection",
        mutate_after_manifest_validation,
    )
    output = assembler.assemble_publication_results(
        manifest_path, tmp_path / "out"
    )
    report = json.loads((output / "publication_results.json").read_text())
    assert report["input_manifest"]["sha256"] == _sha(original_content)
    assert report["input_manifest"]["sha256"] != _file_sha(manifest_path)
    assert report["partition"] == assembler.PARTITION_LABEL


def test_artifact_parsing_and_digest_use_one_authenticated_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    original_artifact_path = assembler._artifact_path
    mutation: dict[str, Any] = {}

    def mutate_report_after_snapshot(*args: Any, **kwargs: Any) -> Any:
        artifact = original_artifact_path(*args, **kwargs)
        label = args[2]
        if label == "report" and not mutation:
            mutation.update(
                {
                    "path": artifact.path,
                    "authenticated_sha256": artifact.sha256,
                }
            )
            artifact.path.write_text('{"tampered":true}\n', encoding="utf-8")
        return artifact

    monkeypatch.setattr(assembler, "_artifact_path", mutate_report_after_snapshot)
    output = assembler.assemble_publication_results(
        manifest_path, tmp_path / "out"
    )
    report = json.loads((output / "publication_results.json").read_text())
    evidence = next(
        row
        for row in report["authenticated_evidence"]
        if row["method_id"] == "C2-FP" and row["seed"] == 42
    )
    assert evidence["report_sha256"] == mutation["authenticated_sha256"]
    assert evidence["report_sha256"] != _file_sha(mutation["path"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("evaluation_ordinal", "999", "evaluation ordinal mismatch"),
        ("input_sha256", "a" * 64, "input SHA-256 mismatch"),
        ("target_sha256", "b" * 64, "target SHA-256 mismatch"),
    ),
)
def test_classical_per_tile_metrics_csv_identity_columns_are_authenticated(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    specification = manifest["classical_evaluation"]["per_tile_metrics_csv"]
    _mutate_csv_cell(
        tmp_path,
        specification,
        row_index=0,
        column=column,
        value=value,
    )
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match=message):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_classical_confusion_csv_ordinal_is_authenticated(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    specification = manifest["classical_evaluation"]["per_tile_confusion_csv"]
    _mutate_csv_cell(
        tmp_path,
        specification,
        row_index=0,
        column="evaluation_ordinal",
        value="999",
    )
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="evaluation ordinal mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


@pytest.mark.parametrize(
    "artifact_key",
    (
        "aggregate_metrics_csv",
        "per_tile_metrics_csv",
        "per_tile_confusion_csv",
    ),
)
def test_classical_csvs_reject_unknown_comparator_identities(
    tmp_path: Path, artifact_key: str
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    specification = manifest["classical_evaluation"][artifact_key]
    _mutate_csv_cell(
        tmp_path,
        specification,
        row_index=0,
        column="comparator",
        value="unknown_comparator",
    )
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="comparator identity set"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_classical_pair_identity_is_recomputed_not_only_cross_compared(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["classical_evaluation"]
    forged_pair = "f" * 64
    record["expected_identity"]["evaluation_pair_identity_sha256"] = forged_pair
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    report["evaluation_pair_identity_sha256"] = forged_pair
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="lock-plus-neural-freeze"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_classical_checkpoint_map_must_match_all_six_neural_cells(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    manifest = _load_manifest(manifest_path)
    record = manifest["classical_evaluation"]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    report["neural_freeze"]["selected_retraining_checkpoint_sha256"][
        "primary_multiscale"
    ][0] = "f" * 64
    record["expected_identity"]["selected_retraining_checkpoint_sha256"][
        "primary_multiscale"
    ][0] = "f" * 64
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="six neural cells"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_all_six_neural_cells_require_distinct_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    manifest_path = _build_fixture(tmp_path, selected_method="C2-FP")
    manifest = _load_manifest(manifest_path)
    primary_records = [
        record
        for record in manifest["neural_evaluations"]
        if record["method_id"] == "C2-FP"
    ]
    duplicated_sha = primary_records[0]["expected_identity"]["checkpoint_sha256"]
    duplicated_record = primary_records[1]
    duplicated_record["expected_identity"]["checkpoint_sha256"] = duplicated_sha
    for neural_record in manifest["neural_evaluations"]:
        neural_record["expected_identity"][
            "selected_retraining_checkpoint_sha256"
        ]["primary_multiscale"][1] = duplicated_sha
        neural_report_path = tmp_path / neural_record["report"]["path"]
        neural_report = json.loads(neural_report_path.read_text())
        neural_report["neural_freeze"][
            "selected_retraining_checkpoint_sha256"
        ]["primary_multiscale"][1] = duplicated_sha
        if neural_record is duplicated_record:
            neural_report["checkpoint"]["sha256"] = duplicated_sha
            neural_report["locked_evaluation_identity"][
                "checkpoint_sha256"
            ] = duplicated_sha
        _write_json(neural_report_path, neural_report)
        _rehash(tmp_path, neural_record["report"])

    classical_record = manifest["classical_evaluation"]
    classical_report_path = tmp_path / classical_record["report"]["path"]
    classical_report = json.loads(classical_report_path.read_text())
    classical_report["neural_freeze"][
        "selected_retraining_checkpoint_sha256"
    ]["primary_multiscale"][1] = duplicated_sha
    classical_record["expected_identity"][
        "selected_retraining_checkpoint_sha256"
    ]["primary_multiscale"][1] = duplicated_sha
    _write_json(classical_report_path, classical_report)
    _rehash(tmp_path, classical_record["report"])
    _save_manifest(manifest_path, manifest)

    with pytest.raises(assembler.ContractError, match="six distinct checkpoints"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


@pytest.mark.parametrize(
    ("artifact_key", "column", "message"),
    (
        ("per_tile_metrics_csv", "image_id", "image_id mismatch"),
        ("per_tile_confusion_csv", "image_id", "image ID mismatch"),
    ),
)
def test_neural_csv_identity_columns_are_authenticated(
    tmp_path: Path, artifact_key: str, column: str, message: str
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    specification = manifest["neural_evaluations"][0][artifact_key]
    _mutate_csv_cell(
        tmp_path,
        specification,
        row_index=0,
        column=column,
        value="999",
    )
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match=message):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


@pytest.mark.parametrize("late_destination", ("empty_directory", "symlink"))
def test_late_created_output_is_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    late_destination: str,
) -> None:
    manifest_path = _build_fixture(tmp_path)
    output = tmp_path / "out"
    symlink_target = tmp_path / "late-symlink-target"
    original_write_tex_tables = assembler._write_tex_tables

    def create_destination_during_assembly(*args: Any, **kwargs: Any) -> None:
        if late_destination == "empty_directory":
            output.mkdir()
        else:
            output.symlink_to(symlink_target, target_is_directory=True)
        original_write_tex_tables(*args, **kwargs)

    monkeypatch.setattr(
        assembler, "_write_tex_tables", create_destination_during_assembly
    )
    with pytest.raises(FileExistsError, match="created during assembly"):
        assembler.assemble_publication_results(manifest_path, output)
    if late_destination == "empty_directory":
        assert output.is_dir()
        assert list(output.iterdir()) == []
    else:
        assert output.is_symlink()
        assert not symlink_target.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("height", "locked native tile shape"),
        ("confusion_pixels", "confusion pixels"),
        ("image_id", "image_id must equal"),
    ),
)
def test_neural_report_native_tile_geometry_is_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["neural_evaluations"][0]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    tile = report["per_tile"][0]
    if mutation == "height":
        tile["height"] = 2047
    elif mutation == "confusion_pixels":
        tile["confusion_matrix"][0][0] += 1
    else:
        tile["image_id"] = 999
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match=message):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")


def test_classical_report_native_tile_shape_is_fail_closed(tmp_path: Path) -> None:
    manifest_path = _build_fixture(tmp_path)
    manifest = _load_manifest(manifest_path)
    record = manifest["classical_evaluation"]
    report_path = tmp_path / record["report"]["path"]
    report = json.loads(report_path.read_text())
    report["runtime"]["native_tile_shape"] = [2047, 2048]
    _write_json(report_path, report)
    _rehash(tmp_path, record["report"])
    _save_manifest(manifest_path, manifest)
    with pytest.raises(assembler.ContractError, match="native tile shape mismatch"):
        assembler.assemble_publication_results(manifest_path, tmp_path / "out")

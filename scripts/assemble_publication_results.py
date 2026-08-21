#!/usr/bin/env python3
"""Assemble authenticated locked-evaluation evidence for publication.

This module is intentionally downstream of the frozen evaluators.  It never
loads images, masks, predictions, checkpoints, or training artefacts.  Every
input file must be named in a small authentication manifest with its SHA-256
and expected scientific identity.  Metrics are reconstructed from the
authenticated per-tile confusion counts, not copied from prose or plots.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import math
import os
import platform
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


INPUT_SCHEMA_VERSION = 2
OUTPUT_SCHEMA_VERSION = 2
GENERATOR_SCHEMA_VERSION = 1
GENERATOR_VERSION = "1.2.0"
GENERATOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARTITION_LABEL = "locked retrospective evaluation partition"
NEURAL_REPORT_SCHEMA_VERSION = "2.0"
NEURAL_EVALUATION_KIND = "locked_held_out_confirmatory_test"
CLASSICAL_REPORT_SCHEMA_VERSION = 1
CLASSICAL_EVALUATION_KIND = "exactly_once_locked_classical_held_out_test"
LOCKED_CURVE_BINS = 4096
EXPECTED_NATIVE_TILE_SHAPE = (2048, 2048)
PAIRED_BOOTSTRAP_REPLICATES = 5000
PAIRED_BOOTSTRAP_SEED = 20260820
PAIRED_BOOTSTRAP_CONFIDENCE = 0.95
NEURAL_SEEDS = (42, 123, 2025)
SELECTED_METHOD_IDS = ("R3", "H3", "C2-P", "C2-F", "C2-FP")
PLAIN_COMPARATOR_METHOD_ID = "plain_unet"
CONTRAST_DIRECTION = "selected_primary_minus_comparator"
CLASSICAL_METHOD_IDS = (
    "B0_small_components",
    "B1_marker_watershed",
    "B2_extra_trees",
)
CLASSICAL_DISPLAY_NAMES = {
    "B0_small_components": "B0 small components",
    "B1_marker_watershed": "B1 marker watershed",
    "B2_extra_trees": "B2 ExtraTrees",
}
CLASS_NAMES = {
    0: "disconnected_pore",
    1: "connected_pore",
    2: "mineral",
}
CLASS_LABELS = {
    0: "C0 disconnected pore",
    1: "C1 connected pore",
    2: "C2 mineral",
}
CLASS_PALETTE = {
    "C0": "#B33A3A",
    "C1": "#2E8B57",
    "C2": "#4C78A8",
}
METRIC_NAMES = ("iou", "dice", "precision", "recall")
PUBLICATION_METRICS = (
    "c0_iou",
    "c0_dice",
    "c0_precision",
    "c0_recall",
    "c1_iou",
    "c1_dice",
    "c1_precision",
    "c1_recall",
    "c0_c1_harmonic_iou",
)
OPTIONALLY_UNDEFINED_PUBLICATION_METRICS = (
    "c0_precision",
    "c1_precision",
)
TEX_FRAGMENT_NAME = "publication_results_tables.tex"
FIGURE_TEX_FRAGMENT_NAME = "publication_results_figures.tex"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
QUALITATIVE_SELECTION_RULE = (
    "lexicographically first file_name in the locked test manifest; "
    "selected before inference without reference to model performance"
)
QUALITATIVE_PDF_PATH = "publication/qualitative_triptych.pdf"
QUALITATIVE_PNG_PATH = "publication/qualitative_triptych.png"
QUALITATIVE_FIGURE_SEED = NEURAL_SEEDS[0]
NEURAL_EVALUATOR_PATH = "scripts/evaluate_confirmatory_checkpoint.py"

NEURAL_AGGREGATE_FIELDS = (
    "scope",
    "class_id",
    "class_name",
    "metric",
    "value",
    "ci_lower",
    "ci_upper",
    "bootstrap_unit",
    "bootstrap_replicates",
    "bootstrap_seed",
    "confidence",
)
CLASSICAL_AGGREGATE_FIELDS = (
    "comparator",
    "scope",
    "class_id",
    "class_name",
    "metric",
    "value",
    "ci_lower",
    "ci_upper",
)
NEURAL_CONFUSION_FIELDS = (
    "image_id",
    "file_name",
    "true_class_id",
    "true_class_name",
    "predicted_class_id",
    "predicted_class_name",
    "pixel_count",
)
CLASSICAL_CONFUSION_FIELDS = (
    "comparator",
    "evaluation_ordinal",
    "file_name",
    "true_class_id",
    "predicted_class_id",
    "pixel_count",
)
HISTOGRAM_FIELDS = (
    "class_id",
    "class_name",
    "bin_id",
    "score_lower",
    "score_upper",
    "positive_pixels",
    "negative_pixels",
)
PRECISION_RECALL_FIELDS = (
    "class_id",
    "class_name",
    "threshold_lower_edge",
    "recall",
    "precision",
    "cumulative_true_positive",
    "cumulative_false_positive",
    "positive_pixels",
    "negative_pixels",
)


class ContractError(ValueError):
    """An authenticated input does not satisfy the frozen assembly contract."""


@dataclass(frozen=True)
class CurveEvidence:
    """Fixed-bin probability evidence authenticated for one neural seed."""

    bins: int
    positive: np.ndarray
    negative: np.ndarray


@dataclass(frozen=True)
class AuthenticatedArtifact:
    """One immutable byte snapshot bound to its path and SHA-256."""

    path: Path
    sha256: str
    content: bytes


@dataclass(frozen=True)
class Evaluation:
    """One authenticated evaluation source at a fixed method and seed."""

    method_id: str
    display_name: str
    source_kind: str
    seed: int | None
    freeze_id: str
    freeze_scientific_identity_sha256: str
    report_sha256: str
    report_directory: Path
    report_outputs: tuple[str, ...]
    qualitative_example: Mapping[str, Any] | None
    artifact_sha256: Mapping[str, str]
    source_identity: Mapping[str, Any]
    tile_names: tuple[str, ...]
    input_sha256: tuple[str, ...]
    target_sha256: tuple[str, ...]
    tile_confusions: np.ndarray
    aggregate_confusion: np.ndarray
    metrics: Mapping[str, float | None]
    curves: CurveEvidence | None


@dataclass(frozen=True)
class QualitativePanel:
    """One authenticated evaluator-produced qualitative figure pair."""

    method_id: str
    seed: int
    file_name: str
    image_id: int
    pdf: AuthenticatedArtifact
    png: AuthenticatedArtifact
    output_pdf_name: str
    output_png_name: str
    publication_figure_contract: Mapping[str, Any]


def _reject_constant(value: str) -> None:
    raise ContractError(f"JSON contains a non-finite constant: {value}")


def _no_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"JSON contains a duplicate key: {key}")
        result[key] = value
    return result


def _load_json_bytes(content: bytes, label: str) -> Any:
    try:
        text = content.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"Cannot read strict JSON from {label}") from error


def _read_file_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ContractError(f"Cannot read {label}") from error


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        missing = sorted(expected_set - observed)
        extra = sorted(observed - expected_set)
        raise ContractError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _require_sha256_map(value: Any, label: str) -> dict[str, str]:
    mapping = _require_mapping(value, label)
    if not mapping:
        raise ContractError(f"{label} must not be empty")
    result: dict[str, str] = {}
    for key, digest in mapping.items():
        identifier = _require_nonempty_string(key, f"{label} key")
        result[identifier] = _require_sha256(digest, f"{label}.{identifier}")
    return result


def _require_checkpoint_sha256_map(value: Any, label: str) -> dict[str, list[str]]:
    mapping = _require_mapping(value, label)
    roles = ("primary_multiscale", "plain_unet_comparator")
    _require_exact_keys(mapping, roles, label)
    result: dict[str, list[str]] = {}
    for role in roles:
        digests = mapping[role]
        if not isinstance(digests, list) or len(digests) != len(NEURAL_SEEDS):
            raise ContractError(f"{label}.{role} must contain three SHA-256 values")
        result[role] = [
            _require_sha256(digest, f"{label}.{role}[{index}]")
            for index, digest in enumerate(digests)
        ]
    return result


def _require_training_source_attestation(value: Any, label: str) -> dict[str, Any]:
    attestation = _require_mapping(value, label)
    _require_exact_keys(
        attestation,
        ("verification_status", "file_count", "files"),
        label,
    )
    if attestation["verification_status"] != (
        "matched_screen_checkpoint_and_live_sources"
    ):
        raise ContractError(f"{label} is not a verified live-source attestation")
    files = _require_sha256_map(attestation["files"], f"{label}.files")
    file_count = _require_int(attestation["file_count"], f"{label}.file_count", minimum=1)
    if file_count != len(files):
        raise ContractError(f"{label}.file_count does not match its source map")
    return {
        "verification_status": attestation["verification_status"],
        "file_count": file_count,
        "files": files,
    }


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{label} must be at least {minimum}")
    return value


def _require_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ContractError(f"{label} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ContractError(f"{label} must be at most {maximum}")
    return number


def _assert_close(observed: Any, expected: float | None, label: str) -> None:
    if expected is None:
        if observed is not None:
            raise ContractError(f"{label} should be undefined")
        return
    number = _require_number(observed, label)
    if not math.isclose(number, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ContractError(
            f"{label} mismatch: observed={number!r}, recomputed={expected!r}"
        )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _harmonic_iou(c0_iou: float | None, c1_iou: float | None) -> float | None:
    if c0_iou is None or c1_iou is None:
        return None
    return float(2.0 * c0_iou * c1_iou / (c0_iou + c1_iou + 1e-8))


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float | None]:
    """Reconstruct the publication metrics from a canonical 3x3 matrix."""

    matrix = np.asarray(confusion)
    if matrix.shape != (3, 3):
        raise ContractError("A confusion matrix must have shape (3, 3)")
    if not np.issubdtype(matrix.dtype, np.integer):
        if not np.all(np.isfinite(matrix)) or not np.all(matrix == np.floor(matrix)):
            raise ContractError("Confusion counts must be integers")
        matrix = matrix.astype(np.int64)
    else:
        matrix = matrix.astype(np.int64, copy=False)
    if np.any(matrix < 0):
        raise ContractError("Confusion counts cannot be negative")

    result: dict[str, float | None] = {}
    class_values: dict[int, dict[str, float | None]] = {}
    for class_id in range(3):
        true_positive = int(matrix[class_id, class_id])
        false_positive = int(matrix[:, class_id].sum() - true_positive)
        false_negative = int(matrix[class_id, :].sum() - true_positive)
        values = {
            "iou": _safe_ratio(
                true_positive, true_positive + false_positive + false_negative
            ),
            "dice": _safe_ratio(
                2 * true_positive,
                2 * true_positive + false_positive + false_negative,
            ),
            "precision": _safe_ratio(
                true_positive, true_positive + false_positive
            ),
            "recall": _safe_ratio(
                true_positive, true_positive + false_negative
            ),
        }
        class_values[class_id] = values
        for metric_name, metric_value in values.items():
            result[f"c{class_id}_{metric_name}"] = metric_value
    result["c0_c1_harmonic_iou"] = _harmonic_iou(
        class_values[0]["iou"], class_values[1]["iou"]
    )
    return result


def _matrix(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise ContractError(f"{label} must be a 3x3 integer array")
    rows: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ContractError(f"{label}[{row_index}] must contain three values")
        rows.append(
            [
                _require_int(item, f"{label}[{row_index}][{column}]", minimum=0)
                for column, item in enumerate(row)
            ]
        )
    return np.asarray(rows, dtype=np.int64)


def _csv_float(value: str, label: str) -> float:
    if value == "":
        raise ContractError(f"{label} is unexpectedly blank")
    try:
        number = float(value)
    except ValueError as error:
        raise ContractError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise ContractError(f"{label} must be finite")
    return number


def _csv_optional_float(value: str, label: str) -> float | None:
    return None if value == "" else _csv_float(value, label)


def _csv_int(value: str, label: str, *, minimum: int = 0) -> int:
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
        raise ContractError(f"{label} is not an integer")
    number = int(value)
    if number < minimum:
        raise ContractError(f"{label} must be at least {minimum}")
    return number


def _read_csv(
    artifact: AuthenticatedArtifact,
    label: str,
    *,
    exact_fields: Sequence[str] | None = None,
    required_fields: Iterable[str] = (),
    allowed_fields: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        text = artifact.content.decode("utf-8")
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ContractError(f"{label} has no header")
            fields = tuple(reader.fieldnames)
            if len(fields) != len(set(fields)):
                raise ContractError(f"{label} has duplicate columns")
            if exact_fields is not None and fields != tuple(exact_fields):
                raise ContractError(
                    f"{label} columns mismatch: {fields!r} != {tuple(exact_fields)!r}"
                )
            if not set(required_fields).issubset(fields):
                raise ContractError(f"{label} is missing required columns")
            if allowed_fields is not None and not set(fields).issubset(
                set(allowed_fields)
            ):
                raise ContractError(f"{label} has unsupported columns")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ContractError(f"{label} row {row_number} has extra values")
                if any(value is None for value in row.values()):
                    raise ContractError(f"{label} row {row_number} is incomplete")
                rows.append(dict(row))
    except (UnicodeError, csv.Error) as error:
        raise ContractError(f"Cannot read {label}") from error
    if not rows:
        raise ContractError(f"{label} has no data rows")
    return fields, rows


def _artifact_path(
    manifest_dir: Path,
    specification: Any,
    label: str,
    *,
    expected_name: str,
) -> AuthenticatedArtifact:
    spec = _require_mapping(specification, label)
    _require_exact_keys(spec, ("path", "sha256"), label)
    relative_text = _require_nonempty_string(spec["path"], f"{label}.path")
    if "\\" in relative_text:
        raise ContractError(f"{label}.path must use POSIX separators")
    relative = Path(relative_text)
    if (
        relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or relative.as_posix() != relative_text
    ):
        raise ContractError(f"{label}.path must be a safe manifest-relative path")
    if relative.name != expected_name:
        raise ContractError(f"{label}.path must end in {expected_name}")

    root = manifest_dir.resolve()
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root)
    except ValueError as error:
        raise ContractError(f"{label}.path escapes the manifest directory") from error
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ContractError(f"{label}.path contains a symbolic link")
    if not candidate.is_file():
        raise ContractError(f"{label}.path is not a regular file")
    expected_sha = _require_sha256(spec["sha256"], f"{label}.sha256")
    content = _read_file_bytes(candidate, label)
    observed_sha = sha256_bytes(content)
    if observed_sha != expected_sha:
        raise ContractError(f"{label} SHA-256 mismatch")
    return AuthenticatedArtifact(
        path=candidate,
        sha256=observed_sha,
        content=content,
    )


def _artifact_group(
    manifest_dir: Path,
    record: Mapping[str, Any],
    *,
    classical: bool,
) -> tuple[
    dict[str, AuthenticatedArtifact | None],
    dict[str, str | None],
]:
    names = {
        "report": "evaluation_summary.json",
        "aggregate_metrics_csv": "aggregate_metrics.csv",
        "per_tile_metrics_csv": "per_tile_metrics.csv",
        "per_tile_confusion_csv": "per_tile_confusion.csv",
    }
    if not classical:
        names.update(
            {
                "probability_histograms_csv": "probability_histograms.csv",
                "precision_recall_curve_csv": "precision_recall_curve.csv",
            }
        )
    artifacts: dict[str, AuthenticatedArtifact | None] = {}
    hashes: dict[str, str | None] = {}
    for key, file_name in names.items():
        value = record[key]
        if value is None:
            if classical or key not in {
                "probability_histograms_csv",
                "precision_recall_curve_csv",
            }:
                raise ContractError(f"{key} cannot be null")
            artifacts[key] = None
            hashes[key] = None
            continue
        artifact = _artifact_path(
            manifest_dir, value, key, expected_name=file_name
        )
        artifacts[key] = artifact
        hashes[key] = artifact.sha256
    report_parent = artifacts["report"].path.parent  # type: ignore[union-attr]
    for key, artifact in artifacts.items():
        if artifact is not None and artifact.path.parent != report_parent:
            raise ContractError(f"{key} must be adjacent to its evaluation report")
    if not classical and (
        (artifacts["probability_histograms_csv"] is None)
        != (artifacts["precision_recall_curve_csv"] is None)
    ):
        raise ContractError("Histogram and precision-recall evidence must appear together")
    return artifacts, hashes


def _report_outputs_include(report: Mapping[str, Any], required: Iterable[str]) -> None:
    outputs = report.get("outputs")
    if (
        not isinstance(outputs, list)
        or any(not isinstance(item, str) for item in outputs)
        or len(outputs) != len(set(outputs))
    ):
        raise ContractError("Evaluation report outputs must be a unique string list")
    missing = sorted(set(required) - set(outputs))
    if missing:
        raise ContractError(f"Evaluation report omits required files: {missing}")


def _json_metric_block(
    block: Any,
    recomputed: Mapping[str, float | None],
    label: str,
    *,
    classical: bool,
) -> None:
    metrics = _require_mapping(block, label)
    per_class = metrics.get("per_class")
    if not isinstance(per_class, list) or len(per_class) != 3:
        raise ContractError(f"{label}.per_class must contain C0, C1, and C2")
    by_id: dict[int, Mapping[str, Any]] = {}
    for item in per_class:
        entry = _require_mapping(item, f"{label}.per_class item")
        class_id = _require_int(entry.get("class_id"), "class_id", minimum=0)
        if class_id not in CLASS_NAMES or class_id in by_id:
            raise ContractError(f"{label}.per_class has invalid or duplicate class IDs")
        if entry.get("class_name") != CLASS_NAMES[class_id]:
            raise ContractError(f"{label}.per_class class name mismatch")
        by_id[class_id] = entry
    for class_id in range(3):
        for metric_name in METRIC_NAMES:
            _assert_close(
                by_id[class_id].get(metric_name),
                recomputed[f"c{class_id}_{metric_name}"],
                f"{label}.C{class_id}.{metric_name}",
            )
    if classical:
        _assert_close(
            metrics.get("balanced_pore_iou"),
            recomputed["c0_c1_harmonic_iou"],
            f"{label}.balanced_pore_iou",
        )
    else:
        selection = _require_mapping(
            metrics.get("selection_metrics"), f"{label}.selection_metrics"
        )
        _assert_close(
            selection.get("c0_c1_harmonic_iou"),
            recomputed["c0_c1_harmonic_iou"],
            f"{label}.selection.c0_c1_harmonic_iou",
        )


def _aggregate_csv_metrics(
    artifact: AuthenticatedArtifact,
    recomputed: Mapping[str, float | None],
    *,
    classical: bool,
    comparator: str | None = None,
) -> None:
    fields = CLASSICAL_AGGREGATE_FIELDS if classical else NEURAL_AGGREGATE_FIELDS
    _, rows = _read_csv(artifact, "aggregate metrics CSV", exact_fields=fields)
    if classical and {row["comparator"] for row in rows} != set(
        CLASSICAL_METHOD_IDS
    ):
        raise ContractError(
            "Classical aggregate CSV comparator identity set mismatch"
        )
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if classical and row["comparator"] != comparator:
            continue
        if row["scope"] != "class":
            continue
        class_id = _csv_int(row["class_id"], "aggregate class_id")
        metric_name = row["metric"]
        if class_id not in CLASS_NAMES or metric_name not in METRIC_NAMES:
            continue
        key = (class_id, metric_name)
        if key in seen:
            raise ContractError(f"Aggregate CSV duplicates C{class_id} {metric_name}")
        seen.add(key)
        _assert_close(
            _csv_optional_float(row["value"], "aggregate value"),
            recomputed[f"c{class_id}_{metric_name}"],
            f"aggregate CSV C{class_id}.{metric_name}",
        )
    expected = {
        (class_id, metric_name)
        for class_id in range(3)
        for metric_name in METRIC_NAMES
    }
    if seen != expected:
        raise ContractError("Aggregate CSV lacks the complete C0/C1/C2 metric set")

    harmonic_rows = [
        row
        for row in rows
        if (not classical or row["comparator"] == comparator)
        and (
            (
                not classical
                and row["scope"] == "selection_and_pore_union"
                and row["metric"] == "c0_c1_harmonic_iou"
            )
            or (
                classical
                and row["scope"] == "pore_focus"
                and row["metric"] == "balanced_pore_iou"
            )
        )
    ]
    if len(harmonic_rows) != 1:
        raise ContractError("Aggregate CSV must contain exactly one C0/C1 harmonic row")
    _assert_close(
        _csv_optional_float(harmonic_rows[0]["value"], "harmonic value"),
        recomputed["c0_c1_harmonic_iou"],
        "aggregate CSV C0/C1 harmonic IoU",
    )


def _neural_per_tile_allowed_fields() -> tuple[set[str], set[str]]:
    required = {
        "image_id",
        "file_name",
        "height",
        "width",
        "pixels",
        "accuracy",
        "selection_c0_c1_harmonic_iou",
        "selection_pore_union_iou",
        "selection_pore_union_agreement",
        "pore_vs_mineral_accuracy",
        "pore_vs_mineral_agreement",
    }
    for scope in ("macro", "weighted", "micro"):
        for metric_name in ("iou", "dice", "precision", "recall", "f1"):
            required.add(f"{scope}_{metric_name}")
    for class_id in range(3):
        required.add(f"class_{class_id}_support_pixels")
        for metric_name in ("iou", "dice", "precision", "recall", "f1"):
            required.add(f"class_{class_id}_{metric_name}")
    for class_id in range(2):
        required.add(f"merged_class_{class_id}_support_pixels")
        for metric_name in ("iou", "dice", "precision", "recall", "f1"):
            required.add(f"merged_class_{class_id}_{metric_name}")
    optional = {
        f"conditional_gate_{name}"
        for name in (
            "mismatched_pixels",
            "mismatch_rate",
            "agreement_rate",
            "reference_pore_gate_mineral_pixels",
            "reference_c0_gate_mineral_pixels",
            "reference_c1_gate_mineral_pixels",
            "reference_mineral_gate_pore_pixels",
        )
    }
    return required, required | optional


def _classical_per_tile_fields() -> tuple[str, ...]:
    fields = [
        "comparator",
        "evaluation_ordinal",
        "file_name",
        "input_sha256",
        "target_sha256",
        "balanced_pore_iou",
        "pore_union_iou",
    ]
    for class_id in range(3):
        for metric_name in METRIC_NAMES:
            fields.append(f"c{class_id}_{metric_name}")
    return tuple(fields)


def _validate_per_tile_metrics_csv(
    artifact: AuthenticatedArtifact,
    evaluation_rows: Sequence[Mapping[str, Any]],
    metrics_by_name: Mapping[str, Mapping[str, float | None]],
    *,
    classical: bool,
    comparator: str | None = None,
) -> None:
    if classical:
        _, rows = _read_csv(
            artifact,
            "classical per-tile metrics CSV",
            exact_fields=_classical_per_tile_fields(),
        )
        if {row["comparator"] for row in rows} != set(CLASSICAL_METHOD_IDS):
            raise ContractError(
                "Classical per-tile metrics CSV comparator identity set mismatch"
            )
        rows = [row for row in rows if row["comparator"] == comparator]
    else:
        required, allowed = _neural_per_tile_allowed_fields()
        _, rows = _read_csv(
            artifact,
            "neural per-tile metrics CSV",
            required_fields=required,
            allowed_fields=allowed,
        )
    if len(rows) != len(evaluation_rows):
        raise ContractError("Per-tile metrics CSV row count mismatch")
    by_name: dict[str, Mapping[str, str]] = {}
    for row in rows:
        name = row["file_name"]
        if name in by_name:
            raise ContractError("Per-tile metrics CSV duplicates a file name")
        by_name[name] = row
    if set(by_name) != set(metrics_by_name):
        raise ContractError("Per-tile metrics CSV file-name set mismatch")
    for tile in evaluation_rows:
        name = str(tile["file_name"])
        row = by_name[name]
        recomputed = metrics_by_name[name]
        if classical:
            expected_ordinal = _require_int(
                tile.get("evaluation_ordinal"),
                f"{name} report evaluation_ordinal",
                minimum=1,
            )
            if _csv_int(
                row["evaluation_ordinal"], f"{name} evaluation_ordinal", minimum=1
            ) != expected_ordinal:
                raise ContractError(
                    f"Per-tile metrics CSV {name} evaluation ordinal mismatch"
                )
            expected_input_sha = _require_sha256(
                tile.get("input_image_sha256"), f"{name} report input SHA-256"
            )
            expected_target_sha = _require_sha256(
                tile.get("target_mask_sha256"), f"{name} report target SHA-256"
            )
            if row["input_sha256"] != expected_input_sha:
                raise ContractError(
                    f"Per-tile metrics CSV {name} input SHA-256 mismatch"
                )
            if row["target_sha256"] != expected_target_sha:
                raise ContractError(
                    f"Per-tile metrics CSV {name} target SHA-256 mismatch"
                )
        else:
            for column, report_key in (
                ("image_id", "image_id"),
                ("height", "height"),
                ("width", "width"),
            ):
                expected_value = _require_int(
                    tile.get(report_key), f"{name} report {report_key}", minimum=1
                )
                if _csv_int(row[column], f"{name} {column}", minimum=1) != (
                    expected_value
                ):
                    raise ContractError(
                        f"Per-tile metrics CSV {name} {column} mismatch"
                    )
            report_confusion = _matrix(
                tile.get("confusion_matrix"), f"{name} report confusion"
            )
            if _csv_int(row["pixels"], f"{name} pixels") != int(
                report_confusion.sum()
            ):
                raise ContractError(
                    f"Per-tile metrics CSV {name} pixel count mismatch"
                )
            for class_id in range(3):
                support_column = f"class_{class_id}_support_pixels"
                if _csv_int(row[support_column], f"{name} {support_column}") != int(
                    report_confusion[class_id, :].sum()
                ):
                    raise ContractError(
                        f"Per-tile metrics CSV {name} C{class_id} support mismatch"
                    )
        for class_id in range(3):
            for metric_name in METRIC_NAMES:
                column = (
                    f"c{class_id}_{metric_name}"
                    if classical
                    else f"class_{class_id}_{metric_name}"
                )
                expected = recomputed[f"c{class_id}_{metric_name}"]
                if expected is None:
                    if row[column] != "":
                        raise ContractError(
                            f"Per-tile CSV {name} {column} should be blank"
                        )
                else:
                    _assert_close(
                        _csv_float(row[column], f"{name} {column}"),
                        expected,
                        f"per-tile CSV {name} {column}",
                    )
        harmonic_column = (
            "balanced_pore_iou"
            if classical
            else "selection_c0_c1_harmonic_iou"
        )
        expected_harmonic = recomputed["c0_c1_harmonic_iou"]
        if expected_harmonic is None:
            if row[harmonic_column] != "":
                raise ContractError(f"Per-tile CSV {name} harmonic should be blank")
        else:
            _assert_close(
                _csv_float(row[harmonic_column], f"{name} harmonic"),
                expected_harmonic,
                f"per-tile CSV {name} harmonic",
            )


def _confusions_from_csv(
    artifact: AuthenticatedArtifact,
    tile_rows: Sequence[Mapping[str, Any]],
    *,
    classical: bool,
    comparator: str | None = None,
) -> dict[str, np.ndarray]:
    exact_fields = CLASSICAL_CONFUSION_FIELDS if classical else NEURAL_CONFUSION_FIELDS
    _, rows = _read_csv(
        artifact, "per-tile confusion CSV", exact_fields=exact_fields
    )
    if classical:
        if {row["comparator"] for row in rows} != set(CLASSICAL_METHOD_IDS):
            raise ContractError(
                "Classical per-tile confusion CSV comparator identity set mismatch"
            )
        rows = [row for row in rows if row["comparator"] == comparator]
    names = [str(tile["file_name"]) for tile in tile_rows]
    tiles_by_name = {str(tile["file_name"]): tile for tile in tile_rows}
    matrices = {name: np.zeros((3, 3), dtype=np.int64) for name in names}
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        name = row["file_name"]
        if name not in matrices:
            raise ContractError("Per-tile confusion CSV contains an unknown file")
        report_tile = tiles_by_name[name]
        if classical:
            expected_ordinal = _require_int(
                report_tile.get("evaluation_ordinal"),
                f"{name} report evaluation_ordinal",
                minimum=1,
            )
            if _csv_int(
                row["evaluation_ordinal"], f"{name} evaluation_ordinal", minimum=1
            ) != expected_ordinal:
                raise ContractError(
                    f"Per-tile confusion CSV {name} evaluation ordinal mismatch"
                )
        else:
            expected_image_id = _require_int(
                report_tile.get("image_id"), f"{name} report image_id", minimum=1
            )
            if _csv_int(row["image_id"], f"{name} image_id", minimum=1) != (
                expected_image_id
            ):
                raise ContractError(
                    f"Per-tile confusion CSV {name} image ID mismatch"
                )
        true_id = _csv_int(row["true_class_id"], "true_class_id")
        predicted_id = _csv_int(row["predicted_class_id"], "predicted_class_id")
        if true_id not in CLASS_NAMES or predicted_id not in CLASS_NAMES:
            raise ContractError("Per-tile confusion CSV class ID is outside 0..2")
        key = (name, true_id, predicted_id)
        if key in seen:
            raise ContractError("Per-tile confusion CSV duplicates a matrix cell")
        seen.add(key)
        matrices[name][true_id, predicted_id] = _csv_int(
            row["pixel_count"], "pixel_count"
        )
        if not classical:
            if row["true_class_name"] != CLASS_NAMES[true_id]:
                raise ContractError("Per-tile confusion true-class name mismatch")
            if row["predicted_class_name"] != CLASS_NAMES[predicted_id]:
                raise ContractError("Per-tile confusion predicted-class name mismatch")
    expected = {
        (name, true_id, predicted_id)
        for name in names
        for true_id in range(3)
        for predicted_id in range(3)
    }
    if seen != expected:
        raise ContractError("Per-tile confusion CSV is incomplete")
    return matrices


def _curves_from_histograms(
    positive_histogram: np.ndarray, negative_histogram: np.ndarray
) -> dict[str, Any]:
    positive = np.asarray(positive_histogram, dtype=np.int64)
    negative = np.asarray(negative_histogram, dtype=np.int64)
    if positive.ndim != 1 or positive.shape != negative.shape:
        raise ContractError("Curve histograms must have matching one-dimensional shape")
    bins = positive.size
    if bins < 2 or np.any(positive < 0) or np.any(negative < 0):
        raise ContractError("Curve histogram counts are invalid")
    positives = int(positive.sum())
    negatives = int(negative.sum())
    cumulative_tp = np.concatenate(([0], np.cumsum(positive[::-1], dtype=np.int64)))
    cumulative_fp = np.concatenate(([0], np.cumsum(negative[::-1], dtype=np.int64)))
    thresholds = np.concatenate(
        ([1.0 + 1.0 / bins], np.arange(bins - 1, -1, -1) / bins)
    )
    predicted_positive = cumulative_tp + cumulative_fp
    precision = np.divide(
        cumulative_tp,
        predicted_positive,
        out=np.ones(cumulative_tp.shape, dtype=np.float64),
        where=predicted_positive != 0,
    )
    recall = (
        cumulative_tp.astype(np.float64) / positives
        if positives
        else np.full(cumulative_tp.shape, np.nan)
    )
    average_precision = (
        float(np.sum(np.diff(recall) * precision[1:])) if positives else None
    )
    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "cumulative_true_positive": cumulative_tp,
        "cumulative_false_positive": cumulative_fp,
        "positive_pixels": positives,
        "negative_pixels": negatives,
        "average_precision": average_precision,
    }


def _load_curve_evidence(
    histogram_artifact: AuthenticatedArtifact,
    pr_artifact: AuthenticatedArtifact,
    report: Mapping[str, Any],
    aggregate_confusion: np.ndarray,
) -> CurveEvidence:
    curves = _require_mapping(report.get("curves"), "neural report curves")
    if curves.get("method") != "one-vs-rest fixed-width probability histograms":
        raise ContractError("Neural curve method mismatch")
    bins = _require_int(curves.get("bins"), "neural curve bins", minimum=2)
    if bins != LOCKED_CURVE_BINS:
        raise ContractError(
            f"Neural curve bins must equal the locked value {LOCKED_CURVE_BINS}"
        )
    _assert_close(
        curves.get("score_bin_width"),
        1.0 / bins,
        "neural curve score-bin width",
    )
    if curves.get("raw_probabilities_persisted") is not False:
        raise ContractError("Neural report must not persist raw probabilities")
    _, histogram_rows = _read_csv(
        histogram_artifact,
        "probability histogram CSV",
        exact_fields=HISTOGRAM_FIELDS,
    )
    positive = np.zeros((3, bins), dtype=np.int64)
    negative = np.zeros((3, bins), dtype=np.int64)
    seen: set[tuple[int, int]] = set()
    for row in histogram_rows:
        class_id = _csv_int(row["class_id"], "histogram class_id")
        bin_id = _csv_int(row["bin_id"], "histogram bin_id")
        if class_id not in CLASS_NAMES or bin_id >= bins:
            raise ContractError("Histogram class or bin ID is outside its contract")
        key = (class_id, bin_id)
        if key in seen:
            raise ContractError("Histogram CSV duplicates a class/bin")
        seen.add(key)
        if row["class_name"] != CLASS_NAMES[class_id]:
            raise ContractError("Histogram class name mismatch")
        _assert_close(
            _csv_float(row["score_lower"], "score_lower"),
            bin_id / bins,
            "histogram score_lower",
        )
        _assert_close(
            _csv_float(row["score_upper"], "score_upper"),
            (bin_id + 1) / bins,
            "histogram score_upper",
        )
        positive[class_id, bin_id] = _csv_int(
            row["positive_pixels"], "positive_pixels"
        )
        negative[class_id, bin_id] = _csv_int(
            row["negative_pixels"], "negative_pixels"
        )
    expected = {
        (class_id, bin_id)
        for class_id in range(3)
        for bin_id in range(bins)
    }
    if seen != expected:
        raise ContractError("Histogram CSV is incomplete")

    total_pixels = int(aggregate_confusion.sum())
    for class_id in range(3):
        support = int(aggregate_confusion[class_id, :].sum())
        if int(positive[class_id].sum()) != support:
            raise ContractError("Histogram positive count disagrees with reference support")
        if int(negative[class_id].sum()) != total_pixels - support:
            raise ContractError("Histogram negative count disagrees with reference support")

    _, pr_rows = _read_csv(
        pr_artifact,
        "precision-recall CSV",
        exact_fields=PRECISION_RECALL_FIELDS,
    )
    by_class: dict[int, list[Mapping[str, str]]] = {0: [], 1: [], 2: []}
    for row in pr_rows:
        class_id = _csv_int(row["class_id"], "PR class_id")
        if class_id not in by_class:
            raise ContractError("PR class ID is outside 0..2")
        if row["class_name"] != CLASS_NAMES[class_id]:
            raise ContractError("PR class name mismatch")
        by_class[class_id].append(row)
    summary = _require_mapping(curves.get("summary"), "neural curve summary")
    for class_id in range(3):
        derived = _curves_from_histograms(positive[class_id], negative[class_id])
        rows = by_class[class_id]
        if len(rows) != bins + 1:
            raise ContractError("PR CSV point count disagrees with histogram bins")
        for index, row in enumerate(rows):
            _assert_close(
                _csv_float(row["threshold_lower_edge"], "PR threshold"),
                float(derived["thresholds"][index]),
                "PR threshold",
            )
            _assert_close(
                _csv_float(row["recall"], "PR recall"),
                float(derived["recall"][index]),
                "PR recall",
            )
            _assert_close(
                _csv_float(row["precision"], "PR precision"),
                float(derived["precision"][index]),
                "PR precision",
            )
            if _csv_int(row["cumulative_true_positive"], "PR cumulative TP") != int(
                derived["cumulative_true_positive"][index]
            ):
                raise ContractError("PR cumulative true-positive count mismatch")
            if _csv_int(row["cumulative_false_positive"], "PR cumulative FP") != int(
                derived["cumulative_false_positive"][index]
            ):
                raise ContractError("PR cumulative false-positive count mismatch")
            if _csv_int(row["positive_pixels"], "PR positive pixels") != int(
                derived["positive_pixels"]
            ):
                raise ContractError("PR positive-pixel total mismatch")
            if _csv_int(row["negative_pixels"], "PR negative pixels") != int(
                derived["negative_pixels"]
            ):
                raise ContractError("PR negative-pixel total mismatch")
        class_summary = _require_mapping(
            summary.get(str(class_id)), f"curve summary C{class_id}"
        )
        if class_summary.get("class_name") != CLASS_NAMES[class_id]:
            raise ContractError("Curve summary class-name mismatch")
        if class_summary.get("positive_pixels") != derived["positive_pixels"]:
            raise ContractError("Curve summary positive count mismatch")
        if class_summary.get("negative_pixels") != derived["negative_pixels"]:
            raise ContractError("Curve summary negative count mismatch")
        _assert_close(
            class_summary.get("average_precision_histogram_approximation"),
            derived["average_precision"],
            f"curve summary C{class_id} AP",
        )
    return CurveEvidence(bins=bins, positive=positive, negative=negative)


def _verify_common_tile_report(
    report: Mapping[str, Any],
    *,
    classical: bool,
    comparator: str | None,
    artifacts: Mapping[str, AuthenticatedArtifact | None],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    dict[str, float | None],
    CurveEvidence | None,
]:
    if classical:
        comparators = _require_mapping(report.get("comparators"), "comparators")
        method = _require_mapping(comparators.get(comparator), f"comparator {comparator}")
        aggregate_json = _matrix(
            method.get("aggregate_confusion_matrix"),
            f"{comparator}.aggregate_confusion_matrix",
        )
        tile_rows_value = method.get("per_tile")
    else:
        aggregate_json = _matrix(
            report.get("aggregate_confusion_matrix"), "aggregate_confusion_matrix"
        )
        tile_rows_value = report.get("per_tile")
    if not isinstance(tile_rows_value, list) or not tile_rows_value:
        raise ContractError("Evaluation report must contain non-empty per-tile records")
    tile_rows = [
        _require_mapping(item, "per-tile record") for item in tile_rows_value
    ]
    names: list[str] = []
    input_hashes: list[str] = []
    target_hashes: list[str] = []
    json_matrices: dict[str, np.ndarray] = {}
    json_metrics: dict[str, Mapping[str, float | None]] = {}
    for expected_ordinal, tile in enumerate(tile_rows, start=1):
        ordinal = _require_int(
            tile.get("evaluation_ordinal"), "evaluation_ordinal", minimum=1
        )
        if ordinal != expected_ordinal:
            raise ContractError("Per-tile evaluation ordinals must be contiguous")
        name = _require_nonempty_string(tile.get("file_name"), "tile file_name")
        if name in json_matrices:
            raise ContractError("Evaluation report duplicates a tile file name")
        input_hash = _require_sha256(
            tile.get("input_image_sha256"), f"{name} input SHA-256"
        )
        target_hash = _require_sha256(
            tile.get("target_mask_sha256"), f"{name} target SHA-256"
        )
        confusion = _matrix(tile.get("confusion_matrix"), f"{name} confusion")
        height = _require_int(tile.get("height"), f"{name} height", minimum=1)
        width = _require_int(tile.get("width"), f"{name} width", minimum=1)
        if (height, width) != EXPECTED_NATIVE_TILE_SHAPE:
            raise ContractError(
                f"{name} dimensions do not match the locked native tile shape"
            )
        if int(confusion.sum()) != height * width:
            raise ContractError(
                f"{name} confusion pixels do not match its reported dimensions"
            )
        if not classical and _require_int(
            tile.get("image_id"), f"{name} image_id", minimum=1
        ) != ordinal:
            raise ContractError(
                f"{name} neural image_id must equal its evaluation ordinal"
            )
        recomputed = metrics_from_confusion(confusion)
        _json_metric_block(
            tile.get("metrics"),
            recomputed,
            f"{name}.metrics",
            classical=classical,
        )
        names.append(name)
        input_hashes.append(input_hash)
        target_hashes.append(target_hash)
        json_matrices[name] = confusion
        json_metrics[name] = recomputed

    csv_matrices = _confusions_from_csv(
        artifacts["per_tile_confusion_csv"],  # type: ignore[arg-type]
        tile_rows,
        classical=classical,
        comparator=comparator,
    )
    for name in names:
        if not np.array_equal(csv_matrices[name], json_matrices[name]):
            raise ContractError(f"{name} confusion differs between JSON and CSV")
    matrices = np.stack([json_matrices[name] for name in names])
    aggregate = matrices.sum(axis=0)
    if not np.array_equal(aggregate, aggregate_json):
        raise ContractError("Aggregate confusion is not the sum of per-tile matrices")
    recomputed_aggregate = metrics_from_confusion(aggregate)
    aggregate_block = (
        _require_mapping(report["comparators"], "comparators")[comparator][
            "aggregate_metrics"
        ]
        if classical
        else report.get("aggregate_metrics")
    )
    _json_metric_block(
        aggregate_block,
        recomputed_aggregate,
        "aggregate metrics",
        classical=classical,
    )
    _aggregate_csv_metrics(
        artifacts["aggregate_metrics_csv"],  # type: ignore[arg-type]
        recomputed_aggregate,
        classical=classical,
        comparator=comparator,
    )
    _validate_per_tile_metrics_csv(
        artifacts["per_tile_metrics_csv"],  # type: ignore[arg-type]
        tile_rows,
        json_metrics,
        classical=classical,
        comparator=comparator,
    )
    publication_metrics: dict[str, float | None] = {}
    for name in PUBLICATION_METRICS:
        value = recomputed_aggregate[name]
        if value is None:
            if name not in OPTIONALLY_UNDEFINED_PUBLICATION_METRICS:
                raise ContractError(
                    f"Aggregate publication metric {name} is undefined"
                )
            publication_metrics[name] = None
            continue
        if not math.isfinite(float(value)):
            raise ContractError(f"Aggregate publication metric {name} is undefined")
        publication_metrics[name] = float(value)

    curve_evidence = None
    if not classical:
        histogram_artifact = artifacts["probability_histograms_csv"]
        pr_artifact = artifacts["precision_recall_curve_csv"]
        if histogram_artifact is not None and pr_artifact is not None:
            curve_evidence = _load_curve_evidence(
                histogram_artifact, pr_artifact, report, aggregate
            )
        elif report.get("curves") is not None:
            raise ContractError(
                "Neural report declares curve evidence but its authenticated CSVs are absent"
            )
    return (
        tuple(names),
        tuple(input_hashes),
        tuple(target_hashes),
        matrices,
        aggregate,
        publication_metrics,
        curve_evidence,
    )


def _load_neural_evaluation(
    manifest_dir: Path, record_value: Any
) -> Evaluation:
    record = _require_mapping(record_value, "neural evaluation record")
    _require_exact_keys(
        record,
        (
            "method_id",
            "display_name",
            "report",
            "aggregate_metrics_csv",
            "per_tile_metrics_csv",
            "per_tile_confusion_csv",
            "probability_histograms_csv",
            "precision_recall_curve_csv",
            "expected_identity",
        ),
        "neural evaluation record",
    )
    method_id = _require_nonempty_string(record["method_id"], "method_id")
    if not METHOD_ID.fullmatch(method_id):
        raise ContractError("method_id has unsupported characters")
    display_name = _require_nonempty_string(record["display_name"], "display_name")
    expected = _require_mapping(record["expected_identity"], "expected_identity")
    _require_exact_keys(
        expected,
        (
            "neural_freeze_manifest_id",
            "neural_freeze_scientific_identity_sha256",
            "architecture_role",
            "cell_index",
            "training_seed",
            "selected_method",
            "checkpoint_sha256",
            "checkpoint_state_dict_semantic_sha256",
            "evaluator_path",
            "evaluator_sha256",
            "training_execution_source_attestation",
            "selected_method_lock_path",
            "selected_method_lock_sha256",
            "neural_freeze_manifest_path",
            "neural_freeze_manifest_file_sha256",
            "selected_retraining_checkpoint_sha256",
        ),
        "neural expected_identity",
    )
    freeze_id = _require_nonempty_string(
        expected["neural_freeze_manifest_id"], "neural freeze manifest ID"
    )
    freeze_sha = _require_sha256(
        expected["neural_freeze_scientific_identity_sha256"],
        "neural freeze scientific identity",
    )
    architecture_role = expected["architecture_role"]
    if architecture_role not in {"primary_multiscale", "plain_unet_comparator"}:
        raise ContractError("Unknown neural architecture role")
    cell_index = _require_int(expected["cell_index"], "cell_index", minimum=0)
    if cell_index not in range(len(NEURAL_SEEDS)):
        raise ContractError("cell_index must be 0, 1, or 2")
    seed = _require_int(expected["training_seed"], "training_seed", minimum=0)
    if seed != NEURAL_SEEDS[cell_index]:
        raise ContractError("training_seed does not match the frozen cell index")
    selected_method = _require_nonempty_string(
        expected["selected_method"], "selected_method"
    )
    checkpoint_sha = _require_sha256(
        expected["checkpoint_sha256"], "checkpoint SHA-256"
    )
    checkpoint_semantic_sha = _require_sha256(
        expected["checkpoint_state_dict_semantic_sha256"],
        "checkpoint state-dict semantic SHA-256",
    )
    evaluator_path = _require_nonempty_string(
        expected["evaluator_path"], "neural evaluator path"
    )
    if evaluator_path != NEURAL_EVALUATOR_PATH:
        raise ContractError("Neural expected identity names a non-canonical evaluator")
    evaluator_sha = _require_sha256(
        expected["evaluator_sha256"], "neural evaluator SHA-256"
    )
    source_attestation = _require_training_source_attestation(
        expected["training_execution_source_attestation"],
        "neural training execution source attestation",
    )
    selected_lock_path = _require_nonempty_string(
        expected["selected_method_lock_path"], "selected-method lock path"
    )
    selected_lock_sha = _require_sha256(
        expected["selected_method_lock_sha256"], "selected-method lock SHA-256"
    )
    freeze_manifest_path = _require_nonempty_string(
        expected["neural_freeze_manifest_path"], "neural-freeze manifest path"
    )
    freeze_file_sha = _require_sha256(
        expected["neural_freeze_manifest_file_sha256"],
        "neural-freeze manifest file SHA-256",
    )
    checkpoint_map = _require_checkpoint_sha256_map(
        expected["selected_retraining_checkpoint_sha256"],
        "selected retraining checkpoint map",
    )
    if checkpoint_map[architecture_role][cell_index] != checkpoint_sha:
        raise ContractError(
            "Neural checkpoint identity does not match the verified retraining map"
        )
    if architecture_role == "primary_multiscale" and method_id != selected_method:
        raise ContractError("Primary method_id must equal the frozen selected method")
    if (
        architecture_role == "plain_unet_comparator"
        and method_id != PLAIN_COMPARATOR_METHOD_ID
    ):
        raise ContractError("Plain-U-Net role must use method_id 'plain_unet'")

    artifacts, hashes = _artifact_group(manifest_dir, record, classical=False)
    report_artifact = artifacts["report"]
    if report_artifact is None:
        raise ContractError("Neural report artifact cannot be null")
    report = _require_mapping(
        _load_json_bytes(report_artifact.content, str(report_artifact.path)),
        "neural evaluation report",
    )
    if report.get("schema_version") != NEURAL_REPORT_SCHEMA_VERSION:
        raise ContractError("Neural report schema mismatch")
    if report.get("evaluation_kind") != NEURAL_EVALUATION_KIND:
        raise ContractError("Neural report evaluation-kind mismatch")
    if report.get("status") != "complete":
        raise ContractError("Neural report is not complete")
    _report_outputs_include(
        report,
        (
            "evaluation_summary.json",
            "aggregate_metrics.csv",
            "per_tile_metrics.csv",
            "per_tile_confusion.csv",
            "probability_histograms.csv",
            "precision_recall_curve.csv",
        ),
    )
    identity = _require_mapping(
        report.get("locked_evaluation_identity"), "locked_evaluation_identity"
    )
    expected_pairs = {
        "neural_freeze_manifest_id": freeze_id,
        "neural_freeze_scientific_identity_sha256": freeze_sha,
        "architecture_role": architecture_role,
        "cell_index": cell_index,
        "training_seed": seed,
        "selected_method": selected_method,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_dict_semantic_sha256": checkpoint_semantic_sha,
    }
    for key, expected_value in expected_pairs.items():
        if identity.get(key) != expected_value:
            raise ContractError(f"Neural locked identity mismatch at {key}")
    if identity.get("verification_status") != (
        "matched_content_addressed_neural_freeze_cell"
    ):
        raise ContractError("Neural locked identity is not verified")
    if identity.get("reservation_key") != (
        f"{freeze_id}:{architecture_role}:cell_{cell_index:02d}"
    ):
        raise ContractError("Neural reservation identity mismatch")
    neural_freeze = _require_mapping(report.get("neural_freeze"), "neural_freeze")
    if (
        neural_freeze.get("manifest_id") != freeze_id
        or neural_freeze.get("manifest_path") != freeze_manifest_path
        or neural_freeze.get("manifest_file_sha256") != freeze_file_sha
        or neural_freeze.get("scientific_identity_sha256") != freeze_sha
        or neural_freeze.get("selected_retraining_checkpoint_sha256")
        != checkpoint_map
    ):
        raise ContractError("Neural report freeze identity mismatch")
    code = _require_mapping(report.get("code"), "neural report code")
    if (
        code.get("evaluator_path") != evaluator_path
        or code.get("evaluator_sha256") != evaluator_sha
        or code.get("training_execution_source_attestation")
        != source_attestation
    ):
        raise ContractError("Neural report source provenance mismatch")
    selected_lock = _require_mapping(
        report.get("selected_method_lock"), "selected_method_lock"
    )
    if (
        selected_lock.get("path") != selected_lock_path
        or selected_lock.get("sha256") != selected_lock_sha
        or selected_lock.get("selected_method") != selected_method
        or selected_lock.get("selected_architecture_role") != architecture_role
    ):
        raise ContractError("Neural selected-method lock mismatch")
    checkpoint = _require_mapping(report.get("checkpoint"), "checkpoint")
    if (
        checkpoint.get("sha256") != checkpoint_sha
        or checkpoint.get("state_dict_semantic_sha256")
        != checkpoint_semantic_sha
    ):
        raise ContractError("Neural checkpoint raw/semantic digest mismatch")
    inference = _require_mapping(report.get("inference"), "inference")
    if (
        inference.get("test_passes") != 1
        or inference.get("post_processing") != "none"
        or inference.get("output_overwrite_allowed") is not False
    ):
        raise ContractError("Neural inference is outside the locked one-pass contract")

    (
        tile_names,
        input_hashes,
        target_hashes,
        matrices,
        aggregate,
        metrics,
        curve_evidence,
    ) = _verify_common_tile_report(
        report,
        classical=False,
        comparator=None,
        artifacts=artifacts,
    )
    data = _require_mapping(report.get("data"), "neural data")
    if (
        data.get("evaluation_split") != "test"
        or data.get("test_image_files") != list(tile_names)
        or data.get("test_tile_count") != len(tile_names)
        or data.get("native_tile_shape") != list(EXPECTED_NATIVE_TILE_SHAPE)
    ):
        raise ContractError("Neural report partition identity mismatch")
    qualitative_example_value = report.get("qualitative_example")
    qualitative_example = (
        None
        if qualitative_example_value is None
        else _require_mapping(
            qualitative_example_value, "neural qualitative_example"
        )
    )
    return Evaluation(
        method_id=method_id,
        display_name=display_name,
        source_kind="neural",
        seed=seed,
        freeze_id=freeze_id,
        freeze_scientific_identity_sha256=freeze_sha,
        report_sha256=str(hashes["report"]),
        report_directory=report_artifact.path.parent,
        report_outputs=tuple(str(value) for value in report["outputs"]),
        qualitative_example=qualitative_example,
        artifact_sha256={
            key: str(value) for key, value in hashes.items() if value is not None
        },
        source_identity={
            "selected_method": selected_method,
            "architecture_role": architecture_role,
            "cell_index": cell_index,
            "training_seed": seed,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_state_dict_semantic_sha256": checkpoint_semantic_sha,
            "evaluator_path": evaluator_path,
            "evaluator_sha256": evaluator_sha,
            "training_execution_source_attestation": source_attestation,
            "selected_method_lock_path": selected_lock_path,
            "selected_method_lock_sha256": selected_lock_sha,
            "neural_freeze_manifest_path": freeze_manifest_path,
            "neural_freeze_manifest_file_sha256": freeze_file_sha,
            "selected_retraining_checkpoint_sha256": checkpoint_map,
        },
        tile_names=tile_names,
        input_sha256=input_hashes,
        target_sha256=target_hashes,
        tile_confusions=matrices,
        aggregate_confusion=aggregate,
        metrics=metrics,
        curves=curve_evidence,
    )


def _load_classical_evaluations(
    manifest_dir: Path, record_value: Any
) -> list[Evaluation]:
    record = _require_mapping(record_value, "classical evaluation record")
    _require_exact_keys(
        record,
        (
            "report",
            "aggregate_metrics_csv",
            "per_tile_metrics_csv",
            "per_tile_confusion_csv",
            "expected_identity",
        ),
        "classical evaluation record",
    )
    expected = _require_mapping(record["expected_identity"], "expected_identity")
    _require_exact_keys(
        expected,
        (
            "fit_id",
            "evaluation_pair_identity_sha256",
            "neural_freeze_manifest_id",
            "neural_freeze_scientific_identity_sha256",
            "classical_lock_identity_sha256",
            "classical_lock_path",
            "classical_lock_raw_file_sha256",
            "classical_lock_source_code_sha256",
            "b2_model_path",
            "b2_model_sha256",
            "b2_model_semantic_sha256",
            "selected_method_lock_path",
            "selected_method_lock_sha256",
            "selected_retraining_checkpoint_sha256",
            "neural_freeze_manifest_path",
            "neural_freeze_manifest_file_sha256",
        ),
        "classical expected_identity",
    )
    fit_id = _require_nonempty_string(expected["fit_id"], "fit_id")
    pair_sha = _require_sha256(
        expected["evaluation_pair_identity_sha256"],
        "evaluation pair identity",
    )
    freeze_id = _require_nonempty_string(
        expected["neural_freeze_manifest_id"], "neural freeze manifest ID"
    )
    freeze_sha = _require_sha256(
        expected["neural_freeze_scientific_identity_sha256"],
        "neural freeze scientific identity",
    )
    lock_sha = _require_sha256(
        expected["classical_lock_identity_sha256"],
        "classical lock identity",
    )
    lock_path = _require_nonempty_string(
        expected["classical_lock_path"], "classical lock path"
    )
    lock_raw_sha = _require_sha256(
        expected["classical_lock_raw_file_sha256"],
        "classical raw lock file SHA-256",
    )
    lock_source_map = _require_sha256_map(
        expected["classical_lock_source_code_sha256"],
        "classical lock source-code map",
    )
    b2_model_path = _require_nonempty_string(
        expected["b2_model_path"], "B2 model path"
    )
    b2_model_sha = _require_sha256(
        expected["b2_model_sha256"], "B2 model SHA-256"
    )
    b2_model_semantic_sha = _require_sha256(
        expected["b2_model_semantic_sha256"], "B2 semantic model SHA-256"
    )
    selected_lock_path = _require_nonempty_string(
        expected["selected_method_lock_path"], "selected-method lock path"
    )
    selected_lock_sha = _require_sha256(
        expected["selected_method_lock_sha256"], "selected-method lock SHA-256"
    )
    checkpoint_map = _require_checkpoint_sha256_map(
        expected["selected_retraining_checkpoint_sha256"],
        "selected retraining checkpoint map",
    )
    freeze_manifest_path = _require_nonempty_string(
        expected["neural_freeze_manifest_path"], "neural-freeze manifest path"
    )
    freeze_file_sha = _require_sha256(
        expected["neural_freeze_manifest_file_sha256"],
        "neural-freeze manifest file SHA-256",
    )
    derived_pair_sha = hashlib.sha256(
        f"{lock_sha}\0{freeze_sha}".encode("ascii")
    ).hexdigest()
    if pair_sha != derived_pair_sha:
        raise ContractError(
            "Classical evaluation-pair identity does not match the frozen "
            "lock-plus-neural-freeze derivation"
        )
    artifacts, hashes = _artifact_group(manifest_dir, record, classical=True)
    report_artifact = artifacts["report"]
    if report_artifact is None:
        raise ContractError("Classical report artifact cannot be null")
    report = _require_mapping(
        _load_json_bytes(report_artifact.content, str(report_artifact.path)),
        "classical evaluation report",
    )
    if report.get("schema_version") != CLASSICAL_REPORT_SCHEMA_VERSION:
        raise ContractError("Classical report schema mismatch")
    if report.get("evaluation_kind") != CLASSICAL_EVALUATION_KIND:
        raise ContractError("Classical report evaluation-kind mismatch")
    if report.get("status") != "complete":
        raise ContractError("Classical report is not complete")
    if (
        report.get("fit_id") != fit_id
        or report.get("evaluation_pair_identity_sha256") != pair_sha
    ):
        raise ContractError("Classical evaluation identity mismatch")
    neural_freeze = _require_mapping(
        report.get("neural_freeze"), "classical neural_freeze"
    )
    if (
        neural_freeze.get("manifest_id") != freeze_id
        or neural_freeze.get("manifest_path") != freeze_manifest_path
        or neural_freeze.get("manifest_file_sha256") != freeze_file_sha
        or neural_freeze.get("scientific_identity_sha256") != freeze_sha
        or neural_freeze.get("selected_retraining_checkpoint_sha256")
        != checkpoint_map
    ):
        raise ContractError("Classical report neural-freeze identity mismatch")
    neural_selected_lock = _require_mapping(
        neural_freeze.get("selected_method_lock"),
        "classical report neural-freeze selected_method_lock",
    )
    if (
        neural_selected_lock.get("repo_relative_identifier")
        != selected_lock_path
        or neural_selected_lock.get("raw_file_sha256") != selected_lock_sha
    ):
        raise ContractError("Classical report selected-method lock mismatch")
    selected_method = _require_nonempty_string(
        neural_freeze.get("selected_method"),
        "classical report neural-freeze selected_method",
    )
    checkpoint_hashes = checkpoint_map
    classical_lock = _require_mapping(report.get("classical_lock"), "classical_lock")
    if (
        classical_lock.get("path") != lock_path
        or classical_lock.get("raw_file_sha256") != lock_raw_sha
        or classical_lock.get("canonical_identity_sha256") != lock_sha
        or classical_lock.get("source_code_sha256") != lock_source_map
    ):
        raise ContractError("Classical lock provenance mismatch")
    b2_model = _require_mapping(report.get("b2_model"), "b2_model")
    if (
        b2_model.get("path") != b2_model_path
        or b2_model.get("sha256") != b2_model_sha
        or b2_model.get("semantic_sha256") != b2_model_semantic_sha
    ):
        raise ContractError("Classical B2 model provenance mismatch")
    if (
        report.get("selection_relationship")
        != "external_only_no_neural_selection_effect"
        or report.get("ranking_policy") != "no_overall_accuracy_or_winner_ranking"
    ):
        raise ContractError("Classical comparison role mismatch")
    runtime = _require_mapping(report.get("runtime"), "classical runtime")
    if runtime.get("overall_accuracy_reported") is not False:
        raise ContractError("Classical report violates its ranking policy")
    if runtime.get("native_tile_shape") != list(EXPECTED_NATIVE_TILE_SHAPE):
        raise ContractError("Classical report native tile shape mismatch")
    curves = _require_mapping(report.get("curves"), "classical curves")
    if curves.get("reported") is not False:
        raise ContractError("Classical hard-label report cannot supply PR evidence")
    _report_outputs_include(
        report,
        (
            "evaluation_summary.json",
            "aggregate_metrics.csv",
            "per_tile_metrics.csv",
            "per_tile_confusion.csv",
        ),
    )
    comparators = _require_mapping(report.get("comparators"), "comparators")
    if set(comparators) != set(CLASSICAL_METHOD_IDS):
        raise ContractError("Classical report comparator set mismatch")

    evaluations = []
    for comparator in CLASSICAL_METHOD_IDS:
        (
            tile_names,
            input_hashes,
            target_hashes,
            matrices,
            aggregate,
            metrics,
            curve_evidence,
        ) = _verify_common_tile_report(
            report,
            classical=True,
            comparator=comparator,
            artifacts=artifacts,
        )
        if curve_evidence is not None:
            raise ContractError("Classical hard-label method has unexpected curves")
        evaluations.append(
            Evaluation(
                method_id=comparator,
                display_name=CLASSICAL_DISPLAY_NAMES[comparator],
                source_kind="classical",
                seed=None,
                freeze_id=freeze_id,
                freeze_scientific_identity_sha256=freeze_sha,
                report_sha256=str(hashes["report"]),
                report_directory=report_artifact.path.parent,
                report_outputs=tuple(str(value) for value in report["outputs"]),
                qualitative_example=None,
                artifact_sha256={
                    key: str(value)
                    for key, value in hashes.items()
                    if value is not None
                },
                source_identity={
                    "fit_id": fit_id,
                    "evaluation_pair_identity_sha256": pair_sha,
                    "classical_lock_identity_sha256": lock_sha,
                    "classical_lock_path": lock_path,
                    "classical_lock_raw_file_sha256": lock_raw_sha,
                    "classical_lock_source_code_sha256": lock_source_map,
                    "b2_model_path": b2_model_path,
                    "b2_model_sha256": b2_model_sha,
                    "b2_model_semantic_sha256": b2_model_semantic_sha,
                    "selected_method_lock_path": selected_lock_path,
                    "selected_method_lock_sha256": selected_lock_sha,
                    "neural_freeze_manifest_path": freeze_manifest_path,
                    "neural_freeze_manifest_file_sha256": freeze_file_sha,
                    "selected_method": selected_method,
                    "selected_retraining_checkpoint_sha256": checkpoint_hashes,
                    "comparator": comparator,
                },
                tile_names=tile_names,
                input_sha256=input_hashes,
                target_sha256=target_hashes,
                tile_confusions=matrices,
                aggregate_confusion=aggregate,
                metrics=metrics,
                curves=None,
            )
        )
    data = _require_mapping(report.get("data"), "classical data")
    if data.get("test_filenames") != list(evaluations[0].tile_names):
        raise ContractError("Classical report partition file list mismatch")
    return evaluations


def _validate_manifest(
    path: Path,
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("Input manifest must be a regular non-symbolic file")
    content = _read_file_bytes(path, "input manifest")
    manifest_sha256 = sha256_bytes(content)
    manifest = _require_mapping(
        _load_json_bytes(content, str(path)), "input manifest"
    )
    _require_exact_keys(
        manifest,
        (
            "schema_version",
            "generator",
            "partition_label",
            "bootstrap",
            "neural_evaluations",
            "classical_evaluation",
            "qualitative_figures",
        ),
        "input manifest",
    )
    if manifest["schema_version"] != INPUT_SCHEMA_VERSION:
        raise ContractError("Input manifest schema mismatch")
    generator = _require_mapping(manifest["generator"], "generator")
    _require_exact_keys(
        generator, ("schema_version", "version", "source_sha256"), "generator"
    )
    if (
        generator["schema_version"] != GENERATOR_SCHEMA_VERSION
        or generator["version"] != GENERATOR_VERSION
        or _require_sha256(
            generator["source_sha256"], "generator.source_sha256"
        )
        != GENERATOR_SOURCE_SHA256
    ):
        raise ContractError("Input manifest generator identity mismatch")
    if manifest["partition_label"] != PARTITION_LABEL:
        raise ContractError("Input manifest partition wording mismatch")
    if not isinstance(manifest["qualitative_figures"], list):
        raise ContractError("qualitative_figures must be a JSON list")
    bootstrap = _require_mapping(manifest["bootstrap"], "bootstrap")
    _require_exact_keys(
        bootstrap, ("replicates", "seed", "confidence"), "bootstrap"
    )
    parsed_bootstrap = {
        "replicates": _require_int(
            bootstrap["replicates"], "bootstrap.replicates", minimum=2
        ),
        "seed": _require_int(bootstrap["seed"], "bootstrap.seed", minimum=0),
        "confidence": _require_number(
            bootstrap["confidence"],
            "bootstrap.confidence",
            minimum=np.finfo(float).eps,
            maximum=1.0 - np.finfo(float).eps,
        ),
    }
    locked_bootstrap = {
        "replicates": PAIRED_BOOTSTRAP_REPLICATES,
        "seed": PAIRED_BOOTSTRAP_SEED,
        "confidence": PAIRED_BOOTSTRAP_CONFIDENCE,
    }
    if parsed_bootstrap != locked_bootstrap:
        raise ContractError(
            "Paired-bootstrap settings must match the prespecified locked "
            f"publication contract: {locked_bootstrap}"
        )
    return manifest, parsed_bootstrap, manifest_sha256


def _resolve_non_symbolic_manifest_path(value: str | Path) -> Path:
    """Reject a symbolic lexical manifest path before canonical resolution."""

    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    lexical = lexical.absolute()
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ContractError(
                "Input manifest path contains a symbolic-link component"
            )
    if not lexical.is_file():
        raise ContractError("Input manifest must be a regular non-symbolic file")
    return lexical.resolve(strict=True)


def _lexical_non_symbolic_output_path(value: str | Path) -> Path:
    """Return an absolute output path without following any existing link."""

    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    lexical = lexical.absolute()
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ContractError(
                "Output path contains a symbolic-link component"
            )
    return lexical


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing any destination overwrite."""

    if sys.platform == "darwin":
        import ctypes

        rename_exclusive = 0x00000004  # RENAME_EXCL from <sys/stdio.h>.
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            os.fsencode(source), os.fsencode(destination), rename_exclusive
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
    elif sys.platform.startswith("linux"):
        import ctypes

        at_fdcwd = -100
        rename_noreplace = 1
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ContractError(
                "Atomic no-replace publication is unavailable on this Linux runtime"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
    elif os.name == "nt":
        try:
            os.rename(source, destination)
            return
        except FileExistsError:
            raise
        except OSError as error:
            error_number = error.errno or errno.EIO
    else:
        raise ContractError(
            "Atomic no-replace publication is unavailable on this platform"
        )

    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"Output path was created during assembly: {destination}",
            str(destination),
        )
    raise OSError(
        error_number,
        f"Cannot atomically publish assembly to {destination}: "
        f"{os.strerror(error_number)}",
        str(destination),
    )


def _validate_evaluation_collection(evaluations: Sequence[Evaluation]) -> str:
    if not evaluations:
        raise ContractError("No evaluation evidence was supplied")
    keys = [(item.method_id, item.seed) for item in evaluations]
    if len(keys) != len(set(keys)):
        raise ContractError("Evaluation method/seed identities must be unique")

    neural = [item for item in evaluations if item.source_kind == "neural"]
    classical = [item for item in evaluations if item.source_kind == "classical"]
    if len(neural) != 2 * len(NEURAL_SEEDS):
        raise ContractError(
            "Exactly three selected-primary and three plain-comparator neural "
            "locked evaluations are required; extra neural evaluations are rejected"
        )
    if len(classical) != len(CLASSICAL_METHOD_IDS):
        raise ContractError("Exactly three authenticated classical comparators are required")

    display_names: dict[str, str] = {}
    for item in evaluations:
        if item.method_id in display_names and display_names[item.method_id] != (
            item.display_name
        ):
            raise ContractError("A method has inconsistent display names")
        display_names[item.method_id] = item.display_name

    selected_methods = {
        str(item.source_identity["selected_method"]) for item in neural
    }
    if len(selected_methods) != 1:
        raise ContractError(
            "All six neural reports tied to one freeze must share one selected method"
        )
    selected_primary_method_id = selected_methods.pop()
    if selected_primary_method_id not in SELECTED_METHOD_IDS:
        raise ContractError("The authenticated selected method is outside the frozen screen")
    canonical_display_names = {
        selected_primary_method_id: selected_primary_method_id,
        PLAIN_COMPARATOR_METHOD_ID: "Plain U-Net",
        **CLASSICAL_DISPLAY_NAMES,
    }
    if display_names != canonical_display_names:
        raise ContractError(
            "Method display names must match the unique canonical publication labels"
        )

    primary = [
        item
        for item in neural
        if item.source_identity["architecture_role"] == "primary_multiscale"
    ]
    plain = [
        item
        for item in neural
        if item.source_identity["architecture_role"] == "plain_unet_comparator"
    ]
    if len(primary) != len(NEURAL_SEEDS) or len(plain) != len(NEURAL_SEEDS):
        raise ContractError(
            "Neural evidence must contain exactly three primary and three plain cells"
        )
    if {item.method_id for item in primary} != {selected_primary_method_id}:
        raise ContractError(
            "Primary locked evaluations must be the authenticated selected winner"
        )
    if {item.method_id for item in plain} != {PLAIN_COMPARATOR_METHOD_ID}:
        raise ContractError(
            "Plain locked evaluations must use only the formulation-matched "
            "plain-U-Net comparator"
        )
    for method_id, items in (
        (selected_primary_method_id, primary),
        (PLAIN_COMPARATOR_METHOD_ID, plain),
    ):
        if {item.seed for item in items} != set(NEURAL_SEEDS):
            raise ContractError(
                f"Neural method {method_id} lacks the complete three-seed evidence"
            )
        curve_presence = {item.curves is not None for item in items}
        if len(curve_presence) != 1:
            raise ContractError(
                f"Neural method {method_id} has partial probability evidence"
            )

    expected_checkpoint_map: dict[str, list[str]] = {}
    for role, items in (
        ("primary_multiscale", primary),
        ("plain_unet_comparator", plain),
    ):
        by_cell = {
            int(item.source_identity["cell_index"]): str(
                item.source_identity["checkpoint_sha256"]
            )
            for item in items
        }
        expected_checkpoint_map[role] = [
            by_cell[cell_index] for cell_index in range(len(NEURAL_SEEDS))
        ]
    all_checkpoint_hashes = [
        checkpoint_sha
        for role in ("primary_multiscale", "plain_unet_comparator")
        for checkpoint_sha in expected_checkpoint_map[role]
    ]
    if len(set(all_checkpoint_hashes)) != 2 * len(NEURAL_SEEDS):
        raise ContractError(
            "The six authenticated neural cells must use six distinct checkpoints"
        )
    neural_binding_keys = (
        "evaluator_path",
        "evaluator_sha256",
        "training_execution_source_attestation",
        "selected_method_lock_path",
        "selected_method_lock_sha256",
        "neural_freeze_manifest_path",
        "neural_freeze_manifest_file_sha256",
        "selected_retraining_checkpoint_sha256",
    )
    shared_neural_bindings = {
        key: neural[0].source_identity[key] for key in neural_binding_keys
    }
    for item in neural:
        observed_bindings = {
            key: item.source_identity[key] for key in neural_binding_keys
        }
        if observed_bindings != shared_neural_bindings:
            raise ContractError(
                "Neural reports disagree on verified source/freeze provenance"
            )
        if item.source_identity["selected_retraining_checkpoint_sha256"] != (
            expected_checkpoint_map
        ):
            raise ContractError(
                "Neural report checkpoint map disagrees with the six neural cells"
            )
    for item in classical:
        if item.source_identity.get("selected_method") != (
            selected_primary_method_id
        ):
            raise ContractError(
                "Classical report selected method disagrees with neural evidence"
            )
        if item.source_identity.get(
            "selected_retraining_checkpoint_sha256"
        ) != expected_checkpoint_map:
            raise ContractError(
                "Classical report checkpoint map disagrees with the six neural cells"
            )
        for key in (
            "selected_method_lock_path",
            "selected_method_lock_sha256",
            "neural_freeze_manifest_path",
            "neural_freeze_manifest_file_sha256",
        ):
            if item.source_identity.get(key) != shared_neural_bindings[key]:
                raise ContractError(
                    "Classical report freeze provenance disagrees with neural evidence"
                )

    first = evaluations[0]
    canonical = (
        first.freeze_id,
        first.freeze_scientific_identity_sha256,
        first.tile_names,
        first.input_sha256,
        first.target_sha256,
    )
    reference_supports = first.tile_confusions.sum(axis=2)
    for item in evaluations[1:]:
        observed = (
            item.freeze_id,
            item.freeze_scientific_identity_sha256,
            item.tile_names,
            item.input_sha256,
            item.target_sha256,
        )
        if observed != canonical:
            raise ContractError(
                "Evaluation sources do not share one authenticated partition identity"
            )
        if not np.array_equal(item.tile_confusions.sum(axis=2), reference_supports):
            raise ContractError(
                "Evaluation sources disagree on per-tile reference-class support"
            )
    return selected_primary_method_id


def _validate_png_snapshot(content: bytes, label: str) -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    if (
        len(content) < 33
        or not content.startswith(signature)
        or content[12:16] != b"IHDR"
    ):
        raise ContractError(f"{label} is not a valid PNG byte snapshot")
    width = int.from_bytes(content[16:20], "big")
    height = int.from_bytes(content[20:24], "big")
    if width <= 0 or height <= 0:
        raise ContractError(f"{label} has invalid PNG dimensions")


def _validate_pdf_snapshot(content: bytes, label: str) -> None:
    if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-1024:]:
        raise ContractError(f"{label} is not a valid PDF byte snapshot")


def _qualitative_figure_contract(
    value: Any, *, method_id: str, image_id: int, file_name: str, label: str
) -> Mapping[str, Any]:
    contract = _require_mapping(value, label)
    _require_exact_keys(
        contract,
        (
            "panel_titles",
            "tile_label",
            "raw_evidence_relationship",
            "raw_evidence_display_encoding",
            "error_overlay_definition",
            "error_category_labels",
            "historical_metric_values_included",
            "physical_scale_author_evidence_gate",
        ),
        label,
    )
    conditional = method_id in {"C2-P", "C2-F", "C2-FP"}
    expected_titles = [
        "(a) Input",
        "(b) Lossless reference",
        (
            "(c) Raw learned P(C0)\ninside fixed pore gate"
            if conditional
            else "(c) Raw learned P(C0)\nnative 3-class softmax"
        ),
        (
            "(d) Raw learned P(C1)\ninside fixed pore gate"
            if conditional
            else "(d) Raw learned P(C1)\nnative 3-class softmax"
        ),
        (
            "(e) Final 3-class prediction\nfixed-gate composition"
            if conditional
            else "(e) Final 3-class prediction\nnative 3-class argmax"
        ),
        "(f) Error overlay\ncolour = reference class",
    ]
    expected_relationship = (
        "Raw C0/C1 evidence is operative only inside the fixed pore gate; "
        "the final map assigns C2 outside that gate."
        if conditional
        else "Raw C0/C1 evidence comes from the same native three-class softmax; "
        "the final map is its pixelwise argmax, including C2."
    )
    if contract["panel_titles"] != expected_titles:
        raise ContractError("Qualitative evaluator panel-title contract mismatch")
    if contract["tile_label"] != f"Tile ID {image_id} | {file_name}":
        raise ContractError("Qualitative evaluator tile-label contract mismatch")
    if contract["raw_evidence_relationship"] != expected_relationship:
        raise ContractError("Qualitative evaluator evidence-relationship mismatch")
    if contract["raw_evidence_display_encoding"] != (
        "round(network probability * 255) for rendering on a fixed 0-1 scale"
    ):
        raise ContractError("Qualitative evaluator evidence-display encoding mismatch")
    if contract["error_overlay_definition"] != (
        "misclassified pixels coloured by their reference class; "
        "correct pixels have no overlay"
    ):
        raise ContractError("Qualitative evaluator error-overlay definition mismatch")
    if contract["error_category_labels"] != {
        "1": "Reference C0 misclassified",
        "2": "Reference C1 misclassified",
        "3": "Reference C2 misclassified",
    }:
        raise ContractError("Qualitative evaluator error-category labels mismatch")
    if contract["historical_metric_values_included"] is not False:
        raise ContractError("Qualitative panel must exclude historical metric values")
    scale_gate = _require_mapping(
        contract["physical_scale_author_evidence_gate"],
        f"{label}.physical_scale_author_evidence_gate",
    )
    expected_scale_gate = {
        "status": "open_author_evidence_required",
        "scale_bar_shown": False,
        "pixel_size_shown": False,
        "required_evidence": "authenticated pixel size or scale-bar metadata",
        "evaluation_blocking": False,
    }
    if dict(scale_gate) != expected_scale_gate:
        raise ContractError("Qualitative physical-scale author gate mismatch")
    return contract


def _load_qualitative_panels(
    manifest_dir: Path,
    record_values: Any,
    evaluations: Sequence[Evaluation],
    selected_primary_method_id: str,
) -> list[QualitativePanel]:
    """Authenticate optional evaluator-produced panels without rereading corpus data."""

    if not isinstance(record_values, list):
        raise ContractError("qualitative_figures must be a JSON list")
    primary_by_cell = {
        (item.method_id, item.seed): item
        for item in evaluations
        if item.source_kind == "neural"
        and item.source_identity.get("architecture_role") == "primary_multiscale"
    }
    if len(record_values) > 1:
        raise ContractError(
            "qualitative_figures may contain at most the prespecified seed-42 panel"
        )
    if not record_values:
        return []
    label = "qualitative_figures[0]"
    record = _require_mapping(record_values[0], label)
    _require_exact_keys(
        record,
        ("method_id", "training_seed", "pdf", "png"),
        label,
    )
    method_id = _require_nonempty_string(record["method_id"], f"{label}.method_id")
    seed = _require_int(record["training_seed"], f"{label}.training_seed", minimum=0)
    if method_id != selected_primary_method_id or seed != QUALITATIVE_FIGURE_SEED:
        raise ContractError(
            "Qualitative figure must reference the selected primary at prespecified seed 42"
        )
    evaluation = primary_by_cell.get((method_id, seed))
    if evaluation is None:
        raise ContractError(
            "Qualitative figure does not identify an authenticated selected-primary evaluation"
        )
    qualitative = evaluation.qualitative_example
    if qualitative is None:
        raise ContractError(
            "Selected evaluator report has no authenticated qualitative_example"
        )
    _require_exact_keys(
        qualitative,
        (
            "selection_rule",
            "image_id",
            "file_name",
            "input_image_sha256",
            "target_mask_sha256",
            "post_processing",
            "publication_figure_contract",
            "publication_files",
        ),
        "neural qualitative_example",
    )
    if qualitative["selection_rule"] != QUALITATIVE_SELECTION_RULE:
        raise ContractError(
            "Qualitative panel selection must be evaluator-authenticated and outcome-independent"
        )
    image_id = _require_int(
        qualitative["image_id"], "qualitative_example.image_id", minimum=1
    )
    file_name = _require_nonempty_string(
        qualitative["file_name"], "qualitative_example.file_name"
    )
    if file_name not in evaluation.tile_names:
        raise ContractError("Qualitative example tile is outside the authenticated partition")
    tile_index = evaluation.tile_names.index(file_name)
    if (
        _require_sha256(
            qualitative["input_image_sha256"],
            "qualitative_example.input_image_sha256",
        )
        != evaluation.input_sha256[tile_index]
        or _require_sha256(
            qualitative["target_mask_sha256"],
            "qualitative_example.target_mask_sha256",
        )
        != evaluation.target_sha256[tile_index]
    ):
        raise ContractError("Qualitative example tile hashes mismatch")
    if qualitative["post_processing"] != "none":
        raise ContractError("Qualitative example violates the no-post-processing contract")
    contract = _qualitative_figure_contract(
        qualitative["publication_figure_contract"],
        method_id=method_id,
        image_id=image_id,
        file_name=file_name,
        label="qualitative publication_figure_contract",
    )
    if qualitative["publication_files"] != [
        QUALITATIVE_PDF_PATH,
        QUALITATIVE_PNG_PATH,
    ]:
        raise ContractError(
            "Qualitative evaluator publication-file contract is absent or incomplete"
        )
    pdf = _artifact_path(
        manifest_dir,
        record["pdf"],
        f"{label}.pdf",
        expected_name=Path(QUALITATIVE_PDF_PATH).name,
    )
    png = _artifact_path(
        manifest_dir,
        record["png"],
        f"{label}.png",
        expected_name=Path(QUALITATIVE_PNG_PATH).name,
    )
    try:
        observed_pdf_path = pdf.path.relative_to(evaluation.report_directory).as_posix()
        observed_png_path = png.path.relative_to(evaluation.report_directory).as_posix()
    except ValueError as error:
        raise ContractError(
            "Qualitative figure files must be inside their evaluator output directory"
        ) from error
    if (
        observed_pdf_path != QUALITATIVE_PDF_PATH
        or observed_png_path != QUALITATIVE_PNG_PATH
    ):
        raise ContractError("Qualitative figure paths do not match the evaluator contract")
    if (
        QUALITATIVE_PDF_PATH not in evaluation.report_outputs
        or QUALITATIVE_PNG_PATH not in evaluation.report_outputs
    ):
        raise ContractError("Qualitative figures must be declared evaluator outputs")
    _validate_pdf_snapshot(pdf.content, f"{label}.pdf")
    _validate_png_snapshot(png.content, f"{label}.png")
    return [
        QualitativePanel(
            method_id=method_id,
            seed=seed,
            file_name=file_name,
            image_id=image_id,
            pdf=pdf,
            png=png,
            output_pdf_name="selected_primary_outcome_independent_qualitative.pdf",
            output_png_name="selected_primary_outcome_independent_qualitative.png",
            publication_figure_contract=contract,
        )
    ]


def _method_order(
    evaluations: Sequence[Evaluation], selected_primary_method_id: str
) -> list[str]:
    method_ids = {item.method_id for item in evaluations}
    classical = [item for item in CLASSICAL_METHOD_IDS if item in method_ids]
    return [selected_primary_method_id, PLAIN_COMPARATOR_METHOD_ID, *classical]


def _evaluation_groups(
    evaluations: Sequence[Evaluation],
) -> dict[str, list[Evaluation]]:
    groups: dict[str, list[Evaluation]] = {}
    for item in evaluations:
        groups.setdefault(item.method_id, []).append(item)
    for items in groups.values():
        items.sort(key=lambda item: (-1 if item.seed is None else item.seed))
    return groups


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _sample_sd(values: Sequence[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def _complete_mean(values: Sequence[float | None]) -> float | None:
    return None if any(value is None for value in values) else _mean(
        [float(value) for value in values if value is not None]
    )


def _complete_sample_sd(values: Sequence[float | None]) -> float | None:
    return None if any(value is None for value in values) else _sample_sd(
        [float(value) for value in values if value is not None]
    )


def _build_aggregate_tables(
    evaluations: Sequence[Evaluation],
    selected_primary_method_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = _evaluation_groups(evaluations)
    reference_by_seed = {
        item.seed: item for item in groups[selected_primary_method_id]
    }
    reference_means = {
        metric: _complete_mean(
            [item.metrics[metric] for item in groups[selected_primary_method_id]]
        )
        for metric in PUBLICATION_METRICS
    }
    per_seed: list[dict[str, Any]] = []
    for method_id in _method_order(evaluations, selected_primary_method_id):
        for item in groups[method_id]:
            if item.seed is None:
                continue
            reference = reference_by_seed.get(item.seed)
            if reference is None:
                raise ContractError(
                    "A neural comparator seed has no same-seed selected-primary reference"
                )
            row: dict[str, Any] = {
                "method_id": method_id,
                "method": item.display_name,
                "seed": item.seed,
                "selected_primary_method_id": selected_primary_method_id,
                "contrast_direction": CONTRAST_DIRECTION,
                "partition": PARTITION_LABEL,
            }
            for metric in PUBLICATION_METRICS:
                value = item.metrics[metric]
                row[metric] = value
                row[f"selected_primary_minus_method_{metric}"] = (
                    None
                    if (
                        method_id == selected_primary_method_id
                        or reference.metrics[metric] is None
                        or value is None
                    )
                    else float(reference.metrics[metric]) - float(value)
                )
            per_seed.append(row)

    per_method: list[dict[str, Any]] = []
    for method_id in _method_order(evaluations, selected_primary_method_id):
        items = groups[method_id]
        row = {
            "method_id": method_id,
            "method": items[0].display_name,
            "source_kind": items[0].source_kind,
            "evaluation_count": len(items),
            "seed_count": sum(item.seed is not None for item in items),
            "selected_primary_method_id": selected_primary_method_id,
            "contrast_direction": CONTRAST_DIRECTION,
            "partition": PARTITION_LABEL,
        }
        for metric in PUBLICATION_METRICS:
            values = [item.metrics[metric] for item in items]
            mean_value = _complete_mean(values)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_sample_sd"] = _complete_sample_sd(values)
            row[f"selected_primary_minus_method_{metric}"] = (
                None
                if (
                    method_id == selected_primary_method_id
                    or reference_means[metric] is None
                    or mean_value is None
                )
                else float(reference_means[metric]) - float(mean_value)
            )
        per_method.append(row)
    return per_seed, per_method


def _bootstrap_draws(replicates: int, tile_count: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(
        0,
        tile_count,
        size=(replicates, tile_count),
        endpoint=False,
        dtype=np.int64,
    )


def _metrics_for_draw(
    item: Evaluation, indices: np.ndarray
) -> Mapping[str, float | None]:
    metrics = metrics_from_confusion(item.tile_confusions[indices].sum(axis=0))
    result: dict[str, float | None] = {}
    for name in PUBLICATION_METRICS:
        value = metrics[name]
        if value is None:
            if name not in OPTIONALLY_UNDEFINED_PUBLICATION_METRICS:
                raise ContractError(f"Bootstrap metric {name} became undefined")
            result[name] = None
            continue
        if not math.isfinite(float(value)):
            raise ContractError(f"Bootstrap metric {name} became undefined")
        result[name] = float(value)
    return result


def _paired_bootstrap_differences(
    evaluations: Sequence[Evaluation],
    bootstrap: Mapping[str, Any],
    per_method: Sequence[Mapping[str, Any]],
    selected_primary_method_id: str,
) -> list[dict[str, Any]]:
    groups = _evaluation_groups(evaluations)
    reference = groups[selected_primary_method_id]
    reference_by_seed = {item.seed: item for item in reference}
    tile_count = len(reference[0].tile_names)
    draws = _bootstrap_draws(
        int(bootstrap["replicates"]), tile_count, int(bootstrap["seed"])
    )
    summary_by_method = {
        str(row["method_id"]): row for row in per_method
    }
    differences: list[dict[str, Any]] = []
    alpha = 1.0 - float(bootstrap["confidence"])
    for method_id in _method_order(evaluations, selected_primary_method_id):
        if method_id == selected_primary_method_id:
            continue
        target = groups[method_id]
        if all(item.seed is not None for item in target):
            target_by_seed = {item.seed: item for item in target}
            if set(target_by_seed) != set(reference_by_seed):
                raise ContractError(
                    f"{method_id} cannot be paired to the complete "
                    "selected-primary seed set"
                )
            seed_pairs = [
                (target_by_seed[seed], reference_by_seed[seed])
                for seed in NEURAL_SEEDS
            ]
            comparison_basis = (
                "same-seed selected primary minus comparator using the same "
                "resampled whole-tile indices"
            )
        elif len(target) == 1 and target[0].seed is None:
            seed_pairs = [(target[0], item) for item in reference]
            comparison_basis = (
                "selected-primary seed mean minus the single deterministic "
                "method using the same resampled whole-tile indices"
            )
        else:
            raise ContractError(f"{method_id} has an unsupported pairing structure")

        series: dict[str, list[float | None]] = {
            metric: [] for metric in PUBLICATION_METRICS
        }
        for indices in draws:
            target_cache: dict[int, Mapping[str, float | None]] = {}
            reference_cache: dict[int, Mapping[str, float | None]] = {}
            replicate_pairs: list[
                tuple[
                    Mapping[str, float | None],
                    Mapping[str, float | None],
                ]
            ] = []
            for target_item, reference_item in seed_pairs:
                target_key = id(target_item)
                reference_key = id(reference_item)
                if target_key not in target_cache:
                    target_cache[target_key] = _metrics_for_draw(
                        target_item, indices
                    )
                if reference_key not in reference_cache:
                    reference_cache[reference_key] = _metrics_for_draw(
                        reference_item, indices
                    )
                replicate_pairs.append(
                    (target_cache[target_key], reference_cache[reference_key])
                )
            for metric in PUBLICATION_METRICS:
                primary_values = [pair[1][metric] for pair in replicate_pairs]
                comparator_values = [pair[0][metric] for pair in replicate_pairs]
                primary_mean = _complete_mean(primary_values)
                comparator_mean = _complete_mean(comparator_values)
                difference = (
                    None
                    if primary_mean is None or comparator_mean is None
                    else primary_mean - comparator_mean
                )
                series[metric].append(difference)

        summary = summary_by_method[method_id]
        for metric in PUBLICATION_METRICS:
            all_values = series[metric]
            if len(all_values) != int(bootstrap["replicates"]):
                raise RuntimeError("Internal paired-bootstrap series shape mismatch")
            values = np.asarray(
                [value for value in all_values if value is not None],
                dtype=np.float64,
            )
            if values.size == int(bootstrap["replicates"]):
                lower, upper = np.quantile(
                    values, [alpha / 2.0, 1.0 - alpha / 2.0]
                )
                lower_value: float | None = float(lower)
                upper_value: float | None = float(upper)
            else:
                lower_value = None
                upper_value = None
            differences.append(
                {
                    "method_id": method_id,
                    "method": target[0].display_name,
                    "reference_method_id": selected_primary_method_id,
                    "contrast_direction": CONTRAST_DIRECTION,
                    "metric": metric,
                    "difference": summary[
                        f"selected_primary_minus_method_{metric}"
                    ],
                    "ci_lower": lower_value,
                    "ci_upper": upper_value,
                    "finite_replicates": int(values.size),
                    "confidence": float(bootstrap["confidence"]),
                    "replicates": int(bootstrap["replicates"]),
                    "bootstrap_seed": int(bootstrap["seed"]),
                    "tile_count": tile_count,
                    "sampling_unit": (
                        "whole native tile from the "
                        f"{PARTITION_LABEL}"
                    ),
                    "pairing": comparison_basis,
                }
            )
    return differences


def _per_tile_diagnostics(
    evaluations: Sequence[Evaluation],
    selected_primary_method_id: str,
) -> list[dict[str, Any]]:
    groups = _evaluation_groups(evaluations)
    reference = groups[selected_primary_method_id]
    reference_by_seed = {item.seed: item for item in reference}
    reference_tile_metrics = [
        [metrics_from_confusion(item.tile_confusions[index]) for item in reference]
        for index in range(len(reference[0].tile_names))
    ]
    rows: list[dict[str, Any]] = []
    for method_id in _method_order(evaluations, selected_primary_method_id):
        for item in groups[method_id]:
            for index, name in enumerate(item.tile_names):
                matrix = item.tile_confusions[index]
                metrics = metrics_from_confusion(matrix)
                if method_id == selected_primary_method_id:
                    reference_metrics = metrics
                    reference_basis = (
                        "selected primary self row; contrast intentionally undefined"
                    )
                elif item.seed is None:
                    reference_metrics = {
                        metric: _complete_mean(
                            [
                                source[metric]
                                for source in reference_tile_metrics[index]
                            ]
                        )
                        for metric in PUBLICATION_METRICS
                    }
                    reference_basis = (
                        "selected-primary seed mean for the same tile index"
                    )
                else:
                    reference_item = reference_by_seed[item.seed]
                    reference_metrics = metrics_from_confusion(
                        reference_item.tile_confusions[index]
                    )
                    reference_basis = (
                        "same-seed selected primary for the same tile index"
                    )
                c0_iou = metrics["c0_iou"]
                c1_iou = metrics["c1_iou"]
                if c0_iou is None or c1_iou is None:
                    worst = "undefined"
                elif math.isclose(c0_iou, c1_iou, rel_tol=0.0, abs_tol=1e-15):
                    worst = "tie"
                else:
                    worst = "C0" if c0_iou < c1_iou else "C1"
                row: dict[str, Any] = {
                    "method_id": method_id,
                    "method": item.display_name,
                    "source_kind": item.source_kind,
                    "seed": item.seed,
                    "evaluation_ordinal": index + 1,
                    "file_name": name,
                    "input_sha256": item.input_sha256[index],
                    "target_sha256": item.target_sha256[index],
                    "total_pixels": int(matrix.sum()),
                    "c0_reference_pixels": int(matrix[0, :].sum()),
                    "c1_reference_pixels": int(matrix[1, :].sum()),
                    "c2_reference_pixels": int(matrix[2, :].sum()),
                    "c0_as_c1_pixels": int(matrix[0, 1]),
                    "c1_as_c0_pixels": int(matrix[1, 0]),
                    "pore_as_mineral_pixels": int(matrix[:2, 2].sum()),
                    "mineral_as_pore_pixels": int(matrix[2, :2].sum()),
                    "worst_pore_iou_class": worst,
                    "reference_basis": reference_basis,
                    "selected_primary_method_id": selected_primary_method_id,
                    "contrast_direction": CONTRAST_DIRECTION,
                    "partition": PARTITION_LABEL,
                }
                for metric in PUBLICATION_METRICS:
                    row[metric] = metrics[metric]
                    reference_value = reference_metrics[metric]
                    row[f"selected_primary_minus_method_{metric}"] = (
                        None
                        if (
                            method_id == selected_primary_method_id
                            or metrics[metric] is None
                            or reference_value is None
                        )
                        else float(reference_value) - float(metrics[metric])
                    )
                rows.append(row)
    return rows


def _aggregate_method_confusions(
    evaluations: Sequence[Evaluation], selected_primary_method_id: str
) -> dict[str, dict[str, Any]]:
    groups = _evaluation_groups(evaluations)
    summaries: dict[str, dict[str, Any]] = {}
    for method_id in _method_order(evaluations, selected_primary_method_id):
        items = groups[method_id]
        matrix = np.sum(
            [item.aggregate_confusion for item in items],
            axis=0,
            dtype=np.int64,
        )
        if matrix.shape != (3, 3) or np.any(matrix < 0):
            raise RuntimeError("Internal aggregate-confusion shape mismatch")
        summaries[method_id] = {
            "method_id": method_id,
            "method": items[0].display_name,
            "source_kind": items[0].source_kind,
            "evaluation_count": len(items),
            "seed_count": sum(item.seed is not None for item in items),
            "confusion": matrix,
        }
    return summaries


def _aggregate_confusion_rows(
    summaries: Mapping[str, Mapping[str, Any]],
    method_order: Sequence[str],
    selected_primary_method_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in method_order:
        summary = summaries[method_id]
        matrix = np.asarray(summary["confusion"], dtype=np.int64)
        for true_class_id in range(3):
            reference_pixels = int(matrix[true_class_id, :].sum())
            if reference_pixels <= 0:
                raise ContractError(
                    f"{method_id} has no authenticated C{true_class_id} reference pixels"
                )
            for predicted_class_id in range(3):
                pixel_count = int(matrix[true_class_id, predicted_class_id])
                rows.append(
                    {
                        "method_id": method_id,
                        "method": summary["method"],
                        "source_kind": summary["source_kind"],
                        "evaluation_count": summary["evaluation_count"],
                        "seed_count": summary["seed_count"],
                        "selected_primary_method_id": selected_primary_method_id,
                        "true_class_id": true_class_id,
                        "true_class_label": CLASS_LABELS[true_class_id],
                        "predicted_class_id": predicted_class_id,
                        "predicted_class_label": CLASS_LABELS[predicted_class_id],
                        "pixel_count": pixel_count,
                        "reference_class_pixels": reference_pixels,
                        "row_fraction": pixel_count / reference_pixels,
                        "outcome": (
                            "correct"
                            if true_class_id == predicted_class_id
                            else "misclassified"
                        ),
                        "aggregation": (
                            "confusion counts pooled over authenticated evaluations "
                            "within method"
                        ),
                        "partition": PARTITION_LABEL,
                    }
                )
    return rows


def _pore_union_metrics_from_confusion(
    confusion: np.ndarray,
) -> dict[str, float | None]:
    """Return pore-vs-mineral metrics from an authenticated 3x3 confusion."""

    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (3, 3) or np.any(matrix < 0):
        raise ContractError("Pore-union metrics require a non-negative 3x3 matrix")
    true_positive = int(matrix[:2, :2].sum())
    false_positive = int(matrix[2, :2].sum())
    false_negative = int(matrix[:2, 2].sum())
    return {
        "iou": _safe_ratio(
            true_positive, true_positive + false_positive + false_negative
        ),
        "dice": _safe_ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "recall": _safe_ratio(true_positive, true_positive + false_negative),
    }


def _full_area_metric_rows(
    summaries: Mapping[str, Mapping[str, Any]],
    method_order: Sequence[str],
    selected_primary_method_id: str,
) -> list[dict[str, Any]]:
    """Derive C2 and pore-union fields only from authenticated confusion counts."""

    conditional_protocol = selected_primary_method_id.startswith("C2-")
    rows: list[dict[str, Any]] = []
    for method_id in method_order:
        summary = summaries[method_id]
        matrix = np.asarray(summary["confusion"], dtype=np.int64)
        class_metrics = metrics_from_confusion(matrix)
        pore_metrics = _pore_union_metrics_from_confusion(matrix)
        prediction_source = (
            "fixed raw-intensity pore gate (rule-assisted C2)"
            if method_id in CLASSICAL_METHOD_IDS or conditional_protocol
            else "native learned three-class prediction"
        )
        row: dict[str, Any] = {
            "method_id": method_id,
            "method": summary["method"],
            "source_kind": summary["source_kind"],
            "evaluation_count": summary["evaluation_count"],
            "seed_count": summary["seed_count"],
            "selected_primary_method_id": selected_primary_method_id,
            "prediction_source": prediction_source,
            "aggregation": (
                "metrics reconstructed from confusion counts pooled over "
                "authenticated evaluations within method"
            ),
            "partition": PARTITION_LABEL,
        }
        for metric_name in METRIC_NAMES:
            row[f"c2_{metric_name}"] = class_metrics[f"c2_{metric_name}"]
            row[f"pore_union_{metric_name}"] = pore_metrics[metric_name]
        rows.append(row)
    return rows


def _save_figure_pair(
    fig: Any,
    output_dir: Path,
    *,
    stem: str,
    title: str,
) -> tuple[str, str]:
    pdf_name = f"{stem}.pdf"
    png_name = f"{stem}.png"
    fixed_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        output_dir / pdf_name,
        bbox_inches="tight",
        metadata={
            "Creator": "assemble_publication_results.py",
            "Producer": "Matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
            "Title": title,
            "Subject": PARTITION_LABEL,
        },
    )
    fig.savefig(
        output_dir / png_name,
        dpi=600,
        bbox_inches="tight",
        metadata={
            "Software": "assemble_publication_results.py",
            "Title": title,
            "Description": PARTITION_LABEL,
        },
    )
    return pdf_name, png_name


def _plot_selected_primary_confusion(
    output_dir: Path,
    summary: Mapping[str, Any],
) -> tuple[str, str]:
    plt = _configure_matplotlib()
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch, Rectangle

    matrix = np.asarray(summary["confusion"], dtype=np.int64)
    row_totals = matrix.sum(axis=1)
    fractions = matrix / row_totals[:, None]
    fig, axis = plt.subplots(figsize=(6.5, 5.35))
    for true_class_id in range(3):
        for predicted_class_id in range(3):
            fraction = float(fractions[true_class_id, predicted_class_id])
            root = np.asarray(
                to_rgb(CLASS_PALETTE[f"C{predicted_class_id}"]),
                dtype=np.float64,
            )
            alpha = 0.12 + 0.82 * fraction
            face = tuple(1.0 - alpha * (1.0 - root))
            is_error = true_class_id != predicted_class_id
            axis.add_patch(
                Rectangle(
                    (predicted_class_id, true_class_id),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="#2F2F2F",
                    linewidth=1.4 if not is_error else 0.7,
                    hatch="////" if is_error else None,
                )
            )
            axis.text(
                predicted_class_id + 0.5,
                true_class_id + 0.5,
                f"{100.0 * fraction:.1f}%\n{int(matrix[true_class_id, predicted_class_id]):,} px",
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="semibold" if not is_error else "normal",
                color="#202020",
            )
    compact_labels = (
        "C0\nDisconnected\npore",
        "C1\nConnected\npore",
        "C2\nMineral",
    )
    axis.set_xlim(0, 3)
    axis.set_ylim(3, 0)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(3) + 0.5, compact_labels)
    axis.set_yticks(np.arange(3) + 0.5, compact_labels)
    for class_id, tick in enumerate(axis.get_xticklabels()):
        tick.set_color(CLASS_PALETTE[f"C{class_id}"])
        tick.set_fontweight("semibold")
    for class_id, tick in enumerate(axis.get_yticklabels()):
        tick.set_color(CLASS_PALETTE[f"C{class_id}"])
        tick.set_fontweight("semibold")
    axis.tick_params(length=0, pad=7)
    axis.set_xlabel("Predicted class", labelpad=10)
    axis.set_ylabel("Reference class", labelpad=10)
    axis.spines[:].set_visible(False)
    axis.set_title(
        f"{summary['method']} pooled over {summary['seed_count']} seeds",
        loc="left",
        fontsize=9,
        pad=16,
    )
    axis.legend(
        handles=(
            Patch(facecolor="white", edgecolor="#2F2F2F", linewidth=1.4, label="Correct"),
            Patch(
                facecolor="white",
                edgecolor="#2F2F2F",
                linewidth=0.7,
                hatch="////",
                label="Misclassified",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Selected-primary aggregate C0/C1/C2 confusion matrix",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.945,
        (
            f"{PARTITION_LABEL}; rows are normalized within reference class; "
            "counts are pooled over the three authenticated seeds"
        ),
        ha="left",
        va="top",
        fontsize=7.7,
        color="#555555",
    )
    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.25, right=0.98)
    names = _save_figure_pair(
        fig,
        output_dir,
        stem="selected_primary_c0_c1_c2_confusion_matrix",
        title="Selected-primary aggregate C0/C1/C2 confusion matrix",
    )
    plt.close(fig)
    return names


def _plot_cross_method_c0_c1_recall_summary(
    output_dir: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    method_order: Sequence[str],
) -> tuple[str, str]:
    plt = _configure_matplotlib()
    from matplotlib.ticker import PercentFormatter

    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    y_positions = np.arange(len(method_order), dtype=np.float64)
    c0_values: list[float] = []
    c1_values: list[float] = []
    labels: list[str] = []
    for method_id in method_order:
        summary = summaries[method_id]
        matrix = np.asarray(summary["confusion"], dtype=np.int64)
        row_totals = matrix.sum(axis=1)
        if row_totals[0] <= 0 or row_totals[1] <= 0:
            raise ContractError(
                f"{method_id} lacks authenticated C0/C1 reference pixels"
            )
        c0_values.append(float(matrix[0, 0] / row_totals[0]))
        c1_values.append(float(matrix[1, 1] / row_totals[1]))
        evaluation_label = (
            f"{summary['seed_count']} seeds"
            if int(summary["seed_count"]) > 0
            else "1 deterministic evaluation"
        )
        labels.append(f"{summary['method']}\n{evaluation_label}")

    c0_array = np.asarray(c0_values, dtype=np.float64)
    c1_array = np.asarray(c1_values, dtype=np.float64)
    for y_position, c0_value, c1_value in zip(
        y_positions, c0_array, c1_array, strict=True
    ):
        axis.plot(
            (c0_value, c1_value),
            (y_position - 0.12, y_position + 0.12),
            color="#9A9A9A",
            linewidth=1.0,
            zorder=1,
        )
    axis.scatter(
        c0_array,
        y_positions - 0.12,
        color=CLASS_PALETTE["C0"],
        marker="o",
        s=43,
        label="C0 disconnected-pore recall",
        zorder=3,
    )
    axis.scatter(
        c1_array,
        y_positions + 0.12,
        facecolor="white",
        edgecolor=CLASS_PALETTE["C1"],
        linewidth=1.6,
        marker="s",
        s=43,
        label="C1 connected-pore recall",
        zorder=3,
    )
    for values, offsets, color, vertical_text_offset in (
        (c0_array, y_positions - 0.12, CLASS_PALETTE["C0"], 8),
        (c1_array, y_positions + 0.12, CLASS_PALETTE["C1"], -8),
    ):
        for value, y_position in zip(values, offsets, strict=True):
            axis.annotate(
                f"{value:.1%}",
                (value, y_position),
                xytext=(-6 if value > 0.9 else 6, vertical_text_offset),
                textcoords="offset points",
                ha="right" if value > 0.9 else "left",
                va="center",
                fontsize=7.6,
                color=color,
                fontweight="semibold",
            )
    axis.set_yticks(y_positions, labels)
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.45, len(method_order) - 0.55)
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_xlabel("Recall (correct share within each reference class)")
    axis.grid(axis="x", color="#E2E2E2", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Cross-method C0 and C1 recall summary",
        x=0.025,
        y=0.985,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.text(
        0.025,
        0.925,
        (
            f"{PARTITION_LABEL}; pooled authenticated confusion counts within method.\n"
            "Each point is the correct-prediction share for its reference class."
        ),
        ha="left",
        va="top",
        fontsize=7.7,
        color="#555555",
    )
    fig.subplots_adjust(top=0.77, bottom=0.22, left=0.27, right=0.985)
    names = _save_figure_pair(
        fig,
        output_dir,
        stem="cross_method_c0_c1_recall_summary",
        title="Cross-method C0 and C1 recall summary",
    )
    plt.close(fig)
    return names


def _aggregate_curve_evidence(
    evaluations: Sequence[Evaluation],
) -> dict[str, dict[str, Any]]:
    groups = _evaluation_groups(evaluations)
    aggregated: dict[str, dict[str, Any]] = {}
    for method_id, items in groups.items():
        if items[0].source_kind != "neural" or any(
            item.curves is None for item in items
        ):
            continue
        evidence = [item.curves for item in items]
        bins = {item.bins for item in evidence if item is not None}
        if len(bins) != 1:
            raise ContractError(f"{method_id} curve-bin counts differ by seed")
        aggregated[method_id] = {
            "display_name": items[0].display_name,
            "bins": bins.pop(),
            "positive": np.sum(
                [item.positive for item in evidence if item is not None], axis=0
            ),
            "negative": np.sum(
                [item.negative for item in evidence if item is not None], axis=0
            ),
            "seed_count": len(items),
        }
    if len({int(item["bins"]) for item in aggregated.values()}) > 1:
        raise ContractError("Neural methods use different probability-histogram bins")
    return aggregated


def _cross_method_pr_rows(
    evidence: Mapping[str, Mapping[str, Any]],
    method_order: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for method_id in method_order:
        if method_id not in evidence:
            continue
        method = evidence[method_id]
        for class_id in (0, 1):
            curve = _curves_from_histograms(
                method["positive"][class_id], method["negative"][class_id]
            )
            for index, threshold in enumerate(curve["thresholds"]):
                rows.append(
                    {
                        "method_id": method_id,
                        "method": method["display_name"],
                        "class_id": class_id,
                        "class_label": CLASS_LABELS[class_id],
                        "threshold_lower_edge": float(threshold),
                        "recall": float(curve["recall"][index]),
                        "precision": float(curve["precision"][index]),
                        "cumulative_true_positive": int(
                            curve["cumulative_true_positive"][index]
                        ),
                        "cumulative_false_positive": int(
                            curve["cumulative_false_positive"][index]
                        ),
                        "positive_pixels": int(curve["positive_pixels"]),
                        "negative_pixels": int(curve["negative_pixels"]),
                        "average_precision_histogram_approximation": float(
                            curve["average_precision"]
                        ),
                        "histogram_bins": int(method["bins"]),
                        "seed_count": int(method["seed_count"]),
                        "partition": PARTITION_LABEL,
                    }
                )
    return rows


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _plot_cross_method_pr(
    output_dir: Path,
    evidence: Mapping[str, Mapping[str, Any]],
    method_order: Sequence[str],
) -> tuple[str, str]:
    plt = _configure_matplotlib()
    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v", "P")
    eligible = [method for method in method_order if method in evidence]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.55),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for class_id, axis in enumerate(axes):
        for method_index, method_id in enumerate(eligible):
            method = evidence[method_id]
            curve = _curves_from_histograms(
                method["positive"][class_id], method["negative"][class_id]
            )
            color = CLASS_PALETTE[f"C{class_id}"]
            point_count = len(curve["recall"])
            axis.plot(
                curve["recall"],
                curve["precision"],
                color=color,
                linestyle=line_styles[method_index % len(line_styles)],
                linewidth=1.8,
                marker=markers[method_index % len(markers)],
                markevery=max(1, point_count // 10),
                markersize=3.8,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                label=(
                    f"{method['display_name']} "
                    f"(AP [{LOCKED_CURVE_BINS}-bin approximation] = "
                    f"{float(curve['average_precision']):.3f})"
                ),
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Recall")
        axis.set_title(CLASS_LABELS[class_id], loc="left", fontweight="semibold")
        axis.grid(True, color="#E2E2E2", linewidth=0.6)
        axis.legend(loc="lower left", frameon=False, fontsize=7.5)
    axes[0].set_ylabel("Precision")
    fig.suptitle(
        "C0 and C1 precision–recall curves by neural method",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    bin_counts = sorted({int(item["bins"]) for item in evidence.values()})
    fig.text(
        0.01,
        0.945,
        (
            f"{PARTITION_LABEL}; fixed-bin probability counts summed across "
            f"three seeds; {bin_counts[0]}-bin score approximation"
        ),
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    pdf_name = "cross_method_c0_c1_precision_recall.pdf"
    png_name = "cross_method_c0_c1_precision_recall.png"
    fixed_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        output_dir / pdf_name,
        bbox_inches="tight",
        metadata={
            "Creator": "assemble_publication_results.py",
            "Producer": "Matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
            "Title": "C0 and C1 precision-recall curves by neural method",
            "Subject": PARTITION_LABEL,
        },
    )
    fig.savefig(
        output_dir / png_name,
        dpi=600,
        bbox_inches="tight",
        metadata={
            "Software": "assemble_publication_results.py",
            "Title": "C0 and C1 precision-recall curves by neural method",
            "Description": PARTITION_LABEL,
        },
    )
    plt.close(fig)
    return pdf_name, png_name


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("Cannot write a non-finite CSV value")
        return f"{value:.12g}"
    return str(value)


def _write_csv_rows(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _mean_sd_tex(mean: float, sample_sd: float | None) -> str:
    return f"{mean:.3f}" if sample_sd is None else f"{mean:.3f} $\\pm$ {sample_sd:.3f}"


def _optional_mean_sd_tex(value: Any, sample_sd: Any) -> str:
    return (
        r"\textemdash{}"
        if value is None
        else _mean_sd_tex(
            float(value), None if sample_sd is None else float(sample_sd)
        )
    )


def _optional_metric_tex(value: Any) -> str:
    return r"\textemdash{}" if value is None else f"{float(value):.3f}"


def _signed_tex(value: float) -> str:
    return f"{value:+.3f}"


def _contrast_tex(value: Any) -> str:
    return r"\textemdash{}" if value is None else _signed_tex(float(value))


def _write_tex_tables(
    path: Path,
    per_seed: Sequence[Mapping[str, Any]],
    per_method: Sequence[Mapping[str, Any]],
    full_area: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    selected_primary_method_id: str,
) -> None:
    paired_lookup = {
        (row["method_id"], row["metric"]): row for row in paired
    }
    if selected_primary_method_id.startswith("C2-"):
        full_area_source_sentence = (
            "The selected conditional protocol and all comparators use the "
            "fixed raw-intensity pore gate for the C2/pore boundary."
        )
    else:
        full_area_source_sentence = (
            "The neural rows use native learned three-class predictions; the "
            "classical rows use the fixed raw-intensity pore gate."
        )
    lines = [
        r"\begin{table*}",
        r"\centering",
        (
            r"\caption{C0 and C1 segmentation on the locked retrospective "
            r"evaluation partition. Neural values are seed means $\pm$ sample "
            r"standard deviations; deterministic classical comparators have one "
            r"evaluation and therefore no seed standard deviation. Contrasts are "
            r"selected primary minus comparator, so positive values favour the "
            r"selected primary; its self-contrast is intentionally omitted.}"
        ),
        r"\label{tab:locked-retrospective-c0-c1}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Method & $n$ & C0 IoU & Primary$-$method & C1 IoU & "
            r"Primary$-$method & Harmonic IoU \\"
        ),
        r"\midrule",
    ]
    for row in per_method:
        lines.append(
            " & ".join(
                (
                    _tex_escape(str(row["method"])),
                    str(row["evaluation_count"]),
                    _mean_sd_tex(
                        float(row["c0_iou_mean"]), row["c0_iou_sample_sd"]
                    ),
                    _contrast_tex(
                        row["selected_primary_minus_method_c0_iou"]
                    ),
                    _mean_sd_tex(
                        float(row["c1_iou_mean"]), row["c1_iou_sample_sd"]
                    ),
                    _contrast_tex(
                        row["selected_primary_minus_method_c1_iou"]
                    ),
                    _mean_sd_tex(
                        float(row["c0_c1_harmonic_iou_mean"]),
                        row["c0_c1_harmonic_iou_sample_sd"],
                    ),
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    lines.extend(
        [
            r"\begin{table*}",
            r"\centering",
            (
                r"\caption{Authenticated C0 and C1 overlap and classification "
                r"metrics on the locked retrospective evaluation partition. "
                r"Values are seed means $\pm$ sample standard deviations for "
                r"neural methods and single deterministic values for classical "
                r"comparators. Undefined precision is shown as an em dash.}"
            ),
            r"\label{tab:locked-retrospective-c0-c1-secondary}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r" & \multicolumn{3}{c}{C0 disconnected pore} & \multicolumn{3}{c}{C1 connected pore} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"Method & Dice & Precision & Recall & Dice & Precision & Recall \\",
            r"\midrule",
        ]
    )
    for row in per_method:
        cells = [_tex_escape(str(row["method"]))]
        for class_id in (0, 1):
            for metric in ("dice", "precision", "recall"):
                cells.append(
                    _optional_mean_sd_tex(
                        row[f"c{class_id}_{metric}_mean"],
                        row[f"c{class_id}_{metric}_sample_sd"],
                    )
                )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table*}",
            r"\centering",
            (
                r"\caption{Full-area C2 and merged-pore metrics reconstructed "
                r"from authenticated confusion counts pooled within each method. "
                + _tex_escape(full_area_source_sentence)
                + r" Undefined precision is shown as an em dash.}"
            ),
            r"\label{tab:locked-retrospective-full-area}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.8pt}",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r" & \multicolumn{4}{c}{C2 mineral} & \multicolumn{4}{c}{Pore union (C0 $\cup$ C1)} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
            r"Method & IoU & Dice & Precision & Recall & IoU & Dice & Precision & Recall \\",
            r"\midrule",
        ]
    )
    for row in full_area:
        lines.append(
            " & ".join(
                (
                    _tex_escape(str(row["method"])),
                    *(
                        _optional_metric_tex(row[f"c2_{metric}"])
                        for metric in METRIC_NAMES
                    ),
                    *(
                        _optional_metric_tex(row[f"pore_union_{metric}"])
                        for metric in METRIC_NAMES
                    ),
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table*}",
            r"\centering",
            (
                r"\caption{Per-seed neural results on the locked retrospective "
                r"evaluation partition. Contrasts are same-seed selected primary "
                r"minus comparator, so positive values favour the selected primary; "
                r"primary self-contrasts are intentionally omitted.}"
            ),
            r"\label{tab:locked-retrospective-per-seed}",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            (
                r"Method (seed) & C0 IoU & Primary$-$method & C1 IoU & "
                r"Primary$-$method \\"
            ),
            r"\midrule",
        ]
    )
    for row in per_seed:
        lines.append(
            " & ".join(
                (
                    _tex_escape(f"{row['method']} ({row['seed']})"),
                    f"{float(row['c0_iou']):.3f}",
                    _contrast_tex(
                        row["selected_primary_minus_method_c0_iou"]
                    ),
                    f"{float(row['c1_iou']):.3f}",
                    _contrast_tex(
                        row["selected_primary_minus_method_c1_iou"]
                    ),
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table*}",
            r"\centering",
            (
                r"\caption{Same-index paired whole-tile bootstrap contrasts on "
                r"the locked retrospective evaluation partition. Each contrast "
                r"is selected primary minus comparator, so positive values favour "
                r"the selected primary. Each bracket is a within-series whole-tile "
                r"sensitivity interval: a "
                f"{100 * PAIRED_BOOTSTRAP_CONFIDENCE:.0f}"
                r"\% equal-tail percentile interval from "
                f"{PAIRED_BOOTSTRAP_REPLICATES:,}"
                r" paired resamples of whole native tiles.}"
            ),
            r"\label{tab:locked-retrospective-paired-differences}",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Method & $\Delta$ C0 IoU & $\Delta$ C1 IoU & $\Delta$ harmonic IoU \\",
            r"\midrule",
        ]
    )
    for row in per_method:
        method_id = str(row["method_id"])
        if method_id == selected_primary_method_id:
            continue
        formatted = []
        for metric in ("c0_iou", "c1_iou", "c0_c1_harmonic_iou"):
            item = paired_lookup[(method_id, metric)]
            formatted.append(
                f"{float(item['difference']):+.3f} "
                f"[{float(item['ci_lower']):+.3f}, "
                f"{float(item['ci_upper']):+.3f}]"
            )
        lines.append(
            " & ".join((_tex_escape(str(row["method"])), *formatted)) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            (
                "% Parameter counts and scheduler-recorded training times are "
                "omitted because the authenticated assembler inputs contain no "
                "such fields."
            ),
            "",
        ]
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def _write_binary_snapshot(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


def _write_tex_figures(
    path: Path,
    selected_primary_method_id: str,
    qualitative_panels: Sequence[QualitativePanel],
    *,
    include_precision_recall: bool,
) -> None:
    lines: list[str] = []
    if include_precision_recall:
        lines.extend(
            [
                r"\begin{figure*}",
                r"\centering",
                r"\includegraphics[width=\textwidth]{cross_method_c0_c1_precision_recall.pdf}",
                (
                    r"\caption{C0 and C1 precision--recall curves for the selected "
                    r"primary and plain-U-Net comparator on the locked retrospective "
                    r"evaluation partition. Fixed-bin probability counts are pooled "
                    r"over the three authenticated seeds within each method; average "
                    f"precision is a {LOCKED_CURVE_BINS}-bin approximation.}}"
                ),
                r"\label{fig:locked-c0-c1-precision-recall}",
                r"\end{figure*}",
                "",
            ]
        )
    lines.extend([
        r"\begin{figure*}",
        r"\centering",
        r"\includegraphics[width=0.78\textwidth]{selected_primary_c0_c1_c2_confusion_matrix.pdf}",
        (
            r"\caption{Aggregate C0/C1/C2 confusion matrix for the selected primary "
            + _tex_escape(selected_primary_method_id)
            + r" on the locked retrospective evaluation partition. Counts are pooled "
            r"over the three authenticated seeds. Rows are reference classes and columns "
            r"are predicted classes; percentages are normalized within each reference "
            r"class, and hatching marks misclassification.}"
        ),
        r"\label{fig:locked-selected-primary-confusion}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}",
        r"\centering",
        r"\includegraphics[width=\textwidth]{cross_method_c0_c1_recall_summary.pdf}",
        (
            r"\caption{Compact cross-method C0 and C1 recall summary on the "
            r"locked retrospective evaluation partition. Each point is the correct "
            r"prediction share within its authenticated reference class, reconstructed "
            r"from confusion counts pooled within method. Red circles denote C0 "
            r"disconnected pores and open green squares denote C1 connected pores.}"
        ),
        r"\label{fig:locked-cross-method-c0-c1-recall}",
        r"\end{figure*}",
        "",
    ])
    for panel in qualitative_panels:
        lines.extend(
            [
                r"\begin{figure*}",
                r"\centering",
                rf"\includegraphics[width=\textwidth]{{{panel.output_pdf_name}}}",
                (
                    r"\caption{Evaluator-produced outcome-independent qualitative panel "
                    r"for "
                    + _tex_escape(panel.method_id)
                    + r" on the locked retrospective evaluation partition. The panel "
                    r"was copied byte-for-byte from the authenticated evaluator output; "
                    r"the assembler did not reread microscopy images, masks, or predictions. "
                    r"No scale bar or pixel size is shown because authenticated physical-scale "
                    r"evidence remains an open author gate.}"
                ),
                r"\label{fig:locked-selected-primary-qualitative}",
                r"\end{figure*}",
                "",
            ]
        )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def _output_fields() -> dict[str, tuple[str, ...]]:
    per_seed_fields = (
        "method_id",
        "method",
        "seed",
        "selected_primary_method_id",
        "contrast_direction",
        "partition",
        *tuple(
            field
            for metric in PUBLICATION_METRICS
            for field in (
                metric,
                f"selected_primary_minus_method_{metric}",
            )
        ),
    )
    per_method_fields = (
        "method_id",
        "method",
        "source_kind",
        "evaluation_count",
        "seed_count",
        "selected_primary_method_id",
        "contrast_direction",
        "partition",
        *tuple(
            field
            for metric in PUBLICATION_METRICS
            for field in (
                f"{metric}_mean",
                f"{metric}_sample_sd",
                f"selected_primary_minus_method_{metric}",
            )
        ),
    )
    paired_fields = (
        "method_id",
        "method",
        "reference_method_id",
        "contrast_direction",
        "metric",
        "difference",
        "ci_lower",
        "ci_upper",
        "finite_replicates",
        "confidence",
        "replicates",
        "bootstrap_seed",
        "tile_count",
        "sampling_unit",
        "pairing",
    )
    diagnostic_fields = (
        "method_id",
        "method",
        "source_kind",
        "seed",
        "evaluation_ordinal",
        "file_name",
        "input_sha256",
        "target_sha256",
        "total_pixels",
        "c0_reference_pixels",
        "c1_reference_pixels",
        "c2_reference_pixels",
        "c0_as_c1_pixels",
        "c1_as_c0_pixels",
        "pore_as_mineral_pixels",
        "mineral_as_pore_pixels",
        "worst_pore_iou_class",
        "reference_basis",
        "selected_primary_method_id",
        "contrast_direction",
        "partition",
        *tuple(
            field
            for metric in PUBLICATION_METRICS
            for field in (
                metric,
                f"selected_primary_minus_method_{metric}",
            )
        ),
    )
    pr_fields = (
        "method_id",
        "method",
        "class_id",
        "class_label",
        "threshold_lower_edge",
        "recall",
        "precision",
        "cumulative_true_positive",
        "cumulative_false_positive",
        "positive_pixels",
        "negative_pixels",
        "average_precision_histogram_approximation",
        "histogram_bins",
        "seed_count",
        "partition",
    )
    aggregate_confusion_fields = (
        "method_id",
        "method",
        "source_kind",
        "evaluation_count",
        "seed_count",
        "selected_primary_method_id",
        "true_class_id",
        "true_class_label",
        "predicted_class_id",
        "predicted_class_label",
        "pixel_count",
        "reference_class_pixels",
        "row_fraction",
        "outcome",
        "aggregation",
        "partition",
    )
    full_area_fields = (
        "method_id",
        "method",
        "source_kind",
        "evaluation_count",
        "seed_count",
        "selected_primary_method_id",
        "prediction_source",
        *tuple(
            f"{scope}_{metric}"
            for scope in ("c2", "pore_union")
            for metric in METRIC_NAMES
        ),
        "aggregation",
        "partition",
    )
    return {
        "per_seed": per_seed_fields,
        "per_method": per_method_fields,
        "paired": paired_fields,
        "diagnostics": diagnostic_fields,
        "pr": pr_fields,
        "aggregate_confusion": aggregate_confusion_fields,
        "full_area": full_area_fields,
    }


def _generator_runtime_versions() -> dict[str, str | None]:
    try:
        matplotlib_version: str | None = importlib_metadata.version("matplotlib")
    except importlib_metadata.PackageNotFoundError:
        matplotlib_version = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib_version,
    }


def assemble_publication_results(
    input_manifest: str | Path, output_dir: str | Path
) -> Path:
    """Validate all evidence and atomically publish deterministic tables/plots."""

    manifest_path = _resolve_non_symbolic_manifest_path(input_manifest)
    output_path = _lexical_non_symbolic_output_path(output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Output path already exists: {output_path}")
    staging = output_path.with_name(f".{output_path.name}.partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Assembly staging path already exists: {staging}")

    manifest, bootstrap, manifest_sha256 = _validate_manifest(manifest_path)
    neural_records = manifest["neural_evaluations"]
    if not isinstance(neural_records, list) or not neural_records:
        raise ContractError("neural_evaluations must be a non-empty list")
    evaluations = [
        _load_neural_evaluation(manifest_path.parent, record)
        for record in neural_records
    ]
    evaluations.extend(
        _load_classical_evaluations(
            manifest_path.parent, manifest["classical_evaluation"]
        )
    )
    selected_primary_method_id = _validate_evaluation_collection(evaluations)
    qualitative_panels = _load_qualitative_panels(
        manifest_path.parent,
        manifest["qualitative_figures"],
        evaluations,
        selected_primary_method_id,
    )

    per_seed, per_method = _build_aggregate_tables(
        evaluations, selected_primary_method_id
    )
    paired = _paired_bootstrap_differences(
        evaluations,
        bootstrap,
        per_method,
        selected_primary_method_id,
    )
    diagnostics = _per_tile_diagnostics(
        evaluations, selected_primary_method_id
    )
    method_order = _method_order(evaluations, selected_primary_method_id)
    confusion_summaries = _aggregate_method_confusions(
        evaluations, selected_primary_method_id
    )
    aggregate_confusion_rows = _aggregate_confusion_rows(
        confusion_summaries,
        method_order,
        selected_primary_method_id,
    )
    selected_confusion_rows = [
        row
        for row in aggregate_confusion_rows
        if row["method_id"] == selected_primary_method_id
    ]
    full_area_rows = _full_area_metric_rows(
        confusion_summaries,
        method_order,
        selected_primary_method_id,
    )
    curve_evidence = _aggregate_curve_evidence(evaluations)
    eligible_curve_methods = [
        method_id for method_id in method_order if method_id in curve_evidence
    ]
    plot_enabled = len(eligible_curve_methods) >= 2
    pr_rows = (
        _cross_method_pr_rows(curve_evidence, method_order)
        if plot_enabled
        else []
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()

    fields = _output_fields()
    _write_csv_rows(
        staging / "per_seed_aggregate.csv", fields["per_seed"], per_seed
    )
    _write_csv_rows(
        staging / "per_method_aggregate.csv", fields["per_method"], per_method
    )
    _write_csv_rows(
        staging / "paired_bootstrap_differences.csv", fields["paired"], paired
    )
    _write_csv_rows(
        staging / "per_tile_diagnostics.csv", fields["diagnostics"], diagnostics
    )
    _write_csv_rows(
        staging / "selected_primary_c0_c1_c2_confusion.csv",
        fields["aggregate_confusion"],
        selected_confusion_rows,
    )
    _write_csv_rows(
        staging / "per_method_c0_c1_c2_prediction_error_summary.csv",
        fields["aggregate_confusion"],
        aggregate_confusion_rows,
    )
    _write_csv_rows(
        staging / "per_method_full_area_aggregate.csv",
        fields["full_area"],
        full_area_rows,
    )
    core_figure_files = [
        *_plot_selected_primary_confusion(
            staging, confusion_summaries[selected_primary_method_id]
        ),
        *_plot_cross_method_c0_c1_recall_summary(
            staging, confusion_summaries, method_order
        ),
    ]
    plot_files: list[str] = []
    if plot_enabled:
        _write_csv_rows(
            staging / "cross_method_c0_c1_precision_recall.csv",
            fields["pr"],
            pr_rows,
        )
        plot_files.extend(_plot_cross_method_pr(staging, curve_evidence, method_order))

    qualitative_output_files: list[str] = []
    for panel in qualitative_panels:
        _write_binary_snapshot(staging / panel.output_pdf_name, panel.pdf.content)
        _write_binary_snapshot(staging / panel.output_png_name, panel.png.content)
        qualitative_output_files.extend(
            (panel.output_pdf_name, panel.output_png_name)
        )

    _write_tex_tables(
        staging / TEX_FRAGMENT_NAME,
        per_seed,
        per_method,
        full_area_rows,
        paired,
        selected_primary_method_id,
    )
    _write_tex_figures(
        staging / FIGURE_TEX_FRAGMENT_NAME,
        selected_primary_method_id,
        qualitative_panels,
        include_precision_recall=plot_enabled,
    )
    evidence_rows = []
    for item in evaluations:
        evidence_rows.append(
            {
                "method_id": item.method_id,
                "seed": item.seed,
                "source_kind": item.source_kind,
                "report_sha256": item.report_sha256,
                "artifact_sha256": dict(item.artifact_sha256),
                "source_identity": dict(item.source_identity),
                "neural_freeze_manifest_id": item.freeze_id,
                "neural_freeze_scientific_identity_sha256": (
                    item.freeze_scientific_identity_sha256
                ),
            }
        )
    expected_outputs = [
        "checksums.sha256",
        "paired_bootstrap_differences.csv",
        "per_method_aggregate.csv",
        "per_method_full_area_aggregate.csv",
        "per_seed_aggregate.csv",
        "per_tile_diagnostics.csv",
        "per_method_c0_c1_c2_prediction_error_summary.csv",
        "selected_primary_c0_c1_c2_confusion.csv",
        "publication_results.json",
        FIGURE_TEX_FRAGMENT_NAME,
        TEX_FRAGMENT_NAME,
        *core_figure_files,
        *qualitative_output_files,
    ]
    if plot_enabled:
        expected_outputs.extend(
            [
                "cross_method_c0_c1_precision_recall.csv",
                *plot_files,
            ]
        )
    report = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "artifact_kind": "deterministic_publication_results_assembly",
        "generator": {
            "schema_version": GENERATOR_SCHEMA_VERSION,
            "version": GENERATOR_VERSION,
            "source_file": Path(__file__).name,
            "source_sha256": GENERATOR_SOURCE_SHA256,
            "runtime_versions": _generator_runtime_versions(),
        },
        "partition": PARTITION_LABEL,
        "selected_primary_method_id": selected_primary_method_id,
        "contrast_direction": CONTRAST_DIRECTION,
        "class_definitions": {
            "C0": CLASS_LABELS[0],
            "C1": CLASS_LABELS[1],
            "C2": CLASS_LABELS[2],
        },
        "class_palette": CLASS_PALETTE,
        "metric_definition": {
            "per_class": (
                "Computed from authenticated confusion counts with rows as "
                "reference classes and columns as predicted classes."
            ),
            "c0_c1_harmonic_iou": (
                "2*C0_IoU*C1_IoU/(C0_IoU+C1_IoU+1e-8), matching the "
                "frozen evaluator."
            ),
            "signed_contrast": (
                "selected-primary value minus comparator value; positive "
                "values favour the selected primary; selected-primary self "
                "contrasts are undefined"
            ),
            "seed_summary": "arithmetic seed mean and sample standard deviation",
            "pore_union": (
                "C0 and C1 are merged as pore; IoU, Dice, precision, and recall "
                "are reconstructed from authenticated 3x3 confusion counts."
            ),
        },
        "paired_bootstrap": {
            **bootstrap,
            "sampling_unit": (
                f"whole native tile from the {PARTITION_LABEL}"
            ),
            "pairing": (
                "Every replicate applies one shared array of tile indices to "
                "the selected primary and comparator before computing selected "
                "primary minus comparator."
            ),
            "random_generator": "NumPy PCG64",
            "interval": "equal-tail percentile",
            "interpretation": "within-series whole-tile sensitivity interval",
        },
        "input_manifest": {
            "schema_version": INPUT_SCHEMA_VERSION,
            "sha256": manifest_sha256,
        },
        "validation_screen_r3_boundary": {
            "status": "not_assembled_from_locked_evaluations",
            "rule": (
                "R3 candidate contrasts belong to the authenticated validation "
                "screen and selected-method lock. A separate R3 evaluation "
                "on this partition is neither requested nor accepted unless R3 "
                "is itself the selected primary."
            ),
        },
        "authenticated_evidence": evidence_rows,
        "per_seed_aggregate": per_seed,
        "per_method_aggregate": per_method,
        "per_method_full_area_aggregate": full_area_rows,
        "paired_bootstrap_differences": paired,
        "per_tile_diagnostics": diagnostics,
        "aggregate_confusion_matrix": {
            "status": "created",
            "method_id": selected_primary_method_id,
            "aggregation": (
                "Authenticated aggregate confusion counts pooled across the "
                "three selected-primary seeds."
            ),
            "normalization": "rows normalized within reference class",
            "rows_are": "reference classes",
            "columns_are": "predicted classes",
            "csv": "selected_primary_c0_c1_c2_confusion.csv",
            "figures": [
                "selected_primary_c0_c1_c2_confusion_matrix.pdf",
                "selected_primary_c0_c1_c2_confusion_matrix.png",
            ],
            "cells": selected_confusion_rows,
        },
        "prediction_error_summary": {
            "status": "created",
            "scope": "non-spatial per-method confusion-derived summary",
            "reason": (
                "Every included method supplies authenticated per-tile C0/C1/C2 "
                "confusion counts. No pixel mask or probability map is reconstructed."
            ),
            "normalization": "rows normalized within reference class",
            "csv": "per_method_c0_c1_c2_prediction_error_summary.csv",
            "figures": [],
            "methods": method_order,
            "cells": aggregate_confusion_rows,
        },
        "c0_c1_recall_summary": {
            "status": "created",
            "scope": "compact cross-method C0/C1 recall comparison",
            "reason": (
                "C0 and C1 recall are correct-prediction shares within each "
                "authenticated reference class, derived from pooled confusion counts."
            ),
            "figures": [
                "cross_method_c0_c1_recall_summary.pdf",
                "cross_method_c0_c1_recall_summary.png",
            ],
            "methods": method_order,
        },
        "full_area_metrics": {
            "status": "created",
            "scope": "C2 mineral and merged-pore metrics",
            "reason": (
                "All fields are reconstructed from authenticated per-method "
                "confusion counts; no masks or predictions are reread."
            ),
            "csv": "per_method_full_area_aggregate.csv",
            "rows": full_area_rows,
        },
        "evidence_gated_omissions": {
            "parameter_count": {
                "status": "omitted",
                "reason": (
                    "The authenticated evaluator and classical-comparator schemas "
                    "contain no parameter-count field."
                ),
            },
            "training_time": {
                "status": "omitted",
                "reason": (
                    "The authenticated inputs contain no scheduler-recorded or "
                    "otherwise comparable training wall-time field."
                ),
            },
        },
        "spatial_prediction_figures": {
            "status": "not_reconstructed",
            "reason": (
                "The assembler never reads or recomposes standalone per-pixel masks, "
                "probability maps, or overlays. When supplied, the evaluator's complete "
                "qualitative PDF/PNG is preserved only by authenticated byte-for-byte copy; "
                "no spatial predictions, scale bars, timings, or metrics are fabricated."
            ),
        },
        "qualitative_figures": {
            "status": "copied" if qualitative_panels else "omitted",
            "reason": (
                "The evaluator-produced outcome-independent PDF and PNG were authenticated "
                "by the assembly manifest, cross-checked against the neural report contract, "
                "then copied byte-for-byte from their validated byte snapshots without "
                "rereading corpus data."
                if qualitative_panels
                else "No manifest-authenticated evaluator-produced qualitative PDF/PNG pair "
                "was supplied; no qualitative figure was created."
            ),
            "selection_rule": QUALITATIVE_SELECTION_RULE,
            "prespecified_training_seed": QUALITATIVE_FIGURE_SEED,
            "source_schema_limitation": (
                "The pinned neural report authenticates the qualitative tile identity, "
                "six-panel construction contract, physical-scale gate, and publication "
                "file paths, but does not embed the PDF/PNG SHA-256 values. Input schema v2 "
                "therefore requires the assembly manifest to bind both figure hashes."
            ),
            "panels": [
                {
                    "method_id": panel.method_id,
                    "training_seed": panel.seed,
                    "image_id": panel.image_id,
                    "file_name": panel.file_name,
                    "source_pdf_sha256": panel.pdf.sha256,
                    "copied_pdf_sha256": panel.pdf.sha256,
                    "source_png_sha256": panel.png.sha256,
                    "copied_png_sha256": panel.png.sha256,
                    "output_pdf": panel.output_pdf_name,
                    "output_png": panel.output_png_name,
                    "publication_figure_contract": dict(
                        panel.publication_figure_contract
                    ),
                    "copy_semantics": "byte-for-byte authenticated snapshot copy",
                }
                for panel in qualitative_panels
            ],
        },
        "precision_recall": {
            "status": "created" if plot_enabled else "omitted",
            "reason": (
                "At least two neural methods supplied authenticated fixed-bin "
                "histograms and matching precision-recall curve tables."
                if plot_enabled
                else "Fewer than two methods supplied complete authenticated "
                "fixed-bin probability evidence; no cross-method curve was drawn."
            ),
            "methods": eligible_curve_methods if plot_enabled else [],
            "construction": (
                "Counts are summed across seeds within a method and the "
                "precision-recall coordinates are reconstructed at the original "
                "bin thresholds without interpolation."
                if plot_enabled
                else None
            ),
            "average_precision_label": (
                f"AP ({LOCKED_CURVE_BINS}-bin approximation)"
                if plot_enabled
                else None
            ),
            "palette": {"C0": CLASS_PALETTE["C0"], "C1": CLASS_PALETTE["C1"]},
            "figure": (
                "cross_method_c0_c1_precision_recall.pdf"
                if plot_enabled
                else None
            ),
        },
        "outputs": sorted(expected_outputs),
    }
    _write_json(staging / "publication_results.json", report)

    output_files = sorted(
        path for path in staging.iterdir() if path.is_file()
    )
    with (staging / "checksums.sha256").open(
        "x", encoding="ascii", newline="\n"
    ) as handle:
        for path in output_files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    observed_outputs = sorted(path.name for path in staging.iterdir() if path.is_file())
    if observed_outputs != sorted(expected_outputs):
        raise RuntimeError(
            f"Internal output inventory mismatch: {observed_outputs!r}"
        )
    _publish_directory_no_replace(staging, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble authenticated neural and classical results into "
            "publication tables, confusion-derived figures, authenticated "
            "qualitative-panel copies, and evidence-gated C0/C1 PR plots"
        )
    )
    parser.add_argument(
        "--input-manifest",
        required=True,
        help="Strict JSON manifest binding every input file to a SHA-256 and identity",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory; existing paths are never overwritten",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = assemble_publication_results(
        args.input_manifest, args.output_dir
    )
    print(f"Wrote publication results assembly to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exactly-once held-out evaluation of the frozen B0/B1/B2 comparators.

This path cannot influence neural selection. It accepts only canonical,
content-addressed classical-fit and neural-freeze identifiers. All data, lock,
model, output, metric, bootstrap, and runtime choices are derived from their
authenticated artifacts and source-controlled constants. No validation image
or mask is opened. The output reservation is created before any held-out image
or mask byte is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_confirmatory_checkpoint import (  # noqa: E402
    CLASS_LABELS,
    LOCKED_BOOTSTRAP_REPLICATES,
    LOCKED_BOOTSTRAP_SEED,
    LOCKED_CONFIDENCE,
    LOCKED_INPUT_ATTESTATIONS,
    LOCKED_RETROSPECTIVE_PARTITION_LABEL,
    LOCKED_SPLIT_MANIFEST_SHA256,
    LOCKED_TARGET_ATTESTATIONS,
    PUBLICATION_CLASS_COLORS,
    PUBLICATION_SANS_SERIF_FONTS,
    WITHIN_SERIES_WHOLE_TILE_INTERVAL_LABEL,
    aggregate_secondary_2d_diagnostics,
    bootstrap_secondary_2d_diagnostics,
    confusion_from_labels,
    gate_reference_diagnostic,
    load_lossless_target_mask_bytes,
    metrics_from_confusion,
    plot_confusion_matrix,
    secondary_2d_operational_diagnostics,
    summarize_gate_reference_confusion,
    write_gate_reference_diagnostics,
    write_secondary_2d_diagnostic_tables,
)
from scripts.fit_classical_comparators import (  # noqa: E402
    CANONICAL_TRAIN_FILENAMES,
    CANONICAL_TRAIN_FILENAME_LIST_SHA256,
    CLASSICAL_LOCK_SCHEMA_VERSION,
    PROTOCOL_ID,
    SOURCE_PATHS,
    _library_versions,
    classical_scientific_identity_sha256,
    frozen_classical_lock_static_contract,
    frozen_one_pass_evaluator_contract,
    validate_campaign_id as validate_fit_campaign_id,
    validate_fit_id,
)
from src.classical.comparators import (  # noqa: E402
    B0_AREA_CUTOFFS,
    B1_AREA_CUTOFFS,
    B1_MARKER_MIN_DISTANCE_PX,
    B1_MARKER_THRESHOLD_ABS_PX,
    CLASSICAL_COMPARATOR_IDS,
    EXTRA_TREES_CONFIG,
    EXTRA_TREES_FEATURE_NAMES,
    FrozenExtraTreesPredictor,
    PORE_THRESHOLD_UINT8,
    balanced_pore_score,
    extra_trees_numeric_semantic_sha256,
    load_extra_trees_numeric,
    predict_b0_small_components,
    predict_b1_marker_watershed,
    predict_b2_extra_trees,
    pore_metrics_from_confusion,
)
from src.training.data_contract import (  # noqa: E402
    CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS,
    CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS,
)
from src.training.neural_freeze import (  # noqa: E402
    load_verified_neural_freeze_manifest,
    validate_manifest_id as validate_neural_freeze_manifest_id,
)


EVALUATOR_SCHEMA_VERSION = 1
EXPECTED_TILE_SHAPE = (2048, 2048)
CANONICAL_LOCK_ROOT = Path("results/classical_comparators/locked")
CANONICAL_OUTPUT_ROOT = Path("results/classical_evaluation/locked")
CANONICAL_IMAGE_DIR = Path("original_images")
CANONICAL_MASK_DIR = Path(
    "results/step2_pore_classification/pore_classifications"
)
CANONICAL_SPLIT_MANIFEST = Path("config/confirmatory_splits.json")
CANONICAL_LOCK_FILENAME = "classical_comparator_lock.json"
CANONICAL_B2_MODEL_FILENAME = "B2_extra_trees.npz"
EXPECTED_TEST_SERIES = ("pdo2_24",)
EXPECTED_SPLIT_COUNTS = {"train": 74, "val": 5, "test": 21}
EXPECTED_TRAIN_GROUP_COUNTS = {
    "pdo1_12": 32,
    "pdo1_7": 39,
    "pdo4_1_140721": 1,
    "pdo4_2_151020": 2,
}
EXPECTED_TEST_INPUT = LOCKED_INPUT_ATTESTATIONS["held_out_test"]
EXPECTED_TEST_TARGET = LOCKED_TARGET_ATTESTATIONS["held_out_test"]
EXPECTED_SLURM_PARTITION = "nodes"
HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CLASS_NAMES = {
    0: "disconnected_pore",
    1: "connected_pore",
    2: "mineral",
}
METRIC_NAMES = ("iou", "dice", "precision", "recall")
RUNTIME_VERSION_KEYS = {
    "python",
    "numpy",
    "pillow",
    "scipy",
    "scikit_image",
    "scikit_learn",
    "opencv",
    "matplotlib",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path, *, label: str) -> Path:
    root_resolved = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{label} escapes its canonical root: {path}") from error
    return resolved


def _repository_relative(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    return _inside(project_root, path, label="Path").relative_to(
        project_root.resolve()
    ).as_posix()


def _canonical_lexical_path(
    root: Path, relative: Path, *, label: str
) -> Path:
    repository_root = root.resolve()
    lexical = Path(os.path.abspath(repository_root / relative))
    try:
        parts = lexical.relative_to(repository_root).parts
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    cursor = repository_root
    for component in parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{label} contains a symbolic-link component")
    return lexical


def canonical_fit_paths(
    fit_id: str, *, project_root: Path = PROJECT_ROOT
) -> Dict[str, Path]:
    canonical_id = validate_fit_id(fit_id)
    root = project_root.resolve()
    lock_dir = _canonical_lexical_path(
        root,
        CANONICAL_LOCK_ROOT / canonical_id,
        label="Classical lock directory",
    )
    return {
        "lock_dir": lock_dir,
        "lock": lock_dir / CANONICAL_LOCK_FILENAME,
        "model": lock_dir / CANONICAL_B2_MODEL_FILENAME,
        "output_fit_root": _canonical_lexical_path(
            root,
            CANONICAL_OUTPUT_ROOT / canonical_id,
            label="Classical output fit root",
        ),
        "image_dir": _canonical_lexical_path(
            root, CANONICAL_IMAGE_DIR, label="Canonical image directory"
        ),
        "mask_dir": _canonical_lexical_path(
            root, CANONICAL_MASK_DIR, label="Canonical mask directory"
        ),
        "manifest": _canonical_lexical_path(
            root,
            CANONICAL_SPLIT_MANIFEST,
            label="Canonical split manifest",
        ),
    }


def _require_canonical_directory(
    path: Path, *, project_root: Path, label: str
) -> Path:
    """Require an existing real directory with no symlinked path component."""

    root = project_root.resolve()
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{label} contains a symbolic-link component")
    if not lexical.is_dir():
        raise ValueError(f"{label} is missing or is not a directory")
    return lexical


def _safe_named_file(root: Path, name: str, *, label: str) -> Path:
    relative = Path(str(name))
    if (
        not str(name)
        or relative == Path(".")
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in str(name)
    ):
        raise ValueError(f"Unsafe {label} filename: {name!r}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link: {name}")
    resolved = _inside(root, candidate, label=label)
    return resolved


def _load_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest_metadata(
    path: Path,
    *,
    expected_sha256: str = LOCKED_SPLIT_MANIFEST_SHA256,
) -> Dict[str, Any]:
    """Authenticate split metadata without resolving or opening any corpus file."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("Canonical split manifest is missing or is a symbolic link")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("Canonical split manifest SHA-256 mismatch")
    manifest = _load_json_object(path)
    normalized: Dict[str, list[tuple[str, Any]]] = {}
    for split_name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        values = manifest.get(split_name)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(
                f"Canonical split '{split_name}' must contain {expected_count} identifiers"
            )
        items = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError(f"Invalid identifier in split '{split_name}'")
            identity = (
                ("id", int(value))
                if isinstance(value, int)
                or (isinstance(value, str) and value.isdigit())
                else ("name", str(value))
            )
            items.append(identity)
        if len(items) != len(set(items)):
            raise ValueError(f"Duplicate identifier in split '{split_name}'")
        normalized[split_name] = items
    memberships: Dict[tuple[str, Any], list[str]] = {}
    for split_name, identities in normalized.items():
        for identity in identities:
            memberships.setdefault(identity, []).append(split_name)
    if any(len(splits) != 1 for splits in memberships.values()):
        raise ValueError("Canonical split manifest contains overlap")

    provenance = manifest.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Canonical split manifest lacks provenance")
    if tuple(provenance.get("test_series", ())) != EXPECTED_TEST_SERIES:
        raise ValueError("Canonical test series must be exactly pdo2_24")
    if set(provenance.get("train_series", ())) != set(EXPECTED_TRAIN_GROUP_COUNTS):
        raise ValueError("Canonical training-series provenance drifted")
    if len(provenance.get("validation_series", ())) != 1:
        raise ValueError("Canonical validation-series provenance drifted")
    return {
        "sha256": observed_sha256,
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "test_identifiers": list(manifest["test"]),
        "provenance": dict(provenance),
        "validation_corpus_bytes_read": 0,
        "held_out_corpus_bytes_read": 0,
    }


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if not HEX_SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{label} is not a non-null SHA-256 digest")
    return text


def _expected_source_paths() -> tuple[str, ...]:
    return tuple(str(value) for value in SOURCE_PATHS)


def _validate_source_hashes(
    record: Any,
    *,
    project_root: Path,
    source_paths: Sequence[str],
) -> Dict[str, str]:
    if not isinstance(record, Mapping) or set(record) != set(source_paths):
        raise ValueError("Classical lock source-path set is incomplete or unexpected")
    verified: Dict[str, str] = {}
    for name in source_paths:
        expected = _require_sha256(record[name], label=f"Source hash for {name}")
        path = _safe_named_file(project_root, name, label="Lock source")
        path = _canonical_lexical_path(
            project_root, Path(name), label="Lock source"
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Lock source is missing or symbolic: {name}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Classical lock source SHA-256 mismatch: {name}")
        verified[name] = observed
    return verified


def _exact_json_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float aliases."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _exact_json_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _exact_json_equal(left, right)
            for left, right in zip(observed, expected)
        )
    return observed == expected


def _assert_exact_mapping(
    observed: Any, expected: Mapping[str, Any], *, label: str
) -> None:
    if not isinstance(observed, dict) or not _exact_json_equal(observed, dict(expected)):
        raise ValueError(f"{label} drifted from the frozen protocol")


def _validate_area_selection(
    record: Any, cutoffs: Sequence[int], *, label: str
) -> int:
    if not isinstance(record, Mapping) or set(record) != {
        "selection_scope",
        "primary",
        "tie_break_1",
        "tie_break_2",
        "selected_area_cutoff_px",
        "candidate_group_summaries",
        "candidate_summaries",
    }:
        raise ValueError(f"{label} train-only selection schema is invalid")
    expected_metadata = {
        "selection_scope": "canonical_training_groups_only",
        "primary": "mean_group_balanced_pore_iou",
        "tie_break_1": "higher_minimum_of_mean_group_iou_c0_and_iou_c1",
        "tie_break_2": "lower_area_cutoff_px",
    }
    if any(record.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"{label} train-only selection rule drifted")
    groups = sorted(EXPECTED_TRAIN_GROUP_COUNTS)
    group_summaries = record.get("candidate_group_summaries")
    if not isinstance(group_summaries, Mapping) or set(group_summaries) != {
        str(value) for value in cutoffs
    }:
        raise ValueError(f"{label} per-series cutoff evidence is incomplete")
    recomputed_means: Dict[int, Dict[str, float]] = {}
    metric_keys = ("iou_c0", "iou_c1", "balanced_pore_iou")
    for cutoff in cutoffs:
        cutoff_groups = group_summaries[str(cutoff)]
        if not isinstance(cutoff_groups, Mapping) or set(cutoff_groups) != set(groups):
            raise ValueError(f"{label} per-series cutoff evidence is incomplete")
        normalized_groups: Dict[str, Dict[str, float]] = {}
        for group in groups:
            summary = cutoff_groups[group]
            if not isinstance(summary, Mapping) or set(summary) != set(metric_keys):
                raise ValueError(f"{label} per-series cutoff schema is invalid")
            if any(
                isinstance(summary[key], bool)
                or not isinstance(summary[key], (int, float))
                for key in metric_keys
            ):
                raise ValueError(f"{label} per-series cutoff metric is non-numeric")
            values = {key: float(summary[key]) for key in metric_keys}
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in values.values()
            ):
                raise ValueError(
                    f"{label} per-series cutoff metric is non-finite or out of range"
                )
            expected_balanced = balanced_pore_score(
                values["iou_c0"], values["iou_c1"]
            )
            if not math.isclose(
                values["balanced_pore_iou"],
                expected_balanced,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{label} per-series cutoff harmonic score is inconsistent"
                )
            normalized_groups[group] = values
        recomputed_means[int(cutoff)] = {
            key: float(
                np.mean([normalized_groups[group][key] for group in groups])
            )
            for key in metric_keys
        }

    summaries = record.get("candidate_summaries")
    if not isinstance(summaries, Mapping) or set(summaries) != {
        str(value) for value in cutoffs
    }:
        raise ValueError(f"{label} train-only candidate summaries are incomplete")
    normalized: Dict[int, Dict[str, float]] = {}
    for cutoff in cutoffs:
        summary = summaries[str(cutoff)]
        if not isinstance(summary, Mapping) or set(summary) != set(metric_keys):
            raise ValueError(f"{label} cutoff summary schema is invalid")
        if any(
            isinstance(summary[key], bool)
            or not isinstance(summary[key], (int, float))
            for key in summary
        ):
            raise ValueError(f"{label} cutoff summary is non-numeric")
        values = {key: float(summary[key]) for key in summary}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError(f"{label} cutoff summary is non-finite or out of range")
        if any(
            not math.isclose(
                values[key],
                recomputed_means[int(cutoff)][key],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in metric_keys
        ):
            raise ValueError(f"{label} cutoff mean metrics are inconsistent")
        normalized[int(cutoff)] = values
    selected = sorted(
        normalized,
        key=lambda cutoff: (
            -normalized[cutoff]["balanced_pore_iou"],
            -min(normalized[cutoff]["iou_c0"], normalized[cutoff]["iou_c1"]),
            int(cutoff),
        ),
    )[0]
    selected_record = record.get("selected_area_cutoff_px")
    if (
        isinstance(selected_record, bool)
        or not isinstance(selected_record, int)
        or selected_record != selected
    ):
        raise ValueError(f"{label} train-only selection record chose the wrong cutoff")
    return int(selected)


def _validate_b2_training_records(record: Mapping[str, Any]) -> None:
    sampling_seed = record.get("sampling_seed")
    sample_limit = record.get("samples_per_class_per_tile")
    if (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or sampling_seed != 20260821
        or isinstance(sample_limit, bool)
        or not isinstance(sample_limit, int)
        or sample_limit != 4096
    ):
        raise ValueError("B2 deterministic sampling contract drifted")
    groups = sorted(EXPECTED_TRAIN_GROUP_COUNTS)
    group_cv = record.get("group_cv")
    if not isinstance(group_cv, Mapping) or set(group_cv) != {
        "scope",
        "evaluation",
        "folds",
        "mean_group_metrics",
    }:
        raise ValueError("B2 grouped-CV record is incomplete")
    if (
        group_cv.get("scope") != "leave_one_canonical_training_series_out"
        or group_cv.get("evaluation")
        != "deterministic_stratified_weighted_sample"
    ):
        raise ValueError("B2 grouped-CV protocol drifted")
    folds = group_cv.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != set(groups):
        raise ValueError("B2 grouped-CV folds are incomplete")
    recomputed = {}
    for held_out_group in groups:
        fold = folds[held_out_group]
        if not isinstance(fold, Mapping) or set(fold) != {
            "held_out_scope",
            "fit_groups",
            "sampled_evaluation_pixels",
            "weighted_confusion",
            "iou_c0",
            "iou_c1",
            "balanced_pore_iou",
        }:
            raise ValueError("B2 grouped-CV fold schema is invalid")
        if (
            fold.get("held_out_scope") != "training_series_only"
            or fold.get("fit_groups")
            != [group for group in groups if group != held_out_group]
            or isinstance(fold.get("sampled_evaluation_pixels"), bool)
            or not isinstance(fold.get("sampled_evaluation_pixels"), int)
            or fold.get("sampled_evaluation_pixels", 0) <= 0
        ):
            raise ValueError("B2 grouped-CV fit/held-out group isolation drifted")
        confusion_record = fold.get("weighted_confusion")
        if (
            not isinstance(confusion_record, list)
            or len(confusion_record) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in confusion_record)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for row in confusion_record
                for value in row
            )
        ):
            raise ValueError("B2 grouped-CV confusion is invalid")
        confusion = np.asarray(confusion_record, dtype=np.float64)
        if (
            confusion.shape != (3, 3)
            or np.any(~np.isfinite(confusion))
            or np.any(confusion < 0)
        ):
            raise ValueError("B2 grouped-CV confusion is invalid")
        metrics = pore_metrics_from_confusion(confusion)
        for key, expected in metrics.items():
            value = fold.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("B2 grouped-CV metric is invalid")
            observed = float(value)
            if not math.isfinite(observed) or not math.isclose(
                observed, expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("B2 grouped-CV metric is inconsistent")
        recomputed[held_out_group] = metrics
    expected_mean = {
        key: float(np.mean([recomputed[group][key] for group in groups]))
        for key in ("iou_c0", "iou_c1", "balanced_pore_iou")
    }
    mean_record = group_cv.get("mean_group_metrics")
    if not isinstance(mean_record, Mapping) or set(mean_record) != set(expected_mean):
        raise ValueError("B2 grouped-CV mean metric schema is invalid")
    if any(
        isinstance(mean_record[key], bool)
        or not isinstance(mean_record[key], (int, float))
        or not math.isfinite(float(mean_record[key]))
        or not math.isclose(
            float(mean_record[key]), expected, rel_tol=0.0, abs_tol=1e-12
        )
        for key, expected in expected_mean.items()
    ):
        raise ValueError("B2 grouped-CV mean metrics are inconsistent")

    final_fit = record.get("final_fit")
    if not isinstance(final_fit, Mapping) or set(final_fit) != {
        "training_sample_count",
        "training_sample_class_counts",
    }:
        raise ValueError("B2 final-fit sample record is incomplete")
    counts = final_fit.get("training_sample_class_counts")
    if not isinstance(counts, Mapping) or set(counts) != {"0", "1"}:
        raise ValueError("B2 final-fit class-count record is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise ValueError("B2 final fit must contain both pore classes")
    total = final_fit.get("training_sample_count")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total != sum(counts.values())
        or total > 74 * 2 * 4096
    ):
        raise ValueError("B2 final-fit sample total is inconsistent")


def verify_classical_lock_document(
    lock: Mapping[str, Any],
    *,
    fit_id: str,
    lock_dir: Path,
    project_root: Path = PROJECT_ROOT,
    source_paths: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Verify the train-only lock and return held-out-safe execution fields."""

    canonical_fit_id = validate_fit_id(fit_id)
    expected_top_level = {
        "schema_version",
        "protocol_id",
        "fit_id",
        "scientific_identity_sha256",
        "campaign_id",
        "status",
        "created_utc",
        "scientific_identity",
        "selection_relationship",
        "data_access",
        "data_provenance",
        "inference_contract",
        "comparators",
        "one_pass_evaluator_contract",
        "reported_selection_metrics",
        "source_code_sha256",
        "environment_versions",
    }
    if set(lock) != expected_top_level:
        raise ValueError("Classical lock top-level schema is incomplete or unexpected")
    if (
        isinstance(lock.get("schema_version"), bool)
        or not isinstance(lock.get("schema_version"), int)
        or lock.get("schema_version") != CLASSICAL_LOCK_SCHEMA_VERSION
        or lock.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError(
            "Classical lock schema/protocol mismatch; a train-only refit is required"
        )
    static = frozen_classical_lock_static_contract()
    embedded_identity = _require_sha256(
        lock.get("scientific_identity_sha256"),
        label="Classical scientific fit identity",
    )
    recomputed_identity = classical_scientific_identity_sha256(lock)
    if (
        embedded_identity != recomputed_identity
        or lock.get("fit_id") != canonical_fit_id
        or canonical_fit_id != f"classical-fit-{embedded_identity[:16]}"
    ):
        raise ValueError("Classical content-addressed fit identity mismatch")
    fit_execution_campaign = lock.get("campaign_id")
    try:
        if not isinstance(fit_execution_campaign, str):
            raise ValueError("campaign must be a string")
        validate_fit_campaign_id(fit_execution_campaign)
    except ValueError as error:
        raise ValueError("Classical fit execution campaign is invalid") from error
    if lock.get("status") != static["status"]:
        raise ValueError("Classical lock is not in the frozen pre-evaluation state")
    if lock.get("scientific_identity") != static["scientific_identity"]:
        raise ValueError("Classical scientific-identity description drifted")
    created_utc = lock.get("created_utc")
    try:
        parsed_created = datetime.fromisoformat(created_utc)
    except (TypeError, ValueError) as error:
        raise ValueError("Classical fit creation timestamp is invalid") from error
    if parsed_created.utcoffset() != timezone.utc.utcoffset(parsed_created):
        raise ValueError("Classical fit creation timestamp must be UTC")
    _assert_exact_mapping(
        lock.get("selection_relationship"),
        static["selection_relationship"],
        label="Classical selection relationship",
    )

    access = lock.get("data_access")
    _assert_exact_mapping(
        access, static["data_access"], label="Classical train-only data access"
    )

    provenance = lock.get("data_provenance")
    expected_provenance_fields = {
        "split_manifest_sha256",
        "training_input",
        "training_target",
        "canonical_training_filename_count",
        "canonical_training_filenames",
        "training_file_name_list_sha256",
        "training_sampling_ids_by_filename",
        "training_group_tile_counts",
        "training_sample_key",
        "source_mask_to_canonical",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance_fields:
        raise ValueError("Classical lock lacks data provenance")
    if provenance.get("split_manifest_sha256") != LOCKED_SPLIT_MANIFEST_SHA256:
        raise ValueError("Classical lock split-manifest SHA-256 mismatch")
    training_input = provenance.get("training_input")
    training_target = provenance.get("training_target")
    if not isinstance(training_input, dict) or not isinstance(training_target, dict):
        raise ValueError("Classical lock lacks training input/target attestations")
    expected_input = CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS["train"]
    expected_target = CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS["train"]
    names = provenance.get("canonical_training_filenames")
    if (
        not isinstance(names, list)
        or len(names) != 74
        or names != sorted(names)
        or len(names) != len(set(names))
    ):
        raise ValueError("Classical lock canonical training filenames are invalid")
    if tuple(names) != CANONICAL_TRAIN_FILENAMES:
        raise ValueError("Classical lock canonical training filename set drifted")
    filename_count = provenance.get("canonical_training_filename_count")
    if (
        isinstance(filename_count, bool)
        or not isinstance(filename_count, int)
        or filename_count != len(names)
    ):
        raise ValueError("Classical lock training filename count mismatch")
    name_list_sha = hashlib.sha256(
        json.dumps(names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_training_input = {
        "scope": "canonical_training_images_only",
        "split_names": ["train"],
        "image_count": int(expected_input["image_count"]),
        "image_aggregate_sha256": expected_input["image_aggregate_sha256"],
        "image_aggregate_sha256_algorithm": (
            "sha256 over lexicographically sorted UTF-8 relative filename, "
            "NUL, raw file bytes, NUL"
        ),
        "file_name_list_sha256": name_list_sha,
    }
    expected_training_target = {
        "target_source": "lossless_png_masks",
        "mask_directory": CANONICAL_MASK_DIR.as_posix(),
        "mask_count": int(expected_target["mask_count"]),
        "mask_aggregate_sha256": expected_target["mask_aggregate_sha256"],
        "mask_aggregate_sha256_algorithm": (
            "sha256 over lexicographically sorted UTF-8 relative filename, NUL, "
            "raw file bytes, NUL"
        ),
        "validated_source_values": [0, 1, 255],
        "canonical_value_mapping": {
            "0": "0 (disconnected_pore)",
            "1": "1 (connected_pore)",
            "255": "2 (mineral) in three-class mode; ignore_index in two-class mode",
        },
    }
    _assert_exact_mapping(
        training_input,
        expected_training_input,
        label="Classical training-image attestation",
    )
    _assert_exact_mapping(
        training_target,
        expected_training_target,
        label="Classical training-mask attestation",
    )
    if (
        provenance.get("training_file_name_list_sha256") != name_list_sha
        or name_list_sha != CANONICAL_TRAIN_FILENAME_LIST_SHA256
    ):
        raise ValueError("Classical lock training filename-list SHA-256 mismatch")
    sampling_ids = provenance.get("training_sampling_ids_by_filename")
    expected_ids = {name: index for index, name in enumerate(names, start=1)}
    if not _exact_json_equal(sampling_ids, expected_ids):
        raise ValueError("Classical lock training sampling-ID map drifted")
    _assert_exact_mapping(
        provenance.get("training_group_tile_counts"),
        EXPECTED_TRAIN_GROUP_COUNTS,
        label="Classical training group counts",
    )
    if provenance.get("training_sample_key") != static["training_sample_key"]:
        raise ValueError("Classical training sample-key description drifted")
    _assert_exact_mapping(
        provenance.get("source_mask_to_canonical"),
        {"0": 0, "1": 1, "255": 2},
        label="Classical source-mask mapping",
    )

    inference = lock.get("inference_contract")
    _assert_exact_mapping(
        inference, static["inference_contract"], label="Classical inference contract"
    )

    comparators = lock.get("comparators")
    if not isinstance(comparators, Mapping) or tuple(comparators) != tuple(
        CLASSICAL_COMPARATOR_IDS
    ):
        raise ValueError("Classical lock comparator order/identity drifted")
    b0 = comparators["B0_small_components"]
    b1 = comparators["B1_marker_watershed"]
    b2 = comparators["B2_extra_trees"]
    if not isinstance(b0, dict) or set(b0) != {
        *static["b0_definition"],
        "candidate_area_cutoffs_px",
        "selection",
        "frozen_area_cutoff_px",
    }:
        raise ValueError("B0 comparator schema drifted")
    for key, expected in static["b0_definition"].items():
        if b0.get(key) != expected:
            raise ValueError("B0 comparator definition drifted")
    if not _exact_json_equal(
        b0.get("candidate_area_cutoffs_px"), list(B0_AREA_CUTOFFS)
    ):
        raise ValueError("B0 cutoff grid drifted")
    b0_cutoff = b0.get("frozen_area_cutoff_px")
    if (
        isinstance(b0_cutoff, bool)
        or not isinstance(b0_cutoff, int)
        or b0_cutoff not in B0_AREA_CUTOFFS
    ):
        raise ValueError("B0 frozen cutoff is outside the prespecified grid")
    if _validate_area_selection(
        b0.get("selection"), B0_AREA_CUTOFFS, label="B0"
    ) != b0_cutoff:
        raise ValueError("B0 train-only selection record does not support its cutoff")
    if not isinstance(b1, dict) or set(b1) != {
        *static["b1_definition"],
        "candidate_area_cutoffs_px",
        "selection",
        "frozen_area_cutoff_px",
    }:
        raise ValueError("B1 comparator schema drifted")
    for key, expected in static["b1_definition"].items():
        if not _exact_json_equal(b1.get(key), expected):
            raise ValueError("B1 comparator definition drifted")
    if not _exact_json_equal(
        b1.get("candidate_area_cutoffs_px"), list(B1_AREA_CUTOFFS)
    ):
        raise ValueError("B1 cutoff grid drifted")
    b1_cutoff = b1.get("frozen_area_cutoff_px")
    if (
        isinstance(b1_cutoff, bool)
        or not isinstance(b1_cutoff, int)
        or b1_cutoff not in B1_AREA_CUTOFFS
    ):
        raise ValueError("B1 frozen cutoff is outside the prespecified grid")
    if _validate_area_selection(
        b1.get("selection"), B1_AREA_CUTOFFS, label="B1"
    ) != b1_cutoff:
        raise ValueError("B1 train-only selection record does not support its cutoff")
    if not isinstance(b2, dict) or set(b2) != {
        "estimator_config",
        "feature_names",
        "sampling_seed",
        "samples_per_class_per_tile",
        "group_cv",
        "final_fit",
        "model_file",
        "model_sha256",
        "model_semantic_sha256",
    }:
        raise ValueError("B2 comparator schema drifted")
    if not _exact_json_equal(b2.get("estimator_config"), EXTRA_TREES_CONFIG):
        raise ValueError("B2 estimator configuration drifted")
    if b2.get("feature_names") != list(EXTRA_TREES_FEATURE_NAMES):
        raise ValueError("B2 feature contract drifted")
    _validate_b2_training_records(b2)
    if b2.get("model_file") != CANONICAL_B2_MODEL_FILENAME:
        raise ValueError("B2 model filename is not canonical")
    model_sha = _require_sha256(b2.get("model_sha256"), label="B2 model hash")
    model_semantic_sha = _require_sha256(
        b2.get("model_semantic_sha256"), label="B2 semantic model hash"
    )
    model_path = _safe_named_file(
        lock_dir, str(b2["model_file"]), label="B2 model artifact"
    )
    if model_path.parent != lock_dir.resolve():
        raise ValueError("B2 model artifact must be directly inside the lock directory")

    evaluator_contract = lock.get("one_pass_evaluator_contract")
    _assert_exact_mapping(
        evaluator_contract,
        frozen_one_pass_evaluator_contract(),
        label="Classical one-pass evaluator contract",
    )
    if lock.get("reported_selection_metrics") != static["reported_selection_metrics"]:
        raise ValueError("Classical reported-selection-metric description drifted")

    environment_versions = lock.get("environment_versions")
    if (
        not isinstance(environment_versions, Mapping)
        or set(environment_versions) != RUNTIME_VERSION_KEYS
        or any(
            not isinstance(value, str) or not value
            for value in environment_versions.values()
        )
    ):
        raise ValueError("Classical lock runtime-version attestation is incomplete")

    verified_sources = _validate_source_hashes(
        lock.get("source_code_sha256"),
        project_root=project_root,
        source_paths=tuple(source_paths or _expected_source_paths()),
    )
    return {
        "fit_id": canonical_fit_id,
        "fit_execution_campaign_id": fit_execution_campaign,
        "b0_area_cutoff_px": int(b0["frozen_area_cutoff_px"]),
        "b1_area_cutoff_px": int(b1["frozen_area_cutoff_px"]),
        "b2_model_path": model_path,
        "b2_model_sha256": model_sha,
        "b2_model_semantic_sha256": model_semantic_sha,
        "source_code_sha256": verified_sources,
        "environment_versions": dict(environment_versions),
        "canonical_lock_identity_sha256": embedded_identity,
    }


def load_verified_classical_lock(
    fit_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    source_paths: Sequence[str] | None = None,
) -> Dict[str, Any]:
    paths = canonical_fit_paths(fit_id, project_root=project_root)
    _require_canonical_directory(
        paths["lock_dir"],
        project_root=project_root,
        label="Canonical classical fit directory",
    )
    lock_path = paths["lock"]
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Canonical classical lock is missing or symbolic")
    raw_sha256 = sha256_file(lock_path)
    lock = _load_json_object(lock_path)
    verified = verify_classical_lock_document(
        lock,
        fit_id=fit_id,
        lock_dir=paths["lock_dir"],
        project_root=project_root,
        source_paths=source_paths,
    )
    verified.update(
        {
            "document": lock,
            "lock_path": lock_path,
            "lock_file_sha256": raw_sha256,
            "paths": paths,
            "project_root": project_root.resolve(),
        }
    )
    return verified


def verify_model_artifact_hash(path: Path, expected_sha256: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Canonical B2 model is missing or symbolic")
    observed = sha256_file(path)
    if observed != _require_sha256(expected_sha256, label="B2 model hash"):
        raise ValueError("B2 numeric-model SHA-256 mismatch")
    return observed


def load_verified_b2_estimator(
    path: Path,
    expected_sha256: str,
    expected_semantic_sha256: str,
    *,
    loader: Callable[[Path], Any] | None = None,
    semantic_digest_loader: Callable[[Any], str] | None = None,
) -> Any:
    verify_model_artifact_hash(path, expected_sha256)
    estimator = (loader or load_extra_trees_numeric)(path)
    if type(estimator) is not FrozenExtraTreesPredictor:
        raise ValueError("B2 artifact is not the validated numeric predictor")
    if any(
        estimator.parameters.get(key) != expected
        for key, expected in EXTRA_TREES_CONFIG.items()
    ):
        raise ValueError("B2 numeric estimator parameter mismatch")
    synthetic = np.zeros((2, len(EXTRA_TREES_FEATURE_NAMES)), dtype=np.float32)
    output = np.asarray(estimator.predict(synthetic), dtype=np.uint8)
    if output.shape != (2,) or not set(int(value) for value in output).issubset({0, 1}):
        raise ValueError("B2 synthetic prediction preflight failed")
    observed_semantic_sha256 = (
        semantic_digest_loader or extra_trees_numeric_semantic_sha256
    )(estimator)
    if observed_semantic_sha256 != _require_sha256(
        expected_semantic_sha256, label="B2 semantic model hash"
    ):
        raise ValueError("B2 semantic model SHA-256 mismatch")
    return estimator


def require_active_nodes_allocation(environment: Mapping[str, str] | None = None) -> None:
    env = os.environ if environment is None else environment
    if not env.get("SLURM_JOB_ID"):
        raise RuntimeError("Classical held-out evaluation requires an active Slurm allocation")
    if env.get("SLURM_JOB_PARTITION") != EXPECTED_SLURM_PARTITION:
        raise RuntimeError("Classical held-out evaluation requires the nodes partition")
    if "SLURM_ARRAY_TASK_ID" in env:
        raise RuntimeError("Classical evaluation is one invocation, not a Slurm array")


def preflight_campaign(
    fit_id: str,
    neural_freeze_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    source_paths: Sequence[str] | None = None,
    model_loader: Callable[[Path], Any] | None = None,
    runtime_versions_loader: Callable[[], Mapping[str, str]] | None = None,
    neural_freeze_loader: Callable[[str, Path], Mapping[str, Any]] | None = None,
    model_semantic_loader: Callable[[Any], str] | None = None,
) -> Dict[str, Any]:
    """Complete every non-held-out check before output reservation."""

    freeze_id = validate_neural_freeze_manifest_id(neural_freeze_id)
    neural_freeze = dict(
        (neural_freeze_loader or load_verified_neural_freeze_manifest)(
            freeze_id, project_root.resolve()
        )
    )
    required_freeze_fields = {
        "manifest_id",
        "manifest_repo_relative_identifier",
        "manifest_file_sha256",
        "scientific_identity_sha256",
        "selected_method",
        "selected_method_lock",
        "selected_retraining_checkpoint_sha256",
        "document",
    }
    if set(neural_freeze) != required_freeze_fields:
        raise ValueError("Verified neural-freeze record is incomplete or unexpected")
    if neural_freeze.get("manifest_id") != freeze_id:
        raise ValueError("Verified neural-freeze ID mismatch")
    for field in ("manifest_file_sha256", "scientific_identity_sha256"):
        _require_sha256(neural_freeze.get(field), label=f"Neural freeze {field}")
    checkpoint_hashes = neural_freeze.get(
        "selected_retraining_checkpoint_sha256"
    )
    if (
        not isinstance(checkpoint_hashes, Mapping)
        or set(checkpoint_hashes)
        != {"primary_multiscale", "plain_unet_comparator"}
        or any(
            not isinstance(values, list)
            or len(values) != 3
            or any(not HEX_SHA256_PATTERN.fullmatch(str(value)) for value in values)
            for values in checkpoint_hashes.values()
        )
    ):
        raise ValueError("Verified neural freeze lacks all six checkpoint hashes")
    flat_checkpoint_hashes = [
        str(value)
        for role in ("primary_multiscale", "plain_unet_comparator")
        for value in checkpoint_hashes[role]
    ]
    if len(set(flat_checkpoint_hashes)) != 6:
        raise ValueError("Verified neural freeze checkpoint hashes are not distinct")

    verified = load_verified_classical_lock(
        fit_id, project_root=project_root, source_paths=source_paths
    )
    _require_canonical_directory(
        verified["paths"]["image_dir"],
        project_root=project_root,
        label="Canonical image directory",
    )
    _require_canonical_directory(
        verified["paths"]["mask_dir"],
        project_root=project_root,
        label="Canonical mask directory",
    )
    current_versions = dict((runtime_versions_loader or _library_versions)())
    if current_versions != verified["environment_versions"]:
        raise ValueError("Classical runtime versions differ from the frozen lock")
    manifest = validate_manifest_metadata(verified["paths"]["manifest"])
    estimator = load_verified_b2_estimator(
        verified["b2_model_path"],
        verified["b2_model_sha256"],
        verified["b2_model_semantic_sha256"],
        loader=model_loader,
        semantic_digest_loader=model_semantic_loader,
    )
    verified["manifest_attestation"] = manifest
    verified["b2_estimator"] = estimator
    verified["neural_freeze"] = neural_freeze
    verified["evaluation_pair_identity_sha256"] = hashlib.sha256(
        (
            f"{verified['canonical_lock_identity_sha256']}\0"
            f"{neural_freeze['scientific_identity_sha256']}"
        ).encode("ascii")
    ).hexdigest()
    verified["held_out_image_bytes_read"] = 0
    verified["held_out_target_bytes_read"] = 0
    verified["validation_image_bytes_read"] = 0
    verified["validation_target_bytes_read"] = 0
    return verified


def reserve_exactly_once_output(preflight: Mapping[str, Any]) -> Path:
    identity = _require_sha256(
        preflight["canonical_lock_identity_sha256"], label="Canonical lock identity"
    )
    neural_freeze = preflight.get("neural_freeze")
    if not isinstance(neural_freeze, Mapping):
        raise ValueError("Classical preflight lacks an authenticated neural freeze")
    freeze_identity = _require_sha256(
        neural_freeze.get("scientific_identity_sha256"),
        label="Neural-freeze scientific identity",
    )
    freeze_file_sha = _require_sha256(
        neural_freeze.get("manifest_file_sha256"),
        label="Neural-freeze manifest hash",
    )
    freeze_id = validate_neural_freeze_manifest_id(
        str(neural_freeze.get("manifest_id", ""))
    )
    fit_id = validate_fit_id(str(preflight.get("fit_id", "")))
    project_root = Path(preflight.get("project_root", PROJECT_ROOT)).resolve()
    expected_paths = canonical_fit_paths(fit_id, project_root=project_root)
    supplied_paths = preflight.get("paths")
    if (
        not isinstance(supplied_paths, Mapping)
        or Path(supplied_paths.get("output_fit_root", Path(".")))
        != expected_paths["output_fit_root"]
    ):
        raise ValueError("Classical evaluator output path is not canonical")
    pair_identity = hashlib.sha256(
        f"{identity}\0{freeze_identity}".encode("ascii")
    ).hexdigest()
    if preflight.get("evaluation_pair_identity_sha256") != pair_identity:
        raise ValueError("Classical/neural evaluation-pair identity mismatch")
    output_fit_root = expected_paths["output_fit_root"]
    canonical_output_root = output_fit_root.parent
    if canonical_output_root.is_symlink():
        raise ValueError("Canonical classical evaluation root is symbolic")
    canonical_output_root.mkdir(parents=True, exist_ok=True)
    output_fit_root.mkdir(exist_ok=True)
    if output_fit_root.is_symlink() or not output_fit_root.is_dir():
        raise ValueError("Canonical classical fit output root is invalid")
    output_dir = output_fit_root / freeze_id
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            "Held-out evaluation was already reserved for this classical-fit/neural-freeze pair"
        ) from error
    guard = output_dir / "held_out_access_guard.json"
    with guard.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "status": "reserved_before_held_out_image_or_mask_access",
                "fit_id": fit_id,
                "fit_execution_campaign_id": preflight[
                    "fit_execution_campaign_id"
                ],
                "canonical_lock_identity_sha256": identity,
                "evaluation_pair_identity_sha256": pair_identity,
                "raw_lock_file_sha256": preflight["lock_file_sha256"],
                "b2_model_file_sha256": preflight["b2_model_sha256"],
                "b2_model_semantic_sha256": preflight[
                    "b2_model_semantic_sha256"
                ],
                "neural_freeze": {
                    "manifest_id": freeze_id,
                    "manifest_file_sha256": freeze_file_sha,
                    "scientific_identity_sha256": freeze_identity,
                    "selected_method_lock_sha256": neural_freeze[
                        "selected_method_lock"
                    ]["raw_file_sha256"],
                    "selected_retraining_checkpoint_sha256": neural_freeze[
                        "selected_retraining_checkpoint_sha256"
                    ],
                },
                "all_comparators_in_one_invocation": list(CLASSICAL_COMPARATOR_IDS),
                "held_out_access_pass_limit": 1,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return output_dir


def _require_reservation(
    output_dir: Path, evaluation_pair_identity_sha256: str
) -> Dict[str, Any]:
    guard = output_dir / "held_out_access_guard.json"
    if not guard.is_file() or guard.is_symlink():
        raise RuntimeError("Held-out bytes cannot be read before canonical reservation")
    record = _load_json_object(guard)
    if (
        record.get("status")
        != "reserved_before_held_out_image_or_mask_access"
        or record.get("evaluation_pair_identity_sha256")
        != evaluation_pair_identity_sha256
    ):
        raise RuntimeError("Held-out reservation does not match the canonical lock")
    return record


def _claim_held_out_corpus_read(
    output_dir: Path,
    evaluation_pair_identity_sha256: str,
    *,
    corpus_label: str,
) -> Path:
    """Atomically consume the sole byte-read pass for one held-out corpus."""

    _require_reservation(output_dir, evaluation_pair_identity_sha256)
    allowed = {
        "input-image": "input_image",
        "target-mask": "target_mask",
    }
    try:
        safe_label = allowed[corpus_label]
    except KeyError as error:
        raise ValueError(f"Unsupported held-out corpus role: {corpus_label}") from error
    claim = output_dir / f"held_out_{safe_label}_read_claim.json"
    try:
        with claim.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "status": "claimed_before_first_byte_read",
                    "corpus_label": corpus_label,
                    "evaluation_pair_identity_sha256": (
                        evaluation_pair_identity_sha256
                    ),
                    "read_pass_limit": 1,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            f"Held-out {corpus_label} byte-read pass was already claimed"
        ) from error
    return claim


def discover_canonical_test_filenames(
    image_root: Path,
    mask_root: Path,
    *,
    output_dir: Path,
    evaluation_pair_identity_sha256: str,
) -> list[str]:
    _require_reservation(output_dir, evaluation_pair_identity_sha256)
    image_names = set()
    mask_names = set()
    for series in EXPECTED_TEST_SERIES:
        for candidate in image_root.glob(f"{series}_*.png"):
            if candidate.is_file():
                image_names.add(candidate.name)
        for candidate in mask_root.glob(f"{series}_*.png"):
            if candidate.is_file():
                mask_names.add(candidate.name)
    if image_names != mask_names:
        raise ValueError("Canonical held-out input/mask filename sets differ")
    names = sorted(image_names)
    if len(names) != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError("Canonical pdo2 held-out filename count is not 21")
    for name in names:
        _safe_named_file(image_root, name, label="Held-out input")
        _safe_named_file(mask_root, name, label="Held-out mask")
    return names


def _update_named_digest(digest: Any, name: str, payload: bytes) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")


def read_attested_held_out_files_once(
    root: Path,
    names: Sequence[str],
    *,
    output_dir: Path,
    evaluation_pair_identity_sha256: str,
    expected_count: int,
    expected_sha256: str,
    corpus_label: str,
    reader: Callable[[Path], bytes] | None = None,
) -> tuple[Dict[str, bytes], Dict[str, str], Dict[str, Any]]:
    """Read each named held-out file exactly once after reservation."""

    _claim_held_out_corpus_read(
        output_dir,
        evaluation_pair_identity_sha256,
        corpus_label=corpus_label,
    )
    if len(names) != expected_count or list(names) != sorted(set(names)):
        raise ValueError(f"{corpus_label} held-out name list is not canonical")
    read = reader or (lambda path: path.read_bytes())
    payloads: Dict[str, bytes] = {}
    file_hashes: Dict[str, str] = {}
    digest = hashlib.sha256()
    for name in names:
        path = _safe_named_file(root, name, label=corpus_label)
        payload = read(path)
        if not isinstance(payload, bytes):
            raise TypeError(f"{corpus_label} reader must return bytes")
        if name in payloads:
            raise RuntimeError(f"{corpus_label} file was read more than once: {name}")
        payloads[name] = payload
        file_hashes[name] = hashlib.sha256(payload).hexdigest()
        _update_named_digest(digest, name, payload)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"Canonical held-out {corpus_label} SHA-256 mismatch")
    return payloads, file_hashes, {
        "scope": "held_out_test",
        "split_names": ["test"],
        "file_count": len(names),
        "aggregate_sha256": observed,
        "file_name_list_sha256": hashlib.sha256(
            json.dumps(list(names), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "read_passes": 1,
        "reservation_verified_before_read": True,
    }


def decode_clean_image_bytes(name: str, payload: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as handle:
            mode = handle.mode
            source = np.asarray(handle)
    except Exception as error:
        raise ValueError(f"Failed to decode held-out input: {name}") from error
    if mode == "L" and source.ndim == 2:
        image = source
    elif mode in ("RGB", "RGBA") and source.ndim == 3:
        expected_channels = 3 if mode == "RGB" else 4
        if source.shape[2] != expected_channels:
            raise ValueError(f"Held-out input has unexpected {mode} shape: {name}")
        rgb = source[:, :, :3]
        if not (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        ):
            raise ValueError(f"Held-out input contains colour/ring information: {name}")
        if mode == "RGBA" and not np.all(source[:, :, 3] == 255):
            raise ValueError(f"Held-out input has non-opaque alpha: {name}")
        image = rgb[:, :, 0]
    else:
        raise ValueError(f"Held-out input mode is not explicitly greyscale: {name}")
    if image.dtype != np.uint8 or image.shape != EXPECTED_TILE_SHAPE:
        raise ValueError(f"Held-out input is not uint8 2048x2048: {name}")
    return np.asarray(image, dtype=np.uint8)


def predict_all_locked_comparators(
    image: np.ndarray, preflight: Mapping[str, Any]
) -> Dict[str, np.ndarray]:
    """Run each frozen comparator exactly once for one native tile."""

    return {
        "B0_small_components": predict_b0_small_components(
            image, area_cutoff_px=int(preflight["b0_area_cutoff_px"])
        ),
        "B1_marker_watershed": predict_b1_marker_watershed(
            image, area_cutoff_px=int(preflight["b1_area_cutoff_px"])
        ),
        "B2_extra_trees": predict_b2_extra_trees(
            image, preflight["b2_estimator"]
        ),
    }


def publication_metrics_from_confusion(confusion: np.ndarray) -> Dict[str, Any]:
    raw = metrics_from_confusion(np.asarray(confusion, dtype=np.int64))
    per_class = []
    for item in raw["per_class"]:
        per_class.append(
            {
                "class_id": int(item["class_id"]),
                "class_name": str(item["class_name"]),
                "support_pixels": int(item["support_pixels"]),
                "tp": int(item["tp"]),
                "fp": int(item["fp"]),
                "fn": int(item["fn"]),
                **{name: float(item[name]) for name in METRIC_NAMES},
            }
        )
    return {
        "total_pixels": int(raw["total_pixels"]),
        "per_class": per_class,
        "balanced_pore_iou": float(
            raw["selection_metrics"]["c0_c1_harmonic_iou"]
        ),
        "pore_union_iou": float(raw["selection_metrics"]["pore_union_iou"]),
        "overall_accuracy_reported": False,
        "ranking_or_selection_role": "none_external_comparator_description_only",
    }


def _metric_lookup(confusion: np.ndarray) -> Dict[str, float]:
    metrics = publication_metrics_from_confusion(confusion)
    values = {
        "balanced_pore_iou": float(metrics["balanced_pore_iou"]),
        "pore_union_iou": float(metrics["pore_union_iou"]),
    }
    for item in metrics["per_class"]:
        for name in METRIC_NAMES:
            values[f"class_{item['class_id']}.{name}"] = float(item[name])
    return values


def whole_tile_bootstrap(
    tile_confusions: np.ndarray,
    *,
    replicates: int = LOCKED_BOOTSTRAP_REPLICATES,
    seed: int = LOCKED_BOOTSTRAP_SEED,
    confidence: float = LOCKED_CONFIDENCE,
) -> Dict[str, Any]:
    matrices = np.asarray(tile_confusions, dtype=np.int64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not len(matrices):
        raise ValueError("Classical bootstrap requires non-empty (tiles,3,3) matrices")
    if replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid classical bootstrap configuration")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(matrices), size=(replicates, len(matrices)))
    series: MutableMapping[str, list[float]] = {}
    for matrix in matrices[draws].sum(axis=1):
        for key, value in _metric_lookup(matrix).items():
            series.setdefault(key, []).append(value)
    alpha = 1.0 - confidence
    intervals = {}
    for key, values in sorted(series.items()):
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if not array.size:
            intervals[key] = {
                "lower": None,
                "upper": None,
                "finite_replicates": 0,
            }
            continue
        lower, upper = np.quantile(array, [alpha / 2.0, 1.0 - alpha / 2.0])
        intervals[key] = {
            "lower": float(lower),
            "upper": float(upper),
            "finite_replicates": int(array.size),
        }
    return {
        "method": "percentile bootstrap",
        "sampling_unit": "held-out native 2048x2048 tile",
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "tile_count": int(len(matrices)),
        "intervals": intervals,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            def csv_value(value: Any) -> Any:
                if value is None:
                    return ""
                if isinstance(value, (float, np.floating)) and not math.isfinite(
                    float(value)
                ):
                    return ""
                return value

            writer.writerow(
                {key: csv_value(row.get(key)) for key in fieldnames}
            )


def write_metric_outputs(
    output_dir: Path, results: Mapping[str, Mapping[str, Any]]
) -> None:
    aggregate_rows = []
    per_tile_rows = []
    aggregate_confusion_rows = []
    per_tile_confusion_rows = []
    for method_id in CLASSICAL_COMPARATOR_IDS:
        method = results[method_id]
        aggregate = method["aggregate_metrics"]
        bootstrap = method["uncertainty"]
        for item in aggregate["per_class"]:
            for metric_name in METRIC_NAMES:
                key = f"class_{item['class_id']}.{metric_name}"
                interval = bootstrap["intervals"][key]
                aggregate_rows.append(
                    {
                        "comparator": method_id,
                        "scope": "class",
                        "class_id": item["class_id"],
                        "class_name": item["class_name"],
                        "metric": metric_name,
                        "value": item[metric_name],
                        "ci_lower": interval["lower"],
                        "ci_upper": interval["upper"],
                    }
                )
        for metric_name in ("balanced_pore_iou", "pore_union_iou"):
            interval = bootstrap["intervals"][metric_name]
            aggregate_rows.append(
                {
                    "comparator": method_id,
                    "scope": "pore_focus",
                    "class_id": "",
                    "class_name": "",
                    "metric": metric_name,
                    "value": aggregate[metric_name],
                    "ci_lower": interval["lower"],
                    "ci_upper": interval["upper"],
                }
            )
        for tile in method["per_tile"]:
            metrics = tile["metrics"]
            row = {
                "comparator": method_id,
                "evaluation_ordinal": tile["evaluation_ordinal"],
                "file_name": tile["file_name"],
                "input_sha256": tile["input_image_sha256"],
                "target_sha256": tile["target_mask_sha256"],
                "balanced_pore_iou": metrics["balanced_pore_iou"],
                "pore_union_iou": metrics["pore_union_iou"],
            }
            for item in metrics["per_class"]:
                for metric_name in METRIC_NAMES:
                    row[f"c{item['class_id']}_{metric_name}"] = item[metric_name]
            per_tile_rows.append(row)
        matrix = np.asarray(method["aggregate_confusion_matrix"], dtype=np.int64)
        for true_id in range(3):
            for predicted_id in range(3):
                aggregate_confusion_rows.append(
                    {
                        "comparator": method_id,
                        "true_class_id": true_id,
                        "true_class_name": CLASS_NAMES[true_id],
                        "predicted_class_id": predicted_id,
                        "predicted_class_name": CLASS_NAMES[predicted_id],
                        "pixel_count": int(matrix[true_id, predicted_id]),
                    }
                )
        for tile in method["per_tile"]:
            matrix = np.asarray(tile["confusion_matrix"], dtype=np.int64)
            for true_id in range(3):
                for predicted_id in range(3):
                    per_tile_confusion_rows.append(
                        {
                            "comparator": method_id,
                            "evaluation_ordinal": tile["evaluation_ordinal"],
                            "file_name": tile["file_name"],
                            "true_class_id": true_id,
                            "predicted_class_id": predicted_id,
                            "pixel_count": int(matrix[true_id, predicted_id]),
                        }
                    )

    bootstrap_fields = (
        "comparator",
        "scope",
        "class_id",
        "class_name",
        "metric",
        "value",
        "ci_lower",
        "ci_upper",
    )
    _write_csv(output_dir / "aggregate_metrics.csv", bootstrap_fields, aggregate_rows)
    _write_csv(
        output_dir / "per_tile_metrics.csv", list(per_tile_rows[0]), per_tile_rows
    )
    _write_csv(
        output_dir / "aggregate_confusion.csv",
        list(aggregate_confusion_rows[0]),
        aggregate_confusion_rows,
    )
    _write_csv(
        output_dir / "per_tile_confusion.csv",
        list(per_tile_confusion_rows[0]),
        per_tile_confusion_rows,
    )


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(PUBLICATION_SANS_SERIF_FONTS),
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
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
        }
    )
    return plt


def write_publication_figures(
    output_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
    qualitative: Mapping[str, np.ndarray],
    qualitative_name: str,
) -> None:
    publication_dir = output_dir / "publication"
    publication_dir.mkdir(exist_ok=False)
    for method_id in CLASSICAL_COMPARATOR_IDS:
        method_dir = publication_dir / method_id
        method_dir.mkdir()
        plot_confusion_matrix(
            np.asarray(results[method_id]["aggregate_confusion_matrix"]),
            method_dir,
            EXPECTED_SPLIT_COUNTS["test"],
        )

    plt = _configure_matplotlib()
    method_labels = ("B0 components", "B1 watershed", "B2 ExtraTrees")
    positions = np.arange(3, dtype=float)
    fig, axis = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    for class_id, offset in ((0, -0.10), (1, 0.10)):
        values = np.asarray(
            [
                results[method]["aggregate_metrics"]["per_class"][class_id]["iou"]
                for method in CLASSICAL_COMPARATOR_IDS
            ]
        )
        lower = np.asarray(
            [
                results[method]["uncertainty"]["intervals"][
                    f"class_{class_id}.iou"
                ]["lower"]
                for method in CLASSICAL_COMPARATOR_IDS
            ],
            dtype=np.float64,
        )
        upper = np.asarray(
            [
                results[method]["uncertainty"]["intervals"][
                    f"class_{class_id}.iou"
                ]["upper"]
                for method in CLASSICAL_COMPARATOR_IDS
            ],
            dtype=np.float64,
        )
        axis.errorbar(
            positions + offset,
            values,
            yerr=np.vstack(
                (np.maximum(0.0, values - lower), np.maximum(0.0, upper - values))
            ),
            fmt="o" if class_id == 0 else "s",
            color=PUBLICATION_CLASS_COLORS[class_id],
            markerfacecolor=(
                PUBLICATION_CLASS_COLORS[class_id]
                if class_id == 0
                else "white"
            ),
            markeredgewidth=1.4,
            markersize=7,
            capsize=4,
            linewidth=0,
            label=f"{CLASS_LABELS[class_id]} IoU",
        )
    axis.set_xticks(positions, method_labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Intersection over union")
    axis.set_title(
        "C0 and C1 IoU by classical comparator",
        loc="left",
        pad=24,
        fontweight="semibold",
    )
    axis.text(
        0,
        1.02,
        f"{LOCKED_RETROSPECTIVE_PARTITION_LABEL}; C0 and C1 shown separately\n"
        f"{WITHIN_SERIES_WHOLE_TILE_INTERVAL_LABEL}",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
        va="bottom",
    )
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.7)
    axis.legend(frameon=False)
    for extension in ("pdf", "png"):
        fig.savefig(
            publication_dir / f"classical_c0_c1_iou.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(PUBLICATION_CLASS_COLORS)
    panels = [
        ("Input", qualitative["image"], "gray"),
        ("Reference", qualitative["target"], cmap),
        ("B0 components", qualitative["B0_small_components"], cmap),
        ("B1 watershed", qualitative["B1_marker_watershed"], cmap),
        ("B2 ExtraTrees", qualitative["B2_extra_trees"], cmap),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.6))
    for axis, (title, array, panel_cmap) in zip(axes, panels):
        vmax = 255 if title == "Input" else 2
        axis.imshow(
            array,
            cmap=panel_cmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.18, top=0.78, wspace=0.04)
    fig.suptitle("Classical comparator segmentation maps", y=0.97, fontsize=11)
    fig.legend(
        handles=[
            Patch(
                facecolor=PUBLICATION_CLASS_COLORS[class_id],
                edgecolor="#444444",
                label=CLASS_LABELS[class_id],
            )
            for class_id in range(3)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
    )
    for extension in ("pdf", "png"):
        fig.savefig(
            publication_dir / f"classical_qualitative_comparison.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def _aggregate_gate_diagnostics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    confusion = np.stack(
        [np.asarray(item["confusion_matrix"], dtype=np.int64) for item in records]
    ).sum(axis=0)
    aggregate = summarize_gate_reference_confusion(confusion)
    for field in (
        "reference_c0_gate_mineral_pixels",
        "reference_c1_gate_mineral_pixels",
    ):
        aggregate[field] = int(sum(int(item[field]) for item in records))
    return aggregate


def _consume_guard(
    output_dir: Path,
    preflight: Mapping[str, Any],
    *,
    tile_count: int,
) -> None:
    _require_reservation(
        output_dir, str(preflight["evaluation_pair_identity_sha256"])
    )
    for role in ("input_image", "target_mask"):
        claim = output_dir / f"held_out_{role}_read_claim.json"
        if claim.is_symlink() or not claim.is_file():
            raise RuntimeError(f"Held-out {role} read claim is missing")
    with (output_dir / "held_out_access_consumed.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "status": "single_held_out_pass_consumed",
                "fit_id": preflight["fit_id"],
                "fit_execution_campaign_id": preflight[
                    "fit_execution_campaign_id"
                ],
                "canonical_lock_identity_sha256": preflight[
                    "canonical_lock_identity_sha256"
                ],
                "evaluation_pair_identity_sha256": preflight[
                    "evaluation_pair_identity_sha256"
                ],
                "test_tile_count": tile_count,
                "test_image_read_passes": 1,
                "test_mask_read_passes": 1,
                "inference_passes_per_comparator": 1,
                "comparators_evaluated_together": list(CLASSICAL_COMPARATOR_IDS),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def evaluate_campaign(fit_id: str, neural_freeze_id: str) -> Path:
    require_active_nodes_allocation()
    preflight = preflight_campaign(fit_id, neural_freeze_id)
    output_dir = reserve_exactly_once_output(preflight)
    identity = str(preflight["canonical_lock_identity_sha256"])
    pair_identity = str(preflight["evaluation_pair_identity_sha256"])
    paths = preflight["paths"]
    names = discover_canonical_test_filenames(
        paths["image_dir"],
        paths["mask_dir"],
        output_dir=output_dir,
        evaluation_pair_identity_sha256=pair_identity,
    )
    image_payloads, image_hashes, input_attestation = (
        read_attested_held_out_files_once(
            paths["image_dir"],
            names,
            output_dir=output_dir,
            evaluation_pair_identity_sha256=pair_identity,
            expected_count=int(EXPECTED_TEST_INPUT["image_count"]),
            expected_sha256=str(EXPECTED_TEST_INPUT["image_aggregate_sha256"]),
            corpus_label="input-image",
        )
    )
    mask_payloads, mask_hashes, target_attestation = (
        read_attested_held_out_files_once(
            paths["mask_dir"],
            names,
            output_dir=output_dir,
            evaluation_pair_identity_sha256=pair_identity,
            expected_count=int(EXPECTED_TEST_TARGET["mask_count"]),
            expected_sha256=str(EXPECTED_TEST_TARGET["mask_aggregate_sha256"]),
            corpus_label="target-mask",
        )
    )

    method_state: Dict[str, Dict[str, Any]] = {
        method: {
            "tile_confusions": [],
            "gate_diagnostics": [],
            "secondary_diagnostics": [],
            "per_tile": [],
        }
        for method in CLASSICAL_COMPARATOR_IDS
    }
    qualitative_name = names[0]
    qualitative: Dict[str, np.ndarray] = {}
    for ordinal, name in enumerate(names, start=1):
        image_payload = image_payloads.pop(name)
        mask_payload = mask_payloads.pop(name)
        image = decode_clean_image_bytes(name, image_payload)
        target = load_lossless_target_mask_bytes(
            _safe_named_file(paths["mask_dir"], name, label="Held-out target"),
            mask_payload,
        )
        predictions = predict_all_locked_comparators(image, preflight)
        tile_gate = gate_reference_diagnostic(image, target)
        for method_id, prediction in predictions.items():
            if (
                prediction.dtype != np.uint8
                or prediction.shape != EXPECTED_TILE_SHAPE
                or not set(int(value) for value in np.unique(prediction)).issubset(
                    {0, 1, 2}
                )
            ):
                raise ValueError(f"{method_id} emitted an invalid held-out mask")
            confusion = confusion_from_labels(target, prediction)
            secondary = secondary_2d_operational_diagnostics(target, prediction)
            state = method_state[method_id]
            state["tile_confusions"].append(confusion)
            state["gate_diagnostics"].append(dict(tile_gate))
            state["secondary_diagnostics"].append(secondary)
            state["per_tile"].append(
                {
                    "evaluation_ordinal": ordinal,
                    "image_id": "",
                    "file_name": name,
                    "input_image_sha256": image_hashes[name],
                    "target_mask_sha256": mask_hashes[name],
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "confusion_matrix": confusion,
                    "metrics": publication_metrics_from_confusion(confusion),
                    "conditional_gate_reference_diagnostic": dict(tile_gate),
                    "secondary_2d_operational_diagnostics": secondary,
                }
            )
        if name == qualitative_name:
            qualitative = {
                "image": image.copy(),
                "target": target.copy(),
                **{method: value.copy() for method, value in predictions.items()},
            }
        del image_payload, mask_payload, image, target, predictions

    if image_payloads or mask_payloads:
        raise RuntimeError("Not every attested held-out byte payload was consumed once")
    _consume_guard(output_dir, preflight, tile_count=len(names))

    results: Dict[str, Dict[str, Any]] = {}
    for method_id in CLASSICAL_COMPARATOR_IDS:
        state = method_state[method_id]
        matrices = np.stack(state["tile_confusions"])
        aggregate_confusion = matrices.sum(axis=0)
        aggregate_gate = _aggregate_gate_diagnostics(state["gate_diagnostics"])
        aggregate_secondary = aggregate_secondary_2d_diagnostics(
            state["secondary_diagnostics"]
        )
        secondary_uncertainty = bootstrap_secondary_2d_diagnostics(
            state["secondary_diagnostics"],
            replicates=LOCKED_BOOTSTRAP_REPLICATES,
            seed=LOCKED_BOOTSTRAP_SEED,
            confidence=LOCKED_CONFIDENCE,
        )
        results[method_id] = {
            "aggregate_confusion_matrix": aggregate_confusion,
            "aggregate_metrics": publication_metrics_from_confusion(
                aggregate_confusion
            ),
            "uncertainty": whole_tile_bootstrap(matrices),
            "gate_reference_diagnostic": aggregate_gate,
            "secondary_2d_operational_diagnostics": {
                "aggregate": aggregate_secondary,
                "uncertainty": secondary_uncertainty,
            },
            "per_tile": state["per_tile"],
        }
        method_dir = output_dir / "methods" / method_id
        method_dir.mkdir(parents=True)
        write_gate_reference_diagnostics(
            method_dir, aggregate_gate, state["per_tile"]
        )
        write_secondary_2d_diagnostic_tables(
            method_dir,
            aggregate_secondary,
            secondary_uncertainty,
            state["per_tile"],
        )

    write_metric_outputs(output_dir, results)
    write_publication_figures(
        output_dir, results, qualitative, qualitative_name=qualitative_name
    )
    report = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "evaluation_kind": "exactly_once_locked_classical_held_out_test",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_id": preflight["fit_id"],
        "fit_execution_campaign_id": preflight["fit_execution_campaign_id"],
        "evaluation_pair_identity_sha256": pair_identity,
        "selection_relationship": "external_only_no_neural_selection_effect",
        "ranking_policy": "no_overall_accuracy_or_winner_ranking",
        "neural_freeze": {
            "manifest_id": preflight["neural_freeze"]["manifest_id"],
            "manifest_path": preflight["neural_freeze"][
                "manifest_repo_relative_identifier"
            ],
            "manifest_file_sha256": preflight["neural_freeze"][
                "manifest_file_sha256"
            ],
            "scientific_identity_sha256": preflight["neural_freeze"][
                "scientific_identity_sha256"
            ],
            "selected_method": preflight["neural_freeze"]["selected_method"],
            "selected_method_lock": preflight["neural_freeze"][
                "selected_method_lock"
            ],
            "selected_retraining_checkpoint_sha256": preflight[
                "neural_freeze"
            ]["selected_retraining_checkpoint_sha256"],
            "neural_held_out_result_required": False,
        },
        "classical_lock": {
            "path": _repository_relative(preflight["lock_path"]),
            "raw_file_sha256": preflight["lock_file_sha256"],
            "canonical_identity_sha256": identity,
            "source_code_sha256": preflight["source_code_sha256"],
        },
        "b2_model": {
            "path": _repository_relative(preflight["b2_model_path"]),
            "sha256": preflight["b2_model_sha256"],
            "semantic_sha256": preflight["b2_model_semantic_sha256"],
        },
        "runtime": {
            "slurm_partition": EXPECTED_SLURM_PARTITION,
            "environment_versions": preflight["environment_versions"],
            "one_invocation_all_comparators": list(CLASSICAL_COMPARATOR_IDS),
            "native_tile_shape": list(EXPECTED_TILE_SHAPE),
            "post_result_tuning": False,
            "overall_accuracy_reported": False,
        },
        "data": {
            "split_manifest": preflight["manifest_attestation"],
            "test_series": list(EXPECTED_TEST_SERIES),
            "test_filenames": names,
            "input_attestation": input_attestation,
            "target_attestation": target_attestation,
            "validation_image_bytes_read": 0,
            "validation_target_bytes_read": 0,
            "held_out_input_read_passes": 1,
            "held_out_target_read_passes": 1,
            "source_mask_to_canonical": {"0": 0, "1": 1, "255": 2},
        },
        "uncertainty_protocol": {
            "method": "deterministic percentile bootstrap",
            "sampling_unit": "whole native held-out tile",
            "replicates": LOCKED_BOOTSTRAP_REPLICATES,
            "seed": LOCKED_BOOTSTRAP_SEED,
            "confidence": LOCKED_CONFIDENCE,
        },
        "comparators": results,
        "qualitative_example": {
            "selection": "lexicographically first locked test filename before targets",
            "file_name": qualitative_name,
        },
        "curves": {
            "reported": False,
            "reason": "hard-label classical comparators do not emit calibrated probabilities",
        },
        "outputs": sorted(
            {
                "evaluation_summary.json",
                *(
                    path.relative_to(output_dir).as_posix()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                ),
            }
        ),
    }
    report_path = output_dir / "evaluation_summary.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(_json_value(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote locked classical evaluation to {_repository_relative(output_dir)}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exactly-once evaluation of the frozen classical comparator lock"
    )
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--neural-freeze-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_campaign(
        validate_fit_id(args.fit_id),
        validate_neural_freeze_manifest_id(args.neural_freeze_id),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit and freeze new task-matched classical comparators on training labels only.

This entry point is deliberately separate from the neural candidate screen. It
opens only the 74 canonical training images and masks authenticated by the
frozen training attestations. Validation and held-out files, masks, and the
shared annotation index are neither resolved on disk nor read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.comparators import (  # noqa: E402
    B0_AREA_CUTOFFS,
    B1_AREA_CUTOFFS,
    B1_MARKER_MIN_DISTANCE_PX,
    B1_MARKER_THRESHOLD_ABS_PX,
    C0_DISCONNECTED,
    C1_CONNECTED,
    C2_MINERAL,
    CLASSICAL_COMPARATOR_IDS,
    EXTRA_TREES_CONFIG,
    EXTRA_TREES_FEATURE_NAMES,
    PORE_THRESHOLD_UINT8,
    b0_component_regions,
    b1_watershed_regions,
    build_extra_trees,
    clean_grayscale_feature_planes,
    confusion_from_labels,
    deterministic_sample_indices,
    feature_rows,
    fixed_pore_gate,
    save_extra_trees_numeric,
    load_clean_grayscale,
    pore_metrics_from_confusion,
    prediction_from_regions,
    select_area_cutoff,
)
from src.training.data_contract import (  # noqa: E402
    CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS,
    CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS,
    CONFIRMATORY_SPLIT_MANIFEST_SHA256,
    aggregate_indexed_file_bytes,
    load_lossless_mask,
    validate_lossless_mask_directory,
)
from scripts.evaluate_confirmatory_checkpoint import (  # noqa: E402
    LOCKED_BOOTSTRAP_REPLICATES,
    LOCKED_BOOTSTRAP_SEED,
    LOCKED_CONFIDENCE,
    LOCKED_INPUT_ATTESTATIONS,
    LOCKED_TARGET_ATTESTATIONS,
)


PROTOCOL_ID = "classical_train_only_comparators_v1"
CLASSICAL_LOCK_SCHEMA_VERSION = 2
CANONICAL_LOCK_ROOT = Path("results/classical_comparators/locked")
CANONICAL_SPLIT_MANIFEST = Path("config/confirmatory_splits.json")
CANONICAL_IMAGE_DIR = Path("original_images")
CANONICAL_MASK_DIR = Path(
    "results/step2_pore_classification/pore_classifications"
)
CAMPAIGN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
FIT_ID_PATTERN = re.compile(r"^classical-fit-[0-9a-f]{16}$")
SAMPLING_SEED = 20260821
SAMPLES_PER_CLASS_PER_TILE = 4096
SOURCE_PATHS = (
    "src/classical/__init__.py",
    "src/classical/comparators.py",
    "src/training/checkpoint_io.py",
    "src/training/data_contract.py",
    "src/training/screen_selection.py",
    "src/training/neural_freeze.py",
    "scripts/fit_classical_comparators.py",
    "scripts/aire_fit_classical_comparators.slurm",
    "scripts/evaluate_locked_classical_comparators.py",
    "scripts/evaluate_confirmatory_checkpoint.py",
    "scripts/aire_locked_classical_evaluation.slurm",
    "scripts/build_neural_freeze_manifest.py",
    "scripts/README.md",
    "docs/CLASSICAL_COMPARATOR_PROTOCOL_2026-08-21.md",
    "docs/CONFIRMATORY_RERUN.md",
)
CANONICAL_TRAIN_FILENAMES = (
    "pdo1_12_segment_10_10.png",
    "pdo1_12_segment_10_5.png",
    "pdo1_12_segment_10_6.png",
    "pdo1_12_segment_10_7.png",
    "pdo1_12_segment_10_8.png",
    "pdo1_12_segment_10_9.png",
    "pdo1_12_segment_2_1.png",
    "pdo1_12_segment_2_10.png",
    "pdo1_12_segment_2_2.png",
    "pdo1_12_segment_2_3.png",
    "pdo1_12_segment_2_4.png",
    "pdo1_12_segment_2_5.png",
    "pdo1_12_segment_2_6.png",
    "pdo1_12_segment_2_7.png",
    "pdo1_12_segment_2_8.png",
    "pdo1_12_segment_2_9.png",
    "pdo1_12_segment_3_0.png",
    "pdo1_12_segment_3_1.png",
    "pdo1_12_segment_3_2.png",
    "pdo1_12_segment_3_3.png",
    "pdo1_12_segment_3_5.png",
    "pdo1_12_segment_3_6.png",
    "pdo1_12_segment_4_4.png",
    "pdo1_12_segment_4_5.png",
    "pdo1_12_segment_5_0.png",
    "pdo1_12_segment_5_1.png",
    "pdo1_12_segment_6_0.png",
    "pdo1_12_segment_9_1.png",
    "pdo1_12_segment_9_6.png",
    "pdo1_12_segment_9_7.png",
    "pdo1_12_segment_9_8.png",
    "pdo1_12_segment_9_9.png",
    "pdo1_7_segment_0_4.png",
    "pdo1_7_segment_0_7.png",
    "pdo1_7_segment_10_0.png",
    "pdo1_7_segment_10_10.png",
    "pdo1_7_segment_10_4.png",
    "pdo1_7_segment_10_6.png",
    "pdo1_7_segment_10_8.png",
    "pdo1_7_segment_10_9.png",
    "pdo1_7_segment_11_0.png",
    "pdo1_7_segment_11_10.png",
    "pdo1_7_segment_11_2.png",
    "pdo1_7_segment_11_3.png",
    "pdo1_7_segment_11_4.png",
    "pdo1_7_segment_11_5.png",
    "pdo1_7_segment_11_6.png",
    "pdo1_7_segment_11_7.png",
    "pdo1_7_segment_11_8.png",
    "pdo1_7_segment_11_9.png",
    "pdo1_7_segment_4_10.png",
    "pdo1_7_segment_4_4.png",
    "pdo1_7_segment_4_8.png",
    "pdo1_7_segment_4_9.png",
    "pdo1_7_segment_5_3.png",
    "pdo1_7_segment_6_0.png",
    "pdo1_7_segment_6_1.png",
    "pdo1_7_segment_6_10.png",
    "pdo1_7_segment_6_2.png",
    "pdo1_7_segment_6_3.png",
    "pdo1_7_segment_6_4.png",
    "pdo1_7_segment_6_5.png",
    "pdo1_7_segment_6_6.png",
    "pdo1_7_segment_6_7.png",
    "pdo1_7_segment_6_8.png",
    "pdo1_7_segment_6_9.png",
    "pdo1_7_segment_8_1.png",
    "pdo1_7_segment_8_11.png",
    "pdo1_7_segment_8_7.png",
    "pdo1_7_segment_9_11.png",
    "pdo1_7_segment_9_3.png",
    "pdo4_1_140721_segment_15_5.png",
    "pdo4_2_151020_segment_0_0.png",
    "pdo4_2_151020_segment_0_7.png",
)
CANONICAL_TRAIN_FILENAME_LIST_SHA256 = hashlib.sha256(
    json.dumps(CANONICAL_TRAIN_FILENAMES, separators=(",", ":")).encode("utf-8")
).hexdigest()


def frozen_classical_lock_static_contract() -> Dict[str, Any]:
    """Return the exact non-result lock fields shared by fitter and evaluator."""

    return {
        "status": "frozen_train_only_choices_pending_one_pass_external_scoring",
        "scientific_identity": (
            "New task-matched classical comparators; not a reproduction of the "
            "companion paper's eight pipelines"
        ),
        "selection_relationship": {
            "eligible_for_five_neural_candidate_screen": False,
            "can_select_neural_winner": False,
            "future_role": "external_comparator_scored_after_freeze",
        },
        "data_access": {
            "full_split_manifest_metadata_read": True,
            "shared_annotation_index_read_count": 0,
            "images_read": ["train"],
            "targets_read": ["train"],
            "training_tile_count": len(CANONICAL_TRAIN_FILENAMES),
            "validation_dataset_constructed": False,
            "validation_image_read_count": 0,
            "validation_target_read_count": 0,
            "held_out_dataset_constructed": False,
            "held_out_image_read_count": 0,
            "held_out_target_read_count": 0,
        },
        "training_sample_key": (
            "one-based index in lexicographically sorted authenticated "
            "training filenames"
        ),
        "inference_contract": {
            "input": (
                "clean uint8 greyscale only; RGB accepted only when all "
                "channels agree"
            ),
            "forbidden_inputs": [
                "yellow ring pixels",
                "ring masks",
                "target masks",
                "validation-derived thresholds",
                "held-out-derived thresholds",
            ],
            "pore_gate": f"raw_uint8 < {PORE_THRESHOLD_UINT8}",
            "outside_gate_class": C2_MINERAL,
            "native_tile_inference": True,
            "tiling": None,
        },
        "b0_definition": {
            "region_partition": "8-connected components of the fixed pore gate",
            "rule": "C0 iff component area is strictly below cutoff; else C1",
        },
        "b1_definition": {
            "region_partition": "marker watershed of fixed-gate Euclidean distance",
            "marker_min_distance_px": B1_MARKER_MIN_DISTANCE_PX,
            "marker_threshold_abs_px": B1_MARKER_THRESHOLD_ABS_PX,
            "fallback": (
                "one distance-maximum marker for every unseeded gate component"
            ),
            "rule": (
                "C0 iff watershed region area is strictly below cutoff; else C1"
            ),
        },
        "reported_selection_metrics": (
            "training-group C0 IoU, C1 IoU, and their harmonic mean only; "
            "not publication test results"
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_campaign_id(value: str) -> str:
    campaign_id = str(value)
    if not CAMPAIGN_PATTERN.fullmatch(campaign_id):
        raise ValueError(
            "Classical campaign ID must be 1-96 public-path-safe characters"
        )
    return campaign_id


def validate_fit_id(value: str) -> str:
    fit_id = str(value)
    if not FIT_ID_PATTERN.fullmatch(fit_id):
        raise ValueError("Classical fit ID is not canonical")
    return fit_id


def classical_scientific_identity_payload(lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the execution-, storage-, and serialization-independent fit state."""

    payload = copy.deepcopy(dict(lock))
    for field in (
        "campaign_id",
        "created_utc",
        "fit_id",
        "scientific_identity_sha256",
    ):
        payload.pop(field, None)
    try:
        payload["comparators"]["B2_extra_trees"].pop("model_sha256", None)
        # This is useful operational provenance, but the authenticated aggregate
        # bytes and filename set—not their mount point—define the training corpus.
        payload["data_provenance"]["training_target"].pop(
            "mask_directory", None
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Classical fit lacks its scientific model/data record"
        ) from error
    return payload


def classical_scientific_identity_sha256(lock: Mapping[str, Any]) -> str:
    payload = json.dumps(
        classical_scientific_identity_payload(lock),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Path must be inside the repository: {resolved}") from error


def _public_declared_path(path: Path) -> str:
    """Return a repository-relative path without leaking an external root."""

    try:
        return _safe_repo_path(path)
    except ValueError:
        return f"<external>/{path.name}"


def _require_canonical_input_path(
    supplied: Path, canonical_relative: Path, *, label: str
) -> Path:
    """Reject alternate roots and every symbolic component in a production fit."""

    root = PROJECT_ROOT.resolve()
    lexical = Path(os.path.abspath(PROJECT_ROOT / supplied))
    expected_lexical = Path(os.path.abspath(PROJECT_ROOT / canonical_relative))
    if lexical != expected_lexical:
        raise ValueError(f"{label} must be exactly {canonical_relative.as_posix()}")
    cursor = root
    for component in canonical_relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{label} contains a symbolic-link component")
    return lexical


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _assign_group(file_name: str, declared_series: Sequence[str]) -> str:
    stem = Path(file_name).stem
    matches = [
        str(series)
        for series in declared_series
        if stem == str(series) or stem.startswith(f"{series}_")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Training tile {file_name!r} matched {matches!r}; "
            "expected one source series"
        )
    return matches[0]


def prepare_train_only_contract(
    *,
    split_manifest_path: Path,
    image_dir: Path,
    mask_dir: Path,
    enforce_confirmatory_attestations: bool = True,
) -> Dict[str, Any]:
    """Authenticate only the declared training-series bytes and metadata.

    The shared COCO annotation JSON is intentionally not an argument: parsing
    that container would load validation/test polygon records even if the code
    subsequently used only training image entries.
    """

    split_manifest_path = split_manifest_path.resolve()
    image_dir = image_dir.resolve()
    mask_dir = mask_dir.resolve()
    manifest = _load_json(split_manifest_path)
    manifest_sha256 = sha256_file(split_manifest_path)
    if (
        enforce_confirmatory_attestations
        and manifest_sha256 != CONFIRMATORY_SPLIT_MANIFEST_SHA256
    ):
        raise ValueError(
            "Classical train-only attestation failed: split manifest SHA-256"
        )

    required_splits = ("train", "val", "test")
    for split_name in required_splits:
        values = manifest.get(split_name)
        if not isinstance(values, list):
            raise ValueError(f"Split manifest lacks list-valued '{split_name}'")
        invalid_identifier = any(
            isinstance(value, bool) or not isinstance(value, (int, str))
            for value in values
        )
        if invalid_identifier:
            raise ValueError(f"Split '{split_name}' contains an invalid identifier")
        normalized = [
            ("id", int(value))
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
            else ("name", str(value))
            for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Split '{split_name}' contains duplicate identifiers")

    normalized_by_split = {
        split_name: {
            ("id", int(value))
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
            else ("name", str(value))
            for value in manifest[split_name]
        }
        for split_name in required_splits
    }
    if any(
        normalized_by_split[left] & normalized_by_split[right]
        for index, left in enumerate(required_splits)
        for right in required_splits[index + 1 :]
    ):
        raise ValueError("Split manifest contains overlapping identifiers")

    provenance = manifest.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Split manifest lacks structured provenance")
    train_series = provenance.get("train_series")
    if not isinstance(train_series, list) or not train_series:
        raise ValueError("Split manifest lacks declared training series")
    normalized_series = []
    for value in train_series:
        series = str(value)
        if (
            not series
            or Path(series).name != series
            or any(character in series for character in "*?[]\\")
        ):
            raise ValueError(f"Unsafe declared training-series name: {series!r}")
        normalized_series.append(series)
    if len(normalized_series) != len(set(normalized_series)):
        raise ValueError("Split manifest repeats a declared training series")

    # Discover only paths bearing a declared training-series prefix. No broad
    # directory traversal or validation/test path construction is required.
    discovered_names = set()
    for series in normalized_series:
        direct = image_dir / f"{series}.png"
        if direct.is_file():
            discovered_names.add(direct.name)
        for candidate in image_dir.glob(f"{series}_*.png"):
            if candidate.is_file():
                discovered_names.add(candidate.name)
    train_names = tuple(sorted(discovered_names))
    if not train_names:
        raise FileNotFoundError("No files match the declared training series")
    if len(train_names) != len(manifest["train"]):
        raise ValueError(
            "Declared training-series file count does not match the split "
            f"manifest: {len(train_names)} != {len(manifest['train'])}"
        )
    if enforce_confirmatory_attestations and train_names != CANONICAL_TRAIN_FILENAMES:
        raise ValueError(
            "Classical train-only attestation failed: canonical training filenames"
        )
    for file_name in train_names:
        image_path = image_dir / file_name
        mask_path = mask_dir / file_name
        if image_path.is_symlink() or mask_path.is_symlink():
            raise ValueError(
                "Classical training corpus cannot contain symbolic image/mask files: "
                f"{file_name}"
            )

    # Stable local IDs exist only to key masks and deterministic sampling. They
    # reveal no validation/test mapping and are locked to the sorted name list.
    train_ids = tuple(range(1, len(train_names) + 1))
    images_by_id = {
        image_id: {"id": image_id, "file_name": file_name}
        for image_id, file_name in zip(train_ids, train_names)
    }
    group_by_id = {
        image_id: _assign_group(
            str(images_by_id[image_id]["file_name"]), normalized_series
        )
        for image_id in train_ids
    }
    if set(group_by_id.values()) != set(normalized_series):
        raise ValueError("Not every declared training series has a training tile")

    # The synthetic selected-ID argument is the access boundary: this validator
    # contains and opens only the authenticated training-file index.
    mask_paths, target_provenance = validate_lossless_mask_directory(
        {"images": list(images_by_id.values())},
        image_dir,
        mask_dir,
        image_ids=train_ids,
    )
    input_provenance = aggregate_indexed_file_bytes(
        image_dir,
        train_names,
        scope="canonical_training_images_only",
        split_names=("train",),
    )
    if enforce_confirmatory_attestations:
        expected_input = CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS["train"]
        expected_target = CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS["train"]
        failures = []
        if input_provenance["image_count"] != expected_input["image_count"]:
            failures.append("training image count")
        if (
            input_provenance["image_aggregate_sha256"]
            != expected_input["image_aggregate_sha256"]
        ):
            failures.append("training image aggregate SHA-256")
        if (
            input_provenance["file_name_list_sha256"]
            != CANONICAL_TRAIN_FILENAME_LIST_SHA256
        ):
            failures.append("training filename-list SHA-256")
        if target_provenance["mask_count"] != expected_target["mask_count"]:
            failures.append("training mask count")
        if (
            target_provenance["mask_aggregate_sha256"]
            != expected_target["mask_aggregate_sha256"]
        ):
            failures.append("training mask aggregate SHA-256")
        if failures:
            raise ValueError(
                "Classical train-only attestation failed: " + ", ".join(failures)
            )

    public_target_provenance = dict(target_provenance)
    public_target_provenance["mask_directory"] = _public_declared_path(mask_dir)
    return {
        "images_by_id": images_by_id,
        "train_ids": train_ids,
        "train_names": train_names,
        "group_by_id": group_by_id,
        "mask_paths": mask_paths,
        "manifest_sha256": manifest_sha256,
        "input_provenance": input_provenance,
        "target_provenance": public_target_provenance,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
    }


def _load_training_tile(
    contract: Mapping[str, Any], image_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    image_info = contract["images_by_id"][int(image_id)]
    image = load_clean_grayscale(
        Path(contract["image_dir"]) / str(image_info["file_name"])
    )
    expected_shape = tuple(int(value) for value in image.shape)
    target = load_lossless_mask(
        contract["mask_paths"][int(image_id)], expected_shape, num_classes=3
    )
    return image, target


def training_contract_attestation(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the byte/provenance state that must remain stable during fitting."""

    return {
        "manifest_sha256": str(contract["manifest_sha256"]),
        "train_names": list(contract["train_names"]),
        "group_by_filename": {
            str(contract["images_by_id"][image_id]["file_name"]): str(
                contract["group_by_id"][image_id]
            )
            for image_id in contract["train_ids"]
        },
        "input_provenance": dict(contract["input_provenance"]),
        "target_provenance": dict(contract["target_provenance"]),
    }


def _zero_confusions(
    candidates: Iterable[int], groups: Iterable[str]
) -> Dict[int, Dict[str, np.ndarray]]:
    return {
        int(candidate): {
            str(group): np.zeros((3, 3), dtype=np.int64) for group in groups
        }
        for candidate in candidates
    }


def _sample_class(
    target: np.ndarray,
    *,
    class_id: int,
    image_id: int,
    limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = np.flatnonzero(target.ravel() == int(class_id))
    selected = deterministic_sample_indices(
        candidates,
        limit=int(limit),
        seed=SAMPLING_SEED,
        image_id=int(image_id),
        class_id=int(class_id),
    )
    if selected.size == 0:
        return selected, np.empty(0, dtype=np.float64)
    weights = np.full(
        selected.size, candidates.size / selected.size, dtype=np.float64
    )
    return selected, weights


def collect_train_only_evidence(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute B0/B1 CV tables and deterministic B2 samples from train tiles."""

    groups = sorted(set(contract["group_by_id"].values()))
    b0_confusions = _zero_confusions(B0_AREA_CUTOFFS, groups)
    b1_confusions = _zero_confusions(B1_AREA_CUTOFFS, groups)
    b2_training: MutableMapping[
        str, list[Tuple[np.ndarray, np.ndarray]]
    ] = defaultdict(list)
    b2_evaluation: MutableMapping[
        str, list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = defaultdict(list)
    group_counts = Counter(contract["group_by_id"].values())

    for image_id in contract["train_ids"]:
        group = str(contract["group_by_id"][int(image_id)])
        image, target = _load_training_tile(contract, int(image_id))
        gate = fixed_pore_gate(image)
        b0_regions = b0_component_regions(image)
        b1_regions = b1_watershed_regions(image)

        for cutoff in B0_AREA_CUTOFFS:
            prediction = prediction_from_regions(
                b0_regions, gate, area_cutoff_px=cutoff
            )
            b0_confusions[cutoff][group] += confusion_from_labels(target, prediction)
        for cutoff in B1_AREA_CUTOFFS:
            prediction = prediction_from_regions(
                b1_regions, gate, area_cutoff_px=cutoff
            )
            b1_confusions[cutoff][group] += confusion_from_labels(target, prediction)

        feature_planes = clean_grayscale_feature_planes(image)
        training_indices = []
        training_targets = []
        for class_id in (C0_DISCONNECTED, C1_CONNECTED):
            in_gate_candidates = np.flatnonzero(
                (target.ravel() == class_id) & gate.ravel()
            )
            selected = deterministic_sample_indices(
                in_gate_candidates,
                limit=SAMPLES_PER_CLASS_PER_TILE,
                seed=SAMPLING_SEED,
                image_id=int(image_id),
                class_id=class_id,
            )
            training_indices.append(selected)
            training_targets.append(
                np.full(selected.size, class_id, dtype=np.uint8)
            )
        model_indices = np.concatenate(training_indices)
        model_targets = np.concatenate(training_targets)
        if model_indices.size:
            b2_training[group].append(
                (feature_rows(feature_planes, model_indices), model_targets)
            )

        evaluation_indices = []
        evaluation_targets = []
        evaluation_weights = []
        for class_id in (C0_DISCONNECTED, C1_CONNECTED):
            selected, weights = _sample_class(
                target,
                class_id=class_id,
                image_id=int(image_id),
                limit=SAMPLES_PER_CLASS_PER_TILE,
            )
            evaluation_indices.append(selected)
            evaluation_targets.append(
                np.full(selected.size, class_id, dtype=np.uint8)
            )
            evaluation_weights.append(weights)
        mineral_inside_gate = np.flatnonzero(
            (target.ravel() == C2_MINERAL) & gate.ravel()
        )
        selected_mineral = deterministic_sample_indices(
            mineral_inside_gate,
            limit=SAMPLES_PER_CLASS_PER_TILE,
            seed=SAMPLING_SEED,
            image_id=int(image_id),
            class_id=C2_MINERAL,
        )
        mineral_weight = (
            np.full(
                selected_mineral.size,
                mineral_inside_gate.size / selected_mineral.size,
                dtype=np.float64,
            )
            if selected_mineral.size
            else np.empty(0, dtype=np.float64)
        )
        evaluation_indices.append(selected_mineral)
        evaluation_targets.append(
            np.full(selected_mineral.size, C2_MINERAL, dtype=np.uint8)
        )
        evaluation_weights.append(mineral_weight)
        sampled_indices = np.concatenate(evaluation_indices)
        b2_evaluation[group].append(
            (
                feature_rows(feature_planes, sampled_indices),
                np.concatenate(evaluation_targets),
                gate.ravel()[sampled_indices],
                np.concatenate(evaluation_weights),
            )
        )

    return {
        "groups": groups,
        "group_tile_counts": {group: int(group_counts[group]) for group in groups},
        "b0_confusions": b0_confusions,
        "b1_confusions": b1_confusions,
        "b2_training": b2_training,
        "b2_evaluation": b2_evaluation,
    }


def _weighted_confusion(
    target: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    if not (target.shape == prediction.shape == weights.shape):
        raise ValueError("Weighted confusion arrays have different shapes")
    confusion = np.zeros((3, 3), dtype=np.float64)
    np.add.at(confusion, (target.astype(int), prediction.astype(int)), weights)
    return confusion


def fit_b2_group_cv(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Run deterministic leave-one-training-series-out B2 diagnostics."""

    groups = list(evidence["groups"])
    folds = {}
    for held_out_group in groups:
        fit_records = [
            record
            for group in groups
            if group != held_out_group
            for record in evidence["b2_training"][group]
        ]
        if not fit_records:
            raise ValueError(
                f"B2 grouped-CV fit is empty when holding out {held_out_group}"
            )
        x_fit = np.concatenate([record[0] for record in fit_records], axis=0)
        y_fit = np.concatenate([record[1] for record in fit_records], axis=0)
        if set(int(value) for value in np.unique(y_fit)) != {
            C0_DISCONNECTED,
            C1_CONNECTED,
        }:
            raise ValueError(
                f"B2 grouped-CV fit lacks a pore class when holding out {held_out_group}"
            )
        estimator = build_extra_trees()
        estimator.fit(x_fit, y_fit)

        confusion = np.zeros((3, 3), dtype=np.float64)
        sampled_pixels = 0
        held_out_records = evidence["b2_evaluation"][held_out_group]
        for features, target, gate, weights in held_out_records:
            prediction = np.full(target.shape, C2_MINERAL, dtype=np.uint8)
            if np.any(gate):
                prediction[gate] = estimator.predict(features[gate]).astype(np.uint8)
            confusion += _weighted_confusion(target, prediction, weights)
            sampled_pixels += int(target.size)
        folds[held_out_group] = {
            "held_out_scope": "training_series_only",
            "fit_groups": [group for group in groups if group != held_out_group],
            "sampled_evaluation_pixels": sampled_pixels,
            "weighted_confusion": confusion.tolist(),
            **pore_metrics_from_confusion(confusion),
        }

    mean_metrics = {
        key: float(np.mean([folds[group][key] for group in groups]))
        for key in ("iou_c0", "iou_c1", "balanced_pore_iou")
    }
    return {
        "scope": "leave_one_canonical_training_series_out",
        "evaluation": "deterministic_stratified_weighted_sample",
        "folds": folds,
        "mean_group_metrics": mean_metrics,
    }


def fit_final_b2(evidence: Mapping[str, Any]) -> Any:
    records = [
        record
        for group in evidence["groups"]
        for record in evidence["b2_training"][group]
    ]
    if not records:
        raise ValueError("B2 final fit has no in-gate pore samples")
    x_fit = np.concatenate([record[0] for record in records], axis=0)
    y_fit = np.concatenate([record[1] for record in records], axis=0)
    if set(int(value) for value in np.unique(y_fit)) != {
        C0_DISCONNECTED,
        C1_CONNECTED,
    }:
        raise ValueError("B2 final fit must contain both pore classes")
    estimator = build_extra_trees()
    estimator.fit(x_fit, y_fit)
    return estimator, {
        "training_sample_count": int(y_fit.size),
        "training_sample_class_counts": {
            str(class_id): int(np.count_nonzero(y_fit == class_id))
            for class_id in (C0_DISCONNECTED, C1_CONNECTED)
        },
    }


def _library_versions() -> Dict[str, str]:
    try:
        import cv2
        import matplotlib
        import PIL
        import scipy
        import sklearn
        import skimage
    except ImportError as error:
        missing = error.name or "an unknown package"
        raise RuntimeError(
            "Classical comparator runtime is incomplete: missing "
            f"{missing}. Install the declared requirements; B1/B2 will not be "
            "silently omitted and no comparator lock can be created."
        ) from error

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "scikit_learn": sklearn.__version__,
        "opencv": cv2.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _source_hashes() -> Dict[str, str]:
    hashes = {}
    for relative_name in SOURCE_PATHS:
        path = PROJECT_ROOT / relative_name
        cursor = PROJECT_ROOT.resolve()
        for component in Path(relative_name).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise ValueError(
                    f"Comparator source/protocol path is symbolic: {relative_name}"
                )
        if not path.is_file():
            raise FileNotFoundError(
                f"Comparator source/protocol file is missing: {relative_name}"
            )
        hashes[relative_name] = sha256_file(path)
    return hashes


def frozen_one_pass_evaluator_contract() -> Dict[str, Any]:
    """Return the static held-out execution schema recorded in every lock."""

    return {
        "schema_version": 1,
        "evaluation_split": "held_out_test",
        "test_series": ["pdo2_24"],
        "test_tile_count": 21,
        "native_tile_shape": [2048, 2048],
        "split_manifest_sha256": CONFIRMATORY_SPLIT_MANIFEST_SHA256,
        "test_input_attestation": dict(
            LOCKED_INPUT_ATTESTATIONS["held_out_test"]
        ),
        "test_target_attestation": dict(
            LOCKED_TARGET_ATTESTATIONS["held_out_test"]
        ),
        "bootstrap": {
            "method": "deterministic_percentile_whole_tile",
            "replicates": LOCKED_BOOTSTRAP_REPLICATES,
            "seed": LOCKED_BOOTSTRAP_SEED,
            "confidence": LOCKED_CONFIDENCE,
        },
        "validation_corpus_bytes_permitted": False,
        "all_comparators_one_invocation": list(CLASSICAL_COMPARATOR_IDS),
        "class_ids": {
            "disconnected_pore_c0": C0_DISCONNECTED,
            "connected_pore_c1": C1_CONNECTED,
            "mineral_c2": C2_MINERAL,
        },
        "prediction_dtype": "uint8",
        "prediction_shape": "same_as_native_input_tile",
        "allowed_prediction_values": [
            C0_DISCONNECTED,
            C1_CONNECTED,
            C2_MINERAL,
        ],
        "predictors": {
            "B0_small_components": {
                "python_callable": (
                    "src.classical.comparators.predict_b0_small_components"
                ),
                "keyword_from_lock": {
                    "area_cutoff_px": (
                        "comparators.B0_small_components.frozen_area_cutoff_px"
                    )
                },
                "serialized_artifact_required": False,
            },
            "B1_marker_watershed": {
                "python_callable": (
                    "src.classical.comparators.predict_b1_marker_watershed"
                ),
                "keyword_from_lock": {
                    "area_cutoff_px": (
                        "comparators.B1_marker_watershed.frozen_area_cutoff_px"
                    )
                },
                "serialized_artifact_required": False,
            },
            "B2_extra_trees": {
                "python_callable": (
                    "src.classical.comparators.predict_b2_extra_trees"
                ),
                "estimator_loader": (
                    "src.classical.comparators.load_extra_trees_numeric"
                ),
                "artifact_from_lock": "comparators.B2_extra_trees.model_file",
                "artifact_sha256_from_lock": (
                    "comparators.B2_extra_trees.model_sha256"
                ),
                "serialized_artifact_required": True,
            },
        },
        "selection_effect": "none_external_comparators_only",
    }


def build_lock(
    *,
    campaign_id: str,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    static = frozen_classical_lock_static_contract()
    selected_b0, b0_selection = select_area_cutoff(evidence["b0_confusions"])
    selected_b1, b1_selection = select_area_cutoff(evidence["b1_confusions"])
    b2_cv = fit_b2_group_cv(evidence)
    b2_model, b2_training = fit_final_b2(evidence)
    model_path = output_dir / "B2_extra_trees.npz"
    model_semantic_sha256 = save_extra_trees_numeric(b2_model, model_path)

    lock = {
        "schema_version": CLASSICAL_LOCK_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": validate_campaign_id(campaign_id),
        "status": static["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_identity": static["scientific_identity"],
        "selection_relationship": static["selection_relationship"],
        "data_access": static["data_access"],
        "data_provenance": {
            "split_manifest_sha256": contract["manifest_sha256"],
            "training_input": contract["input_provenance"],
            "training_target": contract["target_provenance"],
            "canonical_training_filename_count": len(contract["train_names"]),
            "canonical_training_filenames": list(contract["train_names"]),
            "training_file_name_list_sha256": contract["input_provenance"][
                "file_name_list_sha256"
            ],
            "training_sampling_ids_by_filename": {
                str(contract["images_by_id"][image_id]["file_name"]): int(image_id)
                for image_id in contract["train_ids"]
            },
            "training_group_tile_counts": evidence["group_tile_counts"],
            "training_sample_key": static["training_sample_key"],
            "source_mask_to_canonical": {"0": 0, "1": 1, "255": 2},
        },
        "inference_contract": static["inference_contract"],
        "comparators": {
            "B0_small_components": {
                **static["b0_definition"],
                "candidate_area_cutoffs_px": list(B0_AREA_CUTOFFS),
                "selection": b0_selection,
                "frozen_area_cutoff_px": selected_b0,
            },
            "B1_marker_watershed": {
                **static["b1_definition"],
                "candidate_area_cutoffs_px": list(B1_AREA_CUTOFFS),
                "selection": b1_selection,
                "frozen_area_cutoff_px": selected_b1,
            },
            "B2_extra_trees": {
                "estimator_config": dict(EXTRA_TREES_CONFIG),
                "feature_names": list(EXTRA_TREES_FEATURE_NAMES),
                "sampling_seed": SAMPLING_SEED,
                "samples_per_class_per_tile": SAMPLES_PER_CLASS_PER_TILE,
                "group_cv": b2_cv,
                "final_fit": b2_training,
                "model_file": model_path.name,
                "model_sha256": sha256_file(model_path),
                "model_semantic_sha256": model_semantic_sha256,
            },
        },
        "one_pass_evaluator_contract": frozen_one_pass_evaluator_contract(),
        "reported_selection_metrics": static["reported_selection_metrics"],
        "source_code_sha256": _source_hashes(),
        "environment_versions": _library_versions(),
    }
    identity = classical_scientific_identity_sha256(lock)
    lock["fit_id"] = f"classical-fit-{identity[:16]}"
    lock["scientific_identity_sha256"] = identity
    return lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and freeze task-matched classical comparators using training "
            "labels only"
        )
    )
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("config/confirmatory_splits.json")
    )
    parser.add_argument("--image-dir", type=Path, default=Path("original_images"))
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("results/step2_pore_classification/pore_classifications"),
    )
    parser.add_argument("--campaign-id", required=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> int:
    args = parse_args()
    try:
        campaign_id = validate_campaign_id(args.campaign_id)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    lock_root = (PROJECT_ROOT / CANONICAL_LOCK_ROOT).resolve()
    try:
        lock_root.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise SystemExit("Canonical classical lock path escapes the repository") from error
    cursor = PROJECT_ROOT.resolve()
    for component in CANONICAL_LOCK_ROOT.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise SystemExit("Canonical classical lock path contains a symbolic link")

    # Fail before creating an output directory when the declared runtime or
    # source/protocol bundle is incomplete. These snapshots also detect source
    # drift while a long train-only fit is running.
    runtime_before = _library_versions()
    source_hashes_before = _source_hashes()

    try:
        split_manifest_path = _require_canonical_input_path(
            args.split_manifest,
            CANONICAL_SPLIT_MANIFEST,
            label="Classical split manifest",
        )
        image_dir = _require_canonical_input_path(
            args.image_dir, CANONICAL_IMAGE_DIR, label="Classical image directory"
        )
        mask_dir = _require_canonical_input_path(
            args.mask_dir, CANONICAL_MASK_DIR, label="Classical mask directory"
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    contract = prepare_train_only_contract(
        split_manifest_path=split_manifest_path,
        image_dir=image_dir,
        mask_dir=mask_dir,
        enforce_confirmatory_attestations=True,
    )
    data_attestation_before = training_contract_attestation(contract)
    evidence = collect_train_only_evidence(contract)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    lock_root.mkdir(parents=True, exist_ok=True)
    partial_dir = lock_root / f".{campaign_id}.partial-{timestamp}"
    partial_dir.mkdir(exist_ok=False)
    lock = build_lock(
        campaign_id=campaign_id,
        contract=contract,
        evidence=evidence,
        output_dir=partial_dir,
    )
    contract_after = prepare_train_only_contract(
        split_manifest_path=split_manifest_path,
        image_dir=image_dir,
        mask_dir=mask_dir,
        enforce_confirmatory_attestations=True,
    )
    if training_contract_attestation(contract_after) != data_attestation_before:
        raise RuntimeError(
            "Canonical training data drifted during fitting; partial files: "
            f"{_safe_repo_path(partial_dir)}"
        )
    fit_id = validate_fit_id(lock["fit_id"])
    if lock["scientific_identity_sha256"] != classical_scientific_identity_sha256(
        lock
    ):
        raise RuntimeError("Classical scientific fit identity is inconsistent")
    output_dir = lock_root / fit_id
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(
            "Canonical content-addressed classical fit already exists; partial files: "
            f"{_safe_repo_path(partial_dir)}"
        )
    if lock["source_code_sha256"] != source_hashes_before:
        raise RuntimeError(
            "Comparator source/protocol drifted during fitting; partial files: "
            f"{_safe_repo_path(partial_dir)}"
        )
    if lock["environment_versions"] != runtime_before:
        raise RuntimeError(
            "Comparator runtime changed during fitting; partial files: "
            f"{_safe_repo_path(partial_dir)}"
        )
    partial_lock_path = partial_dir / "classical_comparator_lock.json"
    partial_lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial_dir.rename(output_dir)
    lock_path = output_dir / partial_lock_path.name
    print(
        json.dumps(
            {
                "status": lock["status"],
                "protocol_id": lock["protocol_id"],
                "fit_id": fit_id,
                "fit_execution_campaign_id": campaign_id,
                "scientific_identity_sha256": lock[
                    "scientific_identity_sha256"
                ],
                "lock_path": _safe_repo_path(lock_path),
                "lock_sha256": sha256_file(lock_path),
                "data_access": lock["data_access"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

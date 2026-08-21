#!/usr/bin/env python3
"""Report native-tile metrics for the frozen validation-only neural screen.

This script is deliberately separate from the locked retrospective evaluator.
It rebuilds and authenticates the existing selected-method lock, then evaluates
the 15 frozen screen checkpoints on the five validation tiles only.  The
result is model-development evidence: it must not be used to reselect a method
or represented as locked retrospective performance.

The scientific inference primitives are imported from
``evaluate_confirmatory_checkpoint.py``.  In particular, this reporter does
not define a second normalization, target mapping, model constructor, or
conditional output-composition path.

The full annotation index is never opened, hashed, or parsed. Its canonical
SHA-256 is checked only as recorded provenance; a separate byte pass hashes
exactly the 79 recorded train-plus-validation inputs and masks.

Rebuilding the selected-method lock necessarily parses and authenticates its
already-recorded train/validation/test partition metadata.  That metadata-only
operation does not resolve a live locked-retrospective filesystem path,
construct a live locked-retrospective dataset, or open any locked-retrospective
input, target, or annotation-index bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_confirmatory_checkpoint import (  # noqa: E402
    CHECKPOINT_ROLE,
    CONDITIONAL_CANDIDATES,
    EXPECTED_INPUT_NORMALIZATION,
    EXPECTED_TILE_SHAPE,
    LOCKED_ANNOTATION_INDEX_SHA256,
    LOCKED_CUDA_DEVICE_MODEL_TOKEN,
    LOCKED_EVALUATOR_INFERENCE_SEED,
    LOCKED_INFERENCE_PRECISION,
    LOCKED_INPUT_ATTESTATIONS,
    LOCKED_SPLIT_MANIFEST_SHA256,
    LOCKED_TARGET_ATTESTATIONS,
    NORMALIZATION_FORMULA,
    NORMALIZATION_ID,
    _autocast_context,
    _json_value,
    _safe_indexed_path,
    _strip_module_prefix,
    choose_device,
    compose_locked_probabilities,
    confusion_from_labels,
    create_model_from_state,
    infer_model_config_from_state,
    load_lossless_target_mask_bytes,
    metrics_from_confusion,
    preflight_locked_model_output,
    prepare_locked_model_input,
    validate_checkpoint_input_provenance,
    validate_checkpoint_normalization,
    validate_checkpoint_protocol_fields,
    validate_checkpoint_target_provenance,
    validate_checkpoint_training_seed,
    validate_locked_inference_runtime,
    validate_selected_checkpoint,
    validate_source_code_attestation,
)
from src.training.checkpoint_io import load_weights_only_checkpoint  # noqa: E402
from src.training.data_contract import aggregate_indexed_file_bytes  # noqa: E402
from src.training.screen_selection import (  # noqa: E402
    EXECUTION_SOURCE_FILES,
    PROSPECTIVE_METHOD_PROTOCOLS,
    SCREEN_CANDIDATE_ORDER,
    SCREEN_CELL_COUNT,
    SCREEN_SEEDS,
    SELECTED_METHOD_LOCK_SCHEMA_VERSION,
    verify_selected_method_lock_document,
)


REPORT_SCHEMA_VERSION = 1
EVIDENCE_KIND = "validation_screen_model_development_only"
EVIDENCE_LABEL = (
    "Model-development evidence from the frozen validation screen; not locked "
    "retrospective evaluation evidence"
)
CANONICAL_LOCK_IDENTIFIER = "config/selected_method_lock.json"
CANONICAL_IMAGE_ROOT = "results/step3_coco_dataset/images"
CANONICAL_MASK_ROOT = (
    "results/step2_pore_classification/pore_classifications"
)
CANONICAL_OUTPUT_ROOT = "results/validation_screen_model_development"
EXPECTED_VALIDATION_TILE_COUNT = 5
SCREEN_METRIC_ABSOLUTE_TOLERANCE = 5e-7
L40S_ALLOCATION_ENVIRONMENT_KEYS = (
    "PORE_ALLOCATED_GPU_NAME",
    "SLURM_JOB_GRES",
    "SLURM_STEP_GRES",
    "SLURM_TRES_PER_NODE",
)
OUTPUT_FILE_NAMES = (
    "validation_screen_report.json",
    "validation_screen_metrics.csv",
    "r3_signed_margins.csv",
)
MARGIN_METRICS = (
    "overall.accuracy",
    "c0.iou",
    "c0.dice",
    "c0.precision",
    "c0.recall",
    "c1.iou",
    "c1.dice",
    "c1.precision",
    "c1.recall",
    "c2.iou",
    "c2.dice",
    "c2.precision",
    "c2.recall",
    "selection.c0_c1_harmonic_iou",
    "selection.pore_union_iou",
)


def sha256_file(path: Path) -> str:
    """Hash one file without loading it wholesale into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_repository_path(
    repository_root: Path,
    identifier: str | Path,
    *,
    expected_identifier: Optional[str] = None,
) -> Path:
    """Resolve one repository path and reject traversal or symbolic links."""
    root = Path(repository_root).resolve()
    relative = Path(identifier)
    if relative.is_absolute():
        try:
            relative = relative.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"Path is outside the repository: {identifier}") from error
    if (
        not relative.parts
        or relative == Path(".")
        or ".." in relative.parts
        or "\\" in relative.as_posix()
    ):
        raise ValueError(f"Unsafe repository path: {identifier!r}")
    normalized_identifier = relative.as_posix()
    if expected_identifier is not None and normalized_identifier != expected_identifier:
        raise ValueError(
            f"Canonical path drift: {normalized_identifier!r} != "
            f"{expected_identifier!r}"
        )
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"Canonical path contains a symbolic link: {cursor}")
    try:
        cursor.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes the repository: {identifier}") from error
    return cursor


def _screen_cells(verified_lock: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Require the complete successful candidate-major 15-cell matrix."""
    if verified_lock.get("schema_version") != SELECTED_METHOD_LOCK_SCHEMA_VERSION:
        raise ValueError("Selected-method lock has the wrong schema version")
    selected_method = verified_lock.get("selected_method")
    if selected_method not in PROSPECTIVE_METHOD_PROTOCOLS:
        raise ValueError("Selected-method lock names an unknown method")
    if verified_lock.get("resolved_protocol") != PROSPECTIVE_METHOD_PROTOCOLS[
        selected_method
    ]:
        raise ValueError("Selected-method lock protocol drifted")
    provenance = verified_lock.get("screen_selection_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Selected-method lock lacks screen provenance")
    if provenance.get("deterministic_winner") != selected_method:
        raise ValueError("Selected-method lock winner fields disagree")
    cells = provenance.get("screen_cells")
    if not isinstance(cells, list) or len(cells) != SCREEN_CELL_COUNT:
        raise ValueError("Selected-method lock does not contain exactly 15 cells")
    expected_pairs = [
        (candidate, seed)
        for candidate in SCREEN_CANDIDATE_ORDER
        for seed in SCREEN_SEEDS
    ]
    observed_pairs: List[Tuple[object, object]] = []
    validated: List[Dict[str, Any]] = []
    for expected_index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError("Selected-method lock contains a non-object screen cell")
        observed_pairs.append((cell.get("candidate"), cell.get("seed")))
        if int(cell.get("array_index", -1)) != expected_index:
            raise ValueError("Selected-method lock screen-cell indices drifted")
        if cell.get("outcome_status") != "success":
            raise ValueError(
                "The validation-tile report requires all 15 frozen screen "
                "checkpoints; at least one cell was not successful"
            )
        if not isinstance(
            cell.get("selected_checkpoint_repo_relative_identifier"), str
        ) or not _is_sha256(cell.get("selected_checkpoint_sha256")):
            raise ValueError("A frozen screen checkpoint identity is incomplete")
        validated.append(dict(cell))
    if observed_pairs != expected_pairs:
        raise ValueError("Selected-method lock has the wrong candidate/seed matrix")
    return validated


def load_verified_screen_lock(
    lock_path: Path,
    repository_root: Path = PROJECT_ROOT,
) -> Tuple[Dict[str, Any], str, str, List[Dict[str, Any]]]:
    """Rebuild lock/checkpoint metadata without opening live corpus/index bytes."""
    path = _resolve_repository_path(
        repository_root,
        lock_path,
        expected_identifier=CANONICAL_LOCK_IDENTIFIER,
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    verified = verify_selected_method_lock_document(
        document,
        Path(repository_root),
        verify_live_corpus_bytes=False,
    )
    cells = _screen_cells(verified)
    return verified, sha256_file(path), CANONICAL_LOCK_IDENTIFIER, cells


def validation_tile_specs(verified_lock: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return only the five authenticated validation identities.

    The retrospective partition is neither projected into a path list nor
    returned by this function.
    """
    provenance = verified_lock.get("screen_selection_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Selected-method lock lacks screen provenance")
    data_split = provenance.get("resolved_data_split")
    if not isinstance(data_split, Mapping):
        raise ValueError("Selected-method lock lacks the resolved split")
    if (
        data_split.get("manifest_sha256") != LOCKED_SPLIT_MANIFEST_SHA256
        or data_split.get("manifest_repo_relative_identifier")
        != "config/confirmatory_splits.json"
        or data_split.get("annotation_index_sha256")
        != LOCKED_ANNOTATION_INDEX_SHA256
        or data_split.get("annotation_index_repo_relative_identifier")
        != "results/step3_coco_dataset/pore_annotations.json"
        or data_split.get("validation_only") is not True
        or data_split.get("held_out_dataset_constructed") is not False
        or int(data_split.get("held_out_evaluation_count", -1)) != 0
    ):
        raise ValueError(
            "Selected-method lock does not contain the canonical recorded split/index "
            "identity"
        )
    partitions = data_split.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("Selected-method lock lacks exact partition identities")
    validation = partitions.get("val")
    if not isinstance(validation, Mapping):
        raise ValueError("Selected-method lock lacks the validation partition")
    image_ids = validation.get("image_ids")
    image_files = validation.get("image_files")
    if (
        int(validation.get("image_count", -1)) != EXPECTED_VALIDATION_TILE_COUNT
        or not isinstance(image_ids, list)
        or not isinstance(image_files, list)
        or len(image_ids) != EXPECTED_VALIDATION_TILE_COUNT
        or len(image_files) != EXPECTED_VALIDATION_TILE_COUNT
        or len(set(image_ids)) != EXPECTED_VALIDATION_TILE_COUNT
        or len(set(image_files)) != EXPECTED_VALIDATION_TILE_COUNT
    ):
        raise ValueError("Validation partition is not exactly five unique tiles")
    specs: List[Dict[str, Any]] = []
    for ordinal, (image_id, file_name) in enumerate(
        zip(image_ids, image_files), start=1
    ):
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError("Validation image IDs must be integers")
        relative = Path(str(file_name))
        if (
            not str(file_name)
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in str(file_name)
            or not str(file_name).startswith("pdo8_21_segment_")
        ):
            raise ValueError("Validation file identity is not canonical pdo8_21")
        specs.append(
            {
                "validation_ordinal": ordinal,
                "image_id": int(image_id),
                "file_name": relative.as_posix(),
            }
        )
    return specs


def _development_identity(verified_lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Project authenticated dev79 identity without any retrospective bytes."""
    provenance = verified_lock["screen_selection_provenance"]
    input_provenance = provenance.get("input_provenance")
    target_provenance = provenance.get("target_provenance")
    if not isinstance(input_provenance, Mapping) or not isinstance(
        target_provenance, Mapping
    ):
        raise ValueError("Selected-method lock lacks development provenance")
    expected_input = LOCKED_INPUT_ATTESTATIONS[
        "development_train_plus_validation"
    ]
    expected_target = LOCKED_TARGET_ATTESTATIONS[
        "development_train_plus_validation"
    ]
    if (
        int(input_provenance.get("image_count", -1))
        != expected_input["image_count"]
        or input_provenance.get("image_aggregate_sha256")
        != expected_input["image_aggregate_sha256"]
        or input_provenance.get("held_out_bytes_read") != 0
        or int(target_provenance.get("mask_count", -1))
        != expected_target["mask_count"]
        or target_provenance.get("mask_aggregate_sha256")
        != expected_target["mask_aggregate_sha256"]
        or target_provenance.get("held_out_dataset_constructed") is not False
    ):
        raise ValueError("Selected-method lock dev79 identity drifted")
    return {
        "scope": "development_train_plus_validation_only",
        "input_image_count": int(input_provenance["image_count"]),
        "input_image_aggregate_sha256": input_provenance[
            "image_aggregate_sha256"
        ],
        "target_mask_count": int(target_provenance["mask_count"]),
        "target_mask_aggregate_sha256": target_provenance[
            "mask_aggregate_sha256"
        ],
        "held_out_bytes_read": 0,
    }


def _recorded_development_file_names(
    verified_lock: Mapping[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return recorded train/validation names without projecting the test split."""
    provenance = verified_lock.get("screen_selection_provenance")
    data_split = (
        provenance.get("resolved_data_split")
        if isinstance(provenance, Mapping)
        else None
    )
    partitions = data_split.get("partitions") if isinstance(data_split, Mapping) else None
    if not isinstance(partitions, Mapping):
        raise ValueError("Selected-method lock lacks recorded partitions")

    result: List[List[str]] = []
    for split_name, expected_count, allowed_prefixes in (
        (
            "train",
            74,
            (
                "pdo1_12_segment_",
                "pdo1_7_segment_",
                "pdo4_1_140721_segment_",
                "pdo4_2_151020_segment_",
            ),
        ),
        ("val", EXPECTED_VALIDATION_TILE_COUNT, ("pdo8_21_segment_",)),
    ):
        partition = partitions.get(split_name)
        if not isinstance(partition, Mapping):
            raise ValueError(f"Selected-method lock lacks recorded {split_name} files")
        names = partition.get("image_files")
        ids = partition.get("image_ids")
        if (
            int(partition.get("image_count", -1)) != expected_count
            or not isinstance(names, list)
            or not isinstance(ids, list)
            or len(names) != expected_count
            or len(ids) != expected_count
            or len(set(names)) != expected_count
            or len(set(ids)) != expected_count
        ):
            raise ValueError(
                f"Recorded {split_name} partition does not contain exactly "
                f"{expected_count} unique identities"
            )
        safe_names: List[str] = []
        for value in names:
            name = str(value)
            relative = Path(name)
            if (
                not name
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or relative.as_posix() != name
                or not name.startswith(allowed_prefixes)
            ):
                raise ValueError(
                    f"Recorded {split_name} file is not a canonical development identity"
                )
            safe_names.append(name)
        result.append(safe_names)
    train_names, validation_names = result
    if set(train_names) & set(validation_names):
        raise ValueError("Recorded train and validation filenames overlap")
    return train_names, validation_names


def independently_reattest_development_bytes(
    verified_lock: Mapping[str, Any],
    repository_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Hash exactly the recorded 79 train+validation inputs and masks.

    No annotation file, test filename, test path, directory traversal, or
    full-corpus scan is used by this pass.
    """
    train_names, validation_names = _recorded_development_file_names(verified_lock)
    development_names = train_names + validation_names
    if len(development_names) != 79 or len(set(development_names)) != 79:
        raise ValueError("Development re-attestation requires exactly 79 unique files")

    image_root = _resolve_repository_path(
        repository_root,
        CANONICAL_IMAGE_ROOT,
        expected_identifier=CANONICAL_IMAGE_ROOT,
    )
    mask_root = _resolve_repository_path(
        repository_root,
        CANONICAL_MASK_ROOT,
        expected_identifier=CANONICAL_MASK_ROOT,
    )
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("Canonical development image/mask roots are missing")
    for name in development_names:
        image_path = _resolve_repository_path(
            repository_root, Path(CANONICAL_IMAGE_ROOT) / name
        )
        mask_path = _resolve_repository_path(
            repository_root, Path(CANONICAL_MASK_ROOT) / name
        )
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Canonical development pair is missing: {name}")

    observed_inputs = aggregate_indexed_file_bytes(
        image_root,
        development_names,
        scope="development_train_plus_validation",
        split_names=("train", "val"),
    )
    observed_masks = aggregate_indexed_file_bytes(
        mask_root,
        development_names,
        scope="development_train_plus_validation",
        split_names=("train", "val"),
    )
    provenance = verified_lock["screen_selection_provenance"]
    recorded_inputs = provenance.get("input_provenance")
    recorded_targets = provenance.get("target_provenance")
    if not isinstance(recorded_inputs, Mapping) or not isinstance(
        recorded_targets, Mapping
    ):
        raise ValueError("Selected-method lock lacks development provenance")
    input_fields = {
        "scope": observed_inputs["scope"],
        "split_names": observed_inputs["split_names"],
        "image_count": observed_inputs["image_count"],
        "image_aggregate_sha256": observed_inputs["image_aggregate_sha256"],
        "image_aggregate_sha256_algorithm": observed_inputs[
            "image_aggregate_sha256_algorithm"
        ],
        "file_name_list_sha256": observed_inputs["file_name_list_sha256"],
    }
    if any(recorded_inputs.get(key) != value for key, value in input_fields.items()):
        raise ValueError("Live development input bytes differ from recorded dev79")
    target_fields = {
        "mask_count": observed_masks["image_count"],
        "mask_aggregate_sha256": observed_masks["image_aggregate_sha256"],
        "mask_aggregate_sha256_algorithm": observed_masks[
            "image_aggregate_sha256_algorithm"
        ],
    }
    if any(recorded_targets.get(key) != value for key, value in target_fields.items()):
        raise ValueError("Live development target bytes differ from recorded dev79")
    return {
        "verification_status": "matched_recorded_train_plus_validation_bytes",
        "scope": "development_train_plus_validation_only",
        "split_names": ["train", "val"],
        "input_files_read": 79,
        "target_files_read": 79,
        "input_image_aggregate_sha256": observed_inputs[
            "image_aggregate_sha256"
        ],
        "target_mask_aggregate_sha256": observed_masks[
            "image_aggregate_sha256"
        ],
        "input_file_name_list_sha256": observed_inputs["file_name_list_sha256"],
        "annotation_index_bytes_read": 0,
        "annotation_index_parsed": False,
        "locked_retrospective_input_files_read": 0,
        "locked_retrospective_target_files_read": 0,
        "locked_retrospective_paths_constructed": 0,
    }


def load_validation_tiles(
    specs: Sequence[Mapping[str, Any]],
    repository_root: Path = PROJECT_ROOT,
) -> List[Dict[str, Any]]:
    """Read and decode exactly the five canonical validation image/mask pairs."""
    if len(specs) != EXPECTED_VALIDATION_TILE_COUNT:
        raise ValueError("Exactly five validation tile specifications are required")
    image_root = _resolve_repository_path(
        repository_root,
        CANONICAL_IMAGE_ROOT,
        expected_identifier=CANONICAL_IMAGE_ROOT,
    )
    mask_root = _resolve_repository_path(
        repository_root,
        CANONICAL_MASK_ROOT,
        expected_identifier=CANONICAL_MASK_ROOT,
    )
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("Canonical validation image/mask roots are missing")

    validated_paths: List[Tuple[Mapping[str, Any], Path, Path]] = []
    for spec in specs:
        file_name = str(spec["file_name"])
        image_path = _resolve_repository_path(
            repository_root, Path(CANONICAL_IMAGE_ROOT) / file_name
        )
        mask_path = _resolve_repository_path(
            repository_root, Path(CANONICAL_MASK_ROOT) / file_name
        )
        # Retain the locked evaluator's traversal check as an independent
        # assertion while the repository resolver additionally rejects links.
        if (
            image_path != _safe_indexed_path(image_root, file_name)
            or mask_path != _safe_indexed_path(mask_root, file_name)
        ):
            raise ValueError("Validation path resolution disagrees with the locked evaluator")
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Canonical validation pair is missing: {file_name}")
        validated_paths.append((spec, image_path, mask_path))

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Validation reporting requires opencv-python") from error

    tiles: List[Dict[str, Any]] = []
    for spec, image_path, mask_path in validated_paths:
        file_name = str(spec["file_name"])
        image_payload = image_path.read_bytes()
        mask_payload = mask_path.read_bytes()
        image = cv2.imdecode(
            np.frombuffer(image_payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        image = validate_native_validation_image(image, file_name)
        target = load_lossless_target_mask_bytes(mask_path, mask_payload)
        image.setflags(write=False)
        target.setflags(write=False)
        tiles.append(
            {
                **dict(spec),
                "input_image_sha256": hashlib.sha256(image_payload).hexdigest(),
                "target_mask_sha256": hashlib.sha256(mask_payload).hexdigest(),
                "image": image,
                "target": target,
            }
        )
    if len(tiles) != EXPECTED_VALIDATION_TILE_COUNT:
        raise RuntimeError("Validation tile loading did not consume exactly five pairs")
    return tiles


def validate_native_validation_image(image: Any, file_name: str) -> np.ndarray:
    """Require one decoded raw uint8 grayscale image at native 2048 resolution."""
    if image is None:
        raise ValueError(f"Validation image {file_name} could not be decoded")
    array = np.asarray(image)
    if (
        array.dtype != np.uint8
        or array.ndim != 2
        or array.shape != EXPECTED_TILE_SHAPE
    ):
        raise ValueError(
            f"Validation image {file_name} is not a raw uint8 2048x2048 tile"
        )
    return array


def _checkpoint_development_attestations(
    checkpoint: Mapping[str, Any],
    verified_lock: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    provenance = verified_lock["screen_selection_provenance"]
    target = provenance["target_provenance"]
    inputs = provenance["input_provenance"]
    target_attestation = validate_checkpoint_target_provenance(checkpoint, target)
    input_attestation = validate_checkpoint_input_provenance(
        checkpoint,
        {
            "development_train_plus_validation": inputs,
            "training_only": inputs["training_subset"],
        },
    )
    return target_attestation, input_attestation


def _authenticated_screen_checkpoint_file(
    cell: Mapping[str, Any],
    repository_root: Path = PROJECT_ROOT,
) -> Tuple[Path, str, str]:
    """Require the canonical screen-cell checkpoint identifier and live hash."""
    identifier = str(cell["selected_checkpoint_repo_relative_identifier"])
    campaign_id = cell.get("campaign_id")
    cell_index = cell.get("array_index")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or isinstance(cell_index, bool)
        or not isinstance(cell_index, int)
        or cell_index not in range(SCREEN_CELL_COUNT)
    ):
        raise ValueError("Frozen screen cell has no canonical campaign/index identity")
    canonical_identifier = (
        Path("results/patch_training/protocol_runs/validation_screen_cell")
        / campaign_id
        / f"cell_{cell_index:02d}"
        / "checkpoints/best_model.pth"
    ).as_posix()
    if identifier != canonical_identifier:
        raise ValueError(
            f"Frozen screen checkpoint identifier is not canonical: {identifier}"
        )
    checkpoint_path = _resolve_repository_path(repository_root, identifier)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != cell.get("selected_checkpoint_sha256"):
        raise ValueError(f"Frozen screen checkpoint SHA-256 mismatch: {identifier}")
    return checkpoint_path, checkpoint_sha256, identifier


def validate_screen_checkpoint_envelope(
    cell: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    verified_lock: Mapping[str, Any],
    inferred_model: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    checkpoint_identifier: str,
    repository_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Validate checkpoint/source/split/dev79/protocol/seed linkage as one unit."""
    validate_selected_checkpoint(checkpoint)
    normalization = validate_checkpoint_normalization(checkpoint)
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("Screen checkpoint lacks resolved_config")
    expected_identity = {
        "protocol_run_role": "validation_screen_cell",
        "protocol_campaign_id": cell.get("campaign_id"),
        "protocol_cell_index": cell.get("array_index"),
        "protocol_candidate_key": cell.get("candidate"),
    }
    identity_mismatches = {
        key: {"checkpoint": resolved.get(key), "lock_cell": expected}
        for key, expected in expected_identity.items()
        if resolved.get(key) != expected
    }
    try:
        recorded_seed = int(resolved["augmentation"]["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Screen checkpoint lacks its training seed") from error
    if recorded_seed != int(cell["seed"]):
        identity_mismatches["training_seed"] = {
            "checkpoint": recorded_seed,
            "lock_cell": int(cell["seed"]),
        }
    if identity_mismatches:
        raise ValueError(f"Screen checkpoint identity mismatch: {identity_mismatches}")
    if resolved.get("selected_method_lock") is not None:
        raise ValueError("A screen checkpoint unexpectedly embeds a post-screen lock")
    evaluation = resolved.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("mode") != "validation_only"
        or evaluation.get("held_out_dataset_constructed") is not False
        or int(evaluation.get("held_out_evaluation_count", -1)) != 0
    ):
        raise ValueError("Screen checkpoint is not validation-only")

    provenance = verified_lock["screen_selection_provenance"]
    if (
        resolved.get("data_split") != provenance.get("resolved_data_split")
        or checkpoint.get("input_provenance") != provenance.get("input_provenance")
        or resolved.get("input") != provenance.get("input_provenance")
        or checkpoint.get("target_provenance") != provenance.get("target_provenance")
        or resolved.get("target") != provenance.get("target_provenance")
    ):
        raise ValueError("Screen checkpoint split/dev79 provenance differs from the lock")

    source_attestation = validate_source_code_attestation(
        checkpoint,
        verified_lock,
        repository_root=Path(repository_root),
    )
    target_attestation, input_attestation = _checkpoint_development_attestations(
        checkpoint, verified_lock
    )
    protocol_attestation = validate_checkpoint_protocol_fields(
        checkpoint,
        str(cell["candidate"]),
        inferred_model,
        "primary_multiscale",
    )
    seed_attestation = validate_checkpoint_training_seed(
        checkpoint,
        verified_lock,
        candidate=str(cell["candidate"]),
        architecture_role="primary_multiscale",
    )
    if seed_attestation["training_seed"] != int(cell["seed"]):
        raise ValueError("Screen checkpoint seed attestation differs from its lock cell")
    return {
        "checkpoint_path": checkpoint_identifier,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_role": CHECKPOINT_ROLE,
        "normalization": normalization,
        "source": source_attestation,
        "target": target_attestation,
        "input": input_attestation,
        "protocol": protocol_attestation,
        "seed": seed_attestation,
        "inferred_model": inferred_model,
    }


def authenticate_screen_checkpoint(
    cell: Mapping[str, Any],
    verified_lock: Mapping[str, Any],
    repository_root: Path = PROJECT_ROOT,
) -> Tuple[Mapping[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Authenticate one frozen screen checkpoint before model construction."""
    checkpoint_path, checkpoint_sha256, identifier = (
        _authenticated_screen_checkpoint_file(cell, repository_root)
    )
    checkpoint = load_weights_only_checkpoint(checkpoint_path, map_location="cpu")
    state = _strip_module_prefix(checkpoint.get("model_state_dict", {}))
    inferred_model = infer_model_config_from_state(state)
    identity = validate_screen_checkpoint_envelope(
        cell,
        checkpoint,
        verified_lock,
        inferred_model,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_identifier=identifier,
        repository_root=repository_root,
    )
    return checkpoint, state, identity


def _observed_selection_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    by_class = {int(item["class_id"]): item for item in metrics["per_class"]}
    return {
        "score": float(metrics["selection_metrics"]["c0_c1_harmonic_iou"]),
        "c0_iou": float(by_class[0]["iou"]),
        "c1_iou": float(by_class[1]["iou"]),
        "pore_union_iou": float(metrics["selection_metrics"]["pore_union_iou"]),
    }


def _validate_pooled_screen_identity(
    observed: Mapping[str, float],
    recorded: Mapping[str, Any],
) -> None:
    for name in ("score", "c0_iou", "c1_iou", "pore_union_iou"):
        actual = float(observed[name])
        expected = float(recorded[name])
        if not math.isfinite(actual) or not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=SCREEN_METRIC_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                "Re-evaluated validation confusion does not reproduce the frozen "
                f"screen metric {name}: {actual!r} != {expected!r}"
            )


def evaluate_screen_cell(
    cell: Mapping[str, Any],
    verified_lock: Mapping[str, Any],
    tiles: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    device: str,
    repository_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Run one authenticated checkpoint on all five validation tiles."""
    if len(tiles) != EXPECTED_VALIDATION_TILE_COUNT:
        raise ValueError("Each screen checkpoint must receive exactly five tiles")
    checkpoint, state, identity = authenticate_screen_checkpoint(
        cell, verified_lock, repository_root
    )
    candidate = str(cell["candidate"])
    protocol = PROSPECTIVE_METHOD_PROTOCOLS[candidate]
    model, model_config = create_model_from_state(state)
    model = model.to(device)
    model.eval()
    model_preflight = preflight_locked_model_output(
        model,
        torch,
        device=device,
        input_channels=int(protocol["model_input_channels"]),
        output_classes=int(protocol["model_output_classes"]),
        amp_enabled=True,
    )
    tile_reports: List[Dict[str, Any]] = []
    tile_confusions: List[np.ndarray] = []
    expected_output_shape = (
        1,
        int(protocol["model_output_classes"]),
        *EXPECTED_TILE_SHAPE,
    )
    for tile in tiles:
        image = np.asarray(tile["image"])
        target = np.asarray(tile["target"])
        model_input = prepare_locked_model_input(image, candidate)
        tensor = torch.from_numpy(model_input).unsqueeze(0).to(device)
        with torch.inference_mode(), _autocast_context(torch, device, True):
            output = model(tensor)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if tuple(logits.shape) != expected_output_shape:
                raise ValueError(
                    f"Unexpected validation output shape: {tuple(logits.shape)} != "
                    f"{expected_output_shape}"
                )
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("Validation inference produced non-finite logits")
            network_probabilities = (
                torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            )
        probabilities = compose_locked_probabilities(
            image, network_probabilities, candidate
        )
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        confusion = confusion_from_labels(target, prediction)
        metrics = metrics_from_confusion(confusion)
        tile_confusions.append(confusion)
        tile_reports.append(
            {
                "validation_ordinal": int(tile["validation_ordinal"]),
                "image_id": int(tile["image_id"]),
                "file_name": str(tile["file_name"]),
                "input_image_sha256": str(tile["input_image_sha256"]),
                "target_mask_sha256": str(tile["target_mask_sha256"]),
                "native_tile_shape": list(EXPECTED_TILE_SHAPE),
                "confusion_matrix": confusion,
                "metrics": metrics,
            }
        )
        del (
            model_input,
            tensor,
            output,
            logits,
            network_probabilities,
            probabilities,
            prediction,
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if len(tile_reports) != EXPECTED_VALIDATION_TILE_COUNT:
        raise RuntimeError("A screen checkpoint did not evaluate exactly five tiles")
    pooled_confusion = np.stack(tile_confusions).sum(axis=0)
    pooled_metrics = metrics_from_confusion(pooled_confusion)
    observed_selection = _observed_selection_metrics(pooled_metrics)
    recorded_selection = cell.get("selection_metrics")
    if not isinstance(recorded_selection, Mapping):
        raise ValueError("Screen lock cell lacks frozen selection metrics")
    _validate_pooled_screen_identity(observed_selection, recorded_selection)
    del model, checkpoint, state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "array_index": int(cell["array_index"]),
        "candidate": candidate,
        "seed": int(cell["seed"]),
        "screen_campaign_id": str(cell["campaign_id"]),
        "screen_result_artifact": {
            "path": cell["result_artifact_repo_relative_identifier"],
            "sha256": cell["result_artifact_sha256"],
        },
        "checkpoint_identity": identity,
        "model": model_config,
        "synthetic_native_tile_preflight": model_preflight,
        "validation_tile_count": EXPECTED_VALIDATION_TILE_COUNT,
        "pooled_confusion_matrix": pooled_confusion,
        "pooled_metrics": pooled_metrics,
        "frozen_selection_metrics": dict(recorded_selection),
        "pooled_selection_metrics_reproduced": True,
        "per_tile": tile_reports,
    }


def metric_vector(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten the fixed metrics used for signed descriptive R3 margins."""
    by_class = {int(item["class_id"]): item for item in metrics["per_class"]}
    values = {"overall.accuracy": float(metrics["accuracy"])}
    for class_id in range(3):
        for metric_name in ("iou", "dice", "precision", "recall"):
            values[f"c{class_id}.{metric_name}"] = float(
                by_class[class_id][metric_name]
            )
    values["selection.c0_c1_harmonic_iou"] = float(
        metrics["selection_metrics"]["c0_c1_harmonic_iou"]
    )
    values["selection.pore_union_iou"] = float(
        metrics["selection_metrics"]["pore_union_iou"]
    )
    if tuple(values) != MARGIN_METRICS:
        raise RuntimeError("Signed-margin metric order drifted")
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Pooled signed-margin metrics must all be finite")
    return values


def attach_signed_r3_margins(
    cell_reports: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Attach candidate-minus-R3 margins without ranking or selecting methods."""
    if len(cell_reports) != SCREEN_CELL_COUNT:
        raise ValueError("Signed margins require the complete 15-cell report")
    by_pair = {
        (str(cell["candidate"]), int(cell["seed"])): cell
        for cell in cell_reports
    }
    expected_pairs = {
        (candidate, int(seed))
        for candidate in SCREEN_CANDIDATE_ORDER
        for seed in SCREEN_SEEDS
    }
    if set(by_pair) != expected_pairs:
        raise ValueError("Signed margins require the exact candidate/seed matrix")
    vectors = {
        pair: metric_vector(cell["pooled_metrics"])
        for pair, cell in by_pair.items()
    }
    margin_rows: List[Dict[str, Any]] = []
    for candidate in SCREEN_CANDIDATE_ORDER:
        for seed in SCREEN_SEEDS:
            cell = by_pair[(candidate, int(seed))]
            reference = vectors[("R3", int(seed))]
            margins = {
                name: vectors[(candidate, int(seed))][name] - reference[name]
                for name in MARGIN_METRICS
            }
            cell["signed_r3_margins"] = {
                "definition": "candidate_minus_same_seed_R3",
                "positive_direction": "higher_metric_value_than_same_seed_R3",
                "values": margins,
            }
            for name in MARGIN_METRICS:
                margin_rows.append(
                    {
                        "scope": "same_seed",
                        "candidate": candidate,
                        "seed": int(seed),
                        "metric": name,
                        "candidate_value": vectors[(candidate, int(seed))][name],
                        "r3_value": reference[name],
                        "signed_margin_candidate_minus_r3": margins[name],
                    }
                )

    candidate_summaries: List[Dict[str, Any]] = []
    r3_means = {
        name: float(
            np.mean([vectors[("R3", int(seed))][name] for seed in SCREEN_SEEDS])
        )
        for name in MARGIN_METRICS
    }
    for candidate in SCREEN_CANDIDATE_ORDER:
        mean_values: Dict[str, float] = {}
        sample_sd: Dict[str, float] = {}
        signed: Dict[str, float] = {}
        for name in MARGIN_METRICS:
            values = np.asarray(
                [vectors[(candidate, int(seed))][name] for seed in SCREEN_SEEDS],
                dtype=np.float64,
            )
            mean_values[name] = float(values.mean())
            sample_sd[name] = float(values.std(ddof=1))
            signed[name] = mean_values[name] - r3_means[name]
            margin_rows.append(
                {
                    "scope": "three_seed_arithmetic_mean",
                    "candidate": candidate,
                    "seed": "",
                    "metric": name,
                    "candidate_value": mean_values[name],
                    "r3_value": r3_means[name],
                    "signed_margin_candidate_minus_r3": signed[name],
                }
            )
        candidate_summaries.append(
            {
                "candidate": candidate,
                "seeds": list(SCREEN_SEEDS),
                "seed_count": len(SCREEN_SEEDS),
                "three_seed_arithmetic_mean": mean_values,
                "three_seed_sample_standard_deviation": sample_sd,
                "signed_r3_margins": {
                    "definition": "candidate_mean_minus_R3_mean",
                    "positive_direction": "higher_metric_value_than_R3_mean",
                    "values": signed,
                },
            }
        )
    return candidate_summaries, margin_rows


def metric_csv_rows(cell_reports: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build deterministic tidy rows for tile and pooled class metrics."""
    rows: List[Dict[str, Any]] = []
    for cell in cell_reports:
        scopes = [
            (
                "tile",
                tile,
                tile["metrics"],
                tile["confusion_matrix"],
            )
            for tile in cell["per_tile"]
        ]
        scopes.append(
            (
                "pooled_five_validation_tiles",
                None,
                cell["pooled_metrics"],
                cell["pooled_confusion_matrix"],
            )
        )
        for aggregation, tile, metrics, confusion in scopes:
            for class_metric in metrics["per_class"]:
                rows.append(
                    {
                        "evidence_kind": EVIDENCE_KIND,
                        "candidate": cell["candidate"],
                        "seed": int(cell["seed"]),
                        "array_index": int(cell["array_index"]),
                        "aggregation": aggregation,
                        "validation_ordinal": (
                            int(tile["validation_ordinal"]) if tile else ""
                        ),
                        "image_id": int(tile["image_id"]) if tile else "",
                        "file_name": str(tile["file_name"]) if tile else "",
                        "input_image_sha256": (
                            str(tile["input_image_sha256"]) if tile else ""
                        ),
                        "target_mask_sha256": (
                            str(tile["target_mask_sha256"]) if tile else ""
                        ),
                        "class_id": int(class_metric["class_id"]),
                        "class_name": str(class_metric["class_name"]),
                        "support_pixels": int(class_metric["support_pixels"]),
                        "tp": int(class_metric["tp"]),
                        "fp": int(class_metric["fp"]),
                        "fn": int(class_metric["fn"]),
                        "tn": int(class_metric["tn"]),
                        "iou": float(class_metric["iou"]),
                        "dice": float(class_metric["dice"]),
                        "precision": float(class_metric["precision"]),
                        "recall": float(class_metric["recall"]),
                        "confusion_matrix_json": json.dumps(
                            _json_value(confusion),
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    }
                )
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return "" if not math.isfinite(number) else format(number, ".17g")
    if isinstance(value, (int, np.integer)):
        return int(value)
    return "" if value is None else value


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def write_report_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    margin_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    """Atomically write JSON/CSV payloads and their deterministic checksums."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    try:
        report_path = temporary / OUTPUT_FILE_NAMES[0]
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(
                _json_value(report),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        metric_fields = (
            "evidence_kind",
            "candidate",
            "seed",
            "array_index",
            "aggregation",
            "validation_ordinal",
            "image_id",
            "file_name",
            "input_image_sha256",
            "target_mask_sha256",
            "class_id",
            "class_name",
            "support_pixels",
            "tp",
            "fp",
            "fn",
            "tn",
            "iou",
            "dice",
            "precision",
            "recall",
            "confusion_matrix_json",
        )
        _write_csv(temporary / OUTPUT_FILE_NAMES[1], metric_fields, metric_rows)
        margin_fields = (
            "scope",
            "candidate",
            "seed",
            "metric",
            "candidate_value",
            "r3_value",
            "signed_margin_candidate_minus_r3",
        )
        _write_csv(temporary / OUTPUT_FILE_NAMES[2], margin_fields, margin_rows)
        checksums = {
            name: sha256_file(temporary / name) for name in OUTPUT_FILE_NAMES
        }
        with (temporary / "checksums.sha256").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            for name in sorted(checksums):
                handle.write(f"{checksums[name]}  {name}\n")
        temporary.replace(output_dir)
        return checksums
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_report(
    *,
    verified_lock: Mapping[str, Any],
    lock_sha256: str,
    lock_identifier: str,
    cell_reports: Sequence[Dict[str, Any]],
    candidate_summaries: Sequence[Mapping[str, Any]],
    runtime_attestation: Mapping[str, Any],
    development_byte_attestation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the deterministic model-development report document."""
    provenance = verified_lock["screen_selection_provenance"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evidence_kind": EVIDENCE_KIND,
        "evidence_label": EVIDENCE_LABEL,
        "scientific_scope": {
            "model_development_only": True,
            "winner_reselection_performed": False,
            "recorded_partition_metadata_authenticated": True,
            "live_locked_retrospective_dataset_constructed": False,
            "live_locked_retrospective_filesystem_paths_resolved": False,
            "locked_retrospective_input_bytes_read": 0,
            "locked_retrospective_target_bytes_read": 0,
            "locked_retrospective_inference_count": 0,
            "annotation_index_bytes_read": 0,
            "annotation_index_parsed": False,
            "partition_metadata_statement": (
                "Recorded test-partition metadata was necessarily parsed and "
                "authenticated from the selected-method lock; no live locked-"
                "retrospective filesystem path or dataset was resolved or "
                "constructed, and no retrospective input, target, or annotation "
                "bytes were opened"
            ),
            "publication_interpretation": (
                "Validation-screen diagnostics only; do not label as test, held-out, "
                "confirmatory, or unseen-partition performance"
            ),
        },
        "selected_method_lock": {
            "path": lock_identifier,
            "sha256": lock_sha256,
            "schema_version": verified_lock["schema_version"],
            "selected_method_from_existing_lock": verified_lock["selected_method"],
            "screen_campaign_id": provenance["campaign_id"],
            "screen_cell_count": SCREEN_CELL_COUNT,
            "candidate_major_order": list(SCREEN_CANDIDATE_ORDER),
            "seeds": list(SCREEN_SEEDS),
            "selection_performed_by_reporter": False,
        },
        "data": {
            "evaluation_split": "validation",
            "validation_tile_count_per_checkpoint": EXPECTED_VALIDATION_TILE_COUNT,
            "native_tile_shape": list(EXPECTED_TILE_SHAPE),
            "split_manifest_path": "config/confirmatory_splits.json",
            "split_manifest_sha256": LOCKED_SPLIT_MANIFEST_SHA256,
            "recorded_annotation_index_identifier": (
                "results/step3_coco_dataset/pore_annotations.json"
            ),
            "recorded_annotation_index_sha256": LOCKED_ANNOTATION_INDEX_SHA256,
            "annotation_identity_verification": (
                "matched frozen SHA-256 in recorded lock provenance; annotation "
                "file bytes were not opened, read, hashed, or parsed by this reporter"
            ),
            "annotation_index_bytes_read": 0,
            "annotation_index_parsed": False,
            "image_root": CANONICAL_IMAGE_ROOT,
            "target_mask_root": CANONICAL_MASK_ROOT,
            "development_identity": _development_identity(verified_lock),
            "independent_development_byte_attestation": dict(
                development_byte_attestation
            ),
            "target_source": "lossless PNG masks with source values 0, 1, 255",
            "source_to_canonical_mask_value": {"0": 0, "1": 1, "255": 2},
            "live_retrospective_partition_paths_constructed": False,
        },
        "inference": {
            "runtime": dict(runtime_attestation),
            "inference_seed": LOCKED_EVALUATOR_INFERENCE_SEED,
            "batch_size_tiles": 1,
            "resize": None,
            "input_normalization_id": NORMALIZATION_ID,
            "input_normalization": EXPECTED_INPUT_NORMALIZATION,
            "input_normalization_formula": NORMALIZATION_FORMULA,
            "conditional_candidates": sorted(CONDITIONAL_CANDIDATES),
            "normalization_implementation": (
                "scripts.evaluate_confirmatory_checkpoint.prepare_locked_model_input"
            ),
            "output_composition_implementation": (
                "scripts.evaluate_confirmatory_checkpoint.compose_locked_probabilities"
            ),
            "post_processing": "none",
        },
        "signed_r3_margin_definition": {
            "seed_level": "candidate metric minus same-seed R3 metric",
            "candidate_level": (
                "candidate three-seed arithmetic mean minus R3 three-seed "
                "arithmetic mean"
            ),
            "selection_role": "descriptive_only_no_winner_reselection",
        },
        "candidate_summaries": list(candidate_summaries),
        "cells": list(cell_reports),
        "code": {
            "reporter_path": "scripts/report_validation_screen_tiles.py",
            "reporter_sha256": sha256_file(Path(__file__).resolve()),
            "frozen_training_source_sha256": provenance["source_code_sha256"],
        },
        "output_contract": {
            "payload_files": list(OUTPUT_FILE_NAMES),
            "checksum_manifest": "checksums.sha256",
            "checksum_algorithm": "sha256",
            "timestamps_in_payloads": False,
        },
    }


def _early_runtime_intent_guard(requested_device: str) -> Dict[str, Any]:
    """Reject non-scheduler/non-CUDA/non-L40S intent before any file access."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Validation-screen reporting requires an active Slurm GPU allocation"
        )
    requested = str(requested_device).strip().lower()
    if requested not in {"auto", "cuda"} and not (
        requested.startswith("cuda:") and requested[5:].isdigit()
    ):
        raise ValueError(
            "Validation-screen reporting device intent must be auto, cuda, or cuda:N"
        )
    l40s_sources = {
        key: str(os.environ[key])
        for key in L40S_ALLOCATION_ENVIRONMENT_KEYS
        if os.environ.get(key)
        and "l40s" in str(os.environ[key]).lower()
    }
    if not l40s_sources:
        raise RuntimeError(
            "Validation-screen reporting requires recorded Slurm L40S allocation "
            "intent before repository authentication"
        )
    return {
        "verification_status": "early_scheduler_cuda_l40s_intent_passed",
        "requested_device": requested,
        "l40s_intent_source_keys": sorted(l40s_sources),
    }


def _require_scheduler_l40s(torch: Any, requested_device: str) -> Tuple[str, Dict[str, Any]]:
    early = _early_runtime_intent_guard(requested_device)
    device = choose_device(requested_device)
    if not device.startswith("cuda"):
        raise ValueError("Validation-screen reporting requires CUDA")
    cuda_name = torch.cuda.get_device_name(torch.device(device))
    attestation = validate_locked_inference_runtime(
        device,
        True,
        cuda_device_name=cuda_name,
    )
    if (
        attestation.get("precision_protocol") != LOCKED_INFERENCE_PRECISION
        or attestation.get("locked_device_model_token")
        != LOCKED_CUDA_DEVICE_MODEL_TOKEN
    ):
        raise ValueError("Validation inference runtime attestation drifted")
    attestation = {**dict(attestation), "early_intent": early}
    return device, attestation


def run_report(
    lock_path: Path,
    *,
    requested_device: str = "auto",
    repository_root: Path = PROJECT_ROOT,
) -> Path:
    """Authenticate, evaluate, and atomically write the complete report."""
    # This guard intentionally precedes torch import and every repository,
    # lock, corpus, checkpoint, and output operation.
    _early_runtime_intent_guard(requested_device)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Validation reporting requires PyTorch") from error

    device, runtime_attestation = _require_scheduler_l40s(torch, requested_device)
    verified_lock, lock_sha256, lock_identifier, cells = load_verified_screen_lock(
        lock_path, repository_root
    )
    specs = validation_tile_specs(verified_lock)
    _development_identity(verified_lock)
    development_byte_attestation = independently_reattest_development_bytes(
        verified_lock, repository_root
    )

    random.seed(LOCKED_EVALUATOR_INFERENCE_SEED)
    np.random.seed(LOCKED_EVALUATOR_INFERENCE_SEED)
    torch.manual_seed(LOCKED_EVALUATOR_INFERENCE_SEED)
    torch.cuda.manual_seed_all(LOCKED_EVALUATOR_INFERENCE_SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # No retrospective path list is passed to this loader.  It reads only the
    # five validation identities projected above from the verified lock.
    tiles = load_validation_tiles(specs, repository_root)
    cell_reports = [
        evaluate_screen_cell(
            cell,
            verified_lock,
            tiles,
            torch=torch,
            device=device,
            repository_root=repository_root,
        )
        for cell in cells
    ]
    candidate_summaries, margin_rows = attach_signed_r3_margins(cell_reports)
    metric_rows = metric_csv_rows(cell_reports)
    report = build_report(
        verified_lock=verified_lock,
        lock_sha256=lock_sha256,
        lock_identifier=lock_identifier,
        cell_reports=cell_reports,
        candidate_summaries=candidate_summaries,
        runtime_attestation=runtime_attestation,
        development_byte_attestation=development_byte_attestation,
    )
    output_dir = _resolve_repository_path(
        repository_root,
        Path(CANONICAL_OUTPUT_ROOT) / lock_sha256,
    )
    write_report_bundle(output_dir, report, metric_rows, margin_rows)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report per-tile model-development metrics for the complete frozen "
            "15-cell validation screen"
        )
    )
    parser.add_argument(
        "--selected-method-lock",
        default=CANONICAL_LOCK_IDENTIFIER,
        help=(
            "Canonical existing selected-method lock; the reporter verifies but "
            "does not rebuild a winner from its new metrics"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto or cuda[:index]; an allocated NVIDIA L40S is required",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run_report(
        Path(args.selected_method_lock),
        requested_device=args.device,
    )
    print(
        "Wrote validation-screen model-development evidence to "
        f"{output.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()

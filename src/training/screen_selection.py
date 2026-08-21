"""Deterministic validation-screen aggregation and selected-method locks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from src.training.checkpoint_io import load_weights_only_checkpoint
from src.training.data_contract import (
    CONFIRMATORY_ANNOTATION_SHA256,
    CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS,
    CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS,
    CONFIRMATORY_SPLIT_MANIFEST_SHA256,
    aggregate_indexed_file_bytes,
    resolve_split_manifest,
)


SCREEN_CANDIDATE_ORDER = ("R3", "H3", "C2-P", "C2-F", "C2-FP")
SCREEN_SEEDS = (42, 123, 2025)
SCREEN_CELL_COUNT = len(SCREEN_CANDIDATE_ORDER) * len(SCREEN_SEEDS)
SCREEN_RESULT_SCHEMA_VERSION = 1
SELECTED_METHOD_LOCK_SCHEMA_VERSION = 1
SMOKE_PREFLIGHT_SCHEMA_VERSION = 1
SMOKE_CANDIDATE_ORDER = ("R3", "C2-F", "C2-FP")
SMOKE_CELL_COUNT = len(SMOKE_CANDIDATE_ORDER)
CANONICAL_GROUP_MEMBERSHIP_MAP = {
    "train": ["pdo1_12", "pdo1_7", "pdo4_1_140721", "pdo4_2_151020"],
    "val": ["pdo8_21"],
    "test": ["pdo2_24"],
}

# Shared by the trainer, screen/lock builders, and failure recorder.  Keeping
# the list in this lightweight module avoids importing CUDA-heavy training
# code merely to attest a failed scheduler cell.
EXECUTION_SOURCE_FILES = (
    "scripts/train_patches.py",
    "scripts/evaluate_confirmatory_checkpoint.py",
    "src/training/patch_trainer.py",
    "src/training/checkpoint_io.py",
    "src/training/patch_dataset.py",
    "src/training/data_contract.py",
    "src/training/augmentations.py",
    "src/training/screen_selection.py",
    "src/training/neural_freeze.py",
    "src/models/focal_loss.py",
    "src/models/combined_loss.py",
    "src/models/hierarchical_pore_loss.py",
    "src/models/conditional_pore_loss.py",
    "src/models/unet_model.py",
    "src/models/multiscale_attention_unet.py",
    "src/models/pyramid_context.py",
    "config/pipeline_config.yaml",
    "config/confirmatory_splits.json",
    "scripts/aire_confirmatory.slurm",
    "scripts/aire_validation_screen.slurm",
    "scripts/aire_validation_smoke.slurm",
    "scripts/aire_selected_retrain.slurm",
    "scripts/aire_locked_evaluation.slurm",
    "scripts/build_smoke_preflight_manifest.py",
    "scripts/build_selected_method_lock.py",
    "scripts/build_neural_freeze_manifest.py",
    "scripts/record_protocol_cell_failure.py",
)

PROSPECTIVE_METHOD_PROTOCOLS = {
    "R3": {
        "model_type": "multiscale_attention_unet",
        "loss_type": "focal_dice",
        "model_input_channels": 1,
        "model_output_classes": 3,
        "training_patch_size": 683,
        "training_batch_size": 4,
        "evaluation_patch_size": 2048,
        "evaluation_batch_size": 1,
        "conditional_pore_threshold": None,
        "dropout_requested": 0.2,
        "mixed_precision_requested": True,
    },
    "H3": {
        "model_type": "multiscale_attention_unet",
        "loss_type": "hierarchical_pore_connectivity",
        "model_input_channels": 1,
        "model_output_classes": 3,
        "training_patch_size": 683,
        "training_batch_size": 4,
        "evaluation_patch_size": 2048,
        "evaluation_batch_size": 1,
        "conditional_pore_threshold": None,
        "dropout_requested": 0.2,
        "mixed_precision_requested": True,
    },
    "C2-P": {
        "model_type": "multiscale_attention_unet",
        "loss_type": "conditional_pore_focal_dice",
        "model_input_channels": 2,
        "model_output_classes": 2,
        "training_patch_size": 683,
        "training_batch_size": 4,
        "evaluation_patch_size": 2048,
        "evaluation_batch_size": 1,
        "conditional_pore_threshold": 100,
        "dropout_requested": 0.2,
        "mixed_precision_requested": True,
    },
    "C2-F": {
        "model_type": "multiscale_attention_unet",
        "loss_type": "conditional_pore_focal_dice",
        "model_input_channels": 2,
        "model_output_classes": 2,
        "training_patch_size": 2048,
        "training_batch_size": 1,
        "evaluation_patch_size": 2048,
        "evaluation_batch_size": 1,
        "conditional_pore_threshold": 100,
        "dropout_requested": 0.2,
        "mixed_precision_requested": True,
    },
    "C2-FP": {
        "model_type": "multiscale_attention_unet_pyramid",
        "loss_type": "conditional_pore_focal_dice",
        "model_input_channels": 2,
        "model_output_classes": 2,
        "training_patch_size": 2048,
        "training_batch_size": 1,
        "evaluation_patch_size": 2048,
        "evaluation_batch_size": 1,
        "conditional_pore_threshold": 100,
        "dropout_requested": 0.0,
        "mixed_precision_requested": True,
    },
}

SELECTION_RULE = {
    "candidate_aggregation": "arithmetic_mean_over_prespecified_three_seeds",
    "alternative_eligibility": [
        "mean_harmonic_c0_c1_iou_strictly_greater_than_R3",
        "mean_c0_iou_not_lower_than_R3",
        "mean_c1_iou_not_lower_than_R3",
        "all_three_cells_successful_and_scientific_eligible",
    ],
    "winner_order": [
        "highest_mean_harmonic_c0_c1_iou",
        "highest_mean_pore_union_iou",
        "lowest_mean_validation_loss",
    ],
    "unresolved_exact_three_way_tie": "fail_closed_without_lock",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_code_sha256(repository_root: Path) -> Dict[str, Optional[str]]:
    """Return the common repo-relative executable snapshot."""
    root = Path(repository_root)
    return {
        identifier: (
            sha256_file(root / identifier)
            if (root / identifier).is_file()
            else None
        )
        for identifier in EXECUTION_SOURCE_FILES
    }


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repo_relative(path: Path, repository_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"screen result must be inside the repository/staging root: {path}"
        ) from error


def _assert_no_symlink_components(path: Path, repository_root: Path) -> Path:
    root = Path(repository_root).resolve()
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Canonical protocol path escapes repository: {path}") from error
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(
                f"Canonical protocol path contains a symbolic-link component: {cursor}"
            )
    return lexical


def _finite_unit_interval(value, field: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1], got {value!r}")
    return numeric


def _require_close(actual, expected, field: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError(f"{field} mismatch: {actual!r} != {expected!r}")


def _validate_scientific_execution_contract(
    contract: Mapping, candidate: str, seed: int
) -> None:
    """Reject a cell whose live objects diverged from the frozen protocol."""
    if not isinstance(contract, Mapping):
        raise ValueError("screen result lacks scientific execution contract")
    if int(contract.get("epochs_planned", -1)) != 30:
        raise ValueError("screen epoch budget must be 30")
    early = contract.get("early_stopping", {})
    if early != {
        "enabled": True,
        "patience": 10,
        "selection_metric": "validation_c0_c1_iou_harmonic_mean",
    }:
        raise ValueError("screen early-stopping contract mismatch")
    optimizer = contract.get("optimizer", {})
    if optimizer.get("requested") != "adamw" or optimizer.get(
        "implementation_class"
    ) != "AdamW":
        raise ValueError("screen optimizer must resolve to AdamW")
    for key in ("configured_learning_rate", "actual_initial_learning_rate"):
        _require_close(optimizer.get(key), 5e-4, f"optimizer.{key}")
    for key in ("configured_weight_decay", "actual_weight_decay"):
        _require_close(optimizer.get(key), 1e-4, f"optimizer.{key}")
    scheduler = contract.get("scheduler", {})
    if scheduler != {
        "requested": "cosine",
        "implementation_class": "CosineAnnealingLR",
        "t_max": 30,
        "step_unit": "epoch",
    }:
        raise ValueError("screen scheduler contract mismatch")
    if int(contract.get("bootstrap_factor", -1)) != 1:
        raise ValueError("screen bootstrap factor must be one")
    if int(contract.get("workers", -1)) != 8:
        raise ValueError("screen must use eight data-loader workers")
    _require_close(contract.get("gradient_clip_val"), 1.0, "gradient_clip_val")
    if contract.get("mixed_precision_requested") is not True or contract.get(
        "mixed_precision_actual"
    ) is not True:
        raise ValueError("screen requires requested and executed CUDA AMP")
    if int(contract.get("accumulate_grad_batches", -1)) != 1:
        raise ValueError("screen gradient accumulation must be one")
    if contract.get("batch_mixup_enabled") is not False or contract.get(
        "batch_cutmix_enabled"
    ) is not False:
        raise ValueError("screen forbids MixUp and CutMix")

    augmentation = contract.get("augmentation", {})
    if (
        augmentation.get("enabled") is not True
        or augmentation.get("training") is not True
        or augmentation.get("strength") != "light"
        or int(augmentation.get("seed", -1)) != seed
        or augmentation.get("albumentations_version") != "2.0.8"
    ):
        raise ValueError("screen augmentation policy/version/seed mismatch")
    if augmentation.get("transforms") != [
        {"name": "RandomRotate90", "probability": 0.5},
        {"name": "HorizontalFlip", "probability": 0.5},
        {"name": "VerticalFlip", "probability": 0.5},
    ]:
        raise ValueError("screen transform list/probabilities mismatch")
    batch_level = augmentation.get("batch_level", {})
    if (
        batch_level.get("application_probability") != 0.0
        or batch_level.get("mixup_enabled") is not False
        or batch_level.get("cutmix_enabled") is not False
    ):
        raise ValueError("screen batch augmentation mismatch")
    loader = augmentation.get("data_loader", {})
    if (
        loader.get("shuffle_generator_seed") != seed
        or loader.get("num_workers") != 8
        or loader.get("worker_init_function")
        != "src.training.patch_dataset.seed_patch_dataloader_worker"
        or loader.get("effective_worker_seed_formula")
        != "(augmentation_seed + (torch.initial_seed() % 2**32)) % 2**32"
        or loader.get("albumentations_compose_reseeded_for_training") is not True
    ):
        raise ValueError("screen data-loader RNG contract mismatch")

    model = contract.get("model", {})
    protocol = PROSPECTIVE_METHOD_PROTOCOLS[candidate]
    if (
        model.get("architecture_resolved") != protocol["model_type"]
        or model.get("input_channels") != protocol["model_input_channels"]
        or model.get("output_classes") != protocol["model_output_classes"]
        or model.get("deep_supervision") is not False
        or not isinstance(model.get("parameter_count"), int)
        or model.get("parameter_count", 0) <= 0
    ):
        raise ValueError("screen resolved model contract mismatch")
    if candidate == "C2-FP":
        if model.get("dropout_requested") != 0.0:
            raise ValueError("C2-FP requested dropout must be 0.0")
        if model.get("dropout_execution") != {
            "status": "executed_in_pyramid_context",
            "probability": 0.0,
        }:
            raise ValueError("C2-FP executed dropout must be 0.0")
    elif model.get("dropout_execution") != {
        "status": "unused_by_resolved_architecture",
        "probability": None,
    }:
        raise ValueError("non-pyramid dropout request must be marked unused")

    loss = contract.get("loss", {})
    candidate_loss = loss.get("candidate", {})
    if candidate == "R3":
        if (
            loss.get("implementation_class") != "CombinedFocalDiceLoss"
            or loss.get("class_weights_requested") != [3.0, 2.0, 1.0]
            or loss.get("class_weights_actual") != [3.0, 2.0, 1.0]
            or candidate_loss.get("component_weights_actual")
            != {"focal": 0.5, "dice": 0.5}
        ):
            raise ValueError("R3 focal-Dice/class-weight contract mismatch")
        _require_close(
            candidate_loss.get("focal_gamma_actual"), 2.0,
            "R3 focal gamma",
        )
    elif candidate == "H3":
        if (
            loss.get("implementation_class")
            != "HierarchicalPoreConnectivityLoss"
            or loss.get("class_weights_requested") is not None
            or loss.get("class_weights_actual") is not None
        ):
            raise ValueError("H3 loss implementation mismatch")
        for value in candidate_loss.get("component_weights_normalized", []):
            _require_close(value, 1.0 / 3.0, "H3 component weight", 2e-8)
        if len(candidate_loss.get("component_weights_normalized", [])) != 3:
            raise ValueError("H3 must have three equal normalized components")
    else:
        if (
            loss.get("implementation_class") != "ConditionalPoreFocalDiceLoss"
            or loss.get("class_weights_requested") is not None
        ):
            raise ValueError("conditional loss implementation mismatch")
        if candidate_loss.get("component_weights_normalized") != [0.5, 0.5]:
            raise ValueError("conditional component weights must be equal")
        _require_close(
            candidate_loss.get("focal_gamma"), 2.0,
            "conditional focal gamma",
        )
    if candidate != "R3":
        statistics = loss.get("training_class_statistics", {})
        if (
            statistics.get("source") != "authoritative_training_masks_only"
            or not _is_sha256(
                statistics.get("training_mask_aggregate_sha256")
            )
            or statistics.get("training_mask_count")
            != CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS["train"][
                "mask_count"
            ]
            or statistics.get("training_mask_aggregate_sha256")
            != CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS["train"][
                "mask_aggregate_sha256"
            ]
            or candidate_loss.get("class_count_source")
            != "authoritative_training_masks_only"
        ):
            raise ValueError("candidate train-only class-balance provenance mismatch")


def _canonical_cell_identifier(
    role: str, campaign_id: str, cell_index: int, suffix: str
) -> str:
    return (
        Path("results/patch_training/protocol_runs")
        / role
        / campaign_id
        / f"cell_{cell_index:02d}"
        / suffix
    ).as_posix()


def _resolve_authenticated_cell_checkpoint(
    repository_root: Path,
    *,
    role: str,
    campaign_id: str,
    cell_index: int,
    checkpoint_identifier: object,
    expected_sha256: object,
) -> Path:
    canonical = _canonical_cell_identifier(
        role, campaign_id, cell_index, "checkpoints/best_model.pth"
    )
    if checkpoint_identifier != canonical or not _is_sha256(expected_sha256):
        raise ValueError("selected checkpoint identifier/hash is not canonical")
    path = _assert_no_symlink_components(
        Path(repository_root) / canonical, Path(repository_root)
    )
    _repo_relative(path, Path(repository_root))
    if not path.is_file():
        raise FileNotFoundError(f"selected checkpoint is missing: {canonical}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"selected checkpoint SHA-256 mismatch: {actual_sha256} != "
            f"{expected_sha256}"
        )
    return path


def _validate_input_provenance(
    value: Mapping,
    data_split: Mapping,
    repository_root: Path,
    identifier: str,
    *,
    verify_live_corpus_bytes: bool = True,
) -> None:
    expected_train = CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS["train"]
    expected_development = CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS[
        "train_plus_validation"
    ]
    training = value.get("training_subset", {}) if isinstance(value, Mapping) else {}
    if (
        not isinstance(value, Mapping)
        or value.get("input_source") != "indexed_source_images"
        or value.get("scope") != "development_train_plus_validation"
        or value.get("split_names") != ["train", "val"]
        or value.get("held_out_bytes_read") != 0
        or int(value.get("image_count", -1))
        != expected_development["image_count"]
        or value.get("image_aggregate_sha256")
        != expected_development["image_aggregate_sha256"]
        or not isinstance(training, Mapping)
        or training.get("scope") != "training_only"
        or training.get("split_names") != ["train"]
        or int(training.get("image_count", -1)) != expected_train["image_count"]
        or training.get("image_aggregate_sha256")
        != expected_train["image_aggregate_sha256"]
    ):
        raise ValueError(
            f"protocol input-image attestation mismatch: {identifier}"
        )
    partitions = data_split.get("partitions", {})
    train_names = partitions.get("train", {}).get("image_files", [])
    validation_names = partitions.get("val", {}).get("image_files", [])
    if not verify_live_corpus_bytes:
        algorithm = (
            "sha256 over lexicographically sorted UTF-8 relative filename, "
            "NUL, raw file bytes, NUL"
        )
        expected_train_name_sha256 = hashlib.sha256(
            json.dumps(sorted(train_names), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        expected_development_name_sha256 = hashlib.sha256(
            json.dumps(
                sorted(list(train_names) + list(validation_names)),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            training.get("image_aggregate_sha256_algorithm") != algorithm
            or value.get("image_aggregate_sha256_algorithm") != algorithm
            or training.get("file_name_list_sha256")
            != expected_train_name_sha256
            or value.get("file_name_list_sha256")
            != expected_development_name_sha256
        ):
            raise ValueError(
                f"protocol input-image filename provenance mismatch: {identifier}"
            )
        return
    actual_train = aggregate_indexed_file_bytes(
        Path(repository_root) / "results/step3_coco_dataset/images",
        train_names,
        scope="training_only",
        split_names=("train",),
    )
    actual_development = aggregate_indexed_file_bytes(
        Path(repository_root) / "results/step3_coco_dataset/images",
        list(train_names) + list(validation_names),
        scope="development_train_plus_validation",
        split_names=("train", "val"),
    )
    if actual_train != dict(training) or any(
        value.get(key) != actual_development.get(key)
        for key in (
            "scope",
            "split_names",
            "image_count",
            "image_aggregate_sha256",
            "image_aggregate_sha256_algorithm",
            "file_name_list_sha256",
        )
    ):
        raise ValueError(f"protocol input-image bytes changed: {identifier}")


def _validate_data_attestations(
    data_split: Mapping,
    target: Mapping,
    repository_root: Path,
    identifier: str,
    *,
    verify_live_corpus_bytes: bool = True,
) -> None:
    expected_target = CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS[
        "train_plus_validation"
    ]
    partitions = data_split.get("partitions", {}) if isinstance(
        data_split, Mapping
    ) else {}
    if (
        not isinstance(data_split, Mapping)
        or data_split.get("manifest_source") != "explicit_manifest"
        or data_split.get("manifest_sha256")
        != CONFIRMATORY_SPLIT_MANIFEST_SHA256
        or data_split.get("manifest_repo_relative_identifier")
        != "config/confirmatory_splits.json"
        or data_split.get("annotation_index_sha256")
        != CONFIRMATORY_ANNOTATION_SHA256
        or data_split.get("annotation_index_repo_relative_identifier")
        != "results/step3_coco_dataset/pore_annotations.json"
        or not _is_sha256(data_split.get("partition_assignment_sha256"))
        or [partitions.get(name, {}).get("image_count") for name in (
            "train", "val", "test"
        )] != [74, 5, 21]
        or data_split.get("validation_only") is not True
        or data_split.get("held_out_dataset_constructed") is not False
        or int(data_split.get("held_out_evaluation_count", -1)) != 0
        or data_split.get("allocation_unit")
        != "leading_source_identifier_group"
        or data_split.get("observation_unit") != "2048x2048_tile"
        or data_split.get("specimen_independence_confirmation")
        != "pending_data_owner_confirmation"
        or data_split.get("group_semantics")
        != "filename_derived_acquisition_series_kept_wholly_within_one_partition"
    ):
        raise ValueError(f"canonical split attestation mismatch: {identifier}")
    if (
        not isinstance(target, Mapping)
        or target.get("mask_count") != expected_target["mask_count"]
        or target.get("mask_aggregate_sha256")
        != expected_target["mask_aggregate_sha256"]
        or target.get("held_out_dataset_constructed") is not False
    ):
        raise ValueError(f"canonical development-target mismatch: {identifier}")
    if not verify_live_corpus_bytes:
        if set(partitions) != {"train", "val", "test"}:
            raise ValueError(f"recorded partitions are incomplete: {identifier}")
        all_ids = []
        all_files = []
        for split_name in ("train", "val", "test"):
            partition = partitions[split_name]
            if not isinstance(partition, Mapping) or set(partition) != {
                "image_ids",
                "image_files",
                "image_count",
            }:
                raise ValueError(
                    f"recorded partition structure is invalid: {identifier}"
                )
            image_ids = partition["image_ids"]
            image_files = partition["image_files"]
            image_count = partition["image_count"]
            if (
                isinstance(image_count, bool)
                or not isinstance(image_count, int)
                or not isinstance(image_ids, list)
                or not isinstance(image_files, list)
                or len(image_ids) != image_count
                or len(image_files) != image_count
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in image_ids
                )
                or any(
                    not isinstance(name, str)
                    or not name
                    or Path(name).is_absolute()
                    or ".." in Path(name).parts
                    or "\\" in name
                    or Path(name).as_posix() != name
                    for name in image_files
                )
                or len(set(image_ids)) != image_count
                or len(set(image_files)) != image_count
            ):
                raise ValueError(
                    f"recorded partition IDs/files are invalid: {identifier}"
                )
            allowed_prefixes = tuple(
                f"{group}_segment_"
                for group in CANONICAL_GROUP_MEMBERSHIP_MAP[split_name]
            )
            if any(not name.startswith(allowed_prefixes) for name in image_files):
                raise ValueError(
                    f"recorded partition filename/group mapping changed: {identifier}"
                )
            all_ids.extend(image_ids)
            all_files.extend(image_files)
        if (
            len(all_ids) != 100
            or len(set(all_ids)) != 100
            or len(all_files) != 100
            or len(set(all_files)) != 100
            or data_split.get("group_membership_map")
            != CANONICAL_GROUP_MEMBERSHIP_MAP
        ):
            raise ValueError(
                f"recorded partition isolation/group allocation changed: {identifier}"
            )
        expected_assignment_sha256 = hashlib.sha256(
            json.dumps(
                partitions, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if data_split.get("partition_assignment_sha256") != expected_assignment_sha256:
            raise ValueError(
                f"recorded partition assignment hash changed: {identifier}"
            )
        return
    manifest_path = Path(repository_root) / "config/confirmatory_splits.json"
    annotation_path = (
        Path(repository_root)
        / "results/step3_coco_dataset/pore_annotations.json"
    )
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != data_split["manifest_sha256"]
        or not annotation_path.is_file()
        or sha256_file(annotation_path) != data_split["annotation_index_sha256"]
    ):
        raise ValueError(f"split/annotation files changed: {identifier}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_document = json.load(handle)
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation_document = json.load(handle)
    resolved_ids = resolve_split_manifest(
        annotation_document, manifest_document
    )
    file_by_id = {
        int(image["id"]): str(image["file_name"])
        for image in annotation_document.get("images", [])
    }
    expected_partitions = {
        split_name: {
            "image_ids": [int(value) for value in resolved_ids[split_name]],
            "image_files": [
                file_by_id[int(value)] for value in resolved_ids[split_name]
            ],
            "image_count": len(resolved_ids[split_name]),
        }
        for split_name in ("train", "val", "test")
    }
    expected_groups = {
        "train": list(
            manifest_document.get("_provenance", {}).get("train_series", [])
        ),
        "val": list(
            manifest_document.get("_provenance", {}).get(
                "validation_series", []
            )
        ),
        "test": list(
            manifest_document.get("_provenance", {}).get("test_series", [])
        ),
    }
    expected_assignment_sha256 = hashlib.sha256(
        json.dumps(
            expected_partitions, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if (
        partitions != expected_partitions
        or data_split.get("group_membership_map") != expected_groups
        or data_split.get("partition_assignment_sha256")
        != expected_assignment_sha256
    ):
        raise ValueError(
            f"resolved partition IDs/files or group allocation changed: {identifier}"
        )
    train_names = partitions["train"]["image_files"]
    validation_names = partitions["val"]["image_files"]
    actual_targets = aggregate_indexed_file_bytes(
        Path(repository_root)
        / "results/step2_pore_classification/pore_classifications",
        list(train_names) + list(validation_names),
        scope="development_train_plus_validation",
        split_names=("train", "val"),
    )
    if (
        actual_targets["image_count"] != target["mask_count"]
        or actual_targets["image_aggregate_sha256"]
        != target["mask_aggregate_sha256"]
    ):
        raise ValueError(f"development target bytes changed: {identifier}")


def _verified_smoke_reference(
    artifact: Mapping,
    repository_root: Path,
    *,
    verify_live_corpus_bytes: bool = True,
) -> Dict:
    reference = artifact.get("smoke_preflight_manifest")
    if not isinstance(reference, Mapping):
        raise ValueError("screen cell lacks authenticated smoke-preflight manifest")
    identifier = reference.get("repo_relative_identifier")
    expected_sha256 = reference.get("sha256")
    if (
        not isinstance(identifier, str)
        or Path(identifier).is_absolute()
        or not _is_sha256(expected_sha256)
    ):
        raise ValueError("screen smoke-preflight reference is malformed")
    path = _assert_no_symlink_components(
        Path(repository_root) / identifier, Path(repository_root)
    )
    _repo_relative(path, Path(repository_root))
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("screen smoke-preflight manifest is missing or changed")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    verified = verify_smoke_preflight_manifest_document(
        document,
        Path(repository_root),
        verify_live_corpus_bytes=verify_live_corpus_bytes,
    )
    if reference.get("campaign_id") != verified["smoke_campaign_provenance"][
        "campaign_id"
    ]:
        raise ValueError("screen smoke-preflight campaign reference mismatch")
    return {
        "repo_relative_identifier": identifier,
        "sha256": expected_sha256,
        "campaign_id": reference["campaign_id"],
    }


def _load_screen_cell(
    path: Path,
    repository_root: Path,
    *,
    verify_live_corpus_bytes: bool = True,
) -> Dict:
    path = _assert_no_symlink_components(path, repository_root)
    identifier = _repo_relative(path, repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError(f"screen result is not a JSON object: {identifier}")
    if artifact.get("screen_result_schema_version") != SCREEN_RESULT_SCHEMA_VERSION:
        raise ValueError(f"invalid screen result schema: {identifier}")
    candidate = artifact.get("protocol_candidate_key")
    seed = artifact.get("seed")
    if candidate not in PROSPECTIVE_METHOD_PROTOCOLS or seed not in SCREEN_SEEDS:
        raise ValueError(f"invalid candidate/seed in screen result: {identifier}")
    array_index = SCREEN_CANDIDATE_ORDER.index(candidate) * len(SCREEN_SEEDS)
    array_index += SCREEN_SEEDS.index(seed)
    campaign_id = artifact.get("protocol_campaign_id")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or artifact.get("slurm_array_job_id") != campaign_id
        or int(artifact.get("protocol_cell_index", -1)) != array_index
        or int(artifact.get("slurm_array_task_id", -1)) != array_index
        or not artifact.get("slurm_job_id")
    ):
        raise ValueError(f"screen campaign/task provenance mismatch: {identifier}")
    expected_identifier = _canonical_cell_identifier(
        "validation_screen_cell",
        campaign_id,
        array_index,
        "metrics/model_selection.json",
    )
    if identifier != expected_identifier:
        raise ValueError(
            f"screen result is outside its canonical immutable cell: {identifier}"
        )
    if artifact.get("run_role") != "validation_screen_cell":
        raise ValueError(f"result is not a validation screen cell: {identifier}")
    if artifact.get("runtime_protocol") != PROSPECTIVE_METHOD_PROTOCOLS[candidate]:
        raise ValueError(f"runtime protocol mismatch in screen result: {identifier}")
    source_hashes = artifact.get("source_code_sha256")
    if (
        not isinstance(source_hashes, dict)
        or source_hashes != source_code_sha256(repository_root)
        or any(not _is_sha256(value) for value in source_hashes.values())
    ):
        raise ValueError(f"screen source hash mapping is incomplete/stale: {identifier}")
    smoke_reference = _verified_smoke_reference(
        artifact,
        repository_root,
        verify_live_corpus_bytes=verify_live_corpus_bytes,
    )

    common = {
        "array_index": array_index,
        "campaign_id": campaign_id,
        "candidate": candidate,
        "seed": int(seed),
        "result_artifact_repo_relative_identifier": identifier,
        "result_artifact_sha256": sha256_file(path),
        "source_code_sha256": source_hashes,
        "smoke_preflight_manifest": smoke_reference,
    }
    if artifact.get("outcome_status") == "failed":
        if (
            artifact.get("successful_cell") is not False
            or artifact.get("scientific_result_eligible") is not False
            or artifact.get("held_out_dataset_constructed") is not False
            or artifact.get("held_out_test_evaluated") is not False
            or int(artifact.get("held_out_test_evaluation_count", -1)) != 0
            or not isinstance(artifact.get("failure"), Mapping)
            or int(artifact["failure"].get("exit_code", 0)) == 0
        ):
            raise ValueError(f"invalid authenticated failure outcome: {identifier}")
        return {
            **common,
            "outcome_status": "failed",
            "failure": dict(artifact["failure"]),
            "selected_checkpoint_repo_relative_identifier": None,
            "selected_checkpoint_sha256": None,
            "selection_metrics": None,
            "scientific_execution_contract": None,
            "split_manifest_sha256": None,
            "target_mask_aggregate_sha256": None,
            "input_image_aggregate_sha256": None,
            "training_image_aggregate_sha256": None,
        }

    execution_contract = artifact.get("scientific_execution_contract")
    _validate_scientific_execution_contract(execution_contract, candidate, int(seed))
    required_truths = {
        "successful_cell": True,
        "scientific_result_eligible": True,
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_test_evaluated": False,
        "checkpoints_disabled": False,
        "debug_limited": False,
    }
    mismatches = {
        key: artifact.get(key)
        for key, expected in required_truths.items()
        if artifact.get(key) is not expected
    }
    if artifact.get("outcome_status") != "success" or mismatches:
        raise ValueError(
            f"screen result is not a successful scientific cell: {identifier}: "
            f"{mismatches}"
        )
    if int(artifact.get("held_out_test_evaluation_count", -1)) != 0:
        raise ValueError(f"screen result touched held-out evaluation: {identifier}")

    components = artifact.get("best_selection_components")
    if not isinstance(components, dict):
        raise ValueError(f"screen result lacks selection components: {identifier}")
    c0_iou = _finite_unit_interval(components.get("c0_iou"), "c0_iou")
    c1_iou = _finite_unit_interval(components.get("c1_iou"), "c1_iou")
    score = _finite_unit_interval(components.get("score"), "score")
    pore_union_iou = _finite_unit_interval(
        components.get("pore_union_iou"), "pore_union_iou"
    )
    validation_loss = float(components.get("validation_loss"))
    if not math.isfinite(validation_loss):
        raise ValueError(f"validation loss is non-finite: {identifier}")
    expected_score = 2.0 * c0_iou * c1_iou / (c0_iou + c1_iou + 1e-8)
    if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"selection score is inconsistent: {identifier}")
    if not math.isclose(
        float(artifact.get("best_selection_score")),
        score,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"best selection score is inconsistent: {identifier}")

    data_split = artifact.get("data_split")
    target = artifact.get("target_provenance")
    input_provenance = artifact.get("input_provenance")
    if not isinstance(data_split, dict) or not _is_sha256(
        data_split.get("manifest_sha256")
    ):
        raise ValueError(f"screen result lacks split-manifest hash: {identifier}")
    if not isinstance(target, dict) or not _is_sha256(
        target.get("mask_aggregate_sha256")
    ):
        raise ValueError(f"screen result lacks target-mask hash: {identifier}")
    _validate_data_attestations(
        data_split,
        target,
        repository_root,
        identifier,
        verify_live_corpus_bytes=verify_live_corpus_bytes,
    )
    _validate_input_provenance(
        input_provenance,
        data_split,
        repository_root,
        identifier,
        verify_live_corpus_bytes=verify_live_corpus_bytes,
    )

    selected_checkpoint_sha256 = artifact.get("selected_checkpoint_sha256")
    checkpoint_identifier = artifact.get(
        "selected_checkpoint_repo_relative_identifier"
    )
    checkpoint_path = _resolve_authenticated_cell_checkpoint(
        repository_root,
        role="validation_screen_cell",
        campaign_id=campaign_id,
        cell_index=array_index,
        checkpoint_identifier=checkpoint_identifier,
        expected_sha256=selected_checkpoint_sha256,
    )

    checkpoint = load_weights_only_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    if checkpoint.get("checkpoint_role") != "validation_composite_selection":
        raise ValueError(f"selected checkpoint has the wrong role: {checkpoint_identifier}")
    if checkpoint.get("selection_metric_name") != artifact.get("selection_metric_name"):
        raise ValueError(f"selected checkpoint metric mismatch: {checkpoint_identifier}")
    if checkpoint.get("best_selection_epoch") != artifact.get("best_selection_epoch"):
        raise ValueError(f"selected checkpoint epoch mismatch: {checkpoint_identifier}")
    if checkpoint.get("best_selection_components") != components:
        raise ValueError(f"selected checkpoint components mismatch: {checkpoint_identifier}")
    resolved = checkpoint.get("resolved_config", {})
    if (
        resolved.get("source_code_sha256") != source_hashes
        or resolved.get("scientific_execution_contract") != execution_contract
        or resolved.get("data_split") != data_split
        or resolved.get("input") != input_provenance
        or checkpoint.get("input_provenance") != input_provenance
        or checkpoint.get("target_provenance") != target
        or resolved.get("protocol_run_role") != "validation_screen_cell"
        or resolved.get("protocol_campaign_id") != campaign_id
        or int(resolved.get("protocol_cell_index", -1)) != array_index
        or resolved.get("protocol_candidate_key") != candidate
        or resolved.get("evaluation", {}).get("mode") != "validation_only"
        or resolved.get("evaluation", {}).get("held_out_dataset_constructed")
        is not False
        or int(resolved.get("evaluation", {}).get("held_out_evaluation_count", -1))
        != 0
    ):
        raise ValueError(f"selected checkpoint provenance mismatch: {checkpoint_identifier}")

    return {
        **common,
        "outcome_status": "success",
        "selected_checkpoint_repo_relative_identifier": checkpoint_identifier,
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "selection_metrics": {
            "score": score,
            "c0_iou": c0_iou,
            "c1_iou": c1_iou,
            "pore_union_iou": pore_union_iou,
            "validation_loss": validation_loss,
        },
        "scientific_execution_contract": execution_contract,
        "split_manifest_sha256": data_split["manifest_sha256"],
        "target_mask_aggregate_sha256": target["mask_aggregate_sha256"],
        "input_image_aggregate_sha256": input_provenance[
            "image_aggregate_sha256"
        ],
        "training_image_aggregate_sha256": input_provenance[
            "training_subset"
        ]["image_aggregate_sha256"],
        "resolved_data_split": data_split,
        "input_provenance": input_provenance,
        "target_provenance": target,
    }


def _validate_smoke_execution_contract(
    contract: Mapping, candidate: str
) -> None:
    """Validate the bounded plumbing run without making it scientific data."""
    if not isinstance(contract, Mapping):
        raise ValueError("smoke result lacks execution contract")
    if int(contract.get("epochs_planned", -1)) != 1:
        raise ValueError("smoke epoch budget must be one")
    if contract.get("scheduler", {}).get("t_max") != 1:
        raise ValueError("smoke cosine schedule T_max must be one")
    # All remaining objects/settings must match the scientific cell exactly.
    normalized = json.loads(json.dumps(contract))
    normalized["epochs_planned"] = 30
    normalized["scheduler"]["t_max"] = 30
    _validate_scientific_execution_contract(normalized, candidate, 42)


def _load_smoke_cell(
    path: Path,
    repository_root: Path,
    *,
    verify_live_corpus_bytes: bool = True,
) -> Dict:
    path = _assert_no_symlink_components(path, repository_root)
    identifier = _repo_relative(path, repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict) or artifact.get(
        "screen_result_schema_version"
    ) != SCREEN_RESULT_SCHEMA_VERSION:
        raise ValueError(f"invalid smoke result: {identifier}")
    candidate = artifact.get("protocol_candidate_key")
    seed = artifact.get("seed")
    if candidate not in SMOKE_CANDIDATE_ORDER or seed != 42:
        raise ValueError(f"invalid smoke candidate/seed: {identifier}")
    array_index = SMOKE_CANDIDATE_ORDER.index(candidate)
    campaign_id = artifact.get("protocol_campaign_id")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or artifact.get("run_role") != "validation_smoke_cell"
        or artifact.get("slurm_array_job_id") != campaign_id
        or int(artifact.get("protocol_cell_index", -1)) != array_index
        or int(artifact.get("slurm_array_task_id", -1)) != array_index
        or not artifact.get("slurm_job_id")
    ):
        raise ValueError(f"smoke campaign/task provenance mismatch: {identifier}")
    expected_identifier = _canonical_cell_identifier(
        "validation_smoke_cell",
        campaign_id,
        array_index,
        "metrics/model_selection.json",
    )
    if identifier != expected_identifier:
        raise ValueError(f"smoke result is outside its immutable cell: {identifier}")
    required = {
        "outcome_status": "success",
        "successful_smoke": True,
        "successful_cell": False,
        "scientific_result_eligible": False,
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_test_evaluated": False,
        "checkpoints_disabled": False,
        "debug_limited": True,
        "max_batches": 2,
    }
    mismatches = {
        key: artifact.get(key)
        for key, expected in required.items()
        if artifact.get(key) != expected
    }
    if mismatches or int(artifact.get("held_out_test_evaluation_count", -1)) != 0:
        raise ValueError(f"smoke outcome is not successful/bounded: {identifier}: {mismatches}")
    if artifact.get("runtime_protocol") != PROSPECTIVE_METHOD_PROTOCOLS[candidate]:
        raise ValueError(f"smoke runtime protocol mismatch: {identifier}")
    execution_contract = artifact.get("scientific_execution_contract")
    _validate_smoke_execution_contract(execution_contract, candidate)
    environment = artifact.get("execution_environment", {})
    if (
        environment.get("cuda_available") is not True
        or environment.get("device_type") != "cuda"
        or "L40S" not in str(environment.get("gpu_name", ""))
    ):
        raise ValueError(f"smoke did not execute on an allocated L40S: {identifier}")
    source_hashes = artifact.get("source_code_sha256")
    if (
        not isinstance(source_hashes, dict)
        or source_hashes != source_code_sha256(repository_root)
        or any(not _is_sha256(value) for value in source_hashes.values())
    ):
        raise ValueError(f"smoke source snapshot is incomplete/stale: {identifier}")
    data_split = artifact.get("data_split")
    target = artifact.get("target_provenance")
    input_provenance = artifact.get("input_provenance")
    if not isinstance(data_split, Mapping) or not _is_sha256(
        data_split.get("manifest_sha256")
    ):
        raise ValueError(f"smoke lacks split provenance: {identifier}")
    if not isinstance(target, Mapping) or not _is_sha256(
        target.get("mask_aggregate_sha256")
    ):
        raise ValueError(f"smoke lacks target provenance: {identifier}")
    _validate_data_attestations(
        data_split,
        target,
        repository_root,
        identifier,
        verify_live_corpus_bytes=verify_live_corpus_bytes,
    )
    _validate_input_provenance(
        input_provenance,
        data_split,
        repository_root,
        identifier,
        verify_live_corpus_bytes=verify_live_corpus_bytes,
    )

    checkpoint_identifier = artifact.get(
        "selected_checkpoint_repo_relative_identifier"
    )
    checkpoint_sha256 = artifact.get("selected_checkpoint_sha256")
    checkpoint_path = _resolve_authenticated_cell_checkpoint(
        repository_root,
        role="validation_smoke_cell",
        campaign_id=campaign_id,
        cell_index=array_index,
        checkpoint_identifier=checkpoint_identifier,
        expected_sha256=checkpoint_sha256,
    )

    checkpoint = load_weights_only_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    resolved = checkpoint.get("resolved_config", {})
    if (
        checkpoint.get("checkpoint_role") != "validation_composite_selection"
        or checkpoint.get("best_selection_components")
        != artifact.get("best_selection_components")
        or resolved.get("source_code_sha256") != source_hashes
        or resolved.get("scientific_execution_contract") != execution_contract
        or resolved.get("data_split") != data_split
        or resolved.get("input") != input_provenance
        or checkpoint.get("input_provenance") != input_provenance
        or checkpoint.get("target_provenance") != target
        or resolved.get("protocol_run_role") != "validation_smoke_cell"
        or resolved.get("protocol_campaign_id") != campaign_id
        or int(resolved.get("protocol_cell_index", -1)) != array_index
        or resolved.get("protocol_candidate_key") != candidate
        or resolved.get("evaluation", {}).get("mode") != "validation_only"
        or resolved.get("evaluation", {}).get("held_out_dataset_constructed")
        is not False
        or int(resolved.get("evaluation", {}).get("held_out_evaluation_count", -1))
        != 0
    ):
        raise ValueError(f"smoke checkpoint provenance mismatch: {checkpoint_identifier}")
    return {
        "array_index": array_index,
        "campaign_id": campaign_id,
        "candidate": candidate,
        "seed": 42,
        "result_artifact_repo_relative_identifier": identifier,
        "result_artifact_sha256": sha256_file(path),
        "selected_checkpoint_repo_relative_identifier": checkpoint_identifier,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "source_code_sha256": source_hashes,
        "split_manifest_sha256": data_split["manifest_sha256"],
        "target_mask_aggregate_sha256": target["mask_aggregate_sha256"],
        "input_image_aggregate_sha256": input_provenance[
            "image_aggregate_sha256"
        ],
        "training_image_aggregate_sha256": input_provenance[
            "training_subset"
        ]["image_aggregate_sha256"],
        "resolved_data_split": data_split,
        "input_provenance": input_provenance,
        "target_provenance": target,
        "scientific_execution_contract": execution_contract,
        "execution_environment": environment,
    }


def build_smoke_preflight_manifest_document(
    smoke_result_paths: Iterable[Path],
    repository_root: Path,
    *,
    verify_live_corpus_bytes: bool = True,
) -> Dict:
    """Authenticate the one-shot three-path plumbing/memory campaign."""
    paths = [Path(path) for path in smoke_result_paths]
    if len(paths) != SMOKE_CELL_COUNT:
        raise ValueError(f"smoke preflight requires exactly {SMOKE_CELL_COUNT} results")
    cells = sorted(
        (
            _load_smoke_cell(
                path,
                repository_root,
                verify_live_corpus_bytes=verify_live_corpus_bytes,
            )
            for path in paths
        ),
        key=lambda item: item["array_index"],
    )
    if [cell["candidate"] for cell in cells] != list(SMOKE_CANDIDATE_ORDER):
        raise ValueError("smoke must cover R3, C2-F, and C2-FP exactly once")
    if [cell["array_index"] for cell in cells] != list(range(SMOKE_CELL_COUNT)):
        raise ValueError("smoke must cover immutable task indices 0..2")
    campaigns = {cell["campaign_id"] for cell in cells}
    if len(campaigns) != 1:
        raise ValueError("smoke cells must come from one array campaign")
    for field in (
        "source_code_sha256",
        "split_manifest_sha256",
        "target_mask_aggregate_sha256",
        "input_image_aggregate_sha256",
        "training_image_aggregate_sha256",
        "resolved_data_split",
        "input_provenance",
        "target_provenance",
    ):
        if len({json.dumps(cell[field], sort_keys=True) for cell in cells}) != 1:
            raise ValueError(f"smoke cells have inconsistent {field}")
    return {
        "schema_version": SMOKE_PREFLIGHT_SCHEMA_VERSION,
        "smoke_campaign_provenance": {
            "purpose": "bounded_non_scientific_plumbing_and_l40s_memory_preflight",
            "scientific_result_eligible": False,
            "campaign_id": next(iter(campaigns)),
            "candidate_order": list(SMOKE_CANDIDATE_ORDER),
            "seed": 42,
            "expected_cell_count": SMOKE_CELL_COUNT,
            "successful_cell_count": SMOKE_CELL_COUNT,
            "source_code_sha256": cells[0]["source_code_sha256"],
            "split_manifest_sha256": cells[0]["split_manifest_sha256"],
            "target_mask_aggregate_sha256": cells[0][
                "target_mask_aggregate_sha256"
            ],
            "input_image_aggregate_sha256": cells[0][
                "input_image_aggregate_sha256"
            ],
            "training_image_aggregate_sha256": cells[0][
                "training_image_aggregate_sha256"
            ],
            "resolved_data_split": cells[0]["resolved_data_split"],
            "input_provenance": cells[0]["input_provenance"],
            "target_provenance": cells[0]["target_provenance"],
            "smoke_cells": cells,
        },
    }


def verify_smoke_preflight_manifest_document(
    document: Mapping,
    repository_root: Path,
    *,
    verify_live_corpus_bytes: bool = True,
) -> Dict:
    if not isinstance(document, Mapping) or document.get(
        "schema_version"
    ) != SMOKE_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("invalid smoke-preflight manifest schema")
    provenance = document.get("smoke_campaign_provenance")
    cells = provenance.get("smoke_cells") if isinstance(provenance, Mapping) else None
    if not isinstance(cells, list) or len(cells) != SMOKE_CELL_COUNT:
        raise ValueError("smoke-preflight manifest lacks all three cells")
    paths = []
    for cell in cells:
        identifier = cell.get("result_artifact_repo_relative_identifier")
        if not isinstance(identifier, str) or Path(identifier).is_absolute():
            raise ValueError("smoke result identifiers must be repository-relative")
        path = _assert_no_symlink_components(
            Path(repository_root) / identifier, Path(repository_root)
        )
        _repo_relative(path, Path(repository_root))
        if not path.is_file() or sha256_file(path) != cell.get(
            "result_artifact_sha256"
        ):
            raise ValueError(f"smoke result is missing or changed: {identifier}")
        paths.append(path)
    rebuilt = build_smoke_preflight_manifest_document(
        paths,
        Path(repository_root),
        verify_live_corpus_bytes=verify_live_corpus_bytes,
    )
    if dict(document) != rebuilt:
        raise ValueError("smoke-preflight manifest is not its deterministic rebuild")
    return rebuilt


def build_selected_method_lock_document(
    screen_result_paths: Iterable[Path],
    repository_root: Path,
    *,
    verify_live_corpus_bytes: bool = True,
) -> Dict:
    """Aggregate exactly 15 eligible cells and freeze the deterministic winner."""
    paths = [Path(path) for path in screen_result_paths]
    if len(paths) != SCREEN_CELL_COUNT:
        raise ValueError(
            f"selected-method lock requires exactly {SCREEN_CELL_COUNT} screen results"
        )
    cells = [
        _load_screen_cell(
            path,
            repository_root,
            verify_live_corpus_bytes=verify_live_corpus_bytes,
        )
        for path in paths
    ]
    cells.sort(key=lambda item: item["array_index"])
    expected_pairs = [
        (candidate, seed)
        for candidate in SCREEN_CANDIDATE_ORDER
        for seed in SCREEN_SEEDS
    ]
    actual_pairs = [(cell["candidate"], cell["seed"]) for cell in cells]
    if actual_pairs != expected_pairs:
        raise ValueError(
            "screen results must cover the exact candidate-major 15-cell matrix"
        )
    campaign_ids = {cell["campaign_id"] for cell in cells}
    if len(campaign_ids) != 1:
        raise ValueError("screen cells must come from one immutable array campaign")

    source_hashes = cells[0]["source_code_sha256"]
    smoke_reference = cells[0]["smoke_preflight_manifest"]
    for cell in cells[1:]:
        if cell["source_code_sha256"] != source_hashes:
            raise ValueError("screen cells have inconsistent source-code hashes")
        if cell["smoke_preflight_manifest"] != smoke_reference:
            raise ValueError("screen cells use different smoke-preflight campaigns")

    successful_cells = [
        cell for cell in cells if cell["outcome_status"] == "success"
    ]
    reference_cells = [cell for cell in cells if cell["candidate"] == "R3"]
    if any(cell["outcome_status"] != "success" for cell in reference_cells):
        raise ValueError(
            "R3 has a failed seed, so the reference baseline and lock are undefined"
        )
    split_sha256 = successful_cells[0]["split_manifest_sha256"]
    target_sha256 = successful_cells[0]["target_mask_aggregate_sha256"]
    input_sha256 = successful_cells[0]["input_image_aggregate_sha256"]
    training_input_sha256 = successful_cells[0][
        "training_image_aggregate_sha256"
    ]
    resolved_data_split = successful_cells[0]["resolved_data_split"]
    input_provenance = successful_cells[0]["input_provenance"]
    target_provenance = successful_cells[0]["target_provenance"]
    for cell in successful_cells[1:]:
        if cell["split_manifest_sha256"] != split_sha256:
            raise ValueError("successful cells have inconsistent split manifests")
        if cell["target_mask_aggregate_sha256"] != target_sha256:
            raise ValueError("successful cells have inconsistent target-mask hashes")
        if cell["input_image_aggregate_sha256"] != input_sha256:
            raise ValueError("successful cells have inconsistent development images")
        if cell["training_image_aggregate_sha256"] != training_input_sha256:
            raise ValueError("successful cells have inconsistent training images")
        if cell["resolved_data_split"] != resolved_data_split:
            raise ValueError("successful cells have different resolved partitions")
        if cell["input_provenance"] != input_provenance:
            raise ValueError("successful cells have different input provenance")
        if cell["target_provenance"] != target_provenance:
            raise ValueError("successful cells have different target provenance")
    train_mask_hashes = {
        cell["scientific_execution_contract"]["loss"]
        .get("training_class_statistics", {})
        .get("training_mask_aggregate_sha256")
        for cell in successful_cells
        if cell["candidate"] != "R3"
    }
    if train_mask_hashes and (
        len(train_mask_hashes) != 1
        or not _is_sha256(next(iter(train_mask_hashes)))
    ):
        raise ValueError(
            "non-reference screen cells have inconsistent train-only mask hashes"
        )

    aggregates = {}
    for candidate in SCREEN_CANDIDATE_ORDER:
        candidate_cells = [cell for cell in cells if cell["candidate"] == candidate]
        candidate_successes = [
            cell for cell in candidate_cells if cell["outcome_status"] == "success"
        ]
        metrics = [cell["selection_metrics"] for cell in candidate_successes]
        aggregate = {
            "successful_seed_count": len(candidate_successes),
            "failed_seed_count": len(candidate_cells) - len(candidate_successes),
            "failed_seeds": [
                cell["seed"] for cell in candidate_cells
                if cell["outcome_status"] == "failed"
            ],
            "complete_three_seed_result": len(candidate_successes) == 3,
            "mean_score": None,
            "mean_c0_iou": None,
            "mean_c1_iou": None,
            "mean_pore_union_iou": None,
            "mean_validation_loss": None,
        }
        if len(candidate_successes) == 3:
            aggregate.update({
                "mean_score": sum(item["score"] for item in metrics) / 3,
                "mean_c0_iou": sum(item["c0_iou"] for item in metrics) / 3,
                "mean_c1_iou": sum(item["c1_iou"] for item in metrics) / 3,
                "mean_pore_union_iou": (
                    sum(item["pore_union_iou"] for item in metrics) / 3
                ),
                "mean_validation_loss": (
                    sum(item["validation_loss"] for item in metrics) / 3
                ),
            })
        aggregates[candidate] = aggregate

    reference = aggregates["R3"]
    for candidate, aggregate in aggregates.items():
        aggregate["eligible"] = (
            candidate == "R3"
            or (
                aggregate["complete_three_seed_result"]
                and aggregate["mean_score"] > reference["mean_score"]
                and aggregate["mean_c0_iou"] >= reference["mean_c0_iou"]
                and aggregate["mean_c1_iou"] >= reference["mean_c1_iou"]
            )
        )
        aggregate["eligibility_reason"] = (
            "reference_fallback"
            if candidate == "R3"
            else "ineligible_due_to_one_or_more_failed_seeds"
            if not aggregate["complete_three_seed_result"]
            else "passes_all_reference_non_regression_rules"
            if aggregate["eligible"]
            else "fails_one_or_more_reference_non_regression_rules"
        )

    eligible = [
        (candidate, aggregate) for candidate, aggregate in aggregates.items()
        if aggregate["eligible"]
    ]
    winner_key = max(
        (
            aggregate["mean_score"],
            aggregate["mean_pore_union_iou"],
            -aggregate["mean_validation_loss"],
        )
        for _, aggregate in eligible
    )
    winners = [
        candidate for candidate, aggregate in eligible
        if (
            aggregate["mean_score"],
            aggregate["mean_pore_union_iou"],
            -aggregate["mean_validation_loss"],
        ) == winner_key
    ]
    if len(winners) != 1:
        raise ValueError(
            "screen has an unresolved exact candidate-level selection tie; "
            "no selected-method lock may be created"
        )
    winner = winners[0]

    public_cells: List[Dict] = []
    for cell in cells:
        public_cells.append({
            key: value for key, value in cell.items()
            if key not in {
                "source_code_sha256",
                "split_manifest_sha256",
                "target_mask_aggregate_sha256",
                "input_image_aggregate_sha256",
                "training_image_aggregate_sha256",
                "resolved_data_split",
                "input_provenance",
                "target_provenance",
            }
        })
    provenance = {
        "screen_protocol": "balanced_pore_validation_screen_v1",
        "campaign_id": next(iter(campaign_ids)),
        "candidate_major_order": list(SCREEN_CANDIDATE_ORDER),
        "seeds": list(SCREEN_SEEDS),
        "expected_cell_count": SCREEN_CELL_COUNT,
        "successful_cell_count": len(successful_cells),
        "failed_cell_count": len(cells) - len(successful_cells),
        "source_code_sha256": source_hashes,
        "split_manifest_sha256": split_sha256,
        "resolved_data_split": resolved_data_split,
        "target_mask_aggregate_sha256": target_sha256,
        "target_provenance": target_provenance,
        "input_image_aggregate_sha256": input_sha256,
        "training_image_aggregate_sha256": training_input_sha256,
        "input_provenance": input_provenance,
        "training_mask_aggregate_sha256": (
            next(iter(train_mask_hashes)) if train_mask_hashes else None
        ),
        "smoke_preflight_manifest": smoke_reference,
        "screen_cells": public_cells,
        "candidate_aggregates": aggregates,
        "selection_rule": SELECTION_RULE,
        "deterministic_winner": winner,
    }
    return {
        "schema_version": SELECTED_METHOD_LOCK_SCHEMA_VERSION,
        "selected_method": winner,
        "resolved_protocol": PROSPECTIVE_METHOD_PROTOCOLS[winner],
        "screen_selection_provenance": provenance,
    }


def verify_selected_method_lock_document(
    document: Mapping,
    repository_root: Path,
    *,
    verify_live_corpus_bytes: bool = True,
) -> Dict:
    """Rebuild a lock from its hashed cell artifacts and require exact equality."""
    if not isinstance(document, Mapping):
        raise ValueError("selected-method lock must contain one JSON object")
    provenance = document.get("screen_selection_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selected-method lock lacks screen-selection provenance")
    cells = provenance.get("screen_cells")
    if not isinstance(cells, list) or len(cells) != SCREEN_CELL_COUNT:
        raise ValueError("selected-method lock lacks the complete 15-cell screen")
    paths = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("selected-method lock contains an invalid screen cell")
        identifier = cell.get("result_artifact_repo_relative_identifier")
        if not isinstance(identifier, str) or Path(identifier).is_absolute():
            raise ValueError("screen result identifiers must be repository-relative")
        path = _assert_no_symlink_components(
            Path(repository_root) / identifier, Path(repository_root)
        )
        _repo_relative(path, Path(repository_root))
        if not path.is_file():
            raise FileNotFoundError(f"screen result is missing: {identifier}")
        if sha256_file(path) != cell.get("result_artifact_sha256"):
            raise ValueError(f"screen result SHA-256 mismatch: {identifier}")
        paths.append(path)
    rebuilt = build_selected_method_lock_document(
        paths,
        Path(repository_root),
        verify_live_corpus_bytes=verify_live_corpus_bytes,
    )
    if dict(document) != rebuilt:
        raise ValueError(
            "selected-method lock does not equal the deterministic result of its "
            "complete screen artifacts"
        )
    return rebuilt

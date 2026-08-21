"""Deterministic proof that neural selection and retraining are frozen.

The classical held-out evaluator must not expose test results while any neural
choice remains mutable.  This module builds and re-verifies a content-addressed
manifest from the complete selected-method lock plus the three primary and
three plain-U-Net validation-only retraining cells.  It opens no corpus file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from src.training.checkpoint_io import (
    load_weights_only_checkpoint,
    tensor_state_dict_semantic_sha256,
)
from src.training.screen_selection import (
    EXECUTION_SOURCE_FILES,
    PROSPECTIVE_METHOD_PROTOCOLS,
    SCREEN_SEEDS,
    _is_sha256,
    _validate_data_attestations,
    _validate_input_provenance,
    sha256_file,
    source_code_sha256,
    verify_selected_method_lock_document,
)


NEURAL_FREEZE_SCHEMA_VERSION = 2
NEURAL_FREEZE_PROTOCOL_ID = "neural_validation_freeze_v2"
CANONICAL_SELECTED_METHOD_LOCK = "config/selected_method_lock.json"
CANONICAL_SELECTED_RETRAINING_ROOT = (
    "results/patch_training/protocol_runs/selected_winner_retraining"
)
CANONICAL_NEURAL_FREEZE_ROOT = "results/neural_freeze/locked"
CANONICAL_NEURAL_FREEZE_FILENAME = "neural_freeze_manifest.json"
ARCHITECTURE_ROLES = ("primary_multiscale", "plain_unet_comparator")
CAMPAIGN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
MANIFEST_ID_PATTERN = re.compile(r"^neural-freeze-[0-9a-f]{16}$")


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selected_method_scientific_record(
    verified_lock: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return method-selection evidence without execution namespace aliases."""

    provenance = verified_lock.get("screen_selection_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Selected-method lock lacks screen-selection provenance")
    cells = provenance.get("screen_cells")
    if not isinstance(cells, list) or len(cells) != 15:
        raise ValueError("Selected-method lock lacks the complete screen matrix")
    scientific_cells = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("Selected-method lock contains an invalid screen cell")
        scientific_cells.append(
            {
                "array_index": int(cell["array_index"]),
                "candidate": str(cell["candidate"]),
                "seed": int(cell["seed"]),
                "outcome_status": str(cell["outcome_status"]),
                "selection_metrics": deepcopy(cell.get("selection_metrics")),
                "scientific_execution_contract": deepcopy(
                    cell.get("scientific_execution_contract")
                ),
            }
        )
    scientific_cells.sort(key=lambda item: item["array_index"])
    return {
        "selected_method": str(verified_lock["selected_method"]),
        "resolved_protocol": deepcopy(verified_lock["resolved_protocol"]),
        "candidate_major_order": list(provenance["candidate_major_order"]),
        "seeds": [int(value) for value in provenance["seeds"]],
        "screen_cells": scientific_cells,
        "candidate_aggregates": deepcopy(provenance["candidate_aggregates"]),
        "selection_rule": deepcopy(provenance["selection_rule"]),
        "deterministic_winner": str(provenance["deterministic_winner"]),
    }


def validate_campaign_id(value: str) -> str:
    campaign_id = str(value)
    if not CAMPAIGN_PATTERN.fullmatch(campaign_id):
        raise ValueError(
            "Selected-retraining campaign ID must be 1-96 public-path-safe characters"
        )
    return campaign_id


def validate_manifest_id(value: str) -> str:
    manifest_id = str(value)
    if not MANIFEST_ID_PATTERN.fullmatch(manifest_id):
        raise ValueError("Neural-freeze manifest ID is not canonical")
    return manifest_id


def _assert_no_symlink_components(path: Path, repository_root: Path) -> Path:
    """Return a lexical repository path after rejecting every symlink component."""

    root = repository_root.resolve()
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Canonical neural-freeze path escapes repository: {path}") from error
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"Canonical neural-freeze path contains a symlink: {cursor}")
    return lexical


def _canonical_file(
    repository_root: Path, identifier: str, *, label: str
) -> Path:
    relative = Path(identifier)
    if (
        not identifier
        or relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or "\\" in identifier
        or relative.as_posix() != identifier
    ):
        raise ValueError(f"Unsafe canonical {label} identifier: {identifier!r}")
    path = _assert_no_symlink_components(repository_root.resolve() / relative, repository_root)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical {label} is missing: {identifier}")
    return path


def canonical_manifest_path(manifest_id: str, repository_root: Path) -> Path:
    identifier = (
        Path(CANONICAL_NEURAL_FREEZE_ROOT)
        / validate_manifest_id(manifest_id)
        / CANONICAL_NEURAL_FREEZE_FILENAME
    ).as_posix()
    return _assert_no_symlink_components(
        repository_root.resolve() / identifier, repository_root
    )


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    return load_weights_only_checkpoint(path, map_location="cpu")


def _selected_lock_record(repository_root: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    path = _canonical_file(
        repository_root,
        CANONICAL_SELECTED_METHOD_LOCK,
        label="selected-method lock",
    )
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Selected-method lock must contain a JSON object")
    verified = verify_selected_method_lock_document(
        raw, repository_root, verify_live_corpus_bytes=False
    )
    provenance = verified.get("screen_selection_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Selected-method lock lacks screen provenance")
    current_sources = source_code_sha256(repository_root)
    if (
        provenance.get("source_code_sha256") != current_sources
        or set(current_sources) != set(EXECUTION_SOURCE_FILES)
        or any(not _is_sha256(value) for value in current_sources.values())
    ):
        raise ValueError("Selected-method lock does not match the current source snapshot")
    record = {
        "repo_relative_identifier": CANONICAL_SELECTED_METHOD_LOCK,
        "raw_file_sha256": sha256_file(path),
        "canonical_identity_sha256": _canonical_json_sha256(verified),
        "selected_method": verified["selected_method"],
        "screen_campaign_id": provenance["campaign_id"],
    }
    return dict(verified), record


def _development_attestations_from_lock(
    verified_lock: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Dict[str, Mapping[str, Any]]]:
    provenance = verified_lock["screen_selection_provenance"]
    target = provenance["target_provenance"]
    inputs = provenance["input_provenance"]
    development_input = {
        key: inputs[key]
        for key in (
            "scope",
            "split_names",
            "image_count",
            "image_aggregate_sha256",
            "image_aggregate_sha256_algorithm",
            "file_name_list_sha256",
        )
    }
    return target, {
        "development_train_plus_validation": development_input,
        "training_only": dict(inputs["training_subset"]),
    }


def _selected_lock_embedding(
    verified_lock: Mapping[str, Any], lock_record: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "schema_version": int(verified_lock["schema_version"]),
        "selected_method": str(verified_lock["selected_method"]),
        "lock_file_repo_relative_identifier": CANONICAL_SELECTED_METHOD_LOCK,
        "lock_file_sha256": lock_record["raw_file_sha256"],
        "resolved_protocol": dict(verified_lock["resolved_protocol"]),
        "screen_selection_provenance": verified_lock[
            "screen_selection_provenance"
        ],
    }


def _load_selected_retraining_cell(
    *,
    repository_root: Path,
    verified_lock: Mapping[str, Any],
    lock_record: Mapping[str, Any],
    architecture_role: str,
    campaign_id: str,
    cell_index: int,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Authenticate one validation-only selected-retraining outcome and checkpoint."""

    if architecture_role not in ARCHITECTURE_ROLES:
        raise ValueError(f"Unknown selected architecture role: {architecture_role!r}")
    campaign = validate_campaign_id(campaign_id)
    if isinstance(cell_index, bool) or cell_index not in range(len(SCREEN_SEEDS)):
        raise ValueError("Selected-retraining cell index must be 0, 1, or 2")
    seed = int(SCREEN_SEEDS[cell_index])
    cell_root = (
        f"{CANONICAL_SELECTED_RETRAINING_ROOT}/{campaign}/cell_{cell_index:02d}"
    )
    result_identifier = f"{cell_root}/metrics/model_selection.json"
    checkpoint_identifier = f"{cell_root}/checkpoints/best_model.pth"
    result_path = _canonical_file(
        repository_root, result_identifier, label="selected-retraining outcome"
    )
    checkpoint_path = _canonical_file(
        repository_root, checkpoint_identifier, label="selected-retraining checkpoint"
    )
    with result_path.open("r", encoding="utf-8") as handle:
        outcome = json.load(handle)
    if not isinstance(outcome, Mapping):
        raise ValueError("Selected-retraining outcome must be a JSON object")

    expected_runtime = dict(
        PROSPECTIVE_METHOD_PROTOCOLS[str(verified_lock["selected_method"])]
    )
    if architecture_role == "plain_unet_comparator":
        expected_runtime["model_type"] = "plain_unet"
    expected_lock_embedding = _selected_lock_embedding(verified_lock, lock_record)
    required = {
        "screen_result_schema_version": 1,
        "outcome_status": "success",
        "run_role": "selected_winner_retraining",
        "protocol_campaign_id": campaign,
        "protocol_cell_index": cell_index,
        "slurm_array_job_id": campaign,
        "slurm_array_task_id": str(cell_index),
        "selected_architecture_role": architecture_role,
        "protocol_candidate_key": verified_lock["selected_method"],
        "seed": seed,
        "runtime_protocol": expected_runtime,
        "successful_cell": False,
        "successful_smoke": False,
        "scientific_result_eligible": False,
        "validation_only": True,
        "held_out_test_evaluated": False,
        "held_out_test_evaluation_count": 0,
        "evaluation_mode": "validation_only",
        "held_out_dataset_constructed": False,
        "checkpoints_disabled": False,
        "debug_limited": False,
        "max_batches": None,
        "restore_source": "verified_validation_selected_checkpoint",
        "selected_checkpoint_repo_relative_identifier": checkpoint_identifier,
        "selected_method_lock": expected_lock_embedding,
    }
    mismatches = {
        key: {"outcome": outcome.get(key), "expected": value}
        for key, value in required.items()
        if outcome.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Selected-retraining outcome contract mismatch for {architecture_role}/"
            f"cell_{cell_index:02d}: {mismatches}"
        )
    if not outcome.get("slurm_job_id"):
        raise ValueError("Selected-retraining outcome lacks a Slurm job ID")
    checkpoint_sha = outcome.get("selected_checkpoint_sha256")
    if not _is_sha256(checkpoint_sha) or sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError("Selected-retraining checkpoint SHA-256 mismatch")

    current_sources = source_code_sha256(repository_root)
    if outcome.get("source_code_sha256") != current_sources:
        raise ValueError("Selected-retraining outcome source hashes drifted")
    data_split = outcome.get("data_split")
    target = outcome.get("target_provenance")
    input_provenance = outcome.get("input_provenance")
    identifier = result_identifier
    _validate_data_attestations(
        data_split,
        target,
        repository_root,
        identifier,
        verify_live_corpus_bytes=False,
    )
    _validate_input_provenance(
        input_provenance,
        data_split,
        repository_root,
        identifier,
        verify_live_corpus_bytes=False,
    )
    lock_provenance = verified_lock["screen_selection_provenance"]
    if (
        data_split != lock_provenance["resolved_data_split"]
        or target != lock_provenance["target_provenance"]
        or input_provenance != lock_provenance["input_provenance"]
    ):
        raise ValueError("Selected-retraining outcome split/development provenance drifted")

    checkpoint = (checkpoint_loader or _load_torch_checkpoint)(checkpoint_path)
    from scripts.evaluate_confirmatory_checkpoint import (
        infer_model_config_from_state,
        validate_checkpoint_input_provenance,
        validate_checkpoint_normalization,
        validate_checkpoint_protocol_fields,
        validate_checkpoint_selected_method_lock,
        validate_checkpoint_split_isolation,
        validate_checkpoint_target_provenance,
        validate_checkpoint_training_seed,
        validate_source_code_attestation,
    )

    if checkpoint.get("checkpoint_role") != "validation_composite_selection":
        raise ValueError("Selected-retraining checkpoint role is invalid")
    for field in (
        "selection_metric_name",
        "best_selection_epoch",
        "best_selection_components",
    ):
        if checkpoint.get(field) != outcome.get(field):
            raise ValueError(f"Selected-retraining checkpoint/outcome {field} mismatch")
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("Selected-retraining checkpoint lacks resolved_config")
    checkpoint_equalities = {
        "protocol_run_role": "selected_winner_retraining",
        "protocol_campaign_id": campaign,
        "protocol_cell_index": cell_index,
        "selected_architecture_role": architecture_role,
        "protocol_candidate_key": verified_lock["selected_method"],
        "source_code_sha256": current_sources,
        "data_split": data_split,
        "input": input_provenance,
        "target": target,
        "selected_method_lock": expected_lock_embedding,
        "scientific_execution_contract": outcome["scientific_execution_contract"],
    }
    bad_checkpoint = {
        key: {"checkpoint": resolved.get(key), "expected": value}
        for key, value in checkpoint_equalities.items()
        if resolved.get(key) != value
    }
    evaluation = resolved.get("evaluation")
    if bad_checkpoint or evaluation != {
        "mode": "validation_only",
        "held_out_dataset_constructed": False,
        "held_out_evaluation_count": 0,
    }:
        raise ValueError("Selected-retraining checkpoint provenance mismatch")
    if (
        checkpoint.get("source_code_sha256") != current_sources
        or checkpoint.get("input_provenance") != input_provenance
        or checkpoint.get("target_provenance") != target
    ):
        raise ValueError("Selected-retraining checkpoint top-level provenance drifted")

    validate_checkpoint_normalization(checkpoint)
    selected_method, _, observed_role, retraining = (
        validate_checkpoint_selected_method_lock(
            checkpoint,
            verified_lock,
            lock_sha256=str(lock_record["raw_file_sha256"]),
            lock_identifier=CANONICAL_SELECTED_METHOD_LOCK,
        )
    )
    if (
        selected_method != verified_lock["selected_method"]
        or observed_role != architecture_role
        or retraining["campaign_id"] != campaign
        or int(retraining["array_task_index"]) != cell_index
        or int(retraining["training_seed"]) != seed
    ):
        raise ValueError("Selected-retraining checkpoint role/campaign/task/seed drifted")
    partitions = lock_provenance["resolved_data_split"]["partitions"]
    split_ids = {
        name: [int(value) for value in partitions[name]["image_ids"]]
        for name in ("train", "val", "test")
    }
    split_files = {
        name: [str(value) for value in partitions[name]["image_files"]]
        for name in ("train", "val", "test")
    }
    validate_checkpoint_split_isolation(
        checkpoint,
        manifest_sha256=lock_provenance["split_manifest_sha256"],
        split_ids=split_ids,
        split_files=split_files,
    )
    target_attestation, input_attestations = _development_attestations_from_lock(
        verified_lock
    )
    validate_checkpoint_target_provenance(checkpoint, target_attestation)
    validate_checkpoint_input_provenance(checkpoint, input_attestations)
    validate_source_code_attestation(
        checkpoint, verified_lock, repository_root=repository_root
    )
    validate_checkpoint_training_seed(
        checkpoint,
        verified_lock,
        candidate=str(selected_method),
        architecture_role=architecture_role,
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Selected-retraining checkpoint lacks model state")
    state_semantic_sha256 = tensor_state_dict_semantic_sha256(state)
    inferred = infer_model_config_from_state(state)
    validate_checkpoint_protocol_fields(
        checkpoint, str(selected_method), inferred, architecture_role
    )
    model_record = resolved.get("model")
    if (
        not isinstance(model_record, Mapping)
        or model_record.get("base_features") != 32
        or model_record.get("bilinear") is not True
        or inferred.get("base_features") != 32
        or inferred.get("bilinear") is not True
    ):
        raise ValueError("Selected-retraining checkpoint is not the locked 32-feature bilinear model")

    source_identity = _canonical_json_sha256(current_sources)
    return {
        "array_task_index": cell_index,
        "training_seed": seed,
        "architecture_role": architecture_role,
        "campaign_id": campaign,
        "outcome_repo_relative_identifier": result_identifier,
        "outcome_sha256": sha256_file(result_path),
        "checkpoint_repo_relative_identifier": checkpoint_identifier,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_dict_semantic_sha256": state_semantic_sha256,
        "selection_metric_name": outcome["selection_metric_name"],
        "best_selection_epoch": int(outcome["best_selection_epoch"]),
        "scientific_execution_contract_sha256": _canonical_json_sha256(
            outcome["scientific_execution_contract"]
        ),
        "source_code_sha256_identity": source_identity,
        "split_manifest_sha256": data_split["manifest_sha256"],
        "development_input_sha256": input_provenance[
            "image_aggregate_sha256"
        ],
        "development_target_sha256": target["mask_aggregate_sha256"],
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_evaluation_count": 0,
    }


def build_neural_freeze_manifest_document(
    *,
    primary_campaign_id: str,
    plain_campaign_id: str,
    repository_root: Path,
    selected_lock_loader: Callable[[Path], tuple[Dict[str, Any], Dict[str, Any]]] | None = None,
    retraining_cell_loader: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Rebuild the complete neural freeze without opening corpus bytes."""

    root = repository_root.resolve()
    primary = validate_campaign_id(primary_campaign_id)
    plain = validate_campaign_id(plain_campaign_id)
    if primary == plain:
        raise ValueError("Primary and plain selected-retraining campaigns must differ")
    verified_lock, lock_record = (
        selected_lock_loader or _selected_lock_record
    )(root)
    loader = retraining_cell_loader or _load_selected_retraining_cell
    role_campaigns = {
        "primary_multiscale": primary,
        "plain_unet_comparator": plain,
    }
    roles: Dict[str, Any] = {}
    for role in ARCHITECTURE_ROLES:
        campaign = role_campaigns[role]
        cells = [
            loader(
                repository_root=root,
                verified_lock=verified_lock,
                lock_record=lock_record,
                architecture_role=role,
                campaign_id=campaign,
                cell_index=index,
            )
            for index in range(len(SCREEN_SEEDS))
        ]
        expected = [(index, int(seed)) for index, seed in enumerate(SCREEN_SEEDS)]
        observed = [
            (int(cell["array_task_index"]), int(cell["training_seed"]))
            for cell in cells
        ]
        if observed != expected:
            raise ValueError(f"{role} selected-retraining task/seed matrix is incomplete")
        if any(
            cell.get("architecture_role") != role
            or cell.get("campaign_id") != campaign
            or cell.get("validation_only") is not True
            or cell.get("held_out_dataset_constructed") is not False
            or int(cell.get("held_out_evaluation_count", -1)) != 0
            for cell in cells
        ):
            raise ValueError(f"{role} selected-retraining cell provenance is inconsistent")
        roles[role] = {"campaign_id": campaign, "cells": cells}

    checkpoint_hashes = [
        str(cell["checkpoint_sha256"])
        for role in ARCHITECTURE_ROLES
        for cell in roles[role]["cells"]
    ]
    if len(checkpoint_hashes) != 6 or len(set(checkpoint_hashes)) != 6 or any(
        not _is_sha256(value) for value in checkpoint_hashes
    ):
        raise ValueError("Neural freeze requires six distinct authenticated checkpoint hashes")

    semantic_checkpoint_hashes = [
        str(cell["checkpoint_state_dict_semantic_sha256"])
        for role in ARCHITECTURE_ROLES
        for cell in roles[role]["cells"]
    ]
    if len(semantic_checkpoint_hashes) != 6 or any(
        not _is_sha256(value) for value in semantic_checkpoint_hashes
    ):
        raise ValueError("Neural freeze lacks six authenticated semantic model hashes")

    provenance = verified_lock["screen_selection_provenance"]
    current_sources = source_code_sha256(root)
    development_provenance = {
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "input_image_aggregate_sha256": provenance[
            "input_image_aggregate_sha256"
        ],
        "target_mask_aggregate_sha256": provenance[
            "target_mask_aggregate_sha256"
        ],
        "training_image_aggregate_sha256": provenance[
            "training_image_aggregate_sha256"
        ],
        "training_mask_aggregate_sha256": provenance[
            "training_mask_aggregate_sha256"
        ],
    }
    data_access = {
        "selected_method_screen_cell_count": 15,
        "selected_retraining_checkpoint_count": 6,
        "validation_only_retraining": True,
        "held_out_dataset_constructed": False,
        "held_out_image_read_count": 0,
        "held_out_target_read_count": 0,
        "neural_held_out_result_required": False,
    }
    scientific_identity_payload: Dict[str, Any] = {
        "schema_version": NEURAL_FREEZE_SCHEMA_VERSION,
        "protocol_id": NEURAL_FREEZE_PROTOCOL_ID,
        "method_selection": _selected_method_scientific_record(verified_lock),
        "selected_retraining": {
            role: [
                {
                    "array_task_index": int(cell["array_task_index"]),
                    "training_seed": int(cell["training_seed"]),
                    "architecture_role": str(cell["architecture_role"]),
                    "checkpoint_state_dict_semantic_sha256": str(
                        cell["checkpoint_state_dict_semantic_sha256"]
                    ),
                    "scientific_execution_contract_sha256": str(
                        cell["scientific_execution_contract_sha256"]
                    ),
                    "selection_metric_name": str(cell["selection_metric_name"]),
                }
                for cell in roles[role]["cells"]
            ]
            for role in ARCHITECTURE_ROLES
        },
        "source_code_sha256": current_sources,
        "development_provenance": development_provenance,
        "data_access": data_access,
    }
    identity = _canonical_json_sha256(scientific_identity_payload)
    payload: Dict[str, Any] = {
        "schema_version": NEURAL_FREEZE_SCHEMA_VERSION,
        "protocol_id": NEURAL_FREEZE_PROTOCOL_ID,
        "status": "neural_selection_and_validation_only_retraining_frozen",
        "selected_method_lock": lock_record,
        "selected_method": verified_lock["selected_method"],
        "selected_retraining": roles,
        "source_code_sha256": current_sources,
        "development_provenance": development_provenance,
        "data_access": data_access,
        "scientific_identity_payload": scientific_identity_payload,
    }
    manifest_id = f"neural-freeze-{identity[:16]}"
    return {
        **payload,
        "manifest_id": manifest_id,
        "scientific_identity_sha256": identity,
    }


def verify_neural_freeze_manifest_document(
    document: Mapping[str, Any], repository_root: Path
) -> Dict[str, Any]:
    """Require exact equality with a deterministic rebuild from live artifacts."""

    if not isinstance(document, Mapping):
        raise ValueError("Neural-freeze manifest must contain one JSON object")
    if document.get("schema_version") != NEURAL_FREEZE_SCHEMA_VERSION or document.get(
        "protocol_id"
    ) != NEURAL_FREEZE_PROTOCOL_ID:
        raise ValueError("Neural-freeze manifest schema/protocol mismatch")
    manifest_id = validate_manifest_id(str(document.get("manifest_id", "")))
    identity = str(document.get("scientific_identity_sha256", ""))
    if not _is_sha256(identity) or manifest_id != f"neural-freeze-{identity[:16]}":
        raise ValueError("Neural-freeze manifest ID/identity mismatch")
    retraining = document.get("selected_retraining")
    if not isinstance(retraining, Mapping) or tuple(retraining) != ARCHITECTURE_ROLES:
        raise ValueError("Neural-freeze manifest lacks both architecture roles")
    try:
        primary = retraining["primary_multiscale"]["campaign_id"]
        plain = retraining["plain_unet_comparator"]["campaign_id"]
    except (KeyError, TypeError) as error:
        raise ValueError("Neural-freeze manifest lacks retraining campaigns") from error
    rebuilt = build_neural_freeze_manifest_document(
        primary_campaign_id=str(primary),
        plain_campaign_id=str(plain),
        repository_root=repository_root,
    )
    if dict(document) != rebuilt:
        raise ValueError("Neural-freeze manifest is not its deterministic artifact rebuild")
    return rebuilt


def load_verified_neural_freeze_manifest(
    manifest_id: str, repository_root: Path
) -> Dict[str, Any]:
    """Load only the canonical content-addressed manifest and re-verify it."""

    canonical_id = validate_manifest_id(manifest_id)
    path = canonical_manifest_path(canonical_id, repository_root)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical neural-freeze manifest is missing: {canonical_id}")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    verified = verify_neural_freeze_manifest_document(document, repository_root)
    if verified["manifest_id"] != canonical_id:
        raise ValueError("Neural-freeze manifest path/embedded ID mismatch")
    checkpoint_hashes = {
        role: [cell["checkpoint_sha256"] for cell in verified["selected_retraining"][role]["cells"]]
        for role in ARCHITECTURE_ROLES
    }
    return {
        "manifest_id": canonical_id,
        "manifest_repo_relative_identifier": path.relative_to(
            repository_root.resolve()
        ).as_posix(),
        "manifest_file_sha256": sha256_file(path),
        "scientific_identity_sha256": verified["scientific_identity_sha256"],
        "selected_method": verified["selected_method"],
        "selected_method_lock": dict(verified["selected_method_lock"]),
        "selected_retraining_checkpoint_sha256": checkpoint_hashes,
        "document": verified,
    }

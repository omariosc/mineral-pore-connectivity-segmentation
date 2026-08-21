"""Corpus-free tests for the neural-selection freeze required by classical scoring."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.training import neural_freeze
from src.training.neural_freeze import (
    ARCHITECTURE_ROLES,
    CANONICAL_SELECTED_METHOD_LOCK,
    SCREEN_SEEDS,
    _load_selected_retraining_cell,
    build_neural_freeze_manifest_document,
    canonical_manifest_path,
    verify_neural_freeze_manifest_document,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_map() -> dict[str, str]:
    return {"synthetic.py": "a" * 64}


def _verified_lock(campaign_id: str = "screen-100") -> dict[str, Any]:
    candidates = ("R3", "H3", "C2-P", "C2-F", "C2-FP")
    screen_cells = []
    for candidate_index, candidate in enumerate(candidates):
        for seed_index, seed in enumerate(SCREEN_SEEDS):
            screen_cells.append(
                {
                    "array_index": candidate_index * len(SCREEN_SEEDS) + seed_index,
                    "campaign_id": campaign_id,
                    "candidate": candidate,
                    "seed": int(seed),
                    "outcome_status": "success",
                    "selection_metrics": {
                        "score": 0.5,
                        "c0_iou": 0.5,
                        "c1_iou": 0.5,
                        "pore_union_iou": 0.5,
                        "validation_loss": 0.5,
                    },
                    "scientific_execution_contract": {"candidate": candidate},
                }
            )
    return {
        "schema_version": 1,
        "selected_method": "R3",
        "resolved_protocol": {"model_type": "multiscale_attention_unet"},
        "screen_selection_provenance": {
            "campaign_id": campaign_id,
            "candidate_major_order": list(candidates),
            "seeds": list(SCREEN_SEEDS),
            "screen_cells": screen_cells,
            "candidate_aggregates": {
                candidate: {"mean_score": 0.5} for candidate in candidates
            },
            "selection_rule": {"primary": "harmonic_c0_c1_iou"},
            "deterministic_winner": "R3",
            "source_code_sha256": _source_map(),
            "split_manifest_sha256": "b" * 64,
            "resolved_data_split": {
                "manifest_sha256": "b" * 64,
                "partitions": {
                    name: {"image_ids": [], "image_files": [], "image_count": 0}
                    for name in ("train", "val", "test")
                },
            },
            "input_image_aggregate_sha256": "c" * 64,
            "training_image_aggregate_sha256": "d" * 64,
            "target_mask_aggregate_sha256": "e" * 64,
            "training_mask_aggregate_sha256": "f" * 64,
            "input_provenance": {
                "image_aggregate_sha256": "c" * 64,
                "training_subset": {"image_aggregate_sha256": "d" * 64},
            },
            "target_provenance": {"mask_aggregate_sha256": "e" * 64},
        },
    }


def _lock_record(campaign_id: str = "screen-100") -> dict[str, Any]:
    return {
        "repo_relative_identifier": CANONICAL_SELECTED_METHOD_LOCK,
        "raw_file_sha256": "1" * 64,
        "canonical_identity_sha256": "2" * 64,
        "selected_method": "R3",
        "screen_campaign_id": campaign_id,
    }


def _synthetic_cell_loader(**kwargs: Any) -> dict[str, Any]:
    role = kwargs["architecture_role"]
    campaign = kwargs["campaign_id"]
    cell = kwargs["cell_index"]
    hash_digit = cell + (4 if role == "primary_multiscale" else 7)
    return {
        "array_task_index": cell,
        "training_seed": int(SCREEN_SEEDS[cell]),
        "architecture_role": role,
        "campaign_id": campaign,
        "outcome_repo_relative_identifier": f"outcome/{role}/{cell}.json",
        "outcome_sha256": f"{cell + 1:x}" * 64,
        "checkpoint_repo_relative_identifier": f"checkpoint/{role}/{cell}.pth",
        "checkpoint_sha256": f"{hash_digit:x}" * 64,
        "checkpoint_state_dict_semantic_sha256": f"{hash_digit + 1:x}" * 64,
        "selection_metric_name": "validation_c0_c1_iou_harmonic_mean",
        "best_selection_epoch": 1,
        "scientific_execution_contract_sha256": "8" * 64,
        "source_code_sha256_identity": "9" * 64,
        "split_manifest_sha256": "b" * 64,
        "development_input_sha256": "c" * 64,
        "development_target_sha256": "e" * 64,
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_evaluation_count": 0,
    }


def _build_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary_campaign: str = "primary-200",
    plain_campaign: str = "plain-300",
    screen_campaign: str = "screen-100",
) -> dict[str, Any]:
    monkeypatch.setattr(neural_freeze, "source_code_sha256", lambda _: _source_map())
    return build_neural_freeze_manifest_document(
        primary_campaign_id=primary_campaign,
        plain_campaign_id=plain_campaign,
        repository_root=tmp_path,
        selected_lock_loader=lambda _: (
            _verified_lock(screen_campaign),
            _lock_record(screen_campaign),
        ),
        retraining_cell_loader=_synthetic_cell_loader,
    )


def test_manifest_covers_exact_role_seed_matrix_and_is_content_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _build_document(tmp_path, monkeypatch)
    assert tuple(document["selected_retraining"]) == ARCHITECTURE_ROLES
    assert document["manifest_id"] == (
        "neural-freeze-" + document["scientific_identity_sha256"][:16]
    )
    hashes = []
    for role in ARCHITECTURE_ROLES:
        cells = document["selected_retraining"][role]["cells"]
        assert [(cell["array_task_index"], cell["training_seed"]) for cell in cells] == list(
            enumerate(SCREEN_SEEDS)
        )
        hashes.extend(cell["checkpoint_sha256"] for cell in cells)
    assert len(hashes) == len(set(hashes)) == 6
    assert document["data_access"]["held_out_image_read_count"] == 0
    assert document["data_access"]["held_out_target_read_count"] == 0


def test_scientific_identity_rejects_campaign_alias_as_a_new_evaluation_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _build_document(tmp_path, monkeypatch)
    aliased = _build_document(
        tmp_path,
        monkeypatch,
        primary_campaign="copied-primary",
        plain_campaign="copied-plain",
        screen_campaign="copied-screen",
    )
    assert original["selected_retraining"] != aliased["selected_retraining"]
    assert original["selected_method_lock"] != aliased["selected_method_lock"]
    assert original["scientific_identity_payload"] == aliased[
        "scientific_identity_payload"
    ]
    assert original["scientific_identity_sha256"] == aliased[
        "scientific_identity_sha256"
    ]
    assert original["manifest_id"] == aliased["manifest_id"]


def test_selected_lock_rebuild_is_explicitly_corpus_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / CANONICAL_SELECTED_METHOD_LOCK
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    source = {name: "a" * 64 for name in neural_freeze.EXECUTION_SOURCE_FILES}
    verified = _verified_lock()
    verified["screen_selection_provenance"]["source_code_sha256"] = source
    observed = {}

    def verify(document, repository_root, *, verify_live_corpus_bytes=True):
        observed["document"] = document
        observed["root"] = repository_root
        observed["verify_live_corpus_bytes"] = verify_live_corpus_bytes
        return verified

    monkeypatch.setattr(
        neural_freeze, "verify_selected_method_lock_document", verify
    )
    monkeypatch.setattr(neural_freeze, "source_code_sha256", lambda _: source)
    rebuilt, record = neural_freeze._selected_lock_record(tmp_path)
    assert rebuilt == verified
    assert observed["verify_live_corpus_bytes"] is False
    assert record["repo_relative_identifier"] == CANONICAL_SELECTED_METHOD_LOCK
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda cell: cell.__setitem__("training_seed", 999),
            "task/seed matrix",
        ),
        (
            lambda cell: cell.__setitem__("architecture_role", "wrong"),
            "cell provenance",
        ),
        (
            lambda cell: cell.__setitem__("campaign_id", "wrong"),
            "cell provenance",
        ),
        (
            lambda cell: cell.__setitem__("held_out_evaluation_count", 1),
            "cell provenance",
        ),
    ],
)
def test_builder_rejects_role_campaign_seed_and_heldout_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Any,
    message: str,
) -> None:
    monkeypatch.setattr(neural_freeze, "source_code_sha256", lambda _: _source_map())

    def loader(**kwargs: Any) -> dict[str, Any]:
        cell = _synthetic_cell_loader(**kwargs)
        if kwargs["architecture_role"] == "plain_unet_comparator" and kwargs[
            "cell_index"
        ] == 1:
            mutator(cell)
        return cell

    with pytest.raises(ValueError, match=message):
        build_neural_freeze_manifest_document(
            primary_campaign_id="primary-200",
            plain_campaign_id="plain-300",
            repository_root=tmp_path,
            selected_lock_loader=lambda _: (_verified_lock(), _lock_record()),
            retraining_cell_loader=loader,
        )


def test_document_rebuild_rejects_partial_and_tampered_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _build_document(tmp_path, monkeypatch)
    monkeypatch.setattr(
        neural_freeze,
        "build_neural_freeze_manifest_document",
        lambda **_: copy.deepcopy(document),
    )
    assert verify_neural_freeze_manifest_document(document, tmp_path) == document

    tampered = copy.deepcopy(document)
    tampered["selected_retraining"]["primary_multiscale"]["cells"][0][
        "checkpoint_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="deterministic artifact rebuild"):
        verify_neural_freeze_manifest_document(tampered, tmp_path)

    partial = copy.deepcopy(document)
    partial["selected_retraining"].pop("plain_unet_comparator")
    with pytest.raises(ValueError, match="both architecture roles"):
        verify_neural_freeze_manifest_document(partial, tmp_path)


def test_canonical_manifest_path_rejects_traversal_and_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not canonical"):
        canonical_manifest_path("../escape", tmp_path)
    manifest_id = "neural-freeze-" + "a" * 16
    root = tmp_path / "results/neural_freeze/locked"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / manifest_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        canonical_manifest_path(manifest_id, tmp_path)


def _cell_fixture(tmp_path: Path, role: str = "primary_multiscale") -> tuple[dict, dict]:
    campaign = "primary-200" if role == "primary_multiscale" else "plain-300"
    cell = 0
    seed = int(SCREEN_SEEDS[cell])
    root = (
        tmp_path
        / neural_freeze.CANONICAL_SELECTED_RETRAINING_ROOT
        / campaign
        / "cell_00"
    )
    (root / "metrics").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    checkpoint_path = root / "checkpoints/best_model.pth"
    checkpoint_path.write_bytes(b"synthetic checkpoint bytes")
    verified_lock = _verified_lock()
    lock_record = _lock_record()
    lock_embedding = neural_freeze._selected_lock_embedding(verified_lock, lock_record)
    data_split = verified_lock["screen_selection_provenance"]["resolved_data_split"]
    input_provenance = verified_lock["screen_selection_provenance"]["input_provenance"]
    target = verified_lock["screen_selection_provenance"]["target_provenance"]
    source = _source_map()
    runtime = dict(neural_freeze.PROSPECTIVE_METHOD_PROTOCOLS["R3"])
    if role == "plain_unet_comparator":
        runtime["model_type"] = "plain_unet"
    scientific = {"model": {"base_features": 32, "bilinear": True}}
    checkpoint = {
        "checkpoint_role": "validation_composite_selection",
        "selection_metric_name": "validation_c0_c1_iou_harmonic_mean",
        "best_selection_epoch": 1,
        "best_selection_components": {"score": 0.5},
        "source_code_sha256": source,
        "input_provenance": input_provenance,
        "target_provenance": target,
        "model_state_dict": {"synthetic": object()},
        "resolved_config": {
            "protocol_run_role": "selected_winner_retraining",
            "protocol_campaign_id": campaign,
            "protocol_cell_index": cell,
            "selected_architecture_role": role,
            "protocol_candidate_key": "R3",
            "source_code_sha256": source,
            "data_split": data_split,
            "input": input_provenance,
            "target": target,
            "selected_method_lock": lock_embedding,
            "scientific_execution_contract": scientific,
            "evaluation": {
                "mode": "validation_only",
                "held_out_dataset_constructed": False,
                "held_out_evaluation_count": 0,
            },
            "model": {"base_features": 32, "bilinear": True},
        },
    }
    outcome = {
        "screen_result_schema_version": 1,
        "outcome_status": "success",
        "run_role": "selected_winner_retraining",
        "protocol_campaign_id": campaign,
        "protocol_cell_index": cell,
        "slurm_job_id": f"{campaign}_0",
        "slurm_array_job_id": campaign,
        "slurm_array_task_id": "0",
        "selected_architecture_role": role,
        "protocol_candidate_key": "R3",
        "seed": seed,
        "runtime_protocol": runtime,
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
        "selected_checkpoint_repo_relative_identifier": (
            f"{neural_freeze.CANONICAL_SELECTED_RETRAINING_ROOT}/{campaign}/"
            "cell_00/checkpoints/best_model.pth"
        ),
        "selected_checkpoint_sha256": _sha(checkpoint_path.read_bytes()),
        "selected_method_lock": lock_embedding,
        "source_code_sha256": source,
        "data_split": data_split,
        "input_provenance": input_provenance,
        "target_provenance": target,
        "scientific_execution_contract": scientific,
        "selection_metric_name": "validation_c0_c1_iou_harmonic_mean",
        "best_selection_epoch": 1,
        "best_selection_components": {"score": 0.5},
    }
    (root / "metrics/model_selection.json").write_text(
        json.dumps(outcome), encoding="utf-8"
    )
    return outcome, checkpoint


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("source_code_sha256", {"synthetic.py": "0" * 64}, "source hashes"),
        ("data_split", {"manifest_sha256": "0" * 64}, "provenance"),
        ("input_provenance", {"image_aggregate_sha256": "0" * 64}, "provenance"),
        ("target_provenance", {"mask_aggregate_sha256": "0" * 64}, "provenance"),
    ],
)
def test_cell_authentication_rejects_source_split_and_dev_provenance_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    outcome, checkpoint = _cell_fixture(tmp_path)
    outcome[field] = replacement
    outcome_path = (
        tmp_path
        / neural_freeze.CANONICAL_SELECTED_RETRAINING_ROOT
        / "primary-200/cell_00/metrics/model_selection.json"
    )
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    monkeypatch.setattr(neural_freeze, "source_code_sha256", lambda _: _source_map())
    monkeypatch.setattr(
        neural_freeze, "_validate_data_attestations", lambda *a, **_k: None
    )
    monkeypatch.setattr(
        neural_freeze, "_validate_input_provenance", lambda *a, **_k: None
    )
    with pytest.raises(ValueError, match=message):
        _load_selected_retraining_cell(
            repository_root=tmp_path,
            verified_lock=_verified_lock(),
            lock_record=_lock_record(),
            architecture_role="primary_multiscale",
            campaign_id="primary-200",
            cell_index=0,
            checkpoint_loader=lambda _: checkpoint,
        )


def test_cell_authentication_rejects_missing_and_symlinked_checkpoint(
    tmp_path: Path,
) -> None:
    _cell_fixture(tmp_path)
    checkpoint = (
        tmp_path
        / neural_freeze.CANONICAL_SELECTED_RETRAINING_ROOT
        / "primary-200/cell_00/checkpoints/best_model.pth"
    )
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        _load_selected_retraining_cell(
            repository_root=tmp_path,
            verified_lock=_verified_lock(),
            lock_record=_lock_record(),
            architecture_role="primary_multiscale",
            campaign_id="primary-200",
            cell_index=0,
        )
    outside = tmp_path / "alternate.pth"
    outside.write_bytes(b"alternate")
    checkpoint.symlink_to(outside)
    with pytest.raises(ValueError, match="contains a symlink"):
        _load_selected_retraining_cell(
            repository_root=tmp_path,
            verified_lock=_verified_lock(),
            lock_record=_lock_record(),
            architecture_role="primary_multiscale",
            campaign_id="primary-200",
            cell_index=0,
        )

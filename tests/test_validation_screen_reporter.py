from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

import scripts.evaluate_confirmatory_checkpoint as locked_evaluator
import scripts.report_validation_screen_tiles as reporter


def _screen_cells() -> list[dict]:
    cells = []
    for candidate in reporter.SCREEN_CANDIDATE_ORDER:
        for seed in reporter.SCREEN_SEEDS:
            index = len(cells)
            cells.append(
                {
                    "array_index": index,
                    "campaign_id": "12345",
                    "candidate": candidate,
                    "seed": seed,
                    "outcome_status": "success",
                    "result_artifact_repo_relative_identifier": (
                        "results/patch_training/protocol_runs/"
                        f"validation_screen_cell/12345/cell_{index:02d}/"
                        "metrics/model_selection.json"
                    ),
                    "result_artifact_sha256": f"{index + 1:064x}",
                    "selected_checkpoint_repo_relative_identifier": (
                        "results/patch_training/protocol_runs/"
                        f"validation_screen_cell/12345/cell_{index:02d}/"
                        "checkpoints/best_model.pth"
                    ),
                    "selected_checkpoint_sha256": f"{index + 101:064x}",
                    "selection_metrics": {
                        "score": 0.5,
                        "c0_iou": 0.5,
                        "c1_iou": 0.5,
                        "pore_union_iou": 0.7,
                        "validation_loss": 0.4,
                    },
                }
            )
    return cells


def _input_provenance() -> dict:
    development = reporter.LOCKED_INPUT_ATTESTATIONS[
        "development_train_plus_validation"
    ]
    training = reporter.LOCKED_INPUT_ATTESTATIONS["training_only"]
    algorithm = (
        "sha256 over lexicographically sorted UTF-8 relative filename, "
        "NUL, raw file bytes, NUL"
    )
    return {
        "input_source": "indexed_source_images",
        "scope": "development_train_plus_validation",
        "split_names": ["train", "val"],
        "image_count": development["image_count"],
        "image_aggregate_sha256": development["image_aggregate_sha256"],
        "image_aggregate_sha256_algorithm": algorithm,
        "file_name_list_sha256": "a" * 64,
        "training_subset": {
            "scope": "training_only",
            "split_names": ["train"],
            "image_count": training["image_count"],
            "image_aggregate_sha256": training["image_aggregate_sha256"],
            "image_aggregate_sha256_algorithm": algorithm,
            "file_name_list_sha256": "b" * 64,
        },
        "held_out_bytes_read": 0,
        "held_out_scope": "not_read_or_hashed_by_validation_only_trainer",
    }


def _target_provenance() -> dict:
    target = reporter.LOCKED_TARGET_ATTESTATIONS[
        "development_train_plus_validation"
    ]
    return {
        "target_source": "lossless_png_masks",
        "mask_count": target["mask_count"],
        "mask_aggregate_sha256": target["mask_aggregate_sha256"],
        "mask_aggregate_sha256_algorithm": (
            "sha256 over lexicographically sorted UTF-8 relative filename, "
            "NUL, raw file bytes, NUL"
        ),
        "validated_source_values": [0, 1, 255],
        "evaluation_mode": "train_validation_only",
        "held_out_dataset_constructed": False,
        "annotations_role": "image_index_and_metadata_only",
        "canonical_value_mapping": {
            "0": "0 (disconnected_pore)",
            "1": "1 (connected_pore)",
            "255": "2 (mineral) in three-class mode; ignore_index in two-class mode",
        },
    }


def _verified_lock() -> dict:
    validation_files = [f"pdo8_21_segment_{index}_0.png" for index in range(5)]
    resolved_split = {
        "manifest_sha256": reporter.LOCKED_SPLIT_MANIFEST_SHA256,
        "manifest_repo_relative_identifier": "config/confirmatory_splits.json",
        "annotation_index_sha256": reporter.LOCKED_ANNOTATION_INDEX_SHA256,
        "annotation_index_repo_relative_identifier": (
            "results/step3_coco_dataset/pore_annotations.json"
        ),
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_evaluation_count": 0,
        "partitions": {
            "train": {"image_ids": list(range(1, 75)), "image_files": [], "image_count": 74},
            "val": {
                "image_ids": [96, 97, 98, 99, 100],
                "image_files": validation_files,
                "image_count": 5,
            },
            "test": {"image_ids": list(range(72, 93)), "image_files": [], "image_count": 21},
        },
    }
    return {
        "schema_version": reporter.SELECTED_METHOD_LOCK_SCHEMA_VERSION,
        "selected_method": "R3",
        "resolved_protocol": reporter.PROSPECTIVE_METHOD_PROTOCOLS["R3"],
        "screen_selection_provenance": {
            "campaign_id": "12345",
            "deterministic_winner": "R3",
            "screen_cells": _screen_cells(),
            "resolved_data_split": resolved_split,
            "input_provenance": _input_provenance(),
            "target_provenance": _target_provenance(),
            "source_code_sha256": {"source.py": "c" * 64},
        },
    }


def _metrics(confusion: np.ndarray) -> dict:
    return locked_evaluator.metrics_from_confusion(
        np.asarray(confusion, dtype=np.int64)
    )


def _indexed_digest(root: Path, names: list[str]) -> tuple[str, str]:
    digest = hashlib.sha256()
    sorted_names = sorted(names)
    for name in sorted_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    name_digest = hashlib.sha256(
        json.dumps(sorted_names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest.hexdigest(), name_digest


def _cell_reports() -> list[dict]:
    reports = []
    for candidate_index, candidate in enumerate(reporter.SCREEN_CANDIDATE_ORDER):
        for seed_index, seed in enumerate(reporter.SCREEN_SEEDS):
            confusion = np.asarray(
                [
                    [80 + candidate_index + seed_index, 4, 1],
                    [5, 70 + candidate_index + seed_index, 2],
                    [1, 2, 100 + candidate_index],
                ],
                dtype=np.int64,
            )
            tiles = []
            for ordinal in range(1, 6):
                tiles.append(
                    {
                        "validation_ordinal": ordinal,
                        "image_id": 95 + ordinal,
                        "file_name": f"pdo8_21_segment_{ordinal - 1}_0.png",
                        "input_image_sha256": f"{ordinal:064x}",
                        "target_mask_sha256": f"{ordinal + 10:064x}",
                        "confusion_matrix": confusion,
                        "metrics": _metrics(confusion),
                    }
                )
            reports.append(
                {
                    "array_index": len(reports),
                    "candidate": candidate,
                    "seed": seed,
                    "pooled_confusion_matrix": confusion * 5,
                    "pooled_metrics": _metrics(confusion * 5),
                    "per_tile": tiles,
                }
            )
    return reports


def _checkpoint_envelope(
    lock: dict, cell: dict, source_hashes: dict[str, str]
) -> tuple[dict, dict]:
    protocol = reporter.PROSPECTIVE_METHOD_PROTOCOLS["R3"]
    data_loader = {
        "training_patch_size": protocol["training_patch_size"],
        "training_batch_size": protocol["training_batch_size"],
        "evaluation_patch_size": protocol["evaluation_patch_size"],
        "evaluation_batch_size": protocol["evaluation_batch_size"],
        "shuffle_generator_seed": 42,
        "distributed_sampler_seed": None,
    }
    augmentation = {"seed": 42, "data_loader": data_loader}
    resolved_model = {
        "architecture": protocol["model_type"],
        "input_channels": protocol["model_input_channels"],
        "output_classes": protocol["model_output_classes"],
        "dropout_requested": protocol["dropout_requested"],
        "deep_supervision": False,
    }
    scientific_model = {
        "architecture_resolved": protocol["model_type"],
        "input_channels": protocol["model_input_channels"],
        "output_classes": protocol["model_output_classes"],
        "deep_supervision": False,
    }
    execution = {"augmentation": augmentation, "model": scientific_model}
    cell["scientific_execution_contract"] = copy.deepcopy(execution)
    input_provenance = lock["screen_selection_provenance"]["input_provenance"]
    target_provenance = lock["screen_selection_provenance"]["target_provenance"]
    data_split = lock["screen_selection_provenance"]["resolved_data_split"]
    checkpoint = {
        "checkpoint_role": "validation_composite_selection",
        "best_selection_epoch": 7,
        "selection_metric_name": "validation_c0_c1_iou_harmonic_mean",
        "model_state_dict": {},
        "model_type": protocol["model_type"],
        "loss_type": protocol["loss_type"],
        "num_classes": protocol["model_output_classes"],
        "input_normalization": copy.deepcopy(
            reporter.EXPECTED_INPUT_NORMALIZATION
        ),
        "input_provenance": copy.deepcopy(input_provenance),
        "target_provenance": copy.deepcopy(target_provenance),
        "source_code_sha256": dict(source_hashes),
        "resolved_config": {
            "protocol_run_role": "validation_screen_cell",
            "protocol_campaign_id": cell["campaign_id"],
            "protocol_cell_index": cell["array_index"],
            "protocol_candidate_key": cell["candidate"],
            "selected_method_lock": None,
            "input_normalization": copy.deepcopy(
                reporter.EXPECTED_INPUT_NORMALIZATION
            ),
            "evaluation": {
                "mode": "validation_only",
                "held_out_dataset_constructed": False,
                "held_out_evaluation_count": 0,
            },
            "data_split": copy.deepcopy(data_split),
            "input": copy.deepcopy(input_provenance),
            "target": copy.deepcopy(target_provenance),
            "source_code_sha256": dict(source_hashes),
            "augmentation": copy.deepcopy(augmentation),
            "scientific_execution_contract": copy.deepcopy(execution),
            "model": resolved_model,
            "loss": {"type": protocol["loss_type"]},
            "inference": {
                "mode": "native_model_argmax",
                "network_outputs": 3,
                "conditional_pore_threshold": None,
            },
        },
    }
    inferred_model = {
        "architecture": protocol["model_type"],
        "n_channels": protocol["model_input_channels"],
        "num_classes": protocol["model_output_classes"],
        "base_features": 32,
        "bilinear": True,
        "deep_supervision": False,
        "source": "synthetic_tensor_shape_contract",
    }
    return checkpoint, inferred_model


def test_scientific_inference_primitives_are_imported_from_locked_evaluator():
    assert reporter.prepare_locked_model_input is locked_evaluator.prepare_locked_model_input
    assert reporter.compose_locked_probabilities is locked_evaluator.compose_locked_probabilities
    assert reporter.create_model_from_state is locked_evaluator.create_model_from_state
    assert (
        reporter.load_lossless_target_mask_bytes
        is locked_evaluator.load_lossless_target_mask_bytes
    )
    assert reporter.confusion_from_labels is locked_evaluator.confusion_from_labels
    assert reporter.metrics_from_confusion is locked_evaluator.metrics_from_confusion


def test_screen_matrix_requires_all_15_successful_frozen_checkpoints():
    lock = _verified_lock()
    cells = reporter._screen_cells(lock)
    assert len(cells) == 15
    assert [(cell["candidate"], cell["seed"]) for cell in cells] == [
        (candidate, seed)
        for candidate in reporter.SCREEN_CANDIDATE_ORDER
        for seed in reporter.SCREEN_SEEDS
    ]

    failed = copy.deepcopy(lock)
    failed["screen_selection_provenance"]["screen_cells"][8][
        "outcome_status"
    ] = "failed"
    with pytest.raises(ValueError, match="all 15"):
        reporter._screen_cells(failed)

    reordered = copy.deepcopy(lock)
    reordered["screen_selection_provenance"]["screen_cells"][0], reordered[
        "screen_selection_provenance"
    ]["screen_cells"][1] = (
        reordered["screen_selection_provenance"]["screen_cells"][1],
        reordered["screen_selection_provenance"]["screen_cells"][0],
    )
    with pytest.raises(ValueError, match="indices"):
        reporter._screen_cells(reordered)


def test_selected_lock_verifier_errors_fail_closed(tmp_path: Path, monkeypatch):
    lock = tmp_path / "config" / "selected_method_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text("{}\n", encoding="utf-8")
    annotation = tmp_path / "results/step3_coco_dataset/pore_annotations.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text('{"must_not_be_read":true}\n', encoding="utf-8")

    verifier_calls = []

    def verify(document, root, verify_live_corpus_bytes):
        verifier_calls.append(verify_live_corpus_bytes)
        return _verified_lock()

    monkeypatch.setattr(reporter, "verify_selected_method_lock_document", verify)
    original_open = Path.open

    def no_annotation_open(path, *args, **kwargs):
        if path == annotation:  # pragma: no cover - any access is failure
            raise AssertionError("full annotation JSON was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_annotation_open)
    verified, digest, identifier, cells = reporter.load_verified_screen_lock(
        Path("config/selected_method_lock.json"), tmp_path
    )
    assert verified["selected_method"] == "R3"
    assert digest == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert identifier == "config/selected_method_lock.json"
    assert len(cells) == 15
    assert verifier_calls == [False]

    def reject(*args, **kwargs):
        raise ValueError("screen source hash mapping is incomplete/stale")

    monkeypatch.setattr(reporter, "verify_selected_method_lock_document", reject)
    with pytest.raises(ValueError, match="source hash"):
        reporter.load_verified_screen_lock(
            Path("config/selected_method_lock.json"), tmp_path
        )


def test_lock_path_must_be_canonical_and_inside_repository(tmp_path: Path):
    with pytest.raises(ValueError, match="Canonical path drift"):
        reporter.load_verified_screen_lock(Path("results/winner.json"), tmp_path)
    with pytest.raises(ValueError, match="outside the repository"):
        reporter.load_verified_screen_lock(tmp_path.parent / "winner.json", tmp_path)


def test_synthetic_checkpoint_authentication_links_every_frozen_identity(
    tmp_path: Path,
):
    source_hashes = {}
    for identifier in reporter.EXECUTION_SOURCE_FILES:
        source = reporter.PROJECT_ROOT / identifier
        destination = tmp_path / identifier
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[identifier] = reporter.sha256_file(destination)

    lock = _verified_lock()
    lock["screen_selection_provenance"]["source_code_sha256"] = dict(
        source_hashes
    )
    cell = lock["screen_selection_provenance"]["screen_cells"][0]
    checkpoint, inferred_model = _checkpoint_envelope(lock, cell, source_hashes)
    checkpoint_path = (
        tmp_path / cell["selected_checkpoint_repo_relative_identifier"]
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"synthetic checkpoint envelope\n")
    cell["selected_checkpoint_sha256"] = reporter.sha256_file(checkpoint_path)

    path, digest, identifier = reporter._authenticated_screen_checkpoint_file(
        cell, tmp_path
    )
    assert path == checkpoint_path
    assert digest == cell["selected_checkpoint_sha256"]
    identity = reporter.validate_screen_checkpoint_envelope(
        cell,
        checkpoint,
        lock,
        inferred_model,
        checkpoint_sha256=digest,
        checkpoint_identifier=identifier,
        repository_root=tmp_path,
    )
    assert identity["checkpoint_sha256"] == digest
    assert identity["normalization"] == reporter.EXPECTED_INPUT_NORMALIZATION
    assert identity["source"]["file_count"] == len(
        reporter.EXECUTION_SOURCE_FILES
    )
    assert identity["target"]["held_out_test_included"] is False
    assert identity["input"]["held_out_test_included"] is False
    assert identity["protocol"]["candidate"] == "R3"
    assert identity["seed"]["training_seed"] == 42

    bad_identifier = copy.deepcopy(cell)
    bad_identifier["selected_checkpoint_repo_relative_identifier"] = (
        "results/checkpoints/best_model.pth"
    )
    with pytest.raises(ValueError, match="not canonical"):
        reporter._authenticated_screen_checkpoint_file(bad_identifier, tmp_path)
    bad_hash = copy.deepcopy(cell)
    bad_hash["selected_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        reporter._authenticated_screen_checkpoint_file(bad_hash, tmp_path)

    bad_normalization = copy.deepcopy(checkpoint)
    bad_normalization["input_normalization"]["output_range"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="normalization"):
        reporter.validate_screen_checkpoint_envelope(
            cell,
            bad_normalization,
            lock,
            inferred_model,
            checkpoint_sha256=digest,
            checkpoint_identifier=identifier,
            repository_root=tmp_path,
        )
    bad_split = copy.deepcopy(checkpoint)
    bad_split["resolved_config"]["data_split"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="split/dev79"):
        reporter.validate_screen_checkpoint_envelope(
            cell,
            bad_split,
            lock,
            inferred_model,
            checkpoint_sha256=digest,
            checkpoint_identifier=identifier,
            repository_root=tmp_path,
        )
    bad_protocol = copy.deepcopy(checkpoint)
    bad_protocol["resolved_config"]["model"]["architecture"] = "plain_unet"
    with pytest.raises(ValueError, match="protocol mismatch"):
        reporter.validate_screen_checkpoint_envelope(
            cell,
            bad_protocol,
            lock,
            inferred_model,
            checkpoint_sha256=digest,
            checkpoint_identifier=identifier,
            repository_root=tmp_path,
        )
    bad_seed = copy.deepcopy(checkpoint)
    bad_seed["resolved_config"]["augmentation"]["seed"] = 123
    with pytest.raises(ValueError, match="identity mismatch"):
        reporter.validate_screen_checkpoint_envelope(
            cell,
            bad_seed,
            lock,
            inferred_model,
            checkpoint_sha256=digest,
            checkpoint_identifier=identifier,
            repository_root=tmp_path,
        )

    source_to_tamper = tmp_path / reporter.EXECUTION_SOURCE_FILES[0]
    source_to_tamper.write_bytes(source_to_tamper.read_bytes() + b"\n# drift\n")
    with pytest.raises(ValueError, match="source hash drift"):
        reporter.validate_screen_checkpoint_envelope(
            cell,
            checkpoint,
            lock,
            inferred_model,
            checkpoint_sha256=digest,
            checkpoint_identifier=identifier,
            repository_root=tmp_path,
        )


def test_weights_only_checkpoint_deserialization_follows_same_authentication_chain(
    tmp_path: Path,
):
    torch = pytest.importorskip("torch")
    source_hashes = {}
    for identifier in reporter.EXECUTION_SOURCE_FILES:
        source = reporter.PROJECT_ROOT / identifier
        destination = tmp_path / identifier
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[identifier] = reporter.sha256_file(destination)
    lock = _verified_lock()
    lock["screen_selection_provenance"]["source_code_sha256"] = dict(
        source_hashes
    )
    cell = lock["screen_selection_provenance"]["screen_cells"][0]
    checkpoint, _ = _checkpoint_envelope(lock, cell, source_hashes)
    checkpoint["model_state_dict"] = {
        "inc.double_conv.0.weight": torch.zeros((32, 1, 3, 3)),
        "encoder.branch1.synthetic": torch.zeros((1,)),
        "outc.weight": torch.zeros((3, 32, 1, 1)),
    }
    path = tmp_path / cell["selected_checkpoint_repo_relative_identifier"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    cell["selected_checkpoint_sha256"] = reporter.sha256_file(path)
    loaded, state, identity = reporter.authenticate_screen_checkpoint(
        cell, lock, tmp_path
    )
    assert loaded["checkpoint_role"] == "validation_composite_selection"
    assert identity["checkpoint_sha256"] == cell["selected_checkpoint_sha256"]
    assert identity["inferred_model"]["architecture"] == (
        "multiscale_attention_unet"
    )
    assert state["outc.weight"].shape == (3, 32, 1, 1)


def test_canonical_paths_reject_symbolic_links(tmp_path: Path):
    target = tmp_path / "real-lock.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "config" / "selected_method_lock.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        reporter.load_verified_screen_lock(
            Path("config/selected_method_lock.json"), tmp_path
        )


def test_validation_loader_rejects_symlink_before_decoder_import(tmp_path: Path):
    image_root = tmp_path / reporter.CANONICAL_IMAGE_ROOT
    mask_root = tmp_path / reporter.CANONICAL_MASK_ROOT
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    backing = image_root / "backing.png"
    backing.write_bytes(b"not opened")
    specs = []
    for ordinal in range(1, 6):
        name = f"pdo8_21_segment_{ordinal - 1}_0.png"
        image_path = image_root / name
        if ordinal == 1:
            image_path.symlink_to(backing)
        else:
            image_path.write_bytes(b"not opened")
        (mask_root / name).write_bytes(b"not opened")
        specs.append(
            {
                "validation_ordinal": ordinal,
                "image_id": 95 + ordinal,
                "file_name": name,
            }
        )
    with pytest.raises(ValueError, match="symbolic link"):
        reporter.load_validation_tiles(specs, tmp_path)


def test_native_validation_image_contract_is_exact():
    native = np.zeros((2048, 2048), dtype=np.uint8)
    assert reporter.validate_native_validation_image(native, "synthetic.png") is native
    with pytest.raises(ValueError, match="2048x2048"):
        reporter.validate_native_validation_image(
            np.zeros((1024, 2048), dtype=np.uint8), "synthetic.png"
        )
    with pytest.raises(ValueError, match="raw uint8"):
        reporter.validate_native_validation_image(
            np.zeros((2048, 2048), dtype=np.float32), "synthetic.png"
        )


def test_validation_loader_enforces_native_2048_and_rejects_symlink(
    tmp_path: Path,
):
    cv2 = pytest.importorskip("cv2")
    image_root = tmp_path / reporter.CANONICAL_IMAGE_ROOT
    mask_root = tmp_path / reporter.CANONICAL_MASK_ROOT
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    specs = []
    for ordinal in range(1, 6):
        name = f"pdo8_21_segment_{ordinal - 1}_0.png"
        image = np.full((2048, 2048), ordinal * 20, dtype=np.uint8)
        target = np.full((2048, 2048), 255, dtype=np.uint8)
        target[:512] = 0
        target[512:1024] = 1
        assert cv2.imwrite(str(image_root / name), image)
        assert cv2.imwrite(str(mask_root / name), target)
        specs.append(
            {
                "validation_ordinal": ordinal,
                "image_id": 95 + ordinal,
                "file_name": name,
            }
        )
    tiles = reporter.load_validation_tiles(specs, tmp_path)
    assert len(tiles) == 5
    assert all(tile["image"].shape == (2048, 2048) for tile in tiles)
    assert all(tile["target"].shape == (2048, 2048) for tile in tiles)

    first = image_root / specs[0]["file_name"]
    backing = image_root / "backing.png"
    first.replace(backing)
    first.symlink_to(backing)
    with pytest.raises(ValueError, match="symbolic link"):
        reporter.load_validation_tiles(specs, tmp_path)


def test_validation_loader_rejects_non_native_shape(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    image_root = tmp_path / reporter.CANONICAL_IMAGE_ROOT
    mask_root = tmp_path / reporter.CANONICAL_MASK_ROOT
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    specs = []
    for ordinal in range(1, 6):
        name = f"pdo8_21_segment_{ordinal - 1}_0.png"
        shape = (1024, 2048) if ordinal == 1 else (2048, 2048)
        assert cv2.imwrite(
            str(image_root / name), np.zeros(shape, dtype=np.uint8)
        )
        assert cv2.imwrite(
            str(mask_root / name), np.zeros((2048, 2048), dtype=np.uint8)
        )
        specs.append(
            {
                "validation_ordinal": ordinal,
                "image_id": 95 + ordinal,
                "file_name": name,
            }
        )
    with pytest.raises(ValueError, match="2048x2048"):
        reporter.load_validation_tiles(specs, tmp_path)


def test_validation_projection_never_touches_retrospective_partition():
    class RetrospectiveBomb:
        def __getattribute__(self, name):  # pragma: no cover - any touch fails
            raise AssertionError("retrospective partition was touched")

        def __iter__(self):  # pragma: no cover - any touch fails
            raise AssertionError("retrospective partition was iterated")

    lock = _verified_lock()
    lock["screen_selection_provenance"]["resolved_data_split"]["partitions"][
        "test"
    ] = RetrospectiveBomb()
    specs = reporter.validation_tile_specs(lock)
    assert len(specs) == 5
    assert [item["image_id"] for item in specs] == [96, 97, 98, 99, 100]
    assert all(item["file_name"].startswith("pdo8_21_segment_") for item in specs)


def test_validation_projection_rejects_noncanonical_or_wrong_count():
    lock = _verified_lock()
    lock["screen_selection_provenance"]["resolved_data_split"]["partitions"][
        "val"
    ]["image_files"][0] = "pdo2_24_segment_0_0.png"
    with pytest.raises(ValueError, match="pdo8_21"):
        reporter.validation_tile_specs(lock)

    lock = _verified_lock()
    validation = lock["screen_selection_provenance"]["resolved_data_split"][
        "partitions"
    ]["val"]
    validation["image_ids"].pop()
    validation["image_files"].pop()
    validation["image_count"] = 4
    with pytest.raises(ValueError, match="exactly five"):
        reporter.validation_tile_specs(lock)


def test_dev79_identity_is_exact_and_held_out_free():
    lock = _verified_lock()
    identity = reporter._development_identity(lock)
    assert identity["input_image_count"] == 79
    assert identity["target_mask_count"] == 79
    assert identity["held_out_bytes_read"] == 0

    tampered = copy.deepcopy(lock)
    tampered["screen_selection_provenance"]["input_provenance"][
        "image_aggregate_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="dev79"):
        reporter._development_identity(tampered)


def test_independent_dev79_reattest_reads_only_recorded_train_and_val(
    tmp_path: Path, monkeypatch
):
    train_names = [f"pdo1_12_segment_{index}_0.png" for index in range(74)]
    validation_names = [f"pdo8_21_segment_{index}_0.png" for index in range(5)]
    development_names = train_names + validation_names
    image_root = tmp_path / reporter.CANONICAL_IMAGE_ROOT
    mask_root = tmp_path / reporter.CANONICAL_MASK_ROOT
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    for index, name in enumerate(development_names):
        (image_root / name).write_bytes(f"image-{index}".encode())
        (mask_root / name).write_bytes(f"mask-{index}".encode())

    # These sentinels exist specifically to prove that neither the full index
    # nor retrospective files participate in development re-attestation.
    annotation = tmp_path / "results/step3_coco_dataset/pore_annotations.json"
    annotation.write_text('{"forbidden":true}\n', encoding="utf-8")
    forbidden_name = "pdo2_24_segment_0_0.png"
    (image_root / forbidden_name).write_bytes(b"forbidden-test-image")
    (mask_root / forbidden_name).write_bytes(b"forbidden-test-mask")

    input_hash, name_hash = _indexed_digest(image_root, development_names)
    mask_hash, _ = _indexed_digest(mask_root, development_names)
    algorithm = (
        "sha256 over lexicographically sorted UTF-8 relative filename, "
        "NUL, raw file bytes, NUL"
    )

    class NoTestProjection(dict):
        def get(self, key, *args):
            if key == "test":  # pragma: no cover - any access is a failure
                raise AssertionError("test partition was projected")
            return super().get(key, *args)

        def __getitem__(self, key):
            if key == "test":  # pragma: no cover - any access is a failure
                raise AssertionError("test partition was projected")
            return super().__getitem__(key)

    lock = {
        "screen_selection_provenance": {
            "resolved_data_split": {
                "partitions": NoTestProjection(
                    {
                        "train": {
                            "image_ids": list(range(1, 75)),
                            "image_files": train_names,
                            "image_count": 74,
                        },
                        "val": {
                            "image_ids": [96, 97, 98, 99, 100],
                            "image_files": validation_names,
                            "image_count": 5,
                        },
                        "test": object(),
                    }
                )
            },
            "input_provenance": {
                "scope": "development_train_plus_validation",
                "split_names": ["train", "val"],
                "image_count": 79,
                "image_aggregate_sha256": input_hash,
                "image_aggregate_sha256_algorithm": algorithm,
                "file_name_list_sha256": name_hash,
            },
            "target_provenance": {
                "mask_count": 79,
                "mask_aggregate_sha256": mask_hash,
                "mask_aggregate_sha256_algorithm": algorithm,
            },
        }
    }

    original_open = Path.open
    opened = []

    def guarded_open(path, *args, **kwargs):
        text = path.as_posix()
        if text.endswith("pore_annotations.json") or forbidden_name in text:
            raise AssertionError(f"forbidden path opened: {text}")
        opened.append(text)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    attestation = reporter.independently_reattest_development_bytes(lock, tmp_path)
    assert attestation["input_files_read"] == 79
    assert attestation["target_files_read"] == 79
    assert attestation["annotation_index_bytes_read"] == 0
    assert attestation["annotation_index_parsed"] is False
    assert attestation["locked_retrospective_paths_constructed"] == 0
    assert len(opened) == 158


def test_independent_dev79_reattest_rejects_byte_drift(tmp_path: Path):
    train_names = [f"pdo1_12_segment_{index}_0.png" for index in range(74)]
    validation_names = [f"pdo8_21_segment_{index}_0.png" for index in range(5)]
    names = train_names + validation_names
    image_root = tmp_path / reporter.CANONICAL_IMAGE_ROOT
    mask_root = tmp_path / reporter.CANONICAL_MASK_ROOT
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    for name in names:
        (image_root / name).write_bytes(b"input")
        (mask_root / name).write_bytes(b"target")
    image_hash, name_hash = _indexed_digest(image_root, names)
    mask_hash, _ = _indexed_digest(mask_root, names)
    algorithm = (
        "sha256 over lexicographically sorted UTF-8 relative filename, "
        "NUL, raw file bytes, NUL"
    )
    lock = {
        "screen_selection_provenance": {
            "resolved_data_split": {
                "partitions": {
                    "train": {
                        "image_ids": list(range(74)),
                        "image_files": train_names,
                        "image_count": 74,
                    },
                    "val": {
                        "image_ids": list(range(74, 79)),
                        "image_files": validation_names,
                        "image_count": 5,
                    },
                }
            },
            "input_provenance": {
                "scope": "development_train_plus_validation",
                "split_names": ["train", "val"],
                "image_count": 79,
                "image_aggregate_sha256": image_hash,
                "image_aggregate_sha256_algorithm": algorithm,
                "file_name_list_sha256": name_hash,
            },
            "target_provenance": {
                "mask_count": 79,
                "mask_aggregate_sha256": mask_hash,
                "mask_aggregate_sha256_algorithm": algorithm,
            },
        }
    }
    (mask_root / validation_names[-1]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="target bytes"):
        reporter.independently_reattest_development_bytes(lock, tmp_path)


def test_pooled_screen_metric_identity_is_tight_and_fail_closed():
    observed = {
        "score": 0.6,
        "c0_iou": 0.7,
        "c1_iou": 0.5,
        "pore_union_iou": 0.8,
    }
    reporter._validate_pooled_screen_identity(observed, dict(observed))
    drifted = dict(observed)
    drifted["c0_iou"] += 1e-4
    with pytest.raises(ValueError, match="c0_iou"):
        reporter._validate_pooled_screen_identity(drifted, observed)


def test_signed_r3_margins_are_descriptive_candidate_minus_reference():
    cells = _cell_reports()
    summaries, rows = reporter.attach_signed_r3_margins(cells)
    assert [item["candidate"] for item in summaries] == list(
        reporter.SCREEN_CANDIDATE_ORDER
    )
    r3 = next(item for item in summaries if item["candidate"] == "R3")
    assert all(value == pytest.approx(0.0) for value in r3["signed_r3_margins"]["values"].values())
    c2fp = next(item for item in summaries if item["candidate"] == "C2-FP")
    assert c2fp["signed_r3_margins"]["definition"] == (
        "candidate_mean_minus_R3_mean"
    )
    assert "winner" not in json.dumps(summaries).lower()
    assert len(rows) == (15 + 5) * len(reporter.MARGIN_METRICS)
    assert {row["scope"] for row in rows} == {
        "same_seed",
        "three_seed_arithmetic_mean",
    }
    assert all(
        row["evidence_label"] == "validation-only model development"
        for row in rows
    )


def test_metric_csv_has_every_tile_and_pooled_class_row():
    rows = reporter.metric_csv_rows(_cell_reports())
    assert len(rows) == 15 * (5 + 1) * 3
    assert {row["aggregation"] for row in rows} == {
        "tile",
        "pooled_five_validation_tiles",
    }
    assert {row["class_id"] for row in rows} == {0, 1, 2}
    assert all(row["evidence_kind"] == reporter.EVIDENCE_KIND for row in rows)
    assert all(
        row["evidence_label"] == "validation-only model development"
        for row in rows
    )


def test_publication_rows_include_every_candidate_seed_and_fixed_tile():
    cells = _cell_reports()
    summaries, _ = reporter.attach_signed_r3_margins(cells)
    summary_rows = reporter.publication_summary_rows(cells, summaries)
    tile_rows = reporter.publication_tile_rows(cells)

    expected_pairs = [
        (candidate, seed)
        for candidate in reporter.SCREEN_CANDIDATE_ORDER
        for seed in reporter.SCREEN_SEEDS
    ]
    assert [(row["candidate"], row["seed"]) for row in summary_rows] == (
        expected_pairs
    )
    assert len(summary_rows) == 15
    assert len(tile_rows) == 75
    assert all(
        row["evidence_label"] == "validation-only model development"
        for row in (*summary_rows, *tile_rows)
    )
    for candidate, seed in expected_pairs:
        rows = [
            row
            for row in tile_rows
            if (row["candidate"], row["seed"]) == (candidate, seed)
        ]
        assert [row["validation_ordinal"] for row in rows] == [1, 2, 3, 4, 5]
        assert all(row["outcome_dependent_tile_choice_performed"] is False for row in rows)

    for candidate in reporter.SCREEN_CANDIDATE_ORDER:
        candidate_rows = [
            row for row in summary_rows if row["candidate"] == candidate
        ]
        c0_values = np.asarray(
            [row["c0_iou"] for row in candidate_rows], dtype=np.float64
        )
        assert all(
            row["candidate_mean_c0_iou"] == pytest.approx(c0_values.mean())
            for row in candidate_rows
        )
        assert all(
            row["candidate_sample_sd_c0_iou"]
            == pytest.approx(c0_values.std(ddof=1))
            for row in candidate_rows
        )
    r3_rows = [row for row in summary_rows if row["candidate"] == "R3"]
    for row in r3_rows:
        for key, _label, _vector_key, _color in reporter.PUBLICATION_METRICS:
            assert row[
                f"same_seed_{key}_margin_candidate_minus_r3"
            ] == pytest.approx(0.0)
            assert row[
                f"candidate_mean_{key}_margin_candidate_minus_r3"
            ] == pytest.approx(0.0)

    missing_tile = copy.deepcopy(cells)
    missing_tile[0]["per_tile"].pop()
    with pytest.raises(ValueError, match="exactly five"):
        reporter.publication_tile_rows(missing_tile)


def test_report_bundle_is_byte_deterministic_and_refuses_overwrite(tmp_path: Path):
    report = {
        "schema_version": 1,
        "evidence_kind": reporter.EVIDENCE_KIND,
        "scientific_scope": {
            "winner_reselection_performed": False,
            "locked_retrospective_input_bytes_read": 0,
        },
    }
    cells = _cell_reports()
    summaries, margin_rows = reporter.attach_signed_r3_margins(cells)
    metric_rows = reporter.metric_csv_rows(cells)
    summary_rows = reporter.publication_summary_rows(cells, summaries)
    tile_rows = reporter.publication_tile_rows(cells)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_checksums = reporter.write_report_bundle(
        first,
        report,
        metric_rows,
        margin_rows,
        summary_rows,
        tile_rows,
    )
    second_checksums = reporter.write_report_bundle(
        second,
        report,
        metric_rows,
        margin_rows,
        summary_rows,
        tile_rows,
    )
    assert first_checksums == second_checksums
    for name in (*reporter.OUTPUT_FILE_NAMES, "checksums.sha256"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    checksum_lines = (first / "checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    assert checksum_lines == [
        f"{first_checksums[name]}  {name}"
        for name in sorted(reporter.OUTPUT_FILE_NAMES)
    ]
    with (first / reporter.PUBLICATION_SUMMARY_CSV_NAME).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rendered_summary_rows = list(csv.DictReader(handle))
    assert len(rendered_summary_rows) == 15
    assert [
        (row["candidate"], int(row["seed"])) for row in rendered_summary_rows
    ] == [
        (candidate, seed)
        for candidate in reporter.SCREEN_CANDIDATE_ORDER
        for seed in reporter.SCREEN_SEEDS
    ]
    with (first / reporter.PUBLICATION_TILE_CSV_NAME).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rendered_tile_rows = list(csv.DictReader(handle))
    assert len(rendered_tile_rows) == 75
    assert {int(row["validation_ordinal"]) for row in rendered_tile_rows} == {
        1,
        2,
        3,
        4,
        5,
    }
    assert all(
        row["evidence_label"] == "validation-only model development"
        for row in (*rendered_summary_rows, *rendered_tile_rows)
    )

    tex_text = (first / reporter.PUBLICATION_TEX_NAME).read_text(encoding="utf-8")
    lowered_tex = tex_text.lower()
    assert "\\begin{figure}[pos=htbp]" in tex_text
    assert "\\begin{table}[pos=htbp]" in tex_text
    assert "\\begin{figure*}" not in tex_text
    assert "\\begin{table*}" not in tex_text
    assert "[t]" not in tex_text
    assert "validation-only model development" in lowered_tex
    assert "seeds 42, 123, and 2025" in lowered_tex
    assert "all five validation tiles" in lowered_tex
    assert "no re-selection" in lowered_tex
    assert "no outcome-dependent tile choice" in lowered_tex
    assert "held-out" not in lowered_tex
    assert "unseen" not in lowered_tex
    assert "significan" not in lowered_tex
    for candidate in reporter.SCREEN_CANDIDATE_ORDER:
        assert candidate in tex_text

    from PIL import Image

    with Image.open(first / reporter.PUBLICATION_PNG_NAME) as image:
        assert image.size == (
            round(
                reporter.CAS_SC_TEXT_WIDTH_INCHES
                * reporter.PUBLICATION_PNG_DPI
            ),
            round(
                reporter.CAS_SUMMARY_HEIGHT_INCHES
                * reporter.PUBLICATION_PNG_DPI
            ),
        )
        assert image.info["dpi"][0] == pytest.approx(
            reporter.PUBLICATION_PNG_DPI, abs=0.1
        )
        pixels = np.asarray(image.convert("RGB"))
    for color in reporter.PUBLICATION_CLASS_COLORS.values():
        expected_rgb = np.asarray(
            tuple(int(color[index : index + 2], 16) for index in (1, 3, 5)),
            dtype=np.uint8,
        )
        assert np.any(np.all(pixels == expected_rgb, axis=2))

    pdf_bytes = (first / reporter.PUBLICATION_PDF_NAME).read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"/MediaBox [ 0 0 466.56 518.4 ]" in pdf_bytes
    assert b"/FontFile2" in pdf_bytes
    assert b"/Type3" not in pdf_bytes
    if shutil.which("pdftotext"):
        extracted = tmp_path / "summary.txt"
        subprocess.run(
            [
                shutil.which("pdftotext"),
                str(first / reporter.PUBLICATION_PDF_NAME),
                str(extracted),
            ],
            check=True,
        )
        visible_text = extracted.read_text(encoding="utf-8").lower()
        assert "validation-only model development" in visible_text
        assert "seed 42" in visible_text
        assert "seed 123" in visible_text
        assert "seed 2025" in visible_text
        assert all(f"tile {ordinal}" in visible_text for ordinal in range(1, 6))
        assert "held-out" not in visible_text
        assert "unseen" not in visible_text
        assert "significan" not in visible_text

    pdflatex = shutil.which("pdflatex")
    cas_class = reporter.PROJECT_ROOT / "Overleaf" / "cas-sc.cls"
    if pdflatex and cas_class.is_file():
        integration_tex = first / "cas_fragment_integration.tex"
        integration_tex.write_text(
            "\n".join(
                [
                    "\\documentclass[a4paper,fleqn]{cas-sc}",
                    "\\usepackage[authoryear]{natbib}",
                    "\\usepackage{graphicx}",
                    "\\usepackage{booktabs}",
                    "\\begin{document}",
                    "\\typeout{CAS_TEXTWIDTH=\\the\\textwidth}",
                    f"\\input{{{reporter.PUBLICATION_TEX_NAME}}}",
                    "\\clearpage",
                    "\\end{document}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        environment = os.environ.copy()
        prior_texinputs = environment.get("TEXINPUTS", "")
        environment["TEXINPUTS"] = (
            f"{cas_class.parent}//:{prior_texinputs}"
        )
        completed = subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                integration_tex.name,
            ],
            cwd=first,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        log_text = (first / "cas_fragment_integration.log").read_text(
            encoding="utf-8", errors="replace"
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Overleaf/cas-sc.cls" in log_text
        assert "CAS_TEXTWIDTH=468.3324pt" in log_text
        lowered_log = log_text.lower()
        for forbidden in (
            "unknown key",
            "is unknown",
            "unknown option",
            "keyval error",
            "overfull \\hbox",
            "overfull \\vbox",
            "float too large",
        ):
            assert forbidden not in lowered_log

    with pytest.raises(FileExistsError, match="overwrite"):
        reporter.write_report_bundle(
            first,
            report,
            metric_rows,
            margin_rows,
            summary_rows,
            tile_rows,
        )


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, axis):
        return _FakeTensor(np.expand_dims(self.array, axis))

    def to(self, device):
        return self

    def float(self):
        return self

    def __getitem__(self, item):
        return _FakeTensor(self.array[item])

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.array)


class _FakeBoolean:
    def all(self):
        return self

    def item(self):
        return True


class _FakeCuda:
    @staticmethod
    def empty_cache():
        return None


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def from_numpy(array):
        return _FakeTensor(array)

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def isfinite(tensor):
        return _FakeBoolean()

    @staticmethod
    def softmax(tensor, dim):
        return tensor


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, tensor):
        probabilities = np.zeros((1, 3, 2, 2), dtype=np.float32)
        probabilities[:, 0] = 1.0
        return _FakeTensor(probabilities)


def test_cell_inference_consumes_exactly_five_tiles_and_reuses_composition(
    monkeypatch,
):
    cell = _screen_cells()[0]
    confusion_per_tile = np.asarray(
        [[2, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=np.int64
    )
    pooled_metrics = _metrics(confusion_per_tile * 5)
    observed_selection = reporter._observed_selection_metrics(pooled_metrics)
    cell["selection_metrics"] = {
        **observed_selection,
        "validation_loss": 0.4,
    }
    tiles = [
        {
            "validation_ordinal": ordinal,
            "image_id": 95 + ordinal,
            "file_name": f"pdo8_21_segment_{ordinal}_0.png",
            "input_image_sha256": f"{ordinal:064x}",
            "target_mask_sha256": f"{ordinal + 10:064x}",
            "image": np.zeros((2, 2), dtype=np.uint8),
            "target": np.asarray([[0, 0], [1, 1]], dtype=np.uint8),
        }
        for ordinal in range(1, 6)
    ]
    monkeypatch.setattr(reporter, "EXPECTED_TILE_SHAPE", (2, 2))
    monkeypatch.setattr(
        reporter,
        "authenticate_screen_checkpoint",
        lambda *args, **kwargs: ({}, {}, {"checkpoint_sha256": "a" * 64}),
    )
    monkeypatch.setattr(
        reporter,
        "create_model_from_state",
        lambda state: (_FakeModel(), {"architecture": "synthetic"}),
    )
    monkeypatch.setattr(
        reporter,
        "preflight_locked_model_output",
        lambda *args, **kwargs: {"verification_status": "synthetic_passed"},
    )
    monkeypatch.setattr(reporter, "_autocast_context", lambda *args: nullcontext())
    compose_calls = []

    def compose(image, probabilities, candidate):
        compose_calls.append(candidate)
        return probabilities

    monkeypatch.setattr(reporter, "compose_locked_probabilities", compose)
    report = reporter.evaluate_screen_cell(
        cell,
        _verified_lock(),
        tiles,
        torch=_FakeTorch(),
        device="cuda",
    )
    assert report["validation_tile_count"] == 5
    assert len(report["per_tile"]) == 5
    assert compose_calls == ["R3"] * 5
    assert report["pooled_selection_metrics_reproduced"] is True

    with pytest.raises(ValueError, match="exactly five"):
        reporter.evaluate_screen_cell(
            cell,
            _verified_lock(),
            tiles[:4],
            torch=_FakeTorch(),
            device="cuda",
        )


def test_report_scope_is_explicitly_validation_only_and_nonselecting(monkeypatch):
    lock = _verified_lock()
    cells = _cell_reports()
    summaries, _ = reporter.attach_signed_r3_margins(cells)
    monkeypatch.setattr(reporter, "sha256_file", lambda path: "d" * 64)
    document = reporter.build_report(
        verified_lock=lock,
        lock_sha256="e" * 64,
        lock_identifier=reporter.CANONICAL_LOCK_IDENTIFIER,
        cell_reports=cells,
        candidate_summaries=summaries,
        runtime_attestation={"precision_protocol": "cuda_float16_autocast"},
        development_byte_attestation={
            "verification_status": "matched_recorded_train_plus_validation_bytes",
            "annotation_index_bytes_read": 0,
        },
    )
    scope = document["scientific_scope"]
    assert scope["model_development_only"] is True
    assert scope["winner_reselection_performed"] is False
    assert scope["recorded_partition_metadata_authenticated"] is True
    assert scope["live_locked_retrospective_dataset_constructed"] is False
    assert scope["live_locked_retrospective_filesystem_paths_resolved"] is False
    assert "locked_retrospective_partition_constructed" not in scope
    assert scope["locked_retrospective_input_bytes_read"] == 0
    assert scope["locked_retrospective_target_bytes_read"] == 0
    assert scope["annotation_index_bytes_read"] == 0
    assert scope["annotation_index_parsed"] is False
    assert "Recorded test-partition metadata" in scope["partition_metadata_statement"]
    assert "no live locked-retrospective filesystem path" in scope[
        "partition_metadata_statement"
    ]
    assert "no retrospective input, target, or annotation bytes were opened" in scope[
        "partition_metadata_statement"
    ]
    assert document["selected_method_lock"]["selection_performed_by_reporter"] is False
    contract = document["publication_summary_contract"]
    assert contract["label"] == "validation-only model development"
    assert contract["candidate_order"] == list(reporter.SCREEN_CANDIDATE_ORDER)
    assert contract["seed_order"] == [42, 123, 2025]
    assert contract["validation_tile_ordinals"] == [1, 2, 3, 4, 5]
    assert contract["candidate_seed_row_count"] == 15
    assert contract["tile_diagnostic_row_count"] == 75
    assert contract["aggregate_summary"] == (
        "arithmetic_mean_and_sample_standard_deviation"
    )
    assert contract["significance_claims"] is False
    assert contract["selection_fixed_by_existing_lock"] is True
    assert contract["winner_reselection_performed"] is False
    assert contract["outcome_dependent_tile_choice_performed"] is False
    assert contract["figure_geometry_inches"] == {
        "width": 6.48,
        "height": 7.2,
        "cas_layout": "active_cas_sc_text_width",
    }
    assert contract["active_cas_sc_text_width_tex_points"] == 468.3324
    assert contract["minimum_visible_label_points"] == 6.0
    assert contract["png_dpi"] == 600
    assert contract["font_family_preference"] == ["Arial", "Helvetica"]
    assert contract["pdf_font_type"] == 42
    assert contract["class_palette"] == {
        "C0": "#B33A3A",
        "C1": "#2E8B57",
        "C2": "#4C78A8",
    }
    assert set(contract["non_color_seed_encoding"]) == {"42", "123", "2025"}
    assert len(
        {
            (
                value["marker"],
                value["line_style"],
            )
            for value in contract["non_color_seed_encoding"].values()
        }
    ) == 3
    assert "were not opened, read, hashed, or parsed" in document["data"][
        "annotation_identity_verification"
    ]
    assert "validation" in document["evidence_kind"]


def test_run_order_rejects_before_any_authentication_or_corpus_access(monkeypatch):
    calls = []

    def bomb(name):
        def reject(*args, **kwargs):  # pragma: no cover - any call is failure
            calls.append(name)
            raise AssertionError(f"{name} called before the runtime guard")

        return reject

    monkeypatch.setattr(reporter, "load_verified_screen_lock", bomb("lock"))
    monkeypatch.setattr(
        reporter,
        "independently_reattest_development_bytes",
        bomb("development corpus"),
    )
    monkeypatch.setattr(reporter, "load_validation_tiles", bomb("validation corpus"))
    monkeypatch.setattr(
        reporter, "load_weights_only_checkpoint", bomb("checkpoint")
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for key in reporter.L40S_ALLOCATION_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        reporter.run_report(Path("config/selected_method_lock.json"))
    assert calls == []

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="L40S allocation intent"):
        reporter.run_report(Path("config/selected_method_lock.json"))
    assert calls == []

    monkeypatch.setenv("SLURM_JOB_GRES", "gpu:l40s:1")
    with pytest.raises(ValueError, match="device intent"):
        reporter.run_report(
            Path("config/selected_method_lock.json"), requested_device="cpu"
        )
    assert calls == []


def test_scheduler_and_l40s_runtime_are_mandatory(monkeypatch):
    class Torch:
        class cuda:
            @staticmethod
            def get_device_name(device):
                return "NVIDIA L40S"

        @staticmethod
        def device(value):
            return value

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for key in reporter.L40S_ALLOCATION_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        reporter._require_scheduler_l40s(Torch(), "auto")

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="L40S allocation intent"):
        reporter._require_scheduler_l40s(Torch(), "auto")
    monkeypatch.setenv("SLURM_JOB_GRES", "gpu:l40s:1")
    monkeypatch.setattr(reporter, "choose_device", lambda requested: "cuda:0")
    monkeypatch.setattr(
        reporter,
        "validate_locked_inference_runtime",
        lambda device, amp, cuda_device_name: {
            "precision_protocol": reporter.LOCKED_INFERENCE_PRECISION,
            "locked_device_model_token": reporter.LOCKED_CUDA_DEVICE_MODEL_TOKEN,
            "cuda_device_name": cuda_device_name,
        },
    )
    device, attestation = reporter._require_scheduler_l40s(Torch(), "auto")
    assert device == "cuda:0"
    assert "L40S" in attestation["cuda_device_name"]

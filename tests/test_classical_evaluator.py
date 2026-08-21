"""Synthetic-only safeguards for the exactly-once classical evaluator."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts import evaluate_locked_classical_comparators as evaluator
from scripts import fit_classical_comparators as fitter
from scripts.fit_classical_comparators import (
    CANONICAL_TRAIN_FILENAMES,
    CLASSICAL_LOCK_SCHEMA_VERSION,
    PROTOCOL_ID,
    SOURCE_PATHS,
    _source_hashes,
    build_parser as build_fit_parser,
    frozen_one_pass_evaluator_contract,
)
from src.classical.comparators import (
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
)
from src.training.data_contract import (
    CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS,
    CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "synthetic-classical-lock"
NEURAL_FREEZE_ID = "neural-freeze-" + "7" * 16
SYNTHETIC_MODEL_SEMANTIC_SHA256 = "6" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _synthetic_runtime_versions() -> dict[str, str]:
    return {key: "synthetic" for key in evaluator.RUNTIME_VERSION_KEYS}


def _synthetic_neural_freeze() -> dict[str, Any]:
    return {
        "manifest_id": NEURAL_FREEZE_ID,
        "manifest_repo_relative_identifier": (
            f"results/neural_freeze/locked/{NEURAL_FREEZE_ID}/"
            "neural_freeze_manifest.json"
        ),
        "manifest_file_sha256": "7" * 64,
        "scientific_identity_sha256": "8" * 64,
        "selected_method": "R3",
        "selected_method_lock": {
            "raw_file_sha256": "9" * 64,
        },
        "selected_retraining_checkpoint_sha256": {
            "primary_multiscale": ["a" * 64, "b" * 64, "c" * 64],
            "plain_unet_comparator": ["d" * 64, "e" * 64, "f" * 64],
        },
        "document": {"status": "synthetic-neural-freeze"},
    }


def _reidentify(lock: dict[str, Any]) -> str:
    identity = fitter.classical_scientific_identity_sha256(lock)
    lock["scientific_identity_sha256"] = identity
    lock["fit_id"] = f"classical-fit-{identity[:16]}"
    return lock["fit_id"]


def _pair_identity(classical_identity: str) -> str:
    return _sha256(
        f"{classical_identity}\0{'8' * 64}".encode("ascii")
    )


def _reservation_preflight(
    tmp_path: Path, *, fit_id: str, classical_identity: str
) -> dict[str, Any]:
    return {
        "fit_id": fit_id,
        "fit_execution_campaign_id": CAMPAIGN,
        "canonical_lock_identity_sha256": classical_identity,
        "evaluation_pair_identity_sha256": _pair_identity(classical_identity),
        "lock_file_sha256": "e" * 64,
        "b2_model_sha256": "5" * 64,
        "b2_model_semantic_sha256": SYNTHETIC_MODEL_SEMANTIC_SHA256,
        "neural_freeze": _synthetic_neural_freeze(),
        "project_root": tmp_path.resolve(),
        "paths": evaluator.canonical_fit_paths(fit_id, project_root=tmp_path),
    }


def _selection(cutoffs: tuple[int, ...], selected: int) -> dict[str, Any]:
    summaries = {}
    group_summaries = {}
    groups = sorted(evaluator.EXPECTED_TRAIN_GROUP_COUNTS)
    for cutoff in cutoffs:
        c0, c1 = (0.4, 0.6) if cutoff == selected else (0.25, 0.75)
        group_summaries[str(cutoff)] = {
            group: {
                "iou_c0": c0,
                "iou_c1": c1,
                "balanced_pore_iou": balanced_pore_score(c0, c1),
            }
            for group in groups
        }
        summaries[str(cutoff)] = {
            key: float(
                np.mean(
                    [group_summaries[str(cutoff)][group][key] for group in groups]
                )
            )
            for key in ("iou_c0", "iou_c1", "balanced_pore_iou")
        }
    return {
        "selection_scope": "canonical_training_groups_only",
        "primary": "mean_group_balanced_pore_iou",
        "tie_break_1": "higher_minimum_of_mean_group_iou_c0_and_iou_c1",
        "tie_break_2": "lower_area_cutoff_px",
        "selected_area_cutoff_px": selected,
        "candidate_group_summaries": group_summaries,
        "candidate_summaries": summaries,
    }


def _b2_group_cv() -> dict[str, Any]:
    groups = sorted(evaluator.EXPECTED_TRAIN_GROUP_COUNTS)
    confusion = np.diag([10.0, 20.0, 30.0])
    metrics = evaluator.pore_metrics_from_confusion(confusion)
    return {
        "scope": "leave_one_canonical_training_series_out",
        "evaluation": "deterministic_stratified_weighted_sample",
        "folds": {
            held_out: {
                "held_out_scope": "training_series_only",
                "fit_groups": [group for group in groups if group != held_out],
                "sampled_evaluation_pixels": 60,
                "weighted_confusion": confusion.tolist(),
                **metrics,
            }
            for held_out in groups
        },
        "mean_group_metrics": dict(metrics),
    }


def _valid_lock(
    project_root: Path,
    *,
    campaign_id: str = CAMPAIGN,
    source_paths: tuple[str, ...] = ("source.py",),
    model_payload: bytes = b"synthetic locked model",
) -> tuple[dict[str, Any], Path]:
    static = fitter.frozen_classical_lock_static_contract()
    for source_name in source_paths:
        source_path = project_root / source_name
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f"source:{source_name}\n", encoding="utf-8")

    staging_dir = project_root / "synthetic-staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    model_path = staging_dir / evaluator.CANONICAL_B2_MODEL_FILENAME
    model_path.write_bytes(model_payload)

    names = list(CANONICAL_TRAIN_FILENAMES)
    name_list_sha = _sha256(
        json.dumps(names, separators=(",", ":")).encode("utf-8")
    )
    source_hashes = {
        name: evaluator.sha256_file(project_root / name) for name in source_paths
    }
    input_attestation = CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS["train"]
    target_attestation = CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS["train"]
    lock: dict[str, Any] = {
        "schema_version": CLASSICAL_LOCK_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "status": static["status"],
        "created_utc": "2026-08-21T00:00:00+00:00",
        "scientific_identity": static["scientific_identity"],
        "selection_relationship": static["selection_relationship"],
        "data_access": static["data_access"],
        "data_provenance": {
            "split_manifest_sha256": evaluator.LOCKED_SPLIT_MANIFEST_SHA256,
            "training_input": {
                "scope": "canonical_training_images_only",
                "split_names": ["train"],
                "image_count": input_attestation["image_count"],
                "image_aggregate_sha256": input_attestation[
                    "image_aggregate_sha256"
                ],
                "image_aggregate_sha256_algorithm": (
                    "sha256 over lexicographically sorted UTF-8 relative filename, "
                    "NUL, raw file bytes, NUL"
                ),
                "file_name_list_sha256": name_list_sha,
            },
            "training_target": {
                "target_source": "lossless_png_masks",
                "mask_directory": evaluator.CANONICAL_MASK_DIR.as_posix(),
                "mask_count": target_attestation["mask_count"],
                "mask_aggregate_sha256": target_attestation[
                    "mask_aggregate_sha256"
                ],
                "mask_aggregate_sha256_algorithm": (
                    "sha256 over lexicographically sorted UTF-8 relative filename, "
                    "NUL, raw file bytes, NUL"
                ),
                "validated_source_values": [0, 1, 255],
                "canonical_value_mapping": {
                    "0": "0 (disconnected_pore)",
                    "1": "1 (connected_pore)",
                    "255": (
                        "2 (mineral) in three-class mode; ignore_index in "
                        "two-class mode"
                    ),
                },
            },
            "canonical_training_filename_count": 74,
            "canonical_training_filenames": names,
            "training_file_name_list_sha256": name_list_sha,
            "training_sampling_ids_by_filename": {
                name: index for index, name in enumerate(names, start=1)
            },
            "training_group_tile_counts": dict(
                evaluator.EXPECTED_TRAIN_GROUP_COUNTS
            ),
            "training_sample_key": static["training_sample_key"],
            "source_mask_to_canonical": {"0": 0, "1": 1, "255": 2},
        },
        "inference_contract": static["inference_contract"],
        "comparators": {
            "B0_small_components": {
                **static["b0_definition"],
                "candidate_area_cutoffs_px": list(B0_AREA_CUTOFFS),
                "selection": _selection(B0_AREA_CUTOFFS, 100),
                "frozen_area_cutoff_px": 100,
            },
            "B1_marker_watershed": {
                **static["b1_definition"],
                "candidate_area_cutoffs_px": list(B1_AREA_CUTOFFS),
                "selection": _selection(B1_AREA_CUTOFFS, 200),
                "frozen_area_cutoff_px": 200,
            },
            "B2_extra_trees": {
                "estimator_config": dict(EXTRA_TREES_CONFIG),
                "feature_names": list(EXTRA_TREES_FEATURE_NAMES),
                "sampling_seed": 20260821,
                "samples_per_class_per_tile": 4096,
                "group_cv": _b2_group_cv(),
                "final_fit": {
                    "training_sample_count": 2,
                    "training_sample_class_counts": {"0": 1, "1": 1},
                },
                "model_file": evaluator.CANONICAL_B2_MODEL_FILENAME,
                "model_sha256": _sha256(model_payload),
                "model_semantic_sha256": SYNTHETIC_MODEL_SEMANTIC_SHA256,
            },
        },
        "one_pass_evaluator_contract": frozen_one_pass_evaluator_contract(),
        "reported_selection_metrics": static["reported_selection_metrics"],
        "source_code_sha256": source_hashes,
        "environment_versions": _synthetic_runtime_versions(),
    }
    identity = fitter.classical_scientific_identity_sha256(lock)
    lock["fit_id"] = f"classical-fit-{identity[:16]}"
    lock["scientific_identity_sha256"] = identity
    lock_dir = (
        project_root
        / "results/classical_comparators/locked"
        / lock["fit_id"]
    )
    lock_dir.mkdir(parents=True, exist_ok=True)
    model_path.replace(lock_dir / evaluator.CANONICAL_B2_MODEL_FILENAME)
    staging_dir.rmdir()
    return lock, lock_dir


def _write_canonical_lock(project_root: Path) -> tuple[dict[str, Any], Path]:
    lock, lock_dir = _valid_lock(project_root)
    lock_path = lock_dir / evaluator.CANONICAL_LOCK_FILENAME
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock, lock_dir


class ExtraTreesClassifier:
    """Minimal loader result for hash/schema preflight; it is never fitted."""

    n_features_in_ = len(EXTRA_TREES_FEATURE_NAMES)
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        del deep
        return dict(EXTRA_TREES_CONFIG)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=np.uint8)


def _synthetic_numeric_predictor() -> FrozenExtraTreesPredictor:
    tree_count = int(EXTRA_TREES_CONFIG["n_estimators"])
    return FrozenExtraTreesPredictor(
        parameters=dict(EXTRA_TREES_CONFIG),
        classes=np.asarray([0, 1], dtype=np.int64),
        n_classes=np.asarray([2], dtype=np.int64),
        n_outputs=np.asarray([1], dtype=np.int64),
        n_features_in=np.asarray([len(EXTRA_TREES_FEATURE_NAMES)], dtype=np.int64),
        tree_offsets=np.arange(tree_count + 1, dtype=np.int64),
        tree_has_missing=np.zeros(tree_count, dtype=np.uint8),
        children_left=np.full(tree_count, -1, dtype=np.int64),
        children_right=np.full(tree_count, -1, dtype=np.int64),
        feature=np.full(tree_count, -2, dtype=np.int64),
        threshold=np.full(tree_count, -2.0, dtype=np.float64),
        impurity=np.zeros(tree_count, dtype=np.float64),
        n_node_samples=np.ones(tree_count, dtype=np.int64),
        weighted_n_node_samples=np.ones(tree_count, dtype=np.float64),
        value=np.tile(np.asarray([[[1.0, 1.0]]]), (tree_count, 1, 1)),
        missing_go_to_left=np.zeros(tree_count, dtype=np.uint8),
    )


def test_fitter_lock_schema_round_trips_through_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = list(CANONICAL_TRAIN_FILENAMES)
    name_list_sha = _sha256(
        json.dumps(names, separators=(",", ":")).encode("utf-8")
    )
    input_attestation = CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS["train"]
    target_attestation = CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS["train"]
    contract = {
        "train_ids": tuple(range(1, 75)),
        "train_names": tuple(names),
        "images_by_id": {
            index: {"id": index, "file_name": name}
            for index, name in enumerate(names, start=1)
        },
        "manifest_sha256": evaluator.LOCKED_SPLIT_MANIFEST_SHA256,
        "input_provenance": {
            "scope": "canonical_training_images_only",
            "split_names": ["train"],
            "image_count": 74,
            "image_aggregate_sha256": input_attestation[
                "image_aggregate_sha256"
            ],
            "image_aggregate_sha256_algorithm": (
                "sha256 over lexicographically sorted UTF-8 relative filename, "
                "NUL, raw file bytes, NUL"
            ),
            "file_name_list_sha256": name_list_sha,
        },
        "target_provenance": {
            "target_source": "lossless_png_masks",
            "mask_directory": evaluator.CANONICAL_MASK_DIR.as_posix(),
            "mask_count": 74,
            "mask_aggregate_sha256": target_attestation[
                "mask_aggregate_sha256"
            ],
            "mask_aggregate_sha256_algorithm": (
                "sha256 over lexicographically sorted UTF-8 relative filename, "
                "NUL, raw file bytes, NUL"
            ),
            "validated_source_values": [0, 1, 255],
            "canonical_value_mapping": {
                "0": "0 (disconnected_pore)",
                "1": "1 (connected_pore)",
                "255": (
                    "2 (mineral) in three-class mode; ignore_index in "
                    "two-class mode"
                ),
            },
        },
    }
    evidence = {
        "b0_confusions": {cutoff: {} for cutoff in B0_AREA_CUTOFFS},
        "b1_confusions": {cutoff: {} for cutoff in B1_AREA_CUTOFFS},
        "group_tile_counts": dict(evaluator.EXPECTED_TRAIN_GROUP_COUNTS),
    }
    selections = iter(
        (
            (100, _selection(B0_AREA_CUTOFFS, 100)),
            (200, _selection(B1_AREA_CUTOFFS, 200)),
        )
    )
    monkeypatch.setattr(fitter, "select_area_cutoff", lambda _: next(selections))
    monkeypatch.setattr(
        fitter,
        "fit_b2_group_cv",
        lambda _: _b2_group_cv(),
    )
    monkeypatch.setattr(
        fitter,
        "fit_final_b2",
        lambda _: (
            ExtraTreesClassifier(),
            {
                "training_sample_count": 2,
                "training_sample_class_counts": {"0": 1, "1": 1},
            },
        ),
    )
    monkeypatch.setattr(
        fitter,
        "_library_versions",
        _synthetic_runtime_versions,
    )

    def write_synthetic_numeric_model(_model: object, path: Path) -> str:
        path.write_bytes(b"synthetic locked numeric model")
        return SYNTHETIC_MODEL_SEMANTIC_SHA256

    monkeypatch.setattr(
        fitter, "save_extra_trees_numeric", write_synthetic_numeric_model
    )
    output_dir = tmp_path / "lock"
    output_dir.mkdir()
    lock = fitter.build_lock(
        campaign_id=CAMPAIGN,
        contract=contract,
        evidence=evidence,
        output_dir=output_dir,
    )
    verified = evaluator.verify_classical_lock_document(
        lock,
        fit_id=lock["fit_id"],
        lock_dir=output_dir,
        project_root=REPOSITORY_ROOT,
        source_paths=SOURCE_PATHS,
    )
    assert verified["b0_area_cutoff_px"] == 100
    assert verified["b1_area_cutoff_px"] == 200
    loaded = evaluator.load_verified_b2_estimator(
        verified["b2_model_path"],
        verified["b2_model_sha256"],
        verified["b2_model_semantic_sha256"],
        loader=lambda _: _synthetic_numeric_predictor(),
        semantic_digest_loader=lambda _: SYNTHETIC_MODEL_SEMANTIC_SHA256,
    )
    assert type(loaded) is FrozenExtraTreesPredictor


def test_valid_lock_is_content_addressed_independent_of_json_formatting(
    tmp_path: Path,
) -> None:
    lock, lock_dir = _valid_lock(tmp_path)
    compact = json.loads(json.dumps(lock, separators=(",", ":")))
    pretty = json.loads(json.dumps(lock, indent=4, sort_keys=True))
    first = evaluator.verify_classical_lock_document(
        compact,
        fit_id=lock["fit_id"],
        lock_dir=lock_dir,
        project_root=tmp_path,
        source_paths=("source.py",),
    )
    second = evaluator.verify_classical_lock_document(
        pretty,
        fit_id=lock["fit_id"],
        lock_dir=lock_dir,
        project_root=tmp_path,
        source_paths=("source.py",),
    )
    assert first["canonical_lock_identity_sha256"] == second[
        "canonical_lock_identity_sha256"
    ]


def test_area_selection_verifies_mean_of_per_series_harmonics(tmp_path: Path) -> None:
    lock, lock_dir = _valid_lock(tmp_path)
    selection = lock["comparators"]["B0_small_components"]["selection"]
    groups = sorted(evaluator.EXPECTED_TRAIN_GROUP_COUNTS)
    pairs = ((0.4, 0.6), (0.1, 0.9), (0.4, 0.6), (0.1, 0.9))
    cutoff = "25"
    selection["candidate_group_summaries"][cutoff] = {
        group: {
            "iou_c0": c0,
            "iou_c1": c1,
            "balanced_pore_iou": balanced_pore_score(c0, c1),
        }
        for group, (c0, c1) in zip(groups, pairs)
    }
    selection["candidate_summaries"][cutoff] = {
        key: float(
            np.mean(
                [
                    selection["candidate_group_summaries"][cutoff][group][key]
                    for group in groups
                ]
            )
        )
        for key in ("iou_c0", "iou_c1", "balanced_pore_iou")
    }
    aggregate = selection["candidate_summaries"][cutoff]
    assert aggregate["balanced_pore_iou"] != pytest.approx(
        balanced_pore_score(aggregate["iou_c0"], aggregate["iou_c1"])
    )

    fit_id = _reidentify(lock)
    verified = evaluator.verify_classical_lock_document(
        lock,
        fit_id=fit_id,
        lock_dir=lock_dir,
        project_root=tmp_path,
        source_paths=("source.py",),
    )
    assert verified["b0_area_cutoff_px"] == 100


def test_v1_classical_lock_requires_train_only_refit(tmp_path: Path) -> None:
    lock, lock_dir = _valid_lock(tmp_path)
    lock["schema_version"] = 1
    fit_id = _reidentify(lock)
    with pytest.raises(ValueError, match="train-only refit is required"):
        evaluator.verify_classical_lock_document(
            lock,
            fit_id=fit_id,
            lock_dir=lock_dir,
            project_root=tmp_path,
            source_paths=("source.py",),
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda lock: lock["inference_contract"].__setitem__(
                "pore_gate", "raw_uint8 < 101"
            ),
            "inference contract",
        ),
        (
            lambda lock: lock["comparators"]["B0_small_components"].__setitem__(
                "frozen_area_cutoff_px", 500
            ),
            "selection record",
        ),
        (
            lambda lock: lock["one_pass_evaluator_contract"].__setitem__(
                "test_tile_count", 20
            ),
            "one-pass evaluator contract",
        ),
        (
            lambda lock: lock["data_access"].__setitem__(
                "validation_target_read_count", 1
            ),
            "train-only data access",
        ),
        (
            lambda lock: lock.__setitem__("scientific_identity", "alias prose"),
            "scientific-identity description",
        ),
        (
            lambda lock: lock["data_provenance"].__setitem__(
                "training_sample_key", "alias key"
            ),
            "sample-key description",
        ),
        (
            lambda lock: lock["comparators"]["B0_small_components"].__setitem__(
                "region_partition", "alias partition"
            ),
            "B0 comparator definition",
        ),
        (
            lambda lock: lock.__setitem__(
                "reported_selection_metrics", "alias metrics"
            ),
            "reported-selection-metric description",
        ),
        (
            lambda lock: lock["data_provenance"][
                "canonical_training_filenames"
            ].__setitem__(0, "aaa.png"),
            "training filename set drifted",
        ),
        (
            lambda lock: lock["environment_versions"].pop("opencv"),
            "runtime-version attestation",
        ),
        (
            lambda lock: lock["comparators"]["B2_extra_trees"].__setitem__(
                "sampling_seed", 7
            ),
            "sampling contract",
        ),
        (
            lambda lock: lock["comparators"]["B2_extra_trees"].__setitem__(
                "sampling_seed", "20260821"
            ),
            "sampling contract",
        ),
        (
            lambda lock: lock["comparators"]["B2_extra_trees"]["group_cv"][
                "folds"
            ]["pdo1_12"].__setitem__("sampled_evaluation_pixels", "60"),
            "group isolation",
        ),
        (
            lambda lock: lock["comparators"]["B2_extra_trees"]["group_cv"][
                "folds"
            ]["pdo1_12"]["fit_groups"].append("pdo1_12"),
            "group isolation",
        ),
        (
            lambda lock: lock["comparators"]["B2_extra_trees"]["group_cv"][
                "folds"
            ]["pdo1_12"].__setitem__("iou_c0", 0.1),
            "metric is inconsistent",
        ),
        (
            lambda lock: lock["comparators"]["B0_small_components"]["selection"][
                "candidate_group_summaries"
            ]["25"]["pdo1_12"].__setitem__("balanced_pore_iou", 0.2),
            "per-series cutoff harmonic score is inconsistent",
        ),
        (
            lambda lock: lock["comparators"]["B0_small_components"]["selection"][
                "candidate_summaries"
            ]["25"].__setitem__("balanced_pore_iou", 0.2),
            "cutoff mean metrics are inconsistent",
        ),
    ],
)
def test_lock_protocol_tampering_fails_closed(
    tmp_path: Path, mutator: Any, message: str
) -> None:
    lock, lock_dir = _valid_lock(tmp_path)
    mutator(lock)
    fit_id = _reidentify(lock)
    with pytest.raises(ValueError, match=message):
        evaluator.verify_classical_lock_document(
            lock,
            fit_id=fit_id,
            lock_dir=lock_dir,
            project_root=tmp_path,
            source_paths=("source.py",),
        )


def test_source_and_model_byte_tampering_fail_closed(tmp_path: Path) -> None:
    lock, lock_dir = _valid_lock(tmp_path)
    (tmp_path / "source.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        evaluator.verify_classical_lock_document(
            lock,
            fit_id=lock["fit_id"],
            lock_dir=lock_dir,
            project_root=tmp_path,
            source_paths=("source.py",),
        )

    model = lock_dir / evaluator.CANONICAL_B2_MODEL_FILENAME
    model.write_bytes(b"tampered model")
    with pytest.raises(ValueError, match="numeric-model SHA-256 mismatch"):
        evaluator.verify_model_artifact_hash(
            model,
            lock["comparators"]["B2_extra_trees"]["model_sha256"],
        )


def test_alternate_model_paths_and_symlinks_are_rejected(tmp_path: Path) -> None:
    lock, lock_dir = _valid_lock(tmp_path)
    traversal = copy.deepcopy(lock)
    traversal["comparators"]["B2_extra_trees"]["model_file"] = "../model.npz"
    traversal_fit_id = _reidentify(traversal)
    with pytest.raises(ValueError, match="model filename is not canonical"):
        evaluator.verify_classical_lock_document(
            traversal,
            fit_id=traversal_fit_id,
            lock_dir=lock_dir,
            project_root=tmp_path,
            source_paths=("source.py",),
        )

    model = lock_dir / evaluator.CANONICAL_B2_MODEL_FILENAME
    model.unlink()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"synthetic locked model")
    model.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        evaluator.verify_classical_lock_document(
            lock,
            fit_id=lock["fit_id"],
            lock_dir=lock_dir,
            project_root=tmp_path,
            source_paths=("source.py",),
        )


def test_cli_exposes_only_content_addressed_fit_not_output_or_model_override() -> None:
    fit_id = "classical-fit-" + "1" * 16
    for option, value in (
        ("--output-dir", "alternate"),
        ("--model", "alternate.npz"),
        ("--checkpoint", "alternate.pth"),
        ("--image-dir", "alternate-images"),
    ):
        with pytest.raises(SystemExit):
            evaluator.build_parser().parse_args(
                [
                    "--fit-id",
                    fit_id,
                    "--neural-freeze-id",
                    NEURAL_FREEZE_ID,
                    option,
                    value,
                ]
            )
    with pytest.raises(SystemExit):
        evaluator.build_parser().parse_args(["--fit-id", fit_id])
    with pytest.raises(SystemExit):
        build_fit_parser().parse_args(
            ["--campaign-id", CAMPAIGN, "--output-dir", "alternate"]
        )


def test_production_fitter_rejects_alternate_and_symlinked_input_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fitter, "PROJECT_ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/confirmatory_splits.json").touch()
    (tmp_path / "original_images").mkdir()
    canonical = fitter._require_canonical_input_path(
        Path("original_images"),
        fitter.CANONICAL_IMAGE_DIR,
        label="Classical image directory",
    )
    assert canonical == tmp_path / "original_images"
    with pytest.raises(ValueError, match="must be exactly"):
        fitter._require_canonical_input_path(
            Path("copied_images"),
            fitter.CANONICAL_IMAGE_DIR,
            label="Classical image directory",
        )

    (tmp_path / "original_images").rmdir()
    outside = tmp_path / "outside-images"
    outside.mkdir()
    (tmp_path / "original_images").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic-link component"):
        fitter._require_canonical_input_path(
            Path("original_images"),
            fitter.CANONICAL_IMAGE_DIR,
            label="Classical image directory",
        )


@pytest.mark.parametrize(
    "fit_id", ["../escape", "/absolute", "nested/path", "space id", "", CAMPAIGN]
)
def test_fit_identifier_cannot_traverse_or_select_a_path(
    tmp_path: Path, fit_id: str
) -> None:
    with pytest.raises(ValueError, match="not canonical"):
        evaluator.canonical_fit_paths(fit_id, project_root=tmp_path)


def test_global_fit_freeze_pair_blocks_campaign_alias_and_reserialization(
    tmp_path: Path,
) -> None:
    lock, lock_dir = _valid_lock(tmp_path)
    verified = evaluator.verify_classical_lock_document(
        lock,
        fit_id=lock["fit_id"],
        lock_dir=lock_dir,
        project_root=tmp_path,
        source_paths=("source.py",),
    )
    preflight = {
        **verified,
        **_reservation_preflight(
            tmp_path,
            fit_id=lock["fit_id"],
            classical_identity=verified["canonical_lock_identity_sha256"],
        ),
    }
    preflight["lock_file_sha256"] = "a" * 64
    output = evaluator.reserve_exactly_once_output(preflight)
    assert output.name == NEURAL_FREEZE_ID
    assert output.parent.name == lock["fit_id"]

    # Whitespace/key-order serialization changes the raw file hash, not the
    # canonical content identity, and cannot reserve a second pass.
    reserialized = dict(preflight)
    reserialized["lock_file_sha256"] = "b" * 64
    with pytest.raises(FileExistsError, match="already reserved"):
        evaluator.reserve_exactly_once_output(reserialized)

    # Changing a caller campaign or timestamp does not change the fit identity,
    # so there is no alternate evaluator path to reserve.
    alias_lock = copy.deepcopy(lock)
    alias_lock["campaign_id"] = "another-caller-campaign"
    alias_lock["created_utc"] = "2099-01-01T00:00:00+00:00"
    alias_lock["comparators"]["B2_extra_trees"]["model_sha256"] = "0" * 64
    alias_lock["data_provenance"]["training_target"][
        "mask_directory"
    ] = "<external>/copied-masks"
    assert fitter.classical_scientific_identity_sha256(alias_lock) == verified[
        "canonical_lock_identity_sha256"
    ]
    alias = dict(preflight)
    alias["fit_execution_campaign_id"] = "another-caller-campaign"
    with pytest.raises(FileExistsError, match="already reserved"):
        evaluator.reserve_exactly_once_output(alias)

    # A mismatched identity cannot claim the same canonical fit ID.
    changed = dict(preflight)
    changed["canonical_lock_identity_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="evaluation-pair identity mismatch"):
        evaluator.reserve_exactly_once_output(changed)


def test_storage_alias_is_identity_neutral_but_not_an_evaluable_lock(
    tmp_path: Path,
) -> None:
    lock, lock_dir = _valid_lock(tmp_path)
    identity = lock["scientific_identity_sha256"]
    lock["data_provenance"]["training_target"][
        "mask_directory"
    ] = "<external>/copied-masks"
    assert fitter.classical_scientific_identity_sha256(lock) == identity
    with pytest.raises(ValueError, match="training-mask attestation"):
        evaluator.verify_classical_lock_document(
            lock,
            fit_id=lock["fit_id"],
            lock_dir=lock_dir,
            project_root=tmp_path,
            source_paths=("source.py",),
        )


def test_reservation_rejects_alternate_output_root(tmp_path: Path) -> None:
    identity = "2" * 64
    fit_id = "classical-fit-" + identity[:16]
    preflight = _reservation_preflight(
        tmp_path, fit_id=fit_id, classical_identity=identity
    )
    preflight["paths"] = dict(preflight["paths"])
    preflight["paths"]["output_fit_root"] = tmp_path / "alternate-output"
    with pytest.raises(ValueError, match="output path is not canonical"):
        evaluator.reserve_exactly_once_output(preflight)
    assert not (tmp_path / "alternate-output").exists()


def test_canonical_fit_path_rejects_alias_copy_and_symlink(tmp_path: Path) -> None:
    lock, lock_dir = _write_canonical_lock(tmp_path)
    alternate_id = "classical-fit-" + "0" * 16
    alternate = tmp_path / evaluator.CANONICAL_LOCK_ROOT / alternate_id
    alternate.mkdir(parents=True)
    (alternate / evaluator.CANONICAL_LOCK_FILENAME).write_text(
        json.dumps(lock), encoding="utf-8"
    )
    (alternate / evaluator.CANONICAL_B2_MODEL_FILENAME).write_bytes(
        (lock_dir / evaluator.CANONICAL_B2_MODEL_FILENAME).read_bytes()
    )
    with pytest.raises(ValueError, match="content-addressed fit identity mismatch"):
        evaluator.load_verified_classical_lock(
            alternate_id,
            project_root=tmp_path,
            source_paths=("source.py",),
        )

    outside = tmp_path / "outside-fit"
    lock_dir.replace(outside)
    lock_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic-link component"):
        evaluator.load_verified_classical_lock(
            lock["fit_id"],
            project_root=tmp_path,
            source_paths=("source.py",),
        )


def test_no_held_out_byte_read_before_reservation_and_no_second_pass(
    tmp_path: Path,
) -> None:
    root = tmp_path / "heldout"
    root.mkdir()
    name = "pdo2_24_synthetic.png"
    payload = b"attested synthetic bytes"
    (root / name).write_bytes(payload)
    identity = "d" * 64
    fit_id = "classical-fit-" + identity[:16]
    pair_identity = _pair_identity(identity)
    calls: list[Path] = []

    def reader(path: Path) -> bytes:
        calls.append(path)
        return path.read_bytes()

    with pytest.raises(RuntimeError, match="before canonical reservation"):
        evaluator.read_attested_held_out_files_once(
            root,
            [name],
            output_dir=tmp_path / "not-reserved",
            evaluation_pair_identity_sha256=pair_identity,
            expected_count=1,
            expected_sha256=_sha256(name.encode() + b"\0" + payload + b"\0"),
            corpus_label="input-image",
            reader=reader,
        )
    assert calls == []

    preflight = _reservation_preflight(
        tmp_path, fit_id=fit_id, classical_identity=identity
    )
    output = evaluator.reserve_exactly_once_output(preflight)
    expected = _sha256(name.encode() + b"\0" + payload + b"\0")
    records, _, attestation = evaluator.read_attested_held_out_files_once(
        root,
        [name],
        output_dir=output,
        evaluation_pair_identity_sha256=pair_identity,
        expected_count=1,
        expected_sha256=expected,
        corpus_label="input-image",
        reader=reader,
    )
    assert records == {name: payload}
    assert attestation["read_passes"] == 1
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="already claimed"):
        evaluator.read_attested_held_out_files_once(
            root,
            [name],
            output_dir=output,
            evaluation_pair_identity_sha256=pair_identity,
            expected_count=1,
            expected_sha256=expected,
            corpus_label="input-image",
            reader=reader,
        )
    assert len(calls) == 1


def test_discovery_is_reserved_and_exactly_21_pdo2_pairs(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_root.mkdir()
    mask_root.mkdir()
    names = [f"pdo2_24_segment_0_{index}.png" for index in range(21)]
    for name in names:
        (image_root / name).touch()
        (mask_root / name).touch()
    # A validation-series filename must not enter the locked held-out scope.
    (image_root / "pdo8_21_segment_0_0.png").touch()
    (mask_root / "pdo8_21_segment_0_0.png").touch()
    identity = "f" * 64
    fit_id = "classical-fit-" + identity[:16]
    pair_identity = _pair_identity(identity)
    with pytest.raises(RuntimeError, match="before canonical reservation"):
        evaluator.discover_canonical_test_filenames(
            image_root,
            mask_root,
            output_dir=tmp_path / "unreserved",
            evaluation_pair_identity_sha256=pair_identity,
        )
    output = evaluator.reserve_exactly_once_output(
        _reservation_preflight(
            tmp_path, fit_id=fit_id, classical_identity=identity
        )
    )
    assert evaluator.discover_canonical_test_filenames(
        image_root,
        mask_root,
        output_dir=output,
        evaluation_pair_identity_sha256=pair_identity,
    ) == sorted(names)


def test_preflight_succeeds_without_any_validation_or_test_files(
    tmp_path: Path,
) -> None:
    lock, _ = _write_canonical_lock(tmp_path)
    (tmp_path / "original_images").mkdir()
    (tmp_path / "results/step2_pore_classification/pore_classifications").mkdir(
        parents=True
    )
    manifest = tmp_path / "config/confirmatory_splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(
        (REPOSITORY_ROOT / "config/confirmatory_splits.json").read_bytes()
    )

    result = evaluator.preflight_campaign(
        lock["fit_id"],
        NEURAL_FREEZE_ID,
        project_root=tmp_path,
        source_paths=("source.py",),
        model_loader=lambda _: _synthetic_numeric_predictor(),
        model_semantic_loader=lambda _: SYNTHETIC_MODEL_SEMANTIC_SHA256,
        runtime_versions_loader=_synthetic_runtime_versions,
        neural_freeze_loader=lambda *_: _synthetic_neural_freeze(),
    )
    assert result["validation_image_bytes_read"] == 0
    assert result["validation_target_bytes_read"] == 0
    assert result["held_out_image_bytes_read"] == 0
    assert result["held_out_target_bytes_read"] == 0
    assert not result["paths"]["output_fit_root"].exists()
    drifted = _synthetic_runtime_versions()
    drifted["numpy"] = "different"
    with pytest.raises(ValueError, match="runtime versions differ"):
        evaluator.preflight_campaign(
            lock["fit_id"],
            NEURAL_FREEZE_ID,
            project_root=tmp_path,
            source_paths=("source.py",),
            model_loader=lambda _: _synthetic_numeric_predictor(),
            model_semantic_loader=lambda _: SYNTHETIC_MODEL_SEMANTIC_SHA256,
            runtime_versions_loader=lambda: drifted,
            neural_freeze_loader=lambda *_: _synthetic_neural_freeze(),
        )


def test_campaign_reservation_precedes_held_out_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class StopAfterBoundaryCheck(RuntimeError):
        pass

    monkeypatch.setattr(evaluator, "require_active_nodes_allocation", lambda: None)
    monkeypatch.setattr(
        evaluator,
        "preflight_campaign",
        lambda *_: {
            "fit_id": "classical-fit-" + "1" * 16,
            "fit_execution_campaign_id": CAMPAIGN,
            "canonical_lock_identity_sha256": "1" * 64,
            "evaluation_pair_identity_sha256": _pair_identity("1" * 64),
            "neural_freeze": _synthetic_neural_freeze(),
            "paths": {"image_dir": tmp_path, "mask_dir": tmp_path},
        },
    )

    def reserve(_: dict[str, Any]) -> Path:
        events.append("reserved")
        return tmp_path / "reserved-output"

    def discover(*_: Any, **__: Any) -> list[str]:
        events.append("held_out_discovery")
        raise StopAfterBoundaryCheck

    monkeypatch.setattr(evaluator, "reserve_exactly_once_output", reserve)
    monkeypatch.setattr(evaluator, "discover_canonical_test_filenames", discover)
    with pytest.raises(StopAfterBoundaryCheck):
        evaluator.evaluate_campaign("classical-fit-" + "1" * 16, NEURAL_FREEZE_ID)
    assert events == ["reserved", "held_out_discovery"]


def test_symlinked_corpus_escape_is_rejected_in_preflight(tmp_path: Path) -> None:
    lock, _ = _write_canonical_lock(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "original_images").symlink_to(outside, target_is_directory=True)
    (tmp_path / "results/step2_pore_classification/pore_classifications").mkdir(
        parents=True
    )
    manifest = tmp_path / "config/confirmatory_splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(
        (REPOSITORY_ROOT / "config/confirmatory_splits.json").read_bytes()
    )
    with pytest.raises(ValueError, match="symbolic-link component"):
        evaluator.preflight_campaign(
            lock["fit_id"],
            NEURAL_FREEZE_ID,
            project_root=tmp_path,
            source_paths=("source.py",),
            model_loader=lambda _: _synthetic_numeric_predictor(),
            model_semantic_loader=lambda _: SYNTHETIC_MODEL_SEMANTIC_SHA256,
            neural_freeze_loader=lambda *_: _synthetic_neural_freeze(),
        )


def test_active_nodes_allocation_is_mandatory_and_non_array() -> None:
    with pytest.raises(RuntimeError, match="active Slurm"):
        evaluator.require_active_nodes_allocation({})
    with pytest.raises(RuntimeError, match="nodes partition"):
        evaluator.require_active_nodes_allocation(
            {"SLURM_JOB_ID": "1", "SLURM_JOB_PARTITION": "gpu"}
        )
    with pytest.raises(RuntimeError, match="not a Slurm array"):
        evaluator.require_active_nodes_allocation(
            {
                "SLURM_JOB_ID": "1",
                "SLURM_JOB_PARTITION": "nodes",
                "SLURM_ARRAY_TASK_ID": "",
            }
        )
    evaluator.require_active_nodes_allocation(
        {"SLURM_JOB_ID": "1", "SLURM_JOB_PARTITION": "nodes"}
    )


def test_all_three_comparators_execute_once_per_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {method: 0 for method in CLASSICAL_COMPARATOR_IDS}
    image = np.zeros((3, 3), dtype=np.uint8)

    def b0(value: np.ndarray, *, area_cutoff_px: int) -> np.ndarray:
        assert area_cutoff_px == 100
        calls["B0_small_components"] += 1
        return np.zeros_like(value)

    def b1(value: np.ndarray, *, area_cutoff_px: int) -> np.ndarray:
        assert area_cutoff_px == 200
        calls["B1_marker_watershed"] += 1
        return np.ones_like(value)

    def b2(value: np.ndarray, estimator: object) -> np.ndarray:
        assert estimator is not None
        calls["B2_extra_trees"] += 1
        return np.full_like(value, 2)

    monkeypatch.setattr(evaluator, "predict_b0_small_components", b0)
    monkeypatch.setattr(evaluator, "predict_b1_marker_watershed", b1)
    monkeypatch.setattr(evaluator, "predict_b2_extra_trees", b2)
    predictions = evaluator.predict_all_locked_comparators(
        image,
        {
            "b0_area_cutoff_px": 100,
            "b1_area_cutoff_px": 200,
            "b2_estimator": object(),
        },
    )
    assert tuple(predictions) == CLASSICAL_COMPARATOR_IDS
    assert calls == {method: 1 for method in CLASSICAL_COMPARATOR_IDS}


def test_metrics_and_whole_tile_bootstrap_are_deterministic_without_accuracy() -> None:
    matrices = np.asarray(
        [
            [[8, 1, 1], [2, 7, 1], [1, 1, 18]],
            [[4, 2, 0], [1, 5, 0], [0, 1, 11]],
            [[7, 0, 1], [1, 6, 1], [1, 0, 15]],
        ],
        dtype=np.int64,
    )
    metrics = evaluator.publication_metrics_from_confusion(matrices.sum(axis=0))

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert "accuracy" not in keys(metrics)
    assert metrics["overall_accuracy_reported"] is False
    first = evaluator.whole_tile_bootstrap(matrices, replicates=100, seed=19)
    second = evaluator.whole_tile_bootstrap(matrices, replicates=100, seed=19)
    assert first == second
    assert first["sampling_unit"] == "held-out native 2048x2048 tile"


def test_classical_publication_style_reuses_the_locked_class_contract() -> None:
    assert evaluator.PUBLICATION_CLASS_COLORS == (
        "#B33A3A",
        "#2E8B57",
        "#4C78A8",
    )
    assert [evaluator.CLASS_LABELS[index].split()[0] for index in range(3)] == [
        "C0",
        "C1",
        "C2",
    ]
    plt = evaluator._configure_matplotlib()
    assert list(plt.rcParams["font.family"]) == ["sans-serif"]
    assert list(plt.rcParams["font.sans-serif"][:2]) == ["Arial", "Helvetica"]
    plt.close("all")


def test_clean_image_decoder_rejects_colour_and_non_native_tiles() -> None:
    colour = np.zeros((2, 2, 3), dtype=np.uint8)
    colour[:, :, 0] = 255
    buffer = io.BytesIO()
    Image.fromarray(colour).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="colour/ring information"):
        evaluator.decode_clean_image_bytes("colour.png", buffer.getvalue())

    buffer = io.BytesIO()
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="2048x2048"):
        evaluator.decode_clean_image_bytes("small.png", buffer.getvalue())


def test_cpu_slurm_wrapper_is_fixed_non_array_and_has_no_path_overrides() -> None:
    wrapper = REPOSITORY_ROOT / "scripts/aire_locked_classical_evaluation.slurm"
    text = wrapper.read_text(encoding="utf-8")
    assert "#SBATCH --partition=nodes" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --gres" not in text
    assert "#SBATCH --array" not in text
    assert "SLURM_JOB_ID" in text
    assert "SLURM_JOB_PARTITION" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "PORE_CLASSICAL_FIT_ID" in text
    assert "PORE_CLASSICAL_CAMPAIGN_ID" in text  # explicitly forbidden alias
    assert "PORE_NEURAL_FREEZE_ID" in text
    assert '--fit-id "$pore_classical_fit_id"' in text
    assert "--neural-freeze-id \"$pore_neural_freeze_id\"" in text
    assert text.count("evaluate_locked_classical_comparators.py") == 1
    for option in ("--image-dir", "--mask-dir", "--model", "--output-dir"):
        assert option not in text
    subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_train_only_cpu_wrapper_is_fixed_single_fit_with_canonical_paths() -> None:
    wrapper = REPOSITORY_ROOT / "scripts/aire_fit_classical_comparators.slurm"
    text = wrapper.read_text(encoding="utf-8")
    assert "#SBATCH --partition=nodes" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --gres" not in text
    assert "#SBATCH --array" not in text
    assert "SLURM_JOB_ID" in text
    assert "SLURM_JOB_PARTITION" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "${PORE_CLASSICAL_CAMPAIGN_ID:-classical-fit-${SLURM_JOB_ID}}" in text
    assert text.count("fit_classical_comparators.py") == 1
    assert "evaluate_locked_classical_comparators.py" not in text
    assert "--split-manifest config/confirmatory_splits.json" in text
    assert "--image-dir original_images" in text
    assert (
        "--mask-dir results/step2_pore_classification/pore_classifications"
        in text
    )
    assert "--output-dir" not in text
    assert "--annotations" not in text
    for variable in (
        "PORE_PROJECT_DIR",
        "PORE_OUTPUT_DIR",
        "PORE_ANNOTATIONS",
        "PORE_IMAGE_DIR",
        "PORE_MASK_DIR",
        "PORE_SPLIT_MANIFEST",
    ):
        assert variable in text
    subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_real_source_bundle_has_no_missing_or_null_hash() -> None:
    hashes = _source_hashes()
    assert set(hashes) == set(SOURCE_PATHS)
    assert all(evaluator.HEX_SHA256_PATTERN.fullmatch(value) for value in hashes.values())


def test_classical_source_bundle_attests_checkpoint_loader_dependency() -> None:
    """Keep the neural-freeze imports reproducible in the classical bundle."""
    assert "src/training/checkpoint_io.py" in SOURCE_PATHS


def test_scikit_image_is_declared_in_all_supported_environments() -> None:
    for path in ("requirements.txt", "pyproject.toml", "environment.yml"):
        text = (REPOSITORY_ROOT / path).read_text(encoding="utf-8").lower()
        assert "scikit-image" in text

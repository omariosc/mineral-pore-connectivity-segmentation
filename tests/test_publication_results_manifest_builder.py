"""Synthetic tests for the canonical publication-results manifest builder."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from scripts import assemble_publication_results as assembler
from scripts import build_publication_results_manifest as builder
from tests import test_publication_results_assembler as fixtures


CANONICAL_FREEZE_SHA = "1" * 64
CANONICAL_PAIR_SHA = hashlib.sha256(
    f"{fixtures.LOCK_SHA}\0{CANONICAL_FREEZE_SHA}".encode("ascii")
).hexdigest()
NEURAL_SOURCE_MAP = {
    path: hashlib.sha256(f"neural-source:{path}".encode("utf-8")).hexdigest()
    for path in builder.EXECUTION_SOURCE_FILES
}
SELECTED_LOCK_RAW_SHA = "8" * 64
SELECTED_LOCK_RECORD = {
    "repo_relative_identifier": builder.CANONICAL_SELECTED_METHOD_LOCK,
    "raw_file_sha256": SELECTED_LOCK_RAW_SHA,
    "canonical_identity_sha256": "7" * 64,
    "selected_method": "R3",
    "screen_campaign_id": "synthetic-screen",
}
CLASSICAL_SOURCE_MAP = {
    "scripts/evaluate_locked_classical_comparators.py": "6" * 64,
    "scripts/fit_classical_comparators.py": "5" * 64,
}
CLASSICAL_LOCK_BYTES = b"synthetic canonical classical lock\n"
B2_MODEL_BYTES = b"synthetic canonical B2 numeric model\n"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _install_evaluator_outputs(
    project_root: Path,
    *,
    selected_method: str = "R3",
    with_curves: bool = False,
    with_qualitative: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture_root = project_root / "synthetic_fixture"
    source_manifest = fixtures._build_fixture(
        fixture_root,
        selected_method=selected_method,
        with_curves=with_curves,
    )
    if with_qualitative:
        fixtures._attach_qualitative_panel(fixture_root, source_manifest)
    source = _read_json(source_manifest)
    freeze_id = fixtures.FREEZE_ID
    checkpoint_map: dict[str, list[str]] = {
        role: ["" for _ in assembler.NEURAL_SEEDS]
        for role in builder.ARCHITECTURE_ROLES
    }
    neural_destinations: list[tuple[Path, str, int]] = []
    for record in source["neural_evaluations"]:
        expected = record["expected_identity"]
        role = expected["architecture_role"]
        cell_index = expected["cell_index"]
        checkpoint_map[role][cell_index] = expected["checkpoint_sha256"]
        source_directory = fixture_root / record["report"]["path"]
        destination = (
            project_root
            / builder.NEURAL_EVALUATION_ROOT
            / freeze_id
            / role
            / f"cell_{cell_index:02d}"
        )
        shutil.copytree(source_directory.parent, destination)
        neural_destinations.append((destination, role, cell_index))

    selected_lock = {
        **SELECTED_LOCK_RECORD,
        "selected_method": selected_method,
    }
    semantic_map = {
        role: [
            hashlib.sha256(
                f"semantic-checkpoint:{role}:{index}".encode("utf-8")
            ).hexdigest()
            for index in range(len(assembler.NEURAL_SEEDS))
        ]
        for role in builder.ARCHITECTURE_ROLES
    }
    selected_retraining = {
        role: {
            "campaign_id": f"synthetic-{role}",
            "cells": [
                {
                    "array_task_index": index,
                    "training_seed": seed,
                    "architecture_role": role,
                    "campaign_id": f"synthetic-{role}",
                    "checkpoint_sha256": checkpoint_map[role][index],
                    "checkpoint_state_dict_semantic_sha256": semantic_map[role][
                        index
                    ],
                }
                for index, seed in enumerate(assembler.NEURAL_SEEDS)
            ],
        }
        for role in builder.ARCHITECTURE_ROLES
    }
    freeze = {
        "manifest_id": freeze_id,
        "manifest_repo_relative_identifier": (
            f"results/neural_freeze/locked/{freeze_id}/neural_freeze_manifest.json"
        ),
        "manifest_file_sha256": "9" * 64,
        "scientific_identity_sha256": CANONICAL_FREEZE_SHA,
        "selected_method": selected_method,
        "selected_method_lock": selected_lock,
        "selected_retraining_checkpoint_sha256": checkpoint_map,
        "document": {
            "source_code_sha256": NEURAL_SOURCE_MAP,
            "selected_retraining": selected_retraining,
        },
    }
    report_freeze = {
        "manifest_id": freeze_id,
        "manifest_path": freeze["manifest_repo_relative_identifier"],
        "manifest_file_sha256": freeze["manifest_file_sha256"],
        "scientific_identity_sha256": CANONICAL_FREEZE_SHA,
        "selected_retraining_checkpoint_sha256": checkpoint_map,
    }
    source_attestation = {
        "verification_status": "matched_screen_checkpoint_and_live_sources",
        "file_count": len(NEURAL_SOURCE_MAP),
        "files": NEURAL_SOURCE_MAP,
    }
    for destination, role, cell_index in neural_destinations:
        report_path = destination / "evaluation_summary.json"
        report = _read_json(report_path)
        report["code"] = {
            "evaluator_path": builder.NEURAL_EVALUATOR_PATH,
            "evaluator_sha256": NEURAL_SOURCE_MAP[builder.NEURAL_EVALUATOR_PATH],
            "training_execution_source_attestation": source_attestation,
        }
        report["selected_method_lock"].update(
            {
                "path": selected_lock["repo_relative_identifier"],
                "sha256": selected_lock["raw_file_sha256"],
            }
        )
        report["neural_freeze"] = report_freeze
        report["checkpoint"]["state_dict_semantic_sha256"] = semantic_map[role][
            cell_index
        ]
        report["locked_evaluation_identity"][
            "neural_freeze_scientific_identity_sha256"
        ] = CANONICAL_FREEZE_SHA
        report["locked_evaluation_identity"][
            "checkpoint_state_dict_semantic_sha256"
        ] = semantic_map[role][cell_index]
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    classical_source = (
        fixture_root / source["classical_evaluation"]["report"]["path"]
    ).parent
    classical_destination = (
        project_root
        / builder.CLASSICAL_EVALUATION_ROOT
        / fixtures.FIT_ID
        / freeze_id
    )
    shutil.copytree(classical_source, classical_destination)
    classical_paths = builder.canonical_fit_paths(
        fixtures.FIT_ID, project_root=project_root
    )
    classical_paths["lock_dir"].mkdir(parents=True)
    classical_paths["lock"].write_bytes(CLASSICAL_LOCK_BYTES)
    classical_paths["model"].write_bytes(B2_MODEL_BYTES)
    lock_raw_sha = assembler.sha256_file(classical_paths["lock"])
    b2_sha = assembler.sha256_file(classical_paths["model"])
    b2_semantic_sha = "4" * 64
    classical_report_path = classical_destination / "evaluation_summary.json"
    classical_report = _read_json(classical_report_path)
    classical_report["neural_freeze"] = {
        **report_freeze,
        "selected_method": selected_method,
        "selected_method_lock": selected_lock,
        "neural_held_out_result_required": False,
    }
    classical_report["evaluation_pair_identity_sha256"] = CANONICAL_PAIR_SHA
    classical_report["classical_lock"] = {
        "path": classical_paths["lock"].relative_to(project_root).as_posix(),
        "raw_file_sha256": lock_raw_sha,
        "canonical_identity_sha256": fixtures.LOCK_SHA,
        "source_code_sha256": CLASSICAL_SOURCE_MAP,
    }
    classical_report["b2_model"] = {
        "path": classical_paths["model"].relative_to(project_root).as_posix(),
        "sha256": b2_sha,
        "semantic_sha256": b2_semantic_sha,
    }
    classical_report_path.write_text(
        json.dumps(classical_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    classical = {
        "fit_id": fixtures.FIT_ID,
        "canonical_lock_identity_sha256": fixtures.LOCK_SHA,
        "lock_path": classical_paths["lock"],
        "lock_file_sha256": lock_raw_sha,
        "source_code_sha256": CLASSICAL_SOURCE_MAP,
        "b2_model_path": classical_paths["model"],
        "b2_model_sha256": b2_sha,
        "b2_model_semantic_sha256": b2_semantic_sha,
        "paths": classical_paths,
    }
    return freeze, classical


def _patch_verifiers(
    monkeypatch: pytest.MonkeyPatch,
    freeze: dict[str, Any],
    classical: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        builder,
        "load_verified_neural_freeze_manifest",
        lambda manifest_id, project_root: freeze,
    )
    monkeypatch.setattr(
        builder,
        "load_verified_classical_lock",
        lambda fit_id, project_root: classical,
    )


def _neural_report_path(project_root: Path) -> Path:
    return (
        project_root
        / builder.NEURAL_EVALUATION_ROOT
        / fixtures.FREEZE_ID
        / "primary_multiscale"
        / "cell_00"
        / "evaluation_summary.json"
    )


def _classical_report_path(project_root: Path) -> Path:
    return (
        project_root
        / builder.CLASSICAL_EVALUATION_ROOT
        / fixtures.FIT_ID
        / fixtures.FREEZE_ID
        / "evaluation_summary.json"
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_builder_derives_hash_bound_manifest_and_assembler_accepts_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(
        tmp_path, with_curves=True, with_qualitative=True
    )
    _patch_verifiers(monkeypatch, freeze, classical)

    manifest_path = builder.write_manifest(
        fixtures.FREEZE_ID,
        fixtures.FIT_ID,
        project_root=tmp_path,
    )
    assert manifest_path == tmp_path / builder.OUTPUT_NAME
    manifest = _read_json(manifest_path)
    assert manifest["schema_version"] == assembler.INPUT_SCHEMA_VERSION
    assert manifest["generator"] == {
        "schema_version": assembler.GENERATOR_SCHEMA_VERSION,
        "version": assembler.GENERATOR_VERSION,
        "source_sha256": assembler.GENERATOR_SOURCE_SHA256,
    }
    assert len(manifest["neural_evaluations"]) == 6
    assert len(manifest["qualitative_figures"]) == 1
    neural_identity = manifest["neural_evaluations"][0]["expected_identity"]
    assert neural_identity["evaluator_sha256"] == NEURAL_SOURCE_MAP[
        builder.NEURAL_EVALUATOR_PATH
    ]
    assert neural_identity["selected_method_lock_sha256"] == SELECTED_LOCK_RAW_SHA
    assert neural_identity["checkpoint_state_dict_semantic_sha256"] == freeze[
        "document"
    ]["selected_retraining"]["primary_multiscale"]["cells"][0][
        "checkpoint_state_dict_semantic_sha256"
    ]
    assert neural_identity["training_execution_source_attestation"]["files"] == (
        NEURAL_SOURCE_MAP
    )
    classical_identity = manifest["classical_evaluation"]["expected_identity"]
    assert classical_identity["classical_lock_source_code_sha256"] == (
        CLASSICAL_SOURCE_MAP
    )
    assert classical_identity["b2_model_sha256"] == classical["b2_model_sha256"]
    assert classical_identity["selected_retraining_checkpoint_sha256"] == freeze[
        "selected_retraining_checkpoint_sha256"
    ]
    for specification in builder._artifact_specifications(manifest):
        relative = Path(specification["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert specification["sha256"] == assembler.sha256_file(
            tmp_path / relative
        )

    output = assembler.assemble_publication_results(
        manifest_path, tmp_path / "publication_results"
    )
    report = _read_json(output / "publication_results.json")
    assert report["selected_primary_method_id"] == "R3"
    assert report["qualitative_figures"]["status"] == "copied"
    assert report["precision_recall"]["status"] == "created"


def test_builder_output_is_deterministic_across_equivalent_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    installed = [_install_evaluator_outputs(root) for root in roots]

    def freeze_loader(_manifest_id: str, project_root: Path) -> dict[str, Any]:
        return installed[roots.index(project_root)][0]

    def classical_loader(_fit_id: str, project_root: Path) -> dict[str, Any]:
        return installed[roots.index(project_root)][1]

    monkeypatch.setattr(builder, "load_verified_neural_freeze_manifest", freeze_loader)
    monkeypatch.setattr(builder, "load_verified_classical_lock", classical_loader)
    first = builder.write_manifest(
        fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=roots[0]
    )
    second = builder.write_manifest(
        fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=roots[1]
    )
    assert first.read_bytes() == second.read_bytes()


def test_builder_fails_when_report_identity_differs_from_verified_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _neural_report_path(tmp_path)
    report = _read_json(report_path)
    report["checkpoint"]["sha256"] = "f" * 64
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        builder.ManifestBuildError,
        match="assembler preflight rejected.*checkpoint raw/semantic digest mismatch",
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )
    assert not (tmp_path / builder.OUTPUT_NAME).exists()


def test_builder_rejects_neural_evaluator_hash_not_bound_to_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _neural_report_path(tmp_path)
    report = _read_json(report_path)
    report["code"]["evaluator_sha256"] = "f" * 64
    _write_json(report_path, report)

    with pytest.raises(
        builder.ManifestBuildError,
        match="code provenance does not match the verified freeze",
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


def test_builder_rejects_missing_neural_code_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _neural_report_path(tmp_path)
    report = _read_json(report_path)
    del report["code"]
    _write_json(report_path, report)

    with pytest.raises(
        builder.ManifestBuildError,
        match="neural report code provenance must be a mapping",
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


def test_builder_rejects_neural_source_attestation_not_bound_to_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _neural_report_path(tmp_path)
    report = _read_json(report_path)
    report["code"]["training_execution_source_attestation"]["files"][
        builder.NEURAL_EVALUATOR_PATH
    ] = "f" * 64
    _write_json(report_path, report)

    with pytest.raises(
        builder.ManifestBuildError,
        match="code provenance does not match the verified freeze",
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


def test_builder_rejects_neural_lock_hash_not_bound_to_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _neural_report_path(tmp_path)
    report = _read_json(report_path)
    report["selected_method_lock"]["sha256"] = "f" * 64
    _write_json(report_path, report)

    with pytest.raises(
        builder.ManifestBuildError,
        match="selected-method lock does not match the verified freeze",
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


def test_builder_rejects_missing_loaded_checkpoint_semantic_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _neural_report_path(tmp_path)
    report = _read_json(report_path)
    del report["checkpoint"]["state_dict_semantic_sha256"]
    _write_json(report_path, report)

    with pytest.raises(
        builder.ManifestBuildError,
        match="semantic checkpoint identity does not match the verified freeze",
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


def test_builder_rejects_contradictory_locked_checkpoint_semantic_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _neural_report_path(tmp_path)
    report = _read_json(report_path)
    report["locked_evaluation_identity"][
        "checkpoint_state_dict_semantic_sha256"
    ] = "f" * 64
    _write_json(report_path, report)

    with pytest.raises(
        builder.ManifestBuildError,
        match="semantic checkpoint identity does not match the verified freeze",
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("classical_lock", "raw_file_sha256", "lock provenance"),
        ("classical_lock", "source_code_sha256", "lock provenance"),
        ("b2_model", "sha256", "B2 provenance"),
        ("b2_model", "semantic_sha256", "B2 provenance"),
    ],
)
def test_builder_rejects_classical_report_not_bound_to_verified_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    message: str,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _classical_report_path(tmp_path)
    report = _read_json(report_path)
    report[section][field] = (
        {"scripts/forged.py": "f" * 64}
        if field == "source_code_sha256"
        else "f" * 64
    )
    _write_json(report_path, report)

    with pytest.raises(builder.ManifestBuildError, match=message):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


def test_builder_rejects_missing_classical_b2_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    report_path = _classical_report_path(tmp_path)
    report = _read_json(report_path)
    del report["b2_model"]
    _write_json(report_path, report)

    with pytest.raises(
        builder.ManifestBuildError, match="classical report B2 model must be a mapping"
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


def test_builder_rejects_symbolic_evaluator_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    artifact = (
        tmp_path
        / builder.NEURAL_EVALUATION_ROOT
        / fixtures.FREEZE_ID
        / "plain_unet_comparator"
        / "cell_02"
        / "aggregate_metrics.csv"
    )
    external = tmp_path / "copied_aggregate_metrics.csv"
    shutil.copyfile(artifact, external)
    artifact.unlink()
    artifact.symlink_to(external)

    with pytest.raises(
        builder.ManifestBuildError, match="symbolic-link component"
    ):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


def test_builder_never_overwrites_existing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    output = builder.write_manifest(
        fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
    )
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.write_manifest(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )
    assert output.read_bytes() == original


def test_builder_rejects_stale_imported_assembler_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assembler, "GENERATOR_SOURCE_SHA256", "0" * 64)
    with pytest.raises(
        builder.ManifestBuildError, match="assembler source changed after import"
    ):
        builder.build_manifest_document(
            fixtures.FREEZE_ID, fixtures.FIT_ID, project_root=tmp_path
        )


@pytest.mark.parametrize(
    ("freeze_id", "fit_id"),
    [
        ("neural-freeze-not-hex", fixtures.FIT_ID),
        (fixtures.FREEZE_ID, "classical-fit-not-hex"),
    ],
)
def test_builder_accepts_only_canonical_content_addressed_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    freeze_id: str,
    fit_id: str,
) -> None:
    freeze, classical = _install_evaluator_outputs(tmp_path)
    _patch_verifiers(monkeypatch, freeze, classical)
    with pytest.raises((ValueError, builder.ManifestBuildError)):
        builder.write_manifest(freeze_id, fit_id, project_root=tmp_path)

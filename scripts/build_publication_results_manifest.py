#!/usr/bin/env python3
"""Build the strict publication-results input manifest from canonical IDs.

This builder accepts no evaluator paths, checkpoint hashes, method names, or
scientific identities from the caller.  It verifies the content-addressed
neural freeze and classical fit, derives the seven canonical evaluation
directories, hashes every assembler input, and runs the assembler's complete
input validators before creating ``publication_results_inputs.json``.

The builder never opens a microscopy image, target mask, or prediction. The
upstream freeze verifiers authenticate checkpoint and model artifacts, but
neither verifier opens the locked retrospective corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import assemble_publication_results as assembler  # noqa: E402
from scripts.evaluate_confirmatory_checkpoint import (  # noqa: E402
    CANONICAL_LOCKED_EVALUATION_ROOT,
)
from scripts.evaluate_locked_classical_comparators import (  # noqa: E402
    CANONICAL_OUTPUT_ROOT as CLASSICAL_EVALUATION_ROOT,
    canonical_fit_paths,
    load_verified_classical_lock,
)
from scripts.fit_classical_comparators import validate_fit_id  # noqa: E402
from src.training.neural_freeze import (  # noqa: E402
    ARCHITECTURE_ROLES as FROZEN_ARCHITECTURE_ROLES,
    CANONICAL_SELECTED_METHOD_LOCK,
    load_verified_neural_freeze_manifest,
    validate_manifest_id,
)
from src.training.screen_selection import EXECUTION_SOURCE_FILES  # noqa: E402


OUTPUT_NAME = "publication_results_inputs.json"
NEURAL_EVALUATION_ROOT = Path(CANONICAL_LOCKED_EVALUATION_ROOT)
ARCHITECTURE_ROLES = tuple(FROZEN_ARCHITECTURE_ROLES)
CORE_ARTIFACTS = {
    "report": "evaluation_summary.json",
    "aggregate_metrics_csv": "aggregate_metrics.csv",
    "per_tile_metrics_csv": "per_tile_metrics.csv",
    "per_tile_confusion_csv": "per_tile_confusion.csv",
}
CURVE_ARTIFACTS = {
    "probability_histograms_csv": "probability_histograms.csv",
    "precision_recall_curve_csv": "precision_recall_curve.csv",
}
QUALITATIVE_FILES = (
    "publication/qualitative_triptych.pdf",
    "publication/qualitative_triptych.png",
)
NEURAL_EVALUATOR_PATH = "scripts/evaluate_confirmatory_checkpoint.py"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestBuildError(ValueError):
    """Canonical inputs cannot produce an authenticated assembly manifest."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _assert_live_assembler_identity() -> None:
    source = Path(assembler.__file__)
    if not source.is_file() or source.is_symlink():
        raise ManifestBuildError("Publication assembler source is missing or symbolic")
    observed = _sha256_bytes(source.read_bytes())
    if observed != assembler.GENERATOR_SOURCE_SHA256:
        raise ManifestBuildError(
            "Publication assembler source changed after import; restart the builder"
        )


def _real_project_root(value: Path) -> Path:
    lexical = Path(value)
    if lexical.is_symlink() or not lexical.is_dir():
        raise ManifestBuildError("Project root must be a real directory")
    return lexical.resolve()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestBuildError(f"{label} must be a mapping")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise ManifestBuildError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_json_object(content: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = assembler._load_json_bytes(content, label)
    except assembler.ContractError as error:
        raise ManifestBuildError(str(error)) from error
    return _require_mapping(value, label)


def _canonical_file_bytes(
    project_root: Path,
    path: Path,
    *,
    label: str,
) -> tuple[str, bytes]:
    """Read one stable, real file below the repository without following links."""

    root = _real_project_root(project_root)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ManifestBuildError(f"{label} escapes the repository") from error
    if relative == Path(".") or ".." in relative.parts:
        raise ManifestBuildError(f"{label} is not repository-relative")
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ManifestBuildError(f"{label} contains a symbolic-link component")
    if not lexical.is_file():
        raise ManifestBuildError(f"{label} is missing or is not a regular file")
    before = lexical.stat()
    content = lexical.read_bytes()
    after = lexical.stat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ManifestBuildError(f"{label} changed while it was being authenticated")
    return relative.as_posix(), content


def _artifact_specification(
    project_root: Path,
    path: Path,
    *,
    expected_name: str,
    label: str,
) -> dict[str, str]:
    if path.name != expected_name:
        raise ManifestBuildError(f"{label} must end in {expected_name}")
    relative, content = _canonical_file_bytes(project_root, path, label=label)
    return {"path": relative, "sha256": _sha256_bytes(content)}


def _optional_pair(
    project_root: Path,
    directory: Path,
    report: Mapping[str, Any],
) -> dict[str, dict[str, str] | None]:
    curves = report.get("curves")
    paths = [directory / name for name in CURVE_ARTIFACTS.values()]
    observed = [path.is_file() and not path.is_symlink() for path in paths]
    if curves is None:
        if any(observed):
            raise ManifestBuildError(
                "Neural report omits curve evidence but curve artifacts are present"
            )
        return {key: None for key in CURVE_ARTIFACTS}
    if not isinstance(curves, Mapping) or not all(observed):
        raise ManifestBuildError(
            "Neural curve evidence requires both canonical curve artifacts"
        )
    return {
        key: _artifact_specification(
            project_root,
            directory / name,
            expected_name=name,
            label=f"neural {key}",
        )
        for key, name in CURVE_ARTIFACTS.items()
    }


def _verified_freeze(
    neural_freeze_id: str,
    project_root: Path,
) -> Mapping[str, Any]:
    canonical_id = validate_manifest_id(neural_freeze_id)
    verified = _require_mapping(
        load_verified_neural_freeze_manifest(canonical_id, project_root),
        "verified neural freeze",
    )
    required = {
        "manifest_id",
        "manifest_repo_relative_identifier",
        "manifest_file_sha256",
        "scientific_identity_sha256",
        "selected_method",
        "selected_method_lock",
        "selected_retraining_checkpoint_sha256",
        "document",
    }
    if set(verified) != required:
        raise ManifestBuildError(
            "Verified neural-freeze record is incomplete or unexpected"
        )
    identity = _require_sha256(
        verified["scientific_identity_sha256"],
        "neural-freeze scientific identity",
    )
    if (
        verified["manifest_id"] != canonical_id
        or canonical_id != f"neural-freeze-{identity[:16]}"
    ):
        raise ManifestBuildError("Neural-freeze ID and scientific identity mismatch")
    if verified["selected_method"] not in assembler.SELECTED_METHOD_IDS:
        raise ManifestBuildError("Neural freeze names an unsupported selected method")
    checkpoint_map = _require_mapping(
        verified["selected_retraining_checkpoint_sha256"],
        "neural-freeze checkpoint map",
    )
    if tuple(checkpoint_map) != ARCHITECTURE_ROLES:
        raise ManifestBuildError("Neural freeze must contain both canonical roles")
    checkpoint_hashes: list[str] = []
    for role in ARCHITECTURE_ROLES:
        values = checkpoint_map[role]
        if not isinstance(values, list) or len(values) != len(assembler.NEURAL_SEEDS):
            raise ManifestBuildError(f"Neural freeze has an incomplete {role} role")
        checkpoint_hashes.extend(
            _require_sha256(value, f"{role} checkpoint") for value in values
        )
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ManifestBuildError("Neural-freeze checkpoints must be distinct")
    _freeze_expected_provenance(verified)
    return verified


def _freeze_expected_provenance(freeze: Mapping[str, Any]) -> dict[str, Any]:
    document = _require_mapping(freeze.get("document"), "neural-freeze document")
    source_value = _require_mapping(
        document.get("source_code_sha256"), "neural-freeze source map"
    )
    if set(source_value) != set(EXECUTION_SOURCE_FILES):
        raise ManifestBuildError("Neural-freeze source map has an unexpected file set")
    source_map = {
        str(path): _require_sha256(value, f"neural-freeze source {path}")
        for path, value in source_value.items()
    }
    evaluator_sha = source_map.get(NEURAL_EVALUATOR_PATH)
    if evaluator_sha is None:
        raise ManifestBuildError("Neural-freeze source map omits the locked evaluator")

    lock = _require_mapping(
        freeze.get("selected_method_lock"), "neural-freeze selected-method lock"
    )
    lock_path = lock.get("repo_relative_identifier")
    if lock_path != CANONICAL_SELECTED_METHOD_LOCK:
        raise ManifestBuildError(
            "Neural freeze names a non-canonical selected-method lock"
        )
    lock_sha = _require_sha256(
        lock.get("raw_file_sha256"), "selected-method lock raw file"
    )
    manifest_path = freeze.get("manifest_repo_relative_identifier")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ManifestBuildError("Neural freeze lacks its canonical manifest path")
    manifest_sha = _require_sha256(
        freeze.get("manifest_file_sha256"), "neural-freeze manifest file"
    )
    checkpoint_map = {
        role: list(freeze["selected_retraining_checkpoint_sha256"][role])
        for role in ARCHITECTURE_ROLES
    }
    training_attestation = {
        "verification_status": "matched_screen_checkpoint_and_live_sources",
        "file_count": len(source_map),
        "files": source_map,
    }
    return {
        "evaluator_path": NEURAL_EVALUATOR_PATH,
        "evaluator_sha256": evaluator_sha,
        "training_execution_source_attestation": training_attestation,
        "selected_method_lock_path": lock_path,
        "selected_method_lock_sha256": lock_sha,
        "neural_freeze_manifest_path": manifest_path,
        "neural_freeze_manifest_file_sha256": manifest_sha,
        "selected_retraining_checkpoint_sha256": checkpoint_map,
    }


def _freeze_cell_semantic_sha256(
    freeze: Mapping[str, Any],
    *,
    role: str,
    cell_index: int,
    training_seed: int,
    checkpoint_sha256: str,
) -> str:
    document = _require_mapping(freeze.get("document"), "neural-freeze document")
    retraining = _require_mapping(
        document.get("selected_retraining"), "neural-freeze selected retraining"
    )
    if set(retraining) != set(ARCHITECTURE_ROLES):
        raise ManifestBuildError(
            "Neural freeze selected retraining has an unexpected role set"
        )
    role_record = _require_mapping(
        retraining.get(role), f"neural-freeze selected retraining {role}"
    )
    cells = role_record.get("cells")
    if not isinstance(cells, list) or len(cells) != len(assembler.NEURAL_SEEDS):
        raise ManifestBuildError(f"Neural freeze has incomplete {role} cells")
    cell = _require_mapping(
        cells[cell_index], f"neural-freeze {role} cell {cell_index}"
    )
    if (
        cell.get("array_task_index") != cell_index
        or cell.get("training_seed") != training_seed
        or cell.get("architecture_role") != role
        or cell.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ManifestBuildError(
            "Neural freeze semantic checkpoint cell identity is inconsistent"
        )
    return _require_sha256(
        cell.get("checkpoint_state_dict_semantic_sha256"),
        f"neural-freeze {role} cell {cell_index} semantic checkpoint",
    )


def _cross_bind_neural_report(
    report: Mapping[str, Any],
    freeze: Mapping[str, Any],
    expected: Mapping[str, Any],
    checkpoint_state_dict_semantic_sha256: str,
) -> None:
    code = _require_mapping(report.get("code"), "neural report code provenance")
    expected_code = {
        "evaluator_path": expected["evaluator_path"],
        "evaluator_sha256": expected["evaluator_sha256"],
        "training_execution_source_attestation": expected[
            "training_execution_source_attestation"
        ],
    }
    if dict(code) != expected_code:
        raise ManifestBuildError(
            "Neural report code provenance does not match the verified freeze"
        )

    selected_lock = _require_mapping(
        report.get("selected_method_lock"), "neural report selected-method lock"
    )
    if (
        selected_lock.get("path") != expected["selected_method_lock_path"]
        or selected_lock.get("sha256")
        != expected["selected_method_lock_sha256"]
        or selected_lock.get("selected_method") != freeze["selected_method"]
    ):
        raise ManifestBuildError(
            "Neural report selected-method lock does not match the verified freeze"
        )

    report_freeze = _require_mapping(
        report.get("neural_freeze"), "neural report freeze provenance"
    )
    expected_freeze = {
        "manifest_id": freeze["manifest_id"],
        "manifest_path": expected["neural_freeze_manifest_path"],
        "manifest_file_sha256": expected["neural_freeze_manifest_file_sha256"],
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "selected_retraining_checkpoint_sha256": expected[
            "selected_retraining_checkpoint_sha256"
        ],
    }
    if dict(report_freeze) != expected_freeze:
        raise ManifestBuildError(
            "Neural report freeze provenance does not match the verified freeze"
        )
    identity = _require_mapping(
        report.get("locked_evaluation_identity"),
        "neural report locked evaluation identity",
    )
    checkpoint = _require_mapping(report.get("checkpoint"), "neural report checkpoint")
    if (
        identity.get("checkpoint_state_dict_semantic_sha256")
        != checkpoint_state_dict_semantic_sha256
        or checkpoint.get("state_dict_semantic_sha256")
        != checkpoint_state_dict_semantic_sha256
    ):
        raise ManifestBuildError(
            "Neural report semantic checkpoint identity does not match the "
            "verified freeze"
        )


def _neural_records(
    project_root: Path,
    freeze: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Path]:
    freeze_id = str(freeze["manifest_id"])
    freeze_sha = str(freeze["scientific_identity_sha256"])
    selected_method = str(freeze["selected_method"])
    checkpoint_map = _require_mapping(
        freeze["selected_retraining_checkpoint_sha256"],
        "neural-freeze checkpoint map",
    )
    expected_provenance = _freeze_expected_provenance(freeze)
    records: list[dict[str, Any]] = []
    seed_42_report: Mapping[str, Any] | None = None
    seed_42_directory: Path | None = None
    for role in ARCHITECTURE_ROLES:
        method_id = (
            selected_method
            if role == "primary_multiscale"
            else assembler.PLAIN_COMPARATOR_METHOD_ID
        )
        display_name = (
            selected_method if role == "primary_multiscale" else "Plain U-Net"
        )
        hashes = checkpoint_map[role]
        for cell_index, seed in enumerate(assembler.NEURAL_SEEDS):
            checkpoint_sha = _require_sha256(
                hashes[cell_index], f"{role} cell {cell_index} checkpoint"
            )
            checkpoint_semantic_sha = _freeze_cell_semantic_sha256(
                freeze,
                role=role,
                cell_index=cell_index,
                training_seed=seed,
                checkpoint_sha256=checkpoint_sha,
            )
            directory = (
                project_root
                / NEURAL_EVALUATION_ROOT
                / freeze_id
                / role
                / f"cell_{cell_index:02d}"
            )
            report_path = directory / CORE_ARTIFACTS["report"]
            report_spec = _artifact_specification(
                project_root,
                report_path,
                expected_name=CORE_ARTIFACTS["report"],
                label=f"{role} cell {cell_index} report",
            )
            _, report_content = _canonical_file_bytes(
                project_root,
                report_path,
                label=f"{role} cell {cell_index} report",
            )
            report = _strict_json_object(
                report_content, f"{role} cell {cell_index} report"
            )
            _cross_bind_neural_report(
                report,
                freeze,
                expected_provenance,
                checkpoint_semantic_sha,
            )
            record: dict[str, Any] = {
                "method_id": method_id,
                "display_name": display_name,
                "report": report_spec,
                **{
                    key: _artifact_specification(
                        project_root,
                        directory / name,
                        expected_name=name,
                        label=f"{role} cell {cell_index} {key}",
                    )
                    for key, name in CORE_ARTIFACTS.items()
                    if key != "report"
                },
                **_optional_pair(project_root, directory, report),
                "expected_identity": {
                    "neural_freeze_manifest_id": freeze_id,
                    "neural_freeze_scientific_identity_sha256": freeze_sha,
                    "architecture_role": role,
                    "cell_index": cell_index,
                    "training_seed": seed,
                    "selected_method": selected_method,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_state_dict_semantic_sha256": (
                        checkpoint_semantic_sha
                    ),
                    **expected_provenance,
                },
            }
            records.append(record)
            if (
                role == "primary_multiscale"
                and seed == assembler.QUALITATIVE_FIGURE_SEED
            ):
                seed_42_report = report
                seed_42_directory = directory
    if seed_42_report is None or seed_42_directory is None:
        raise ManifestBuildError("Selected-primary seed-42 evaluation is absent")
    return records, seed_42_report, seed_42_directory


def _classical_record(
    project_root: Path,
    fit_id: str,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        canonical_fit_id = validate_fit_id(fit_id)
    except ValueError as error:
        raise ManifestBuildError("Classical fit ID is not canonical") from error
    if canonical_fit_id != fit_id:
        raise ManifestBuildError("Classical fit ID is not canonical")
    expected_paths = canonical_fit_paths(fit_id, project_root=project_root)
    verified = _require_mapping(
        load_verified_classical_lock(fit_id, project_root=project_root),
        "verified classical fit",
    )
    if verified.get("fit_id") != fit_id:
        raise ManifestBuildError("Verified classical fit ID mismatch")
    observed_paths = _require_mapping(
        verified.get("paths"), "verified classical canonical paths"
    )
    if Path(observed_paths.get("output_fit_root", Path("."))) != expected_paths[
        "output_fit_root"
    ]:
        raise ManifestBuildError("Verified classical output root is not canonical")
    if Path(verified.get("lock_path", Path("."))) != expected_paths["lock"]:
        raise ManifestBuildError("Verified classical lock path is not canonical")
    if Path(verified.get("b2_model_path", Path("."))) != expected_paths["model"]:
        raise ManifestBuildError("Verified B2 model path is not canonical")

    lock_sha = _require_sha256(
        verified.get("canonical_lock_identity_sha256"),
        "classical lock scientific identity",
    )
    lock_raw_sha = _require_sha256(
        verified.get("lock_file_sha256"), "classical lock raw file"
    )
    lock_path, lock_content = _canonical_file_bytes(
        project_root, expected_paths["lock"], label="canonical classical lock"
    )
    if _sha256_bytes(lock_content) != lock_raw_sha:
        raise ManifestBuildError("Canonical classical lock byte hash mismatch")

    source_value = _require_mapping(
        verified.get("source_code_sha256"), "classical source-code map"
    )
    if not source_value:
        raise ManifestBuildError("Classical source-code map is empty")
    source_map = {
        str(path): _require_sha256(value, f"classical source {path}")
        for path, value in source_value.items()
    }
    b2_sha = _require_sha256(verified.get("b2_model_sha256"), "B2 model file")
    b2_semantic_sha = _require_sha256(
        verified.get("b2_model_semantic_sha256"), "B2 model semantic identity"
    )
    b2_path, b2_content = _canonical_file_bytes(
        project_root, expected_paths["model"], label="canonical B2 model"
    )
    if _sha256_bytes(b2_content) != b2_sha:
        raise ManifestBuildError("Canonical B2 model byte hash mismatch")

    freeze_provenance = _freeze_expected_provenance(freeze)
    freeze_sha = _require_sha256(
        freeze["scientific_identity_sha256"],
        "neural-freeze scientific identity",
    )
    pair_sha = hashlib.sha256(f"{lock_sha}\0{freeze_sha}".encode("ascii")).hexdigest()
    directory = expected_paths["output_fit_root"] / str(freeze["manifest_id"])
    report_path = directory / CORE_ARTIFACTS["report"]
    report_spec = _artifact_specification(
        project_root,
        report_path,
        expected_name=CORE_ARTIFACTS["report"],
        label="classical report",
    )
    _, report_content = _canonical_file_bytes(
        project_root, report_path, label="classical report"
    )
    report = _strict_json_object(report_content, "classical report")

    expected_report_lock = {
        "path": lock_path,
        "raw_file_sha256": lock_raw_sha,
        "canonical_identity_sha256": lock_sha,
        "source_code_sha256": source_map,
    }
    if dict(
        _require_mapping(report.get("classical_lock"), "classical report lock")
    ) != expected_report_lock:
        raise ManifestBuildError(
            "Classical report lock provenance does not match the verified fit"
        )
    expected_report_b2 = {
        "path": b2_path,
        "sha256": b2_sha,
        "semantic_sha256": b2_semantic_sha,
    }
    if dict(_require_mapping(report.get("b2_model"), "classical report B2 model")) != (
        expected_report_b2
    ):
        raise ManifestBuildError(
            "Classical report B2 provenance does not match the verified fit"
        )
    expected_report_freeze = {
        "manifest_id": freeze["manifest_id"],
        "manifest_path": freeze_provenance["neural_freeze_manifest_path"],
        "manifest_file_sha256": freeze_provenance[
            "neural_freeze_manifest_file_sha256"
        ],
        "scientific_identity_sha256": freeze_sha,
        "selected_method": freeze["selected_method"],
        "selected_method_lock": dict(freeze["selected_method_lock"]),
        "selected_retraining_checkpoint_sha256": freeze_provenance[
            "selected_retraining_checkpoint_sha256"
        ],
        "neural_held_out_result_required": False,
    }
    if dict(
        _require_mapping(
            report.get("neural_freeze"), "classical report neural freeze"
        )
    ) != expected_report_freeze:
        raise ManifestBuildError(
            "Classical report neural-freeze provenance does not match the "
            "verified freeze"
        )

    expected_provenance = {
        "classical_lock_path": lock_path,
        "classical_lock_raw_file_sha256": lock_raw_sha,
        "classical_lock_source_code_sha256": source_map,
        "b2_model_path": b2_path,
        "b2_model_sha256": b2_sha,
        "b2_model_semantic_sha256": b2_semantic_sha,
        "selected_method_lock_path": freeze_provenance[
            "selected_method_lock_path"
        ],
        "selected_method_lock_sha256": freeze_provenance[
            "selected_method_lock_sha256"
        ],
        "selected_retraining_checkpoint_sha256": freeze_provenance[
            "selected_retraining_checkpoint_sha256"
        ],
        "neural_freeze_manifest_path": freeze_provenance[
            "neural_freeze_manifest_path"
        ],
        "neural_freeze_manifest_file_sha256": freeze_provenance[
            "neural_freeze_manifest_file_sha256"
        ],
    }
    return {
        "report": report_spec,
        **{
            key: _artifact_specification(
                project_root,
                directory / name,
                expected_name=name,
                label=f"classical {key}",
            )
            for key, name in CORE_ARTIFACTS.items()
            if key != "report"
        },
        "expected_identity": {
            "fit_id": fit_id,
            "evaluation_pair_identity_sha256": pair_sha,
            "neural_freeze_manifest_id": freeze["manifest_id"],
            "neural_freeze_scientific_identity_sha256": freeze_sha,
            "classical_lock_identity_sha256": lock_sha,
            **expected_provenance,
        },
    }


def _qualitative_records(
    project_root: Path,
    selected_method: str,
    report: Mapping[str, Any],
    directory: Path,
) -> list[dict[str, Any]]:
    paths = [directory / identifier for identifier in QUALITATIVE_FILES]
    present = [path.is_file() and not path.is_symlink() for path in paths]
    qualitative_value = report.get("qualitative_example")
    if qualitative_value is None:
        if any(present):
            raise ManifestBuildError(
                "Qualitative artifacts are present without an evaluator contract"
            )
        return []
    qualitative = _require_mapping(
        qualitative_value, "selected seed-42 qualitative_example"
    )
    declared = qualitative.get("publication_files")
    if not isinstance(declared, list):
        raise ManifestBuildError("qualitative_example.publication_files must be a list")
    if declared == []:
        if any(present):
            raise ManifestBuildError(
                "Undeclared qualitative publication artifacts are present"
            )
        return []
    if declared != list(QUALITATIVE_FILES) or not all(present):
        raise ManifestBuildError(
            "The evaluator-declared qualitative PDF/PNG pair is incomplete"
        )
    return [
        {
            "method_id": selected_method,
            "training_seed": assembler.QUALITATIVE_FIGURE_SEED,
            "pdf": _artifact_specification(
                project_root,
                paths[0],
                expected_name=paths[0].name,
                label="selected seed-42 qualitative PDF",
            ),
            "png": _artifact_specification(
                project_root,
                paths[1],
                expected_name=paths[1].name,
                label="selected seed-42 qualitative PNG",
            ),
        }
    ]


def _artifact_specifications(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    specifications: list[Mapping[str, Any]] = []
    for record in manifest["neural_evaluations"]:
        for key in (*CORE_ARTIFACTS, *CURVE_ARTIFACTS):
            value = record[key]
            if value is not None:
                specifications.append(value)
    for key in CORE_ARTIFACTS:
        specifications.append(manifest["classical_evaluation"][key])
    for record in manifest["qualitative_figures"]:
        specifications.extend((record["pdf"], record["png"]))
    return specifications


def _validate_complete_document(
    project_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    try:
        evaluations = [
            assembler._load_neural_evaluation(project_root, record)
            for record in manifest["neural_evaluations"]
        ]
        evaluations.extend(
            assembler._load_classical_evaluations(
                project_root, manifest["classical_evaluation"]
            )
        )
        selected_method = assembler._validate_evaluation_collection(evaluations)
        assembler._load_qualitative_panels(
            project_root,
            manifest["qualitative_figures"],
            evaluations,
            selected_method,
        )
    except assembler.ContractError as error:
        raise ManifestBuildError(
            f"Publication assembler preflight rejected the canonical evidence: {error}"
        ) from error


def _verify_bound_bytes_unchanged(
    project_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    for specification in _artifact_specifications(manifest):
        path = project_root / str(specification["path"])
        _, content = _canonical_file_bytes(
            project_root, path, label=str(specification["path"])
        )
        if _sha256_bytes(content) != specification["sha256"]:
            raise ManifestBuildError(
                "Authenticated input changed before publication: "
                f"{specification['path']}"
            )


def build_manifest_document(
    neural_freeze_id: str,
    classical_fit_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return one fully authenticated strict-v2 assembler input document."""

    root = _real_project_root(project_root)
    _assert_live_assembler_identity()
    freeze = _verified_freeze(neural_freeze_id, root)
    neural, qualitative_report, qualitative_directory = _neural_records(root, freeze)
    classical = _classical_record(root, classical_fit_id, freeze)
    qualitative = _qualitative_records(
        root,
        str(freeze["selected_method"]),
        qualitative_report,
        qualitative_directory,
    )
    manifest = {
        "schema_version": assembler.INPUT_SCHEMA_VERSION,
        "generator": {
            "schema_version": assembler.GENERATOR_SCHEMA_VERSION,
            "version": assembler.GENERATOR_VERSION,
            "source_sha256": assembler.GENERATOR_SOURCE_SHA256,
        },
        "partition_label": assembler.PARTITION_LABEL,
        "bootstrap": {
            "replicates": assembler.PAIRED_BOOTSTRAP_REPLICATES,
            "seed": assembler.PAIRED_BOOTSTRAP_SEED,
            "confidence": assembler.PAIRED_BOOTSTRAP_CONFIDENCE,
        },
        "neural_evaluations": neural,
        "classical_evaluation": classical,
        "qualitative_figures": qualitative,
    }
    _validate_complete_document(root, manifest)
    _verify_bound_bytes_unchanged(root, manifest)
    _assert_live_assembler_identity()
    return manifest


def write_manifest(
    neural_freeze_id: str,
    classical_fit_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Create the canonical repository-root manifest without overwriting."""

    root = _real_project_root(project_root)
    output = root / OUTPUT_NAME
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    document = build_manifest_document(
        neural_freeze_id, classical_fit_id, project_root=root
    )
    payload = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the strict v2 publication-results input manifest from one "
            "canonical neural-freeze ID and one canonical classical-fit ID"
        )
    )
    parser.add_argument("--neural-freeze-id", required=True)
    parser.add_argument("--classical-fit-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = write_manifest(args.neural_freeze_id, args.classical_fit_id)
    print(f"Wrote authenticated publication-results manifest to {output.name}")


if __name__ == "__main__":
    main()

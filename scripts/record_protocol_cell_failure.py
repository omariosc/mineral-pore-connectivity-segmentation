#!/usr/bin/env python3
"""Write one immutable authenticated failure outcome for a Slurm array cell."""

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.screen_selection import (
    PROSPECTIVE_METHOD_PROTOCOLS,
    SCREEN_CANDIDATE_ORDER,
    SCREEN_RESULT_SCHEMA_VERSION,
    SCREEN_SEEDS,
    SMOKE_CANDIDATE_ORDER,
    sha256_file,
    source_code_sha256,
    verify_smoke_preflight_manifest_document,
)


def _safe_campaign(value: str) -> str:
    if (
        not value
        or len(value) > 96
        or not value[0].isalnum()
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value)
    ):
        raise ValueError("campaign ID is not public-path safe")
    return value


def _smoke_reference(value: str) -> dict:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        identifier = path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("smoke manifest must be inside the staging root") from error
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    verified = verify_smoke_preflight_manifest_document(document, PROJECT_ROOT)
    return {
        "repo_relative_identifier": identifier,
        "sha256": sha256_file(path),
        "campaign_id": verified["smoke_campaign_provenance"]["campaign_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-role",
        required=True,
        choices=["validation_screen_cell", "validation_smoke_cell"],
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--smoke-preflight-manifest")
    args = parser.parse_args()

    campaign = _safe_campaign(args.campaign_id)
    if args.exit_code == 0:
        parser.error("failure outcome requires a non-zero exit code")
    if args.run_role == "validation_screen_cell":
        if args.candidate not in SCREEN_CANDIDATE_ORDER or args.seed not in SCREEN_SEEDS:
            parser.error("invalid screen candidate/seed")
        expected_index = SCREEN_CANDIDATE_ORDER.index(args.candidate) * 3
        expected_index += SCREEN_SEEDS.index(args.seed)
        if args.cell_index != expected_index:
            parser.error("screen candidate/seed does not match task index")
        if not args.smoke_preflight_manifest:
            parser.error("screen failures require the authenticated smoke manifest")
        smoke_reference = _smoke_reference(args.smoke_preflight_manifest)
        debug_limited = False
        max_batches = None
    else:
        if args.candidate not in SMOKE_CANDIDATE_ORDER or args.seed != 42:
            parser.error("invalid smoke candidate/seed")
        if args.cell_index != SMOKE_CANDIDATE_ORDER.index(args.candidate):
            parser.error("smoke candidate does not match task index")
        if args.smoke_preflight_manifest:
            parser.error("a smoke cell cannot depend on a smoke manifest")
        smoke_reference = None
        debug_limited = True
        max_batches = 2

    relative_cell = (
        Path("results/patch_training/protocol_runs")
        / args.run_role
        / campaign
        / f"cell_{args.cell_index:02d}"
    )
    cell = (PROJECT_ROOT / relative_cell).resolve()
    try:
        cell.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("failure cell escapes staging root") from error
    if not cell.exists():
        cell.parent.mkdir(parents=True, exist_ok=True)
        cell.mkdir(exist_ok=False)
    metrics = cell / "metrics"
    metrics.mkdir(exist_ok=True)
    outcome = metrics / "model_selection.json"
    if outcome.exists():
        raise FileExistsError(
            "refusing to overwrite an existing immutable cell outcome: "
            f"{outcome.relative_to(PROJECT_ROOT)}"
        )

    artifact = {
        "screen_result_schema_version": SCREEN_RESULT_SCHEMA_VERSION,
        "outcome_status": "failed",
        "run_role": args.run_role,
        "protocol_campaign_id": campaign,
        "protocol_cell_index": args.cell_index,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "protocol_candidate_key": args.candidate,
        "seed": args.seed,
        "runtime_protocol": PROSPECTIVE_METHOD_PROTOCOLS[args.candidate],
        "scientific_execution_contract": None,
        "successful_cell": False,
        "successful_smoke": False,
        "scientific_result_eligible": False,
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_test_evaluated": False,
        "held_out_test_evaluation_count": 0,
        "checkpoints_disabled": False,
        "debug_limited": debug_limited,
        "max_batches": max_batches,
        "source_code_sha256": source_code_sha256(PROJECT_ROOT),
        "smoke_preflight_manifest": smoke_reference,
        "execution_environment": {
            "cuda_available": True,
            "device_type": "cuda",
            "gpu_name": os.environ.get("PORE_ALLOCATED_GPU_NAME"),
        },
        "failure": {
            "exit_code": args.exit_code,
            "reason": "training_command_nonzero_exit",
            "scheduler_outcome": "failed_no_scientific_metrics",
            "rerun_policy": "whole_campaign_only_new_slurm_array_job_id",
        },
        "selected_checkpoint_repo_relative_identifier": None,
        "selected_checkpoint_sha256": None,
        "best_selection_components": None,
        "best_selection_score": None,
        "data_split": None,
        "target_provenance": None,
        "input_provenance": None,
    }
    with outcome.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    print(f"Recorded immutable failed cell: {outcome.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

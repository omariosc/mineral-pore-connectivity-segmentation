#!/usr/bin/env python3
"""Freeze neural selection plus six validation-only retraining checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.neural_freeze import (  # noqa: E402
    build_neural_freeze_manifest_document,
    canonical_manifest_path,
    load_verified_neural_freeze_manifest,
    validate_campaign_id,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the content-addressed proof that neural selection and both "
            "three-seed validation-only retraining campaigns are frozen"
        )
    )
    parser.add_argument("--primary-campaign-id", required=True)
    parser.add_argument("--plain-campaign-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        primary = validate_campaign_id(args.primary_campaign_id)
        plain = validate_campaign_id(args.plain_campaign_id)
        document = build_neural_freeze_manifest_document(
            primary_campaign_id=primary,
            plain_campaign_id=plain,
            repository_root=PROJECT_ROOT,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    output_path = canonical_manifest_path(document["manifest_id"], PROJECT_ROOT)
    output_dir = output_path.parent
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(
            f"Canonical neural-freeze manifest directory already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    # Rebuild from all source artifacts once more after the exclusive write.
    verified = load_verified_neural_freeze_manifest(
        document["manifest_id"], PROJECT_ROOT
    )
    print(
        json.dumps(
            {
                "status": document["status"],
                "manifest_id": document["manifest_id"],
                "manifest_path": verified["manifest_repo_relative_identifier"],
                "manifest_sha256": verified["manifest_file_sha256"],
                "scientific_identity_sha256": document[
                    "scientific_identity_sha256"
                ],
                "selected_method": document["selected_method"],
                "checkpoint_count": 6,
                "held_out_bytes_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize the reviewed public allowlist into a new, clean directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.audit_public_snapshot import (
    PROJECT_ROOT,
    audit_asset_approvals,
    audit_materialized_boundary,
    audit_paths,
    audit_snapshot_content,
    expand_allowlist,
    read_allowlist,
    snapshot_tree_sha256,
)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def export_public_snapshot(
    source_root: Path,
    output_dir: Path,
    allowlist_path: Path,
    approval_manifest_path: Path | None = None,
) -> list[str]:
    """Copy exactly the reviewed paths and verify the resulting complete tree."""
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    allowlist_path = Path(allowlist_path).resolve()

    if not source_root.is_dir():
        raise ValueError(f"source root is not a directory: {source_root}")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    if _is_within(output_dir, source_root):
        raise ValueError("output directory must be outside the mixed research checkout")

    patterns = read_allowlist(allowlist_path)
    source_paths, missing = expand_allowlist(patterns, source_root)
    findings = [f"missing required public source path: {path}" for path in missing]
    findings.extend(audit_paths(source_paths, source_root))
    findings.extend(
        audit_asset_approvals(
            source_paths,
            source_root,
            manifest_path=approval_manifest_path,
        )
    )
    if findings:
        raise ValueError("source selection is unsafe:\n- " + "\n- ".join(findings))

    output_dir.mkdir(parents=True)
    copied = []
    for source_path in source_paths:
        relative = source_path.relative_to(source_root)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination, follow_symlinks=False)
        copied.append(relative.as_posix())

    snapshot_paths, snapshot_missing = expand_allowlist(patterns, output_dir)
    verification = [
        f"missing required public snapshot path: {path}" for path in snapshot_missing
    ]
    verification.extend(audit_paths(snapshot_paths, output_dir))
    verification.extend(audit_materialized_boundary(snapshot_paths, output_dir))
    verification.extend(
        audit_snapshot_content(source_paths, source_root, output_dir)
    )
    if verification:
        raise RuntimeError(
            "materialized snapshot failed verification:\n- "
            + "\n- ".join(verification)
        )
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Mixed source checkout containing the reviewed files",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Allowlist file (default: <source-root>/config/public_release_allowlist.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to create; it must not already exist",
    )
    parser.add_argument(
        "--approval-manifest",
        type=Path,
        default=None,
        help=(
            "Completed JSON-compatible YAML approval manifest. Required when "
            "the allowlist selects paper_assets/; defaults to "
            "<source-root>/config/public_asset_approvals.yml"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    allowlist_path = (
        args.allowlist.resolve()
        if args.allowlist is not None
        else source_root / "config" / "public_release_allowlist.txt"
    )
    try:
        copied = export_public_snapshot(
            source_root,
            args.output_dir,
            allowlist_path,
            approval_manifest_path=args.approval_manifest,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Public snapshot export failed: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "passed",
                "file_count": len(copied),
                "output_dir": str(args.output_dir.resolve()),
                "tree_sha256": snapshot_tree_sha256(
                    [args.output_dir.resolve() / relative for relative in copied],
                    args.output_dir.resolve(),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

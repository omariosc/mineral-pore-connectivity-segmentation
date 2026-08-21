#!/usr/bin/env python3
"""Fail closed when the proposed public snapshot contains unsafe artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = PROJECT_ROOT / "config" / "public_release_allowlist.txt"
DEFAULT_ASSET_APPROVAL_MANIFEST = Path("config/public_asset_approvals.yml")
ASSET_APPROVAL_SCHEMA_VERSION = 1
ASSET_APPROVAL_REQUIRED_TEXT_FIELDS = (
    "asset_kind",
    "source_description",
    "copyright_owner",
    "licence_identifier",
    "written_approval_reference",
    "approved_by",
    "approval_date",
)

FORBIDDEN_PATH_GLOBS = (
    "Overleaf/**",
    "original_images/**",
    "labelled images/**",
    "results/**",
    "logs/**",
    "papers/**",
    "tasks/**",
    "figures/**",
    "tables/**",
    "scripts/cron/**",
    "scripts/rerun_*.sh",
    "**/__pycache__/**",
    "**/.DS_Store",
    "**/*.pptx",
    "**/*.zip",
    "**/*.docx",
    "**/*.pth",
    "**/*.pt",
    "**/*.ckpt",
    "**/*.onnx",
)

FORBIDDEN_PAPER_ASSETS = (
    "paper_assets/figures/fig_02_*",
    "paper_assets/figures/fig_03_*",
    "paper_assets/figures/fig_04_*",
    "paper_assets/figures/fig_05_*",
    "paper_assets/figures/fig_07_*",
    "paper_assets/figures/fig_08_*",
    "paper_assets/tables/experiment_summary.csv",
    "paper_assets/tables/top_experiments.md",
    "paper_assets/manifest.json",
)

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".slurm",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CONTENT_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    (
        "private key",
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    ),
    (
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    (
        "OpenAI-style key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "literal credential assignment",
        re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*"
            r"['\"](?![<$\{])[^'\"]{8,}['\"]"
        ),
    ),
    (
        "Weights & Biases API key",
        re.compile(
            r"(?i)\bWANDB_API_KEY\b\s*[:=]\s*['\"]?[a-f0-9]{40}\b"
        ),
    ),
    (
        "email address",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    ("macOS home path", re.compile(r"/Users/[A-Za-z0-9._-]+/")),
    ("Linux home path", re.compile(r"/home/[A-Za-z0-9._-]+/")),
    ("HPC scratch path", re.compile(r"/mnt/(?:scratch|lustre|nobackup)/")),
    ("institutional username", re.compile(r"\bsc20osc\b", re.IGNORECASE)),
)


def read_allowlist(path: Path) -> List[str]:
    patterns = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            if Path(line).is_absolute() or ".." in Path(line).parts:
                raise ValueError(f"Unsafe allowlist entry: {line}")
            patterns.append(line)
    if not patterns:
        raise ValueError("Public release allowlist is empty")
    return patterns


def expand_allowlist(
    patterns: Iterable[str], root: Path = PROJECT_ROOT
) -> Tuple[List[Path], List[str]]:
    root = Path(root).resolve()
    paths = set()
    missing_exact = []
    for pattern in patterns:
        matches = [path for path in root.glob(pattern) if path.is_file()]
        if not matches and not any(character in pattern for character in "*?["):
            missing_exact.append(pattern)
        for path in matches:
            paths.add(path)
    return sorted(paths), missing_exact


def _validated_approval_path(value: Any) -> Tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "path must be a non-empty string"
    if value != value.strip() or "\\" in value:
        return None, f"path is not canonical: {value!r}"
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        return None, f"path is unsafe: {value!r}"
    canonical = path.as_posix()
    if canonical != value or not canonical.startswith("paper_assets/"):
        return None, f"path must be canonical and under paper_assets/: {value!r}"
    return canonical, None


def audit_asset_approvals(
    paths: Iterable[Path],
    root: Path,
    manifest_path: Path | None = None,
) -> List[str]:
    """Require a complete written approval record for every selected asset.

    Approval manifests use JSON syntax, which is valid YAML 1.2, so this
    boundary can parse them with the standard library and never instantiate
    YAML tags or require an extra runtime package.
    """
    root = Path(root).resolve()
    selected_assets = sorted(
        path.relative_to(root).as_posix()
        for path in paths
        if path.relative_to(root).as_posix().startswith("paper_assets/")
    )
    if not selected_assets:
        return []

    manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else root / DEFAULT_ASSET_APPROVAL_MANIFEST
    )
    if manifest.is_symlink():
        return [f"asset approval manifest must not be a symbolic link: {manifest}"]
    if not manifest.is_file():
        return [
            "selected public assets require a completed approval manifest: "
            f"{manifest}"
        ]
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"asset approval manifest is not valid JSON-compatible YAML: {error}"]

    if not isinstance(document, Mapping):
        return ["asset approval manifest root must be an object"]

    findings: List[str] = []
    if document.get("schema_version") != ASSET_APPROVAL_SCHEMA_VERSION:
        findings.append(
            "asset approval manifest schema_version must equal "
            f"{ASSET_APPROVAL_SCHEMA_VERSION}"
        )
    if document.get("template_only") is not False:
        findings.append(
            "asset approval manifest is a template or does not explicitly set "
            "template_only to false"
        )
    records = document.get("assets")
    if not isinstance(records, list):
        findings.append("asset approval manifest assets must be a list")
        return findings

    authorised_paths: set[str] = set()
    recorded_paths: set[str] = set()
    for index, record in enumerate(records):
        label = f"asset approval entry {index}"
        if not isinstance(record, Mapping):
            findings.append(f"{label} must be an object")
            continue

        entry_findings: List[str] = []
        raw_paths = record.get("paths")
        valid_paths: List[str] = []
        if not isinstance(raw_paths, list) or not raw_paths:
            entry_findings.append("paths must be a non-empty list")
        else:
            for raw_path in raw_paths:
                canonical, error = _validated_approval_path(raw_path)
                if error:
                    entry_findings.append(error)
                    continue
                assert canonical is not None
                if canonical in recorded_paths:
                    entry_findings.append(
                        f"path appears in more than one approval entry: {canonical}"
                    )
                else:
                    recorded_paths.add(canonical)
                    valid_paths.append(canonical)

        if record.get("approval_status") != "approved":
            entry_findings.append("approval_status must equal 'approved'")
        for field in ASSET_APPROVAL_REQUIRED_TEXT_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                entry_findings.append(f"{field} must be a non-empty string")
        for field in (
            "contains_source_image_pixels",
            "contains_annotation_or_mask_pixels",
        ):
            if type(record.get(field)) is not bool:
                entry_findings.append(f"{field} must be true or false")

        approval_date = record.get("approval_date")
        if isinstance(approval_date, str) and approval_date.strip():
            try:
                dt.date.fromisoformat(approval_date)
            except ValueError:
                entry_findings.append("approval_date must use YYYY-MM-DD")

        findings.extend(f"{label}: {finding}" for finding in entry_findings)
        if not entry_findings:
            authorised_paths.update(valid_paths)

    findings.extend(
        f"selected public asset has no complete approved entry: {path}"
        for path in selected_assets
        if path not in authorised_paths
    )
    return findings


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _has_symlink_component(path: Path, root: Path) -> bool:
    root = Path(root).resolve()
    cursor = root
    for component in Path(path).absolute().relative_to(root).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            return True
    return False


def audit_paths(
    paths: Iterable[Path], root: Path = PROJECT_ROOT
) -> List[str]:
    root = Path(root).resolve()
    findings = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if _has_symlink_component(path, root):
            findings.append(f"{relative}: symbolic links are not allowed")
            continue
        if _matches_any(relative, FORBIDDEN_PATH_GLOBS):
            findings.append(f"{relative}: forbidden private, bulky, or legacy path")
        if _matches_any(relative, FORBIDDEN_PAPER_ASSETS):
            findings.append(f"{relative}: diagnostic or non-evidential paper asset")
        if path.suffix.lower() == ".pdf" and not relative.startswith("paper_assets/"):
            findings.append(f"{relative}: PDF is outside the curated paper asset directory")

        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                findings.append(f"{relative}: expected text file is not valid UTF-8")
                continue
        else:
            # Paths, emails, and common tokens in PDF/PNG metadata are usually
            # stored as plain bytes. Latin-1 preserves those bytes one-to-one;
            # this is a metadata leak check, not a semantic binary parser.
            content = path.read_bytes().decode("latin-1", errors="ignore")
        for label, pattern in CONTENT_PATTERNS:
            match = pattern.search(content)
            if match:
                line = content.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: contains {label}")
    return findings


def audit_tracked_boundary(
    paths: Iterable[Path], root: Path = PROJECT_ROOT
) -> List[str]:
    """Require a Git tracked tree to equal the allowlisted tree."""
    root = Path(root).resolve()
    if not (root / ".git").exists():
        return [
            "release boundary is unavailable: no Git metadata exists; use "
            "--selection-only for a source-content check or --snapshot-root "
            "for a complete materialized-tree audit"
        ]
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        return [f"could not inspect Git-tracked paths: {detail}"]

    tracked = {
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    }
    allowed = {path.relative_to(root).as_posix() for path in paths}
    findings = [
        f"tracked path is outside public allowlist: {path}"
        for path in sorted(tracked - allowed)
    ]
    findings.extend(
        f"allowlisted path is not tracked: {path}"
        for path in sorted(allowed - tracked)
    )
    return findings


def materialized_tree_paths(root: Path) -> set[str]:
    """Return every file or symlink outside Git's internal metadata."""
    root = Path(root).resolve()
    paths = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_file() or path.is_symlink():
            paths.add(relative.as_posix())
    return paths


def audit_materialized_boundary(
    paths: Iterable[Path], root: Path
) -> List[str]:
    """Require a materialized directory to contain exactly allowlisted files."""
    root = Path(root).resolve()
    allowed = {path.relative_to(root).as_posix() for path in paths}
    materialized = materialized_tree_paths(root)
    findings = [
        f"materialized path is outside public allowlist: {path}"
        for path in sorted(materialized - allowed)
    ]
    findings.extend(
        f"allowlisted path is missing from materialized snapshot: {path}"
        for path in sorted(allowed - materialized)
    )
    return findings


def audit_snapshot_content(
    source_paths: Iterable[Path], source_root: Path, snapshot_root: Path
) -> List[str]:
    """Require copied allowlisted files to be byte-identical to their sources."""
    source_root = Path(source_root).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    findings = []
    for source_path in source_paths:
        relative = source_path.relative_to(source_root)
        snapshot_path = snapshot_root / relative
        if snapshot_path.is_file() and source_path.read_bytes() != snapshot_path.read_bytes():
            findings.append(
                f"materialized file differs from reviewed source: {relative.as_posix()}"
            )
    return findings


def snapshot_tree_sha256(paths: Iterable[Path], root: Path) -> str:
    """Hash canonical paths and file bytes independent of mtimes and modes.

    Versioned length prefixes make the byte stream unambiguous. The caller is
    expected to audit the tree boundary and reject symbolic links first.
    """
    root = Path(root).resolve()
    digest = hashlib.sha256(b"public-snapshot-tree-v1\0")
    ordered_paths = sorted(
        (Path(path) for path in paths),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    for path in ordered_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"tree hash requires a regular file: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        file_digest = hashlib.sha256()
        file_size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
                file_size += len(chunk)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_size.to_bytes(8, "big"))
        digest.update(file_digest.digest())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument(
        "--approval-manifest",
        type=Path,
        default=None,
        help=(
            "Completed JSON-compatible YAML approval manifest; required only "
            "when a custom allowlist selects paper_assets/"
        ),
    )
    parser.add_argument("--list", action="store_true", help="Print every selected path")
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "Scan resolved source files without claiming that a complete public "
            "snapshot has been verified"
        ),
    )
    boundary.add_argument(
        "--snapshot-root",
        type=Path,
        help=(
            "Audit an already materialized directory against the reviewed source "
            "allowlist, including complete-tree and byte-equality checks"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowlist_path = args.allowlist.resolve()
    patterns = read_allowlist(allowlist_path)
    source_paths, source_missing = expand_allowlist(patterns, PROJECT_ROOT)
    findings = [
        f"missing required public source path: {path}" for path in source_missing
    ]
    findings.extend(audit_paths(source_paths, PROJECT_ROOT))
    findings.extend(
        audit_asset_approvals(
            source_paths,
            PROJECT_ROOT,
            manifest_path=args.approval_manifest,
        )
    )

    audit_root = PROJECT_ROOT
    paths = source_paths
    if args.snapshot_root is not None:
        audit_root = args.snapshot_root.resolve()
        if not audit_root.is_dir():
            findings.append(f"snapshot root is not a directory: {audit_root}")
            paths = []
        else:
            paths, missing_exact = expand_allowlist(patterns, audit_root)
            findings.extend(
                f"missing required public snapshot path: {path}"
                for path in missing_exact
            )
            findings.extend(audit_paths(paths, audit_root))
            findings.extend(audit_materialized_boundary(paths, audit_root))
            findings.extend(
                audit_snapshot_content(source_paths, PROJECT_ROOT, audit_root)
            )
    elif not args.selection_only:
        findings.extend(audit_tracked_boundary(paths, audit_root))

    if args.list:
        for path in paths:
            print(path.relative_to(audit_root).as_posix())
    if findings:
        print("Public snapshot audit failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    if args.selection_only:
        print(
            "Public source selection audit passed: "
            f"{len(paths)} allowlisted files; "
            f"selection_sha256={snapshot_tree_sha256(paths, audit_root)}; "
            "no complete snapshot was claimed"
        )
    else:
        print(
            f"Public snapshot audit passed: {len(paths)} allowlisted files; "
            f"tree_sha256={snapshot_tree_sha256(paths, audit_root)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

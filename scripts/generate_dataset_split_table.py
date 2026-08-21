#!/usr/bin/env python3
"""Generate the authenticated descriptive class-balance table bundle.

This utility reads only the frozen split/index metadata and authoritative
lossless target masks.  It never reads predictions, checkpoints, metrics, or
model-selection artifacts, and its output is descriptive dataset accounting
only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("paper_assets/tables/dataset_split_class_summary")
GENERATOR_SCHEMA = "geoscience.dataset_split_class_summary"
GENERATOR_VERSION = 1
GENERATOR_RELATIVE_PATH = Path("scripts/generate_dataset_split_table.py")
OUTPUT_FILENAMES = (
    "dataset_split_class_summary.csv",
    "dataset_split_class_summary_rows.tex",
    "dataset_split_class_summary_manifest.json",
)
SPLIT_ORDER = (
    ("train", "Training"),
    ("val", "Validation"),
    ("test", "Locked retrospective"),
)
SOURCE_TO_CLASS = {0: "C0", 1: "C1", 255: "C2"}
MASK_AGGREGATE_ALGORITHM = (
    "sha256 over lexicographically sorted UTF-8 relative filename, NUL, "
    "raw file bytes, NUL"
)


@dataclass(frozen=True)
class DatasetContract:
    split_manifest: Path
    annotation_index: Path
    mask_directory: Path
    split_manifest_sha256: str
    annotation_index_sha256: str
    mask_aggregate_sha256: str
    image_count: int


CANONICAL_CONTRACT = DatasetContract(
    split_manifest=Path("config/confirmatory_splits.json"),
    annotation_index=Path("results/step3_coco_dataset/pore_annotations.json"),
    mask_directory=Path(
        "results/step2_pore_classification/pore_classifications"
    ),
    split_manifest_sha256=(
        "4f274f0eced3dfb096ff2e49fe2b9cac3901664fef2bf6da6943ccabad64c703"
    ),
    annotation_index_sha256=(
        "fd79f8820b44ed1e8eed880be699f7d3428fcb18907ab8627d52ba4f0b1c471a"
    ),
    mask_aggregate_sha256=(
        "36e6e8fde61fe8dc6dd75c74d8a098a4c215dbd84ddb1d132f9c2d4dce6dd830"
    ),
    image_count=100,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"JSON document contains duplicate key {key!r}")
        document[key] = value
    return document


def _load_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_relative_path(relative: Path, *, label: str) -> None:
    rendered = relative.as_posix()
    if (
        not rendered
        or relative == Path(".")
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in rendered
    ):
        raise ValueError(f"Unsafe {label} path: {rendered!r}")


def _require_path(
    project_root: Path,
    relative: Path,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    _validate_relative_path(relative, label=label)
    root = project_root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Project root must be an existing non-symlink directory")

    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{label} path contains a symbolic link: {relative}")

    try:
        cursor.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{label} is absent or escapes the project root") from error

    expected = cursor.is_dir() if directory else cursor.is_file()
    if not expected:
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{label} must be an existing {kind}: {relative}")
    return cursor


def _authenticated_json(
    project_root: Path,
    relative: Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    path = _require_path(project_root, relative, label=label)
    payload = _read_bytes(path)
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return _load_json(payload, label=label), payload


def _annotation_images(
    annotation: Mapping[str, Any], *, expected_count: int
) -> dict[int, dict[str, Any]]:
    values = annotation.get("images")
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(
            f"Annotation index must contain exactly {expected_count} image records"
        )

    images: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Every annotation image record must be an object")
        image_id = value.get("id")
        file_name = value.get("file_name")
        width = value.get("width")
        height = value.get("height")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError("Annotation image IDs must be integers")
        if image_id in images:
            raise ValueError(f"Duplicate annotation image ID: {image_id}")
        if not isinstance(file_name, str):
            raise ValueError(f"Annotation image {image_id} has no string filename")
        relative_name = Path(file_name)
        _validate_relative_path(relative_name, label="annotation filename")
        if relative_name.parent != Path(".") or relative_name.suffix.lower() != ".png":
            raise ValueError(
                f"Annotation filename must be a top-level PNG basename: {file_name!r}"
            )
        if file_name in names:
            raise ValueError(f"Duplicate annotation filename: {file_name}")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ValueError(f"Annotation image {image_id} has invalid dimensions")
        images[image_id] = {
            "file_name": file_name,
            "width": width,
            "height": height,
        }
        names.add(file_name)
    return images


def _resolved_splits(
    manifest: Mapping[str, Any], images: Mapping[int, Mapping[str, Any]]
) -> dict[str, list[int]]:
    resolved: dict[str, list[int]] = {}
    memberships: dict[int, str] = {}
    known_ids = set(images)
    for split_name, _ in SPLIT_ORDER:
        values = manifest.get(split_name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Split {split_name!r} must be a non-empty list")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"Split {split_name!r} must contain only integer image IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"Split {split_name!r} contains duplicate image IDs")
        unknown = sorted(set(values) - known_ids)
        if unknown:
            raise ValueError(f"Split {split_name!r} contains unknown IDs: {unknown}")
        for image_id in values:
            if image_id in memberships:
                raise ValueError(
                    f"Image ID {image_id} occurs in both {memberships[image_id]} "
                    f"and {split_name}"
                )
            memberships[image_id] = split_name
        resolved[split_name] = list(values)
    missing = sorted(known_ids - set(memberships))
    if missing:
        raise ValueError(f"Split manifest leaves image IDs unassigned: {missing}")
    return resolved


def _safe_mask_paths(
    mask_directory: Path, images: Mapping[int, Mapping[str, Any]]
) -> dict[int, Path]:
    expected_names = {str(value["file_name"]) for value in images.values()}
    entries = list(mask_directory.iterdir())
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"Mask corpus contains a symbolic link: {entry.name}")
        if not entry.is_file():
            raise ValueError(f"Mask corpus contains a non-file entry: {entry.name}")
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"Mask corpus filenames do not exactly match the annotation index; "
            f"missing={missing}, extra={extra}"
        )
    return {
        image_id: mask_directory / str(image["file_name"])
        for image_id, image in images.items()
    }


def _authenticate_mask_bytes(
    paths: Mapping[int, Path],
    images: Mapping[int, Mapping[str, Any]],
    expected_sha256: str,
) -> dict[int, bytes]:
    id_by_name = {
        str(image["file_name"]): image_id for image_id, image in images.items()
    }
    payloads: dict[int, bytes] = {}
    digest = hashlib.sha256()
    for name in sorted(id_by_name):
        image_id = id_by_name[name]
        path = paths[image_id]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unsafe or absent mask file: {name}")
        payload = _read_bytes(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        payloads[image_id] = payload
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            "Mask aggregate SHA-256 mismatch: "
            f"expected {expected_sha256}, got {observed}"
        )
    return payloads


def _mask_histogram(
    payload: bytes, image: Mapping[str, Any], *, file_name: str
) -> dict[int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format != "PNG":
                raise ValueError(f"Authoritative mask is not a PNG: {file_name}")
            array = np.asarray(source)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Failed to decode authoritative mask: {file_name}") from error
    expected_shape = (int(image["height"]), int(image["width"]))
    if array.ndim != 2 or tuple(array.shape) != expected_shape:
        raise ValueError(
            f"Mask {file_name} must be single-channel with shape {expected_shape}; "
            f"got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"Mask {file_name} must decode as uint8; got {array.dtype}")
    values, counts = np.unique(array, return_counts=True)
    histogram = {int(value): int(count) for value, count in zip(values, counts)}
    invalid = sorted(set(histogram) - set(SOURCE_TO_CLASS))
    if invalid:
        raise ValueError(
            f"Mask {file_name} contains invalid values {invalid}; allowed values are "
            "[0, 1, 255]"
        )
    return {value: histogram.get(value, 0) for value in SOURCE_TO_CLASS}


def _percentage(count: int, total: int, places: int) -> str:
    if total <= 0:
        raise ValueError("Cannot calculate a percentage for zero pixels")
    quantum = Decimal(1).scaleb(-places)
    value = (Decimal(count) * Decimal(100) / Decimal(total)).quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    return f"{value:.{places}f}"


def build_summary(
    project_root: Path = PROJECT_ROOT,
    contract: DatasetContract = CANONICAL_CONTRACT,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Authenticate canonical inputs, then count target-mask class values."""
    split_document, split_payload = _authenticated_json(
        project_root,
        contract.split_manifest,
        contract.split_manifest_sha256,
        label="Split manifest",
    )
    annotation_document, annotation_payload = _authenticated_json(
        project_root,
        contract.annotation_index,
        contract.annotation_index_sha256,
        label="Annotation index",
    )
    images = _annotation_images(
        annotation_document, expected_count=contract.image_count
    )
    splits = _resolved_splits(split_document, images)
    mask_directory = _require_path(
        project_root,
        contract.mask_directory,
        label="Mask directory",
        directory=True,
    )
    mask_paths = _safe_mask_paths(mask_directory, images)

    # Pixel values are decoded only after all byte-level provenance is authentic.
    mask_payloads = _authenticate_mask_bytes(
        mask_paths, images, contract.mask_aggregate_sha256
    )

    rows: list[dict[str, Any]] = []
    totals = {0: 0, 1: 0, 255: 0}
    total_tiles = 0
    for split_name, display_name in SPLIT_ORDER:
        counts = {0: 0, 1: 0, 255: 0}
        for image_id in splits[split_name]:
            image = images[image_id]
            histogram = _mask_histogram(
                mask_payloads[image_id],
                image,
                file_name=str(image["file_name"]),
            )
            for value in counts:
                counts[value] += histogram[value]
        pixel_total = sum(counts.values())
        row: dict[str, Any] = {
            "split": split_name,
            "display_name": display_name,
            "tile_count": len(splits[split_name]),
            "total_pixels": pixel_total,
        }
        for value, class_name in SOURCE_TO_CLASS.items():
            row[f"{class_name.lower()}_pixels"] = counts[value]
            row[f"{class_name.lower()}_percent"] = _percentage(
                counts[value], pixel_total, 6
            )
            totals[value] += counts[value]
        rows.append(row)
        total_tiles += len(splits[split_name])

    grand_total = sum(totals.values())
    total_row: dict[str, Any] = {
        "split": "total",
        "display_name": "Total",
        "tile_count": total_tiles,
        "total_pixels": grand_total,
    }
    for value, class_name in SOURCE_TO_CLASS.items():
        total_row[f"{class_name.lower()}_pixels"] = totals[value]
        total_row[f"{class_name.lower()}_percent"] = _percentage(
            totals[value], grand_total, 6
        )
    rows.append(total_row)

    provenance = {
        "split_manifest_sha256": _sha256_bytes(split_payload),
        "annotation_index_sha256": _sha256_bytes(annotation_payload),
        "mask_aggregate_sha256": contract.mask_aggregate_sha256,
    }
    return rows, provenance


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "split",
        "tile_count",
        "total_pixels",
        "c0_pixels",
        "c0_percent",
        "c1_pixels",
        "c1_percent",
        "c2_pixels",
        "c2_percent",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue().encode("utf-8")


def _tex_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        "% Generated by scripts/generate_dataset_split_table.py; do not edit.",
    ]
    for index, row in enumerate(rows):
        if index == len(rows) - 1:
            lines.append(r"\midrule")
        percentages = [
            _percentage(int(row[f"c{class_id}_pixels"]), int(row["total_pixels"]), 3)
            for class_id in range(3)
        ]
        lines.append(
            f"{row['display_name']} & {row['tile_count']} & "
            + " & ".join(percentages)
            + r" \\"
        )
    # Keep the closing rule inside the included fragment. Modern LaTeX file hooks can
    # otherwise insert non-alignment tokens between a returned \input and a following
    # \bottomrule in the parent tabular.
    lines.append(r"\bottomrule")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_bundle(
    project_root: Path = PROJECT_ROOT,
    contract: DatasetContract = CANONICAL_CONTRACT,
) -> dict[str, bytes]:
    rows, provenance = build_summary(project_root, contract)
    csv_payload = _csv_bytes(rows)
    tex_payload = _tex_bytes(rows)
    generator_path = _require_path(
        project_root, GENERATOR_RELATIVE_PATH, label="Generator script"
    )
    manifest = {
        "schema_version": 1,
        "generator": {
            "schema": GENERATOR_SCHEMA,
            "version": GENERATOR_VERSION,
            "script": {
                "path": GENERATOR_RELATIVE_PATH.as_posix(),
                "sha256": _sha256_bytes(_read_bytes(generator_path)),
            },
        },
        "purpose": "descriptive_dataset_accounting_only",
        "prohibited_uses": [
            "model_selection",
            "hyperparameter_tuning",
            "post_processing_tuning",
        ],
        "prediction_or_metric_inputs_read": False,
        "source_value_to_class": {"0": "C0", "1": "C1", "255": "C2"},
        "inputs": {
            "split_manifest": {
                "path": contract.split_manifest.as_posix(),
                "sha256": provenance["split_manifest_sha256"],
            },
            "annotation_index": {
                "path": contract.annotation_index.as_posix(),
                "role": "image ID, filename, and dimensions only",
                "sha256": provenance["annotation_index_sha256"],
            },
            "mask_directory": {
                "path": contract.mask_directory.as_posix(),
                "image_count": contract.image_count,
                "aggregate_sha256": provenance["mask_aggregate_sha256"],
                "aggregate_sha256_algorithm": MASK_AGGREGATE_ALGORITHM,
            },
        },
        "outputs": {
            OUTPUT_FILENAMES[0]: {"sha256": _sha256_bytes(csv_payload)},
            OUTPUT_FILENAMES[1]: {"sha256": _sha256_bytes(tex_payload)},
        },
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {
        OUTPUT_FILENAMES[0]: csv_payload,
        OUTPUT_FILENAMES[1]: tex_payload,
        OUTPUT_FILENAMES[2]: manifest_payload,
    }


def _output_path(project_root: Path, relative: Path) -> Path:
    _validate_relative_path(relative, label="output directory")
    required_prefix = Path("paper_assets/tables")
    if relative == required_prefix or required_prefix not in relative.parents:
        raise ValueError("Output directory must be below paper_assets/tables")
    root = project_root.absolute()
    parent = root
    for component in relative.parent.parts:
        parent = parent / component
        if parent.is_symlink():
            raise ValueError("Output parent path contains a symbolic link")
        if parent.exists() and not parent.is_dir():
            raise ValueError("Output parent path contains a non-directory")
        parent.mkdir(exist_ok=True)
    output = root / relative
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {relative}")
    return output


def write_bundle(
    output_directory: Path = DEFAULT_OUTPUT_DIR,
    *,
    project_root: Path = PROJECT_ROOT,
    contract: DatasetContract = CANONICAL_CONTRACT,
) -> tuple[Path, dict[str, str]]:
    payloads = render_bundle(project_root, contract)
    output = _output_path(project_root, output_directory)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        for name in OUTPUT_FILENAMES:
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output_directory}"
            )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output, {name: _sha256_bytes(payload) for name, payload in payloads.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an authenticated descriptive dataset split table bundle"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "New repository-relative directory below paper_assets/tables; "
            "existing paths are never overwritten"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output, hashes = write_bundle(args.output_dir)
    report = {
        "output_directory": output.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

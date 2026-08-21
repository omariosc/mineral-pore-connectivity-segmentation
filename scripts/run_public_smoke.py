#!/usr/bin/env python3
"""Generate and validate a tiny synthetic lossless-mask dataset.

The smoke corpus is created from deterministic arrays and contains no research
microscopy.  It checks the same split resolver, mask-directory validator, and
mask-value loader used by the training path.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.data_contract import (  # noqa: E402
    CANONICAL_CLASS_NAMES,
    load_lossless_mask,
    resolve_split_manifest,
    validate_lossless_mask_directory,
)


SMOKE_FILENAMES = (
    "synthetic_train.png",
    "synthetic_validation.png",
    "synthetic_test.png",
)


def _synthetic_image(index: int, size: int = 9) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.uint16)
    return ((17 * x + 29 * y + 31 * index) % 256).astype(np.uint8)


def _synthetic_mask(index: int, size: int = 9) -> np.ndarray:
    y, x = np.indices((size, size))
    mask = np.full((size, size), 255, dtype=np.uint8)
    mask[(x + y + index) % 7 == 0] = 0
    mask[(2 * x + y + index) % 5 == 0] = 1
    return mask


def build_synthetic_smoke_dataset(output_dir: Path) -> Dict[str, Any]:
    """Create the smoke corpus, validate it, and return its report."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for image_id, file_name in enumerate(SMOKE_FILENAMES, start=1):
        Image.fromarray(_synthetic_image(image_id)).save(images_dir / file_name)
        Image.fromarray(_synthetic_mask(image_id)).save(masks_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": 9,
                "height": 9,
            }
        )

    coco_data = {
        "info": {
            "description": "Deterministic synthetic smoke corpus; not research data",
            "version": "1",
        },
        "images": images,
        "categories": [
            {"id": class_id, "name": class_name}
            for class_id, class_name in CANONICAL_CLASS_NAMES.items()
        ],
        "annotations": [],
    }
    split_manifest = {
        "_provenance": {
            "status": "synthetic_smoke_only",
            "split_unit": "one generated image",
        },
        "train": [SMOKE_FILENAMES[0]],
        "val": [SMOKE_FILENAMES[1]],
        "test": [SMOKE_FILENAMES[2]],
    }

    annotations_path = output_dir / "annotations.json"
    split_path = output_dir / "split_manifest.json"
    annotations_path.write_text(
        json.dumps(coco_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    split_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resolved_splits = resolve_split_manifest(coco_data, split_path)
    mask_paths, provenance = validate_lossless_mask_directory(
        coco_data, images_dir, masks_dir
    )

    class_counts = {str(class_id): 0 for class_id in CANONICAL_CLASS_NAMES}
    for image_id in sorted(mask_paths):
        target = load_lossless_mask(mask_paths[image_id], (9, 9), num_classes=3)
        observed = set(int(value) for value in np.unique(target))
        if observed != set(CANONICAL_CLASS_NAMES):
            raise RuntimeError(
                f"Synthetic target {image_id} has classes {sorted(observed)}, expected [0, 1, 2]"
            )
        for class_id in CANONICAL_CLASS_NAMES:
            class_counts[str(class_id)] += int(np.count_nonzero(target == class_id))

    split_sets = [set(resolved_splits[name]) for name in ("train", "val", "test")]
    if any(
        split_sets[left] & split_sets[right]
        for left in range(len(split_sets))
        for right in range(left + 1, len(split_sets))
    ):
        raise RuntimeError("Synthetic split validation unexpectedly allowed leakage")

    report = {
        "status": "passed",
        "data_classification": "synthetic_smoke_only_not_scientific_evidence",
        "image_count": len(images),
        "image_shape": [9, 9],
        "resolved_splits": resolved_splits,
        "source_mask_values": [0, 1, 255],
        "mapped_target_values": [0, 1, 2],
        "mapped_class_pixel_counts": class_counts,
        "mask_aggregate_sha256": provenance["mask_aggregate_sha256"],
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public synthetic data-contract smoke test"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory in which to retain the generated smoke corpus",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="mineral-pore-smoke-") as temp_dir:
            report = build_synthetic_smoke_dataset(Path(temp_dir))
    else:
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            raise SystemExit(f"Output directory already exists: {output_dir}")
        report = build_synthetic_smoke_dataset(output_dir)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Dependency-light contracts for confirmatory images, masks, and splits.

This module deliberately avoids importing PyTorch or the augmentation stack so
the public data contract can be checked in a small CPU-only smoke test.  The
training dataset imports the same functions, keeping the smoke test and the
executed loader on one implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


CANONICAL_CLASS_NAMES = {
    0: "disconnected_pore",
    1: "connected_pore",
    2: "mineral",
}

LOSSLESS_MASK_SOURCE_VALUES = frozenset({0, 1, 255})
INPUT_NORMALIZATION = {
    "name": "grayscale_uint8_to_minus_one_one",
    "input_channels": 1,
    "source_dtype": "uint8",
    "scale_divisor": 255.0,
    "mean": [0.5],
    "std": [0.5],
    "output_range": [-1.0, 1.0],
}

SplitManifest = Union[str, Path, Mapping[str, Sequence[Union[int, str]]]]

CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS = {
    "train": {
        "image_count": 74,
        "image_aggregate_sha256": (
            "88e8392fedbb1309e779acc28693010475c41065bdff17b5a4dd8c26f3200079"
        ),
    },
    "train_plus_validation": {
        "image_count": 79,
        "image_aggregate_sha256": (
            "f5e594e1589c75964af66ec0ddd1e73270c4cb78b447d29014869d696d4ed13d"
        ),
    },
}
CONFIRMATORY_ANNOTATION_SHA256 = (
    "fd79f8820b44ed1e8eed880be699f7d3428fcb18907ab8627d52ba4f0b1c471a"
)
CONFIRMATORY_SPLIT_MANIFEST_SHA256 = (
    "4f274f0eced3dfb096ff2e49fe2b9cac3901664fef2bf6da6943ccabad64c703"
)
CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS = {
    "train": {
        "mask_count": 74,
        "mask_aggregate_sha256": (
            "659ff82b0730b6d6062437170749eeb98c65f10ef42d9287202b2551147239e9"
        ),
    },
    "train_plus_validation": {
        "mask_count": 79,
        "mask_aggregate_sha256": (
            "7f22592baf583d88da6d77658b2898eb1448fc2603fd7d3e3f589a91cb0d5de9"
        ),
    },
}


def aggregate_indexed_file_bytes(
    root: Union[str, Path],
    relative_names: Sequence[str],
    *,
    scope: str,
    split_names: Sequence[str],
) -> Dict[str, Any]:
    """Hash a named subset without traversing or opening any other files."""
    declared_root = Path(root).resolve()
    names = sorted(str(value) for value in relative_names)
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{scope} requires a non-empty list of unique files")
    digest = hashlib.sha256()
    for name in names:
        relative = Path(name)
        if (
            not name
            or relative == Path(".")
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
        ):
            raise ValueError(f"Unsafe indexed file name: {name!r}")
        path = (declared_root / relative).resolve()
        try:
            path.relative_to(declared_root)
        except ValueError as error:
            raise ValueError(
                f"Indexed file escapes its declared directory: {name!r}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"Indexed file does not exist: {name}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "scope": str(scope),
        "split_names": [str(value) for value in split_names],
        "image_count": len(names),
        "image_aggregate_sha256": digest.hexdigest(),
        "image_aggregate_sha256_algorithm": (
            "sha256 over lexicographically sorted UTF-8 relative filename, "
            "NUL, raw file bytes, NUL"
        ),
        "file_name_list_sha256": hashlib.sha256(
            json.dumps(names, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _read_single_channel_mask(mask_path: Path) -> np.ndarray:
    """Read a mask without silently converting a colour image to grayscale."""
    try:
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
    except Exception as exc:
        raise ValueError(f"Failed to load authoritative mask: {mask_path}") from exc

    if mask.ndim != 2:
        raise ValueError(
            f"Authoritative mask must be single-channel: {mask_path} has shape {mask.shape}"
        )
    return mask


def load_lossless_mask(
    mask_path: Union[str, Path],
    expected_shape: Tuple[int, int],
    *,
    num_classes: int = 3,
    mineral_class_id: int = 2,
    ignore_index: int = -100,
) -> np.ndarray:
    """Load and map one authoritative PNG mask to model target IDs.

    Source values are fixed at 0 (disconnected pore), 1 (connected pore), and
    255 (mineral in three-class mode or ignored in two-class mode).
    """
    if num_classes not in (2, 3):
        raise ValueError(f"Lossless masks support 2 or 3 classes, got {num_classes}")

    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Authoritative lossless mask does not exist: {path}")

    mask = _read_single_channel_mask(path)
    if tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"Image/mask shape mismatch for {path.name}: expected {expected_shape}, "
            f"mask {mask.shape}"
        )

    observed_values = {int(value) for value in np.unique(mask)}
    invalid_values = sorted(observed_values - LOSSLESS_MASK_SOURCE_VALUES)
    if invalid_values:
        raise ValueError(
            f"Authoritative mask {path} contains invalid values {invalid_values}; "
            "allowed values are [0, 1, 255]"
        )

    target_dtype = np.uint8 if num_classes == 3 else np.int64
    target = mask.astype(target_dtype, copy=True)
    target[target == 255] = mineral_class_id if num_classes == 3 else ignore_index
    return target


def validate_lossless_mask_directory(
    coco_data: Mapping[str, Any],
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    image_ids: Optional[Sequence[int]] = None,
) -> Tuple[Dict[int, Path], Dict[str, Any]]:
    """Validate authoritative lossless masks and return deterministic provenance."""
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Lossless mask directory does not exist: {mask_root}")

    images_by_id = {
        int(image["id"]): image for image in coco_data.get("images", [])
    }
    selected_ids = (
        sorted(images_by_id)
        if image_ids is None
        else [int(image_id) for image_id in image_ids]
    )
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Mask validation image IDs must be unique")
    unknown_ids = sorted(set(selected_ids) - set(images_by_id))
    if unknown_ids:
        raise ValueError(
            "Mask validation received image IDs absent from COCO data: "
            + ", ".join(map(str, unknown_ids))
        )

    ordered_images = sorted(
        (images_by_id[image_id] for image_id in selected_ids),
        key=lambda image: str(image["file_name"]),
    )
    mask_paths: Dict[int, Path] = {}
    aggregate_digest = hashlib.sha256()

    for image_info in ordered_images:
        image_id = int(image_info["id"])
        file_name = str(image_info["file_name"])
        relative_name = Path(file_name)
        if (
            not file_name
            or relative_name == Path(".")
            or relative_name.is_absolute()
            or ".." in relative_name.parts
            or "\\" in file_name
        ):
            raise ValueError(f"Unsafe COCO image file name: {file_name!r}")

        image_path = image_root / relative_name
        mask_path = mask_root / relative_name
        for declared_root, candidate in (
            (image_root, image_path),
            (mask_root, mask_path),
        ):
            try:
                candidate.resolve().relative_to(declared_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"COCO image file escapes its declared directory: {file_name!r}"
                ) from exc
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image does not exist: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Authoritative lossless mask does not exist: {mask_path}"
            )

        try:
            with Image.open(image_path) as image:
                image_shape = (image.height, image.width)
                image.verify()
        except Exception as exc:
            raise ValueError(f"Failed to load source image: {image_path}") from exc

        mask = _read_single_channel_mask(mask_path)
        expected_shape = (
            int(image_info.get("height", image_shape[0])),
            int(image_info.get("width", image_shape[1])),
        )
        if image_shape != expected_shape:
            raise ValueError(
                f"COCO metadata/image shape mismatch for {image_path}: "
                f"metadata {expected_shape}, image {image_shape}"
            )
        if mask.shape != image_shape:
            raise ValueError(
                f"Image/mask shape mismatch for {relative_name}: "
                f"image {image_shape}, mask {mask.shape}"
            )

        observed_values = {int(value) for value in np.unique(mask)}
        invalid_values = sorted(observed_values - LOSSLESS_MASK_SOURCE_VALUES)
        if invalid_values:
            raise ValueError(
                f"Authoritative mask {mask_path} contains invalid values "
                f"{invalid_values}; allowed values are [0, 1, 255]"
            )

        relative_bytes = relative_name.as_posix().encode("utf-8")
        aggregate_digest.update(relative_bytes)
        aggregate_digest.update(b"\0")
        with mask_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                aggregate_digest.update(chunk)
        aggregate_digest.update(b"\0")
        mask_paths[image_id] = mask_path

    provenance = {
        "target_source": "lossless_png_masks",
        "mask_directory": str(mask_root),
        "mask_count": len(mask_paths),
        "mask_aggregate_sha256": aggregate_digest.hexdigest(),
        "mask_aggregate_sha256_algorithm": (
            "sha256 over lexicographically sorted UTF-8 relative filename, NUL, "
            "raw file bytes, NUL"
        ),
        "validated_source_values": [0, 1, 255],
        "canonical_value_mapping": {
            "0": "0 (disconnected_pore)",
            "1": "1 (connected_pore)",
            "255": "2 (mineral) in three-class mode; ignore_index in two-class mode",
        },
    }
    return mask_paths, provenance


def resolve_split_manifest(
    coco_data: Mapping[str, Any], split_manifest: SplitManifest
) -> Dict[str, List[int]]:
    """Load and validate a train/validation/test image manifest."""
    if isinstance(split_manifest, (str, Path)):
        manifest_path = Path(split_manifest)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = dict(split_manifest)

    required_splits = ("train", "val", "test")
    missing_splits = [name for name in required_splits if name not in manifest]
    if missing_splits:
        raise ValueError(f"Split manifest is missing: {', '.join(missing_splits)}")

    images = list(coco_data.get("images", []))
    known_ids = {int(image["id"]) for image in images}
    if len(known_ids) != len(images):
        raise ValueError("COCO image IDs must be unique")
    id_by_filename = {str(image["file_name"]): int(image["id"]) for image in images}
    if len(id_by_filename) != len(images):
        raise ValueError("COCO image file names must be unique")

    resolved: Dict[str, List[int]] = {}
    for split_name in required_splits:
        values = manifest[split_name]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Split '{split_name}' must be a list of image identifiers")
        image_ids: List[int] = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError(f"Invalid boolean image identifier in split '{split_name}'")
            if isinstance(value, int):
                image_id = value
            elif isinstance(value, str) and value in id_by_filename:
                image_id = id_by_filename[value]
            elif isinstance(value, str) and value.isdigit():
                image_id = int(value)
            else:
                raise ValueError(
                    f"Unknown image identifier {value!r} in split '{split_name}'"
                )
            if image_id not in known_ids:
                raise ValueError(
                    f"Image ID {image_id} in split '{split_name}' is not in the COCO data"
                )
            image_ids.append(image_id)
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"Split '{split_name}' contains duplicate image identifiers")
        resolved[split_name] = image_ids

    memberships: Dict[int, List[str]] = {}
    for split_name, image_ids in resolved.items():
        for image_id in image_ids:
            memberships.setdefault(image_id, []).append(split_name)
    overlaps = {
        image_id: names for image_id, names in memberships.items() if len(names) > 1
    }
    if overlaps:
        detail = ", ".join(
            f"{image_id} ({'/'.join(names)})"
            for image_id, names in sorted(overlaps.items())
        )
        raise ValueError(f"Split manifest contains image leakage: {detail}")

    assigned_ids = set(memberships)
    unassigned_ids = sorted(known_ids - assigned_ids)
    if unassigned_ids:
        raise ValueError(
            "Split manifest does not assign all COCO images; missing IDs: "
            + ", ".join(map(str, unassigned_ids))
        )
    return resolved


def create_deterministic_image_splits(
    coco_data: Mapping[str, Any],
    val_split: float,
    test_split: float,
    seed: int,
) -> Dict[str, List[int]]:
    """Create deterministic, mutually exclusive splits of whole COCO images."""
    if not 0 <= val_split < 1 or not 0 <= test_split < 1:
        raise ValueError("val_split and test_split must be in [0, 1)")
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split must be less than 1")

    image_ids = np.array(
        sorted(int(image["id"]) for image in coco_data.get("images", []))
    )
    if len(image_ids) < 3:
        raise ValueError("At least three images are required for train/val/test splitting")
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)

    n_images = len(image_ids)
    n_test = max(1, int(round(n_images * test_split))) if test_split > 0 else 0
    n_val = max(1, int(round(n_images * val_split))) if val_split > 0 else 0
    if n_val + n_test >= n_images:
        raise ValueError("Split fractions leave no images for training")

    return {
        "train": image_ids[: n_images - n_val - n_test].tolist(),
        "val": image_ids[n_images - n_val - n_test : n_images - n_test].tolist(),
        "test": image_ids[n_images - n_test :].tolist() if n_test else [],
    }

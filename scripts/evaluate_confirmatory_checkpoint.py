#!/usr/bin/env python3
"""Locked evaluation of a validation-selected segmentation checkpoint.

The evaluator deliberately has no thresholding, morphology, or other
post-processing path. It evaluates the ``test`` image IDs in a complete,
disjoint split manifest once, as native 2048 x 2048 tiles. Targets come only
from the lossless step-2 PNG masks: source values 0/1/255 are mapped to the
canonical disconnected-pore/connected-pore/mineral IDs 0/1/2. COCO is used
only as the image-ID and file-name index; its simplified polygons are not used
as confirmatory ground truth.

ROC and precision-recall curves are accumulated into deterministic, fixed-width
probability histograms. Raw per-pixel probabilities are never written to disk or
retained across tiles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.screen_selection import (  # noqa: E402
    EXECUTION_SOURCE_FILES,
    PROSPECTIVE_METHOD_PROTOCOLS,
    SCREEN_CANDIDATE_ORDER,
    SCREEN_CELL_COUNT,
    SCREEN_SEEDS,
    SELECTED_METHOD_LOCK_SCHEMA_VERSION,
    verify_selected_method_lock_document,
)
from src.training.checkpoint_io import load_weights_only_checkpoint  # noqa: E402
from src.training.neural_freeze import (  # noqa: E402
    ARCHITECTURE_ROLES,
    load_verified_neural_freeze_manifest,
)

CLASS_NAMES = {
    0: "disconnected_pore",
    1: "connected_pore",
    2: "mineral",
}
CLASS_LABELS = {
    0: "C0 disconnected pore",
    1: "C1 connected pore",
    2: "C2 mineral",
}
EXPECTED_TILE_SHAPE = (2048, 2048)
EXPECTED_INPUT_NORMALIZATION = {
    "name": "grayscale_uint8_to_minus_one_one",
    "input_channels": 1,
    "source_dtype": "uint8",
    "scale_divisor": 255.0,
    "mean": [0.5],
    "std": [0.5],
    "output_range": [-1.0, 1.0],
}
NORMALIZATION_ID = EXPECTED_INPUT_NORMALIZATION["name"]
NORMALIZATION_FORMULA = "(uint8 / 255.0 - 0.5) / 0.5"
CHECKPOINT_ROLE = "validation_composite_selection"
SELECTION_METRIC_NAME = "validation_c0_c1_iou_harmonic_mean"
LOCKED_SPLIT_MANIFEST_SHA256 = (
    "4f274f0eced3dfb096ff2e49fe2b9cac3901664fef2bf6da6943ccabad64c703"
)
LOCKED_TARGET_ATTESTATIONS = {
    "development_train_plus_validation": {
        "split_names": ["train", "val"],
        "mask_count": 79,
        "mask_aggregate_sha256": (
            "7f22592baf583d88da6d77658b2898eb1448fc2603fd7d3e3f589a91cb0d5de9"
        ),
    },
    "held_out_test": {
        "split_names": ["test"],
        "mask_count": 21,
        "mask_aggregate_sha256": (
            "18d457d036294479fbf404a649487a9c0d61bec5e804883a1dda977c854922a2"
        ),
    },
    "full_indexed_corpus": {
        "split_names": ["train", "val", "test"],
        "mask_count": 100,
        "mask_aggregate_sha256": (
            "36e6e8fde61fe8dc6dd75c74d8a098a4c215dbd84ddb1d132f9c2d4dce6dd830"
        ),
    },
}
LOCKED_INPUT_ATTESTATIONS = {
    "training_only": {
        "split_names": ["train"],
        "image_count": 74,
        "image_aggregate_sha256": (
            "88e8392fedbb1309e779acc28693010475c41065bdff17b5a4dd8c26f3200079"
        ),
    },
    "validation_only": {
        "split_names": ["val"],
        "image_count": 5,
        "image_aggregate_sha256": (
            "c5a3854996ff59762918d8bf9b422408d90974fcea4c32f086834acb2198442c"
        ),
    },
    "development_train_plus_validation": {
        "split_names": ["train", "val"],
        "image_count": 79,
        "image_aggregate_sha256": (
            "f5e594e1589c75964af66ec0ddd1e73270c4cb78b447d29014869d696d4ed13d"
        ),
    },
    "held_out_test": {
        "split_names": ["test"],
        "image_count": 21,
        "image_aggregate_sha256": (
            "713bbc4fdae22b393237e9c8f2384262693a5ade4c319443246890dbdda057ea"
        ),
    },
    "full_indexed_corpus": {
        "split_names": ["train", "val", "test"],
        "image_count": 100,
        "image_aggregate_sha256": (
            "05ab6ab685a37015f9dd43294b10dc1c044fe3b104e08d3076f1ab333c265992"
        ),
    },
}
CONDITIONAL_CANDIDATES = frozenset({"C2-P", "C2-F", "C2-FP"})
CONDITIONAL_PORE_THRESHOLD_UINT8 = 100
LOCKED_INFERENCE_PRECISION = "cuda_float16_autocast"
LOCKED_CUDA_DEVICE_MODEL_TOKEN = "L40S"
LOCKED_EVALUATOR_INFERENCE_SEED = 0
LOCKED_BOOTSTRAP_SEED = 20260820
LOCKED_BOOTSTRAP_REPLICATES = 5000
LOCKED_CONFIDENCE = 0.95
LOCKED_CURVE_BINS = 4096
LOCKED_ANNOTATION_INDEX_SHA256 = (
    "fd79f8820b44ed1e8eed880be699f7d3428fcb18907ab8627d52ba4f0b1c471a"
)
CANONICAL_SELECTED_RETRAINING_ROOT = (
    "results/patch_training/protocol_runs/selected_winner_retraining"
)
CANONICAL_LOCKED_EVALUATION_ROOT = "results/confirmatory_evaluation/locked"
PUBLICATION_CLASS_COLORS = ("#B33A3A", "#2E8B57", "#4C78A8")
PUBLICATION_CLASS_LINE_STYLES = ("-", "--", "-.")
PUBLICATION_CLASS_MARKERS = ("o", "s", "^")
PUBLICATION_CURVE_ORDER = ("precision_recall", "roc")
PUBLICATION_SANS_SERIF_FONTS = (
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "DejaVu Sans",
)
# Kept as an alias for callers that previously imported the evaluator palette.
PALETTE = PUBLICATION_CLASS_COLORS


def _project_path(value: str | Path) -> Path:
    """Resolve a repository-relative path without relying on the shell cwd."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _repository_relative(path: Path) -> str:
    """Return a public-safe repository-relative provenance path."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Path is outside the repository: {path}") from error


def _safe_indexed_path(root: Path, file_name: Any) -> Path:
    """Resolve a COCO-indexed file while rejecting traversal and escaping links."""
    relative = Path(str(file_name))
    if (
        not str(file_name)
        or relative == Path(".")
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in str(file_name)
    ):
        raise ValueError(f"Unsafe COCO image file name: {file_name!r}")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"COCO image file escapes its declared directory: {file_name!r}") from error
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_complete_manifest(
    coco: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Dict[str, List[int]]:
    """Resolve and validate a complete, leakage-free train/val/test manifest."""
    required = ("train", "val", "test")
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"Split manifest is missing: {', '.join(missing)}")

    images = list(coco.get("images", []))
    known_ids = {int(image["id"]) for image in images}
    id_by_name = {str(image["file_name"]): int(image["id"]) for image in images}
    if len(id_by_name) != len(images):
        raise ValueError("COCO image file names must be unique")

    resolved: Dict[str, List[int]] = {}
    for split_name in required:
        values = manifest[split_name]
        if not isinstance(values, list):
            raise ValueError(f"Split '{split_name}' must be a JSON list")
        ids: List[int] = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError(f"Invalid boolean image identifier in '{split_name}'")
            if isinstance(value, int):
                image_id = value
            elif isinstance(value, str) and value in id_by_name:
                image_id = id_by_name[value]
            elif isinstance(value, str) and value.isdigit():
                image_id = int(value)
            else:
                raise ValueError(f"Unknown image identifier {value!r} in '{split_name}'")
            if image_id not in known_ids:
                raise ValueError(f"Image ID {image_id} in '{split_name}' is absent from COCO")
            ids.append(image_id)
        if len(ids) != len(set(ids)):
            raise ValueError(f"Split '{split_name}' contains duplicate image IDs")
        resolved[split_name] = ids

    memberships: Dict[int, List[str]] = {}
    for split_name, image_ids in resolved.items():
        for image_id in image_ids:
            memberships.setdefault(image_id, []).append(split_name)
    overlaps = {key: value for key, value in memberships.items() if len(value) > 1}
    if overlaps:
        detail = ", ".join(
            f"{image_id} ({'/'.join(names)})"
            for image_id, names in sorted(overlaps.items())
        )
        raise ValueError(f"Split manifest contains image leakage: {detail}")
    missing_ids = sorted(known_ids - set(memberships))
    if missing_ids:
        raise ValueError(
            "Split manifest does not assign every COCO image; missing IDs: "
            + ", ".join(map(str, missing_ids))
        )
    if not resolved["test"]:
        raise ValueError("The held-out test split is empty")
    return resolved


def mask_corpus_sha256(mask_paths: Sequence[Path], mask_root: Path) -> str:
    """Match the training loader's deterministic lossless-mask corpus hash."""
    digest = hashlib.sha256()
    for path in sorted(mask_paths, key=lambda item: item.relative_to(mask_root).as_posix()):
        relative_name = path.relative_to(mask_root).as_posix()
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _update_indexed_corpus_digest(
    digest: "hashlib._Hash", relative_name: str, payload: bytes
) -> None:
    digest.update(relative_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")


def _validate_indexed_names(
    corpus_root: Path,
    split_files: Mapping[str, Sequence[str]],
    *,
    corpus_label: str,
) -> Dict[str, str]:
    """Validate filenames and return relative-name to split membership."""
    required_splits = ("train", "val", "test")
    if set(split_files) != set(required_splits):
        raise ValueError(
            f"{corpus_label} attestation requires exact train/val/test file lists"
        )
    membership: Dict[str, str] = {}
    for split_name in required_splits:
        for value in split_files[split_name]:
            relative_name = Path(str(value)).as_posix()
            _safe_indexed_path(corpus_root, relative_name)
            if relative_name in membership:
                raise ValueError(
                    f"{corpus_label} file {relative_name!r} appears in multiple split lists"
                )
            membership[relative_name] = split_name

    indexed_names = set(membership)
    actual_names = {
        path.relative_to(corpus_root).as_posix()
        for path in corpus_root.rglob("*.png")
        if path.is_file()
    }
    missing = sorted(indexed_names - actual_names)
    unexpected = sorted(actual_names - indexed_names)
    if missing or unexpected:
        raise ValueError(
            f"{corpus_label} corpus does not match the locked image index; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return membership


def _attestation_record(
    scope: str,
    split_names: Sequence[str],
    names: Sequence[str],
    digest: "hashlib._Hash",
    expectation: Mapping[str, Any],
    *,
    count_key: str,
    aggregate_key: str,
    algorithm_key: str,
    corpus_label: str,
) -> Dict[str, Any]:
    observed_count = len(names)
    observed_hash = digest.hexdigest()
    expected_count = int(expectation[count_key])
    expected_hash = str(expectation[aggregate_key])
    if observed_count != expected_count or observed_hash != expected_hash:
        raise ValueError(
            f"Locked {scope} {corpus_label} attestation mismatch: "
            f"count {observed_count} != {expected_count} or "
            f"SHA-256 {observed_hash} != {expected_hash}"
        )
    return {
        "scope": scope,
        "split_names": list(split_names),
        count_key: observed_count,
        aggregate_key: observed_hash,
        algorithm_key: (
            "sha256 over lexicographically sorted UTF-8 relative filename, "
            "NUL, raw file bytes, NUL"
        ),
        "file_name_list_sha256": hashlib.sha256(
            json.dumps(list(names), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "verification_status": "matched_locked_attestation",
    }


def _attest_development_corpus_once(
    corpus_root: Path,
    split_files: Mapping[str, Sequence[str]],
    *,
    expected_attestations: Mapping[str, Mapping[str, Any]],
    count_key: str,
    aggregate_key: str,
    algorithm_key: str,
    corpus_label: str,
    read_bytes: Optional[Callable[[Path], bytes]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bytes]]:
    """Read development bytes only, returning attestation records and a cache."""
    membership = _validate_indexed_names(
        corpus_root, split_files, corpus_label=corpus_label
    )
    development_names = sorted(
        name for name, split_name in membership.items() if split_name != "test"
    )
    reader = read_bytes or (lambda path: path.read_bytes())
    payloads: Dict[str, bytes] = {}
    for relative_name in development_names:
        payload = reader(_safe_indexed_path(corpus_root, relative_name))
        if not isinstance(payload, bytes):
            raise TypeError(f"{corpus_label} byte reader must return bytes")
        payloads[relative_name] = payload

    records: Dict[str, Dict[str, Any]] = {}
    scopes = {
        "development_train_plus_validation": ("train", "val"),
        "training_only": ("train",),
        "validation_only": ("val",),
    }
    for scope, scope_splits in scopes.items():
        if scope not in expected_attestations:
            continue
        names = sorted(
            name for name, split_name in membership.items() if split_name in scope_splits
        )
        digest = hashlib.sha256()
        for relative_name in names:
            _update_indexed_corpus_digest(digest, relative_name, payloads[relative_name])
        records[scope] = _attestation_record(
            scope,
            scope_splits,
            names,
            digest,
            expected_attestations[scope],
            count_key=count_key,
            aggregate_key=aggregate_key,
            algorithm_key=algorithm_key,
            corpus_label=corpus_label,
        )
        records[scope]["held_out_bytes_read"] = 0
    return records, payloads


def _attest_held_out_and_full_corpus_once(
    corpus_root: Path,
    split_files: Mapping[str, Sequence[str]],
    development_payloads: Mapping[str, bytes],
    *,
    expected_attestations: Mapping[str, Mapping[str, Any]],
    count_key: str,
    aggregate_key: str,
    algorithm_key: str,
    corpus_label: str,
    read_bytes: Optional[Callable[[Path], bytes]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bytes], Dict[str, str]]:
    """After reservation, read each held-out file once and finish attestations."""
    membership = _validate_indexed_names(
        corpus_root, split_files, corpus_label=corpus_label
    )
    development_names = sorted(
        name for name, split_name in membership.items() if split_name != "test"
    )
    if sorted(development_payloads) != development_names:
        raise ValueError(
            f"Development {corpus_label} payload cache does not match the locked subset"
        )
    test_names = sorted(
        name for name, split_name in membership.items() if split_name == "test"
    )
    reader = read_bytes or (lambda path: path.read_bytes())
    test_payloads: Dict[str, bytes] = {}
    test_file_sha256: Dict[str, str] = {}
    test_digest = hashlib.sha256()
    for relative_name in test_names:
        payload = reader(_safe_indexed_path(corpus_root, relative_name))
        if not isinstance(payload, bytes):
            raise TypeError(f"{corpus_label} byte reader must return bytes")
        _update_indexed_corpus_digest(test_digest, relative_name, payload)
        test_payloads[relative_name] = payload
        test_file_sha256[relative_name] = hashlib.sha256(payload).hexdigest()

    all_payloads = {**development_payloads, **test_payloads}
    full_names = sorted(all_payloads)
    full_digest = hashlib.sha256()
    for relative_name in full_names:
        _update_indexed_corpus_digest(full_digest, relative_name, all_payloads[relative_name])
    attestations = {
        "held_out_test": _attestation_record(
            "held_out_test",
            ("test",),
            test_names,
            test_digest,
            expected_attestations["held_out_test"],
            count_key=count_key,
            aggregate_key=aggregate_key,
            algorithm_key=algorithm_key,
            corpus_label=corpus_label,
        ),
        "full_indexed_corpus": _attestation_record(
            "full_indexed_corpus",
            ("train", "val", "test"),
            full_names,
            full_digest,
            expected_attestations["full_indexed_corpus"],
            count_key=count_key,
            aggregate_key=aggregate_key,
            algorithm_key=algorithm_key,
            corpus_label=corpus_label,
        ),
    }
    attestations["held_out_test"]["checkpoint_development_attested_scope"] = False
    attestations["held_out_test"]["read_passes"] = 1
    attestations["full_indexed_corpus"]["checkpoint_development_attested_scope"] = False
    attestations["full_indexed_corpus"]["test_bytes_supplied_by_evaluator_pass"] = 1
    return attestations, test_payloads, test_file_sha256


def attest_development_masks_once(
    mask_root: Path,
    split_files: Mapping[str, Sequence[str]],
    *,
    expected_attestations: Mapping[str, Mapping[str, Any]] = LOCKED_TARGET_ATTESTATIONS,
    read_bytes: Optional[Callable[[Path], bytes]] = None,
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Read and authenticate only train+validation, retaining bytes for full hash."""
    records, payloads = _attest_development_corpus_once(
        mask_root,
        split_files,
        expected_attestations=expected_attestations,
        count_key="mask_count",
        aggregate_key="mask_aggregate_sha256",
        algorithm_key="mask_aggregate_sha256_algorithm",
        corpus_label="target-mask",
        read_bytes=read_bytes,
    )
    return records["development_train_plus_validation"], payloads


def attest_held_out_and_full_once(
    mask_root: Path,
    split_files: Mapping[str, Sequence[str]],
    development_payloads: Mapping[str, bytes],
    *,
    expected_attestations: Mapping[str, Mapping[str, Any]] = LOCKED_TARGET_ATTESTATIONS,
    read_bytes: Optional[Callable[[Path], bytes]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bytes], Dict[str, str]]:
    """After reservation, read test once and complete test/full attestations."""
    return _attest_held_out_and_full_corpus_once(
        mask_root,
        split_files,
        development_payloads,
        expected_attestations=expected_attestations,
        count_key="mask_count",
        aggregate_key="mask_aggregate_sha256",
        algorithm_key="mask_aggregate_sha256_algorithm",
        corpus_label="target-mask",
        read_bytes=read_bytes,
    )


def attest_development_inputs_once(
    image_root: Path,
    split_files: Mapping[str, Sequence[str]],
    *,
    expected_attestations: Mapping[str, Mapping[str, Any]] = LOCKED_INPUT_ATTESTATIONS,
    read_bytes: Optional[Callable[[Path], bytes]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bytes]]:
    """Authenticate train, validation, and dev79 inputs without reading test."""
    return _attest_development_corpus_once(
        image_root,
        split_files,
        expected_attestations=expected_attestations,
        count_key="image_count",
        aggregate_key="image_aggregate_sha256",
        algorithm_key="image_aggregate_sha256_algorithm",
        corpus_label="input-image",
        read_bytes=read_bytes,
    )


def attest_held_out_inputs_and_full_once(
    image_root: Path,
    split_files: Mapping[str, Sequence[str]],
    development_payloads: Mapping[str, bytes],
    *,
    expected_attestations: Mapping[str, Mapping[str, Any]] = LOCKED_INPUT_ATTESTATIONS,
    read_bytes: Optional[Callable[[Path], bytes]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bytes], Dict[str, str]]:
    """Read each test input once and authenticate test21 plus full100."""
    return _attest_held_out_and_full_corpus_once(
        image_root,
        split_files,
        development_payloads,
        expected_attestations=expected_attestations,
        count_key="image_count",
        aggregate_key="image_aggregate_sha256",
        algorithm_key="image_aggregate_sha256_algorithm",
        corpus_label="input-image",
        read_bytes=read_bytes,
    )


def load_lossless_target_mask_bytes(mask_path: Path, payload: bytes) -> np.ndarray:
    """Decode attested 0/1/255 PNG bytes and map 255 to mineral class 2."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required to load confirmatory masks") from error
    source = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise ValueError(f"Failed to load target mask: {mask_path}")
    if source.ndim != 2:
        raise ValueError(f"Target mask is not single-channel: {mask_path.name}")
    if source.shape != EXPECTED_TILE_SHAPE:
        raise ValueError(
            f"Target mask {mask_path.name} is {source.shape[1]}x{source.shape[0]}; "
            "expected 2048x2048"
        )
    values = set(int(value) for value in np.unique(source))
    if not values or not values.issubset({0, 1, 255}):
        raise ValueError(
            f"Target mask {mask_path.name} has values {sorted(values)}; "
            "the locked protocol permits only a nonempty subset of [0, 1, 255]"
        )
    target = source.astype(np.uint8, copy=True)
    target[target == 255] = 2
    return target


def confusion_from_labels(
    target: np.ndarray, prediction: np.ndarray, num_classes: int = 3
) -> np.ndarray:
    if target.shape != prediction.shape:
        raise ValueError(f"Target/prediction shape mismatch: {target.shape} != {prediction.shape}")
    valid = (target >= 0) & (target < num_classes)
    if not np.all(valid):
        raise ValueError("Target contains values outside the canonical class range")
    if np.any((prediction < 0) | (prediction >= num_classes)):
        raise ValueError("Prediction contains values outside the canonical class range")
    encoded = target.astype(np.int64).ravel() * num_classes + prediction.astype(np.int64).ravel()
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def pore_vs_mineral_metrics(confusion: np.ndarray) -> Dict[str, Any]:
    """Merge C0/C1 into pore and retain C2 as mineral for full-area agreement."""
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (3, 3):
        raise ValueError("Pore-union metrics require a canonical 3x3 confusion matrix")
    merged = np.asarray(
        [
            [matrix[:2, :2].sum(), matrix[:2, 2].sum()],
            [matrix[2, :2].sum(), matrix[2, 2]],
        ],
        dtype=np.int64,
    )
    total = int(merged.sum())
    per_class = []
    for class_id, class_name in ((0, "pore_union"), (1, "mineral")):
        tp = int(merged[class_id, class_id])
        fn = int(merged[class_id, :].sum() - tp)
        fp = int(merged[:, class_id].sum() - tp)
        tn = total - tp - fn - fp
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support_pixels": int(merged[class_id, :].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "iou": _safe_ratio(tp, tp + fp + fn),
                "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
                "precision": _safe_ratio(tp, tp + fp),
                "recall": _safe_ratio(tp, tp + fn),
                "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
            }
        )
    return {
        "class_order": ["pore_union", "mineral"],
        "confusion_matrix": merged,
        "accuracy": _safe_ratio(np.diag(merged).sum(), total),
        "agreement": _safe_ratio(np.diag(merged).sum(), total),
        "per_class": per_class,
        "pore_union_iou": per_class[0]["iou"],
        "pore_union_dice": per_class[0]["dice"],
    }


def summarize_gate_reference_confusion(confusion: np.ndarray) -> Dict[str, Any]:
    """Describe fixed raw-intensity gate disagreement with authoritative pore union."""
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (2, 2) or np.any(matrix < 0):
        raise ValueError("Gate/reference confusion must be a non-negative 2x2 matrix")
    total = int(matrix.sum())
    mismatched = int(matrix[0, 1] + matrix[1, 0])
    return {
        "reference_order": ["pore_union", "mineral"],
        "gate_order": ["raw_uint8_lt_100_pore", "raw_uint8_ge_100_mineral"],
        "confusion_matrix": matrix,
        "total_pixels": total,
        "matched_pixels": total - mismatched,
        "mismatched_pixels": mismatched,
        "mismatch_rate": _safe_ratio(mismatched, total),
        "agreement_rate": _safe_ratio(total - mismatched, total),
        "reference_pore_gate_mineral_pixels": int(matrix[0, 1]),
        "reference_mineral_gate_pore_pixels": int(matrix[1, 0]),
        "reference_pore_pixels": int(matrix[0, :].sum()),
        "reference_mineral_pixels": int(matrix[1, :].sum()),
        "gate_pore_pixels": int(matrix[:, 0].sum()),
        "gate_mineral_pixels": int(matrix[:, 1].sum()),
        "threshold_uint8": CONDITIONAL_PORE_THRESHOLD_UINT8,
        "threshold_rule": "raw grayscale intensity < 100 is gate-pore",
        "interpretation": (
            "Intrinsic disagreement between the frozen raw-intensity gate and the "
            "authoritative lossless pore union; the gate is not exact label reconstruction."
        ),
    }


def gate_reference_diagnostic(image: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """Compute the fixed-gate diagnostic from one held-out tile."""
    if image.dtype != np.uint8 or image.shape != target.shape:
        raise ValueError("Gate diagnostic requires aligned raw uint8 image and target")
    if np.any((target < 0) | (target > 2)):
        raise ValueError("Gate diagnostic target is outside canonical C0/C1/C2")
    reference_is_mineral = target == 2
    gate_is_mineral = image >= CONDITIONAL_PORE_THRESHOLD_UINT8
    matrix = np.asarray(
        [
            [
                np.count_nonzero(~reference_is_mineral & ~gate_is_mineral),
                np.count_nonzero(~reference_is_mineral & gate_is_mineral),
            ],
            [
                np.count_nonzero(reference_is_mineral & ~gate_is_mineral),
                np.count_nonzero(reference_is_mineral & gate_is_mineral),
            ],
        ],
        dtype=np.int64,
    )
    diagnostic = summarize_gate_reference_confusion(matrix)
    diagnostic["reference_c0_gate_mineral_pixels"] = int(
        np.count_nonzero((target == 0) & gate_is_mineral)
    )
    diagnostic["reference_c1_gate_mineral_pixels"] = int(
        np.count_nonzero((target == 1) & gate_is_mineral)
    )
    return diagnostic


def _largest_component_and_count(binary_mask: np.ndarray) -> Tuple[int, int]:
    """Return largest-pixel area and unfiltered 8-connected component count."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for 2D component diagnostics") from error
    count_with_background, labels = cv2.connectedComponents(
        np.asarray(binary_mask, dtype=np.uint8), connectivity=8
    )
    component_count = int(count_with_background - 1)
    if component_count == 0:
        return 0, 0
    sizes = np.bincount(labels.ravel(), minlength=count_with_background)[1:]
    return int(sizes.max()), component_count


def _binary_boundary_counts(
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    tolerance_pixels: int = 2,
) -> Dict[str, Any]:
    """Match internal 8-neighbour boundaries within fixed Chebyshev tolerance."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for boundary diagnostics") from error
    if tolerance_pixels != 2:
        raise ValueError("Confirmatory pore-union boundary tolerance is locked to 2 px")
    reference = np.asarray(reference_mask, dtype=np.uint8)
    prediction = np.asarray(prediction_mask, dtype=np.uint8)
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Boundary masks must be aligned two-dimensional arrays")
    neighbourhood = np.ones((3, 3), dtype=np.uint8)
    reference_boundary = reference & (
        1 - cv2.erode(reference, neighbourhood, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    )
    prediction_boundary = prediction & (
        1 - cv2.erode(prediction, neighbourhood, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    )
    tolerance_kernel = np.ones((2 * tolerance_pixels + 1,) * 2, dtype=np.uint8)
    reference_tolerance = cv2.dilate(reference_boundary, tolerance_kernel)
    prediction_tolerance = cv2.dilate(prediction_boundary, tolerance_kernel)
    reference_boundary_pixels = int(np.count_nonzero(reference_boundary))
    prediction_boundary_pixels = int(np.count_nonzero(prediction_boundary))
    matched_reference = int(np.count_nonzero(reference_boundary & prediction_tolerance))
    matched_prediction = int(np.count_nonzero(prediction_boundary & reference_tolerance))
    if prediction_boundary_pixels:
        precision = matched_prediction / prediction_boundary_pixels
    else:
        precision = 1.0 if reference_boundary_pixels == 0 else 0.0
    if reference_boundary_pixels:
        recall = matched_reference / reference_boundary_pixels
    else:
        recall = 1.0 if prediction_boundary_pixels == 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else (1.0 if not reference_boundary_pixels and not prediction_boundary_pixels else 0.0)
    )
    return {
        "tolerance_pixels": tolerance_pixels,
        "distance_neighbourhood": "Chebyshev distance using a fixed 5x5 square dilation",
        "boundary_definition": (
            "foreground pore-union pixels removed by 3x3 8-neighbour erosion; "
            "no component or boundary filtering"
        ),
        "reference_boundary_pixels": reference_boundary_pixels,
        "prediction_boundary_pixels": prediction_boundary_pixels,
        "matched_reference_boundary_pixels": matched_reference,
        "matched_prediction_boundary_pixels": matched_prediction,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def secondary_2d_operational_diagnostics(
    target: np.ndarray, prediction: np.ndarray
) -> Dict[str, Any]:
    """Compute prespecified 2D agreement diagnostics without filtering or tuning."""
    reference = np.asarray(target)
    predicted = np.asarray(prediction)
    if reference.shape != predicted.shape or reference.ndim != 2:
        raise ValueError("2D diagnostics require aligned target and prediction masks")
    if np.any((reference < 0) | (reference > 2)) or np.any(
        (predicted < 0) | (predicted > 2)
    ):
        raise ValueError("2D diagnostics require canonical C0/C1/C2 labels")
    pixel_count = int(reference.size)
    reference_c0 = int(np.count_nonzero(reference == 0))
    prediction_c0 = int(np.count_nonzero(predicted == 0))
    reference_c1 = int(np.count_nonzero(reference == 1))
    prediction_c1 = int(np.count_nonzero(predicted == 1))
    reference_c1_largest, _ = _largest_component_and_count(reference == 1)
    prediction_c1_largest, _ = _largest_component_and_count(predicted == 1)
    _, reference_c0_components = _largest_component_and_count(reference == 0)
    _, prediction_c0_components = _largest_component_and_count(predicted == 0)
    boundary = _binary_boundary_counts(reference < 2, predicted < 2)

    def paired(reference_value: float, prediction_value: float) -> Dict[str, float]:
        return {
            "reference": float(reference_value),
            "prediction": float(prediction_value),
            "signed_error_prediction_minus_reference": float(
                prediction_value - reference_value
            ),
            "absolute_error": float(abs(prediction_value - reference_value)),
        }

    c0_fraction = paired(reference_c0 / pixel_count, prediction_c0 / pixel_count)
    c1_fraction = paired(reference_c1 / pixel_count, prediction_c1 / pixel_count)
    largest_c1_fraction = paired(
        _safe_ratio(reference_c1_largest, reference_c1),
        _safe_ratio(prediction_c1_largest, prediction_c1),
    )
    c0_components_per_megapixel = paired(
        reference_c0_components * 1_000_000.0 / pixel_count,
        prediction_c0_components * 1_000_000.0 / pixel_count,
    )
    return {
        "scope": "two_dimensional_operational_agreement",
        "selection_role": "secondary_only_never_used_for_model_selection",
        "scientific_interpretation": (
            "2D operational segmentation agreement only; not permeability, flow, "
            "transport, or 3D pore-connectivity evidence"
        ),
        "component_connectivity": 8,
        "component_filtering": "none",
        "c0_area_fraction": c0_fraction,
        "c1_area_fraction": c1_fraction,
        "largest_c1_8_connected_component_c1_fraction": largest_c1_fraction,
        "c0_8_connected_components_per_megapixel": c0_components_per_megapixel,
        "pore_union_boundary_f1_at_2px": boundary,
        "sufficient_statistics": {
            "pixel_count": pixel_count,
            "reference_c0_pixels": reference_c0,
            "prediction_c0_pixels": prediction_c0,
            "reference_c1_pixels": reference_c1,
            "prediction_c1_pixels": prediction_c1,
            "reference_largest_c1_component_pixels": reference_c1_largest,
            "prediction_largest_c1_component_pixels": prediction_c1_largest,
            "reference_c0_component_count": reference_c0_components,
            "prediction_c0_component_count": prediction_c0_components,
            "reference_boundary_pixels": boundary["reference_boundary_pixels"],
            "prediction_boundary_pixels": boundary["prediction_boundary_pixels"],
            "matched_reference_boundary_pixels": boundary[
                "matched_reference_boundary_pixels"
            ],
            "matched_prediction_boundary_pixels": boundary[
                "matched_prediction_boundary_pixels"
            ],
        },
    }


def aggregate_secondary_2d_diagnostics(
    tile_diagnostics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Pool count-based diagnostics and tile-average the largest-component fraction."""
    if not tile_diagnostics:
        raise ValueError("At least one tile diagnostic is required")
    statistics = [item["sufficient_statistics"] for item in tile_diagnostics]
    pixels = sum(int(item["pixel_count"]) for item in statistics)

    def paired(reference_value: float, prediction_value: float) -> Dict[str, float]:
        return {
            "reference": float(reference_value),
            "prediction": float(prediction_value),
            "signed_error_prediction_minus_reference": float(
                prediction_value - reference_value
            ),
            "absolute_error": float(abs(prediction_value - reference_value)),
        }

    c0_area = paired(
        sum(int(item["reference_c0_pixels"]) for item in statistics) / pixels,
        sum(int(item["prediction_c0_pixels"]) for item in statistics) / pixels,
    )
    c1_area = paired(
        sum(int(item["reference_c1_pixels"]) for item in statistics) / pixels,
        sum(int(item["prediction_c1_pixels"]) for item in statistics) / pixels,
    )
    largest = paired(
        _safe_ratio(
            sum(
                int(item["reference_largest_c1_component_pixels"])
                for item in statistics
            ),
            sum(int(item["reference_c1_pixels"]) for item in statistics),
        ),
        _safe_ratio(
            sum(
                int(item["prediction_largest_c1_component_pixels"])
                for item in statistics
            ),
            sum(int(item["prediction_c1_pixels"]) for item in statistics),
        ),
    )
    c0_components = paired(
        sum(int(item["reference_c0_component_count"]) for item in statistics)
        * 1_000_000.0
        / pixels,
        sum(int(item["prediction_c0_component_count"]) for item in statistics)
        * 1_000_000.0
        / pixels,
    )
    reference_boundary = sum(int(item["reference_boundary_pixels"]) for item in statistics)
    prediction_boundary = sum(
        int(item["prediction_boundary_pixels"]) for item in statistics
    )
    matched_reference = sum(
        int(item["matched_reference_boundary_pixels"]) for item in statistics
    )
    matched_prediction = sum(
        int(item["matched_prediction_boundary_pixels"]) for item in statistics
    )
    boundary_precision = (
        matched_prediction / prediction_boundary
        if prediction_boundary
        else (1.0 if reference_boundary == 0 else 0.0)
    )
    boundary_recall = (
        matched_reference / reference_boundary
        if reference_boundary
        else (1.0 if prediction_boundary == 0 else 0.0)
    )
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall
        / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else (1.0 if not reference_boundary and not prediction_boundary else 0.0)
    )
    return {
        "scope": "aggregate_held_out_two_dimensional_operational_agreement",
        "selection_role": "secondary_only_never_used_for_model_selection",
        "scientific_interpretation": (
            "2D operational segmentation agreement only; not permeability, flow, "
            "transport, or 3D pore-connectivity evidence"
        ),
        "tile_count": len(tile_diagnostics),
        "pixel_count": pixels,
        "c0_area_fraction": c0_area,
        "c1_area_fraction": c1_area,
        "largest_c1_8_connected_component_c1_fraction": largest,
        "largest_component_aggregation": (
            "sum of each tile's largest C1 component pixels divided by pooled C1 pixels"
        ),
        "c0_8_connected_components_per_megapixel": c0_components,
        "component_density_aggregation": "total unfiltered components divided by total megapixels",
        "pore_union_boundary_f1_at_2px": {
            "precision": float(boundary_precision),
            "recall": float(boundary_recall),
            "f1": float(boundary_f1),
            "reference_boundary_pixels": reference_boundary,
            "prediction_boundary_pixels": prediction_boundary,
            "matched_reference_boundary_pixels": matched_reference,
            "matched_prediction_boundary_pixels": matched_prediction,
            "tolerance_pixels": 2,
            "aggregation": "micro-pooled boundary-match counts across held-out tiles",
        },
    }


def _secondary_diagnostic_metric_lookup(
    diagnostic: Mapping[str, Any],
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for name in (
        "c0_area_fraction",
        "c1_area_fraction",
        "largest_c1_8_connected_component_c1_fraction",
        "c0_8_connected_components_per_megapixel",
    ):
        for field in (
            "reference",
            "prediction",
            "signed_error_prediction_minus_reference",
            "absolute_error",
        ):
            values[f"{name}.{field}"] = float(diagnostic[name][field])
    for field in ("precision", "recall", "f1"):
        values[f"pore_union_boundary_f1_at_2px.{field}"] = float(
            diagnostic["pore_union_boundary_f1_at_2px"][field]
        )
    return values


def bootstrap_secondary_2d_diagnostics(
    tile_diagnostics: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> Dict[str, Any]:
    """Deterministically resample whole tiles for secondary-diagnostic intervals."""
    if not tile_diagnostics or replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid secondary-diagnostic bootstrap configuration")
    rng = np.random.default_rng(seed)
    series: Dict[str, List[float]] = {}
    tile_count = len(tile_diagnostics)
    for indices in rng.integers(0, tile_count, size=(replicates, tile_count)):
        aggregate = aggregate_secondary_2d_diagnostics(
            [tile_diagnostics[int(index)] for index in indices]
        )
        for key, value in _secondary_diagnostic_metric_lookup(aggregate).items():
            series.setdefault(key, []).append(value)
    alpha = 1.0 - confidence
    intervals = {}
    for key, values in sorted(series.items()):
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size:
            lower, upper = np.quantile(array, [alpha / 2.0, 1.0 - alpha / 2.0])
            intervals[key] = {
                "lower": float(lower),
                "upper": float(upper),
                "finite_replicates": int(array.size),
            }
        else:
            intervals[key] = {
                "lower": None,
                "upper": None,
                "finite_replicates": 0,
            }
    return {
        "method": "percentile bootstrap",
        "sampling_unit": "held-out 2048x2048 tile",
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "tile_count": tile_count,
        "intervals": intervals,
    }


def metrics_from_confusion(
    confusion: np.ndarray, class_names: Mapping[int, str] = CLASS_NAMES
) -> Dict[str, Any]:
    """Calculate per-class and aggregate segmentation metrics from one matrix."""
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if np.any(confusion < 0):
        raise ValueError("Confusion counts cannot be negative")

    total = int(confusion.sum())
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    false_negative = support - true_positive
    false_positive = predicted - true_positive
    true_negative = total - true_positive - false_negative - false_positive

    per_class: List[Dict[str, Any]] = []
    for class_id in range(confusion.shape[0]):
        tp = true_positive[class_id]
        fp = false_positive[class_id]
        fn = false_negative[class_id]
        tn = true_negative[class_id]
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"class_{class_id}"),
                "support_pixels": int(support[class_id]),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "iou": _safe_ratio(tp, tp + fp + fn),
                "dice": f1,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    metric_names = ("iou", "dice", "precision", "recall", "f1")
    macro = {
        name: _finite_mean(item[name] for item in per_class)
        for name in metric_names
    }
    support_total = float(support.sum())
    weighted = {
        name: _safe_ratio(
            sum(
                item[name] * item["support_pixels"]
                for item in per_class
                if math.isfinite(item[name])
            ),
            sum(
                item["support_pixels"]
                for item in per_class
                if math.isfinite(item[name])
            ),
        )
        for name in metric_names
    }
    sum_tp = float(true_positive.sum())
    sum_fp = float(false_positive.sum())
    sum_fn = float(false_negative.sum())
    micro_precision = _safe_ratio(sum_tp, sum_tp + sum_fp)
    micro_recall = _safe_ratio(sum_tp, sum_tp + sum_fn)
    micro_f1 = _safe_ratio(2 * sum_tp, 2 * sum_tp + sum_fp + sum_fn)
    micro = {
        "iou": _safe_ratio(sum_tp, sum_tp + sum_fp + sum_fn),
        "dice": micro_f1,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
    }
    c0_iou = float(per_class[0]["iou"])
    c1_iou = float(per_class[1]["iou"])
    harmonic_iou = _safe_ratio(
        2.0 * c0_iou * c1_iou,
        c0_iou + c1_iou + 1e-8,
    )
    merged = pore_vs_mineral_metrics(confusion)
    return {
        "total_pixels": total,
        "correct_pixels": int(true_positive.sum()),
        "accuracy": _safe_ratio(true_positive.sum(), support_total),
        "per_class": per_class,
        "macro": macro,
        "weighted": weighted,
        "micro": micro,
        "selection_metrics": {
            "c0_c1_harmonic_iou": harmonic_iou,
            "pore_union_iou": merged["pore_union_iou"],
            "pore_union_agreement": merged["agreement"],
        },
        "pore_vs_mineral": merged,
    }


def _metric_lookup(metrics: Mapping[str, Any]) -> Dict[str, float]:
    values = {"overall.accuracy": float(metrics["accuracy"])}
    for scope in ("macro", "weighted", "micro"):
        for name, value in metrics[scope].items():
            values[f"{scope}.{name}"] = float(value)
    for item in metrics["per_class"]:
        for name in ("iou", "dice", "precision", "recall", "f1"):
            values[f"class_{item['class_id']}.{name}"] = float(item[name])
    for name, value in metrics["selection_metrics"].items():
        values[f"selection.{name}"] = float(value)
    values["pore_vs_mineral.accuracy"] = float(
        metrics["pore_vs_mineral"]["accuracy"]
    )
    values["pore_vs_mineral.agreement"] = float(
        metrics["pore_vs_mineral"]["agreement"]
    )
    for item in metrics["pore_vs_mineral"]["per_class"]:
        for name in ("iou", "dice", "precision", "recall", "f1"):
            values[f"merged_class_{item['class_id']}.{name}"] = float(item[name])
    return values


def tile_bootstrap_intervals(
    tile_confusions: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Percentile intervals from deterministic whole-tile resampling."""
    matrices = np.asarray(tile_confusions, dtype=np.int64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("tile_confusions must have shape (tiles, 3, 3)")
    if not matrices.shape[0]:
        raise ValueError("At least one tile is required for bootstrap intervals")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("Bootstrap confidence must be between zero and one")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, matrices.shape[0], size=(replicates, matrices.shape[0]))
    replicate_confusions = matrices[draws].sum(axis=1)
    series: Dict[str, List[float]] = {}
    for matrix in replicate_confusions:
        for key, value in _metric_lookup(metrics_from_confusion(matrix)).items():
            series.setdefault(key, []).append(value)

    alpha = 1.0 - confidence
    intervals: Dict[str, Dict[str, Optional[float]]] = {}
    for key, values in sorted(series.items()):
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size:
            lower, upper = np.quantile(array, [alpha / 2.0, 1.0 - alpha / 2.0])
            intervals[key] = {
                "lower": float(lower),
                "upper": float(upper),
                "finite_replicates": int(array.size),
            }
        else:
            intervals[key] = {
                "lower": None,
                "upper": None,
                "finite_replicates": 0,
            }
    return {
        "method": "percentile bootstrap",
        "sampling_unit": "held-out 2048x2048 tile",
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "tile_count": int(matrices.shape[0]),
        "intervals": intervals,
    }


def update_probability_histograms(
    positive_histograms: np.ndarray,
    negative_histograms: np.ndarray,
    probabilities: np.ndarray,
    target: np.ndarray,
) -> None:
    """Accumulate one-vs-rest score evidence without retaining raw scores."""
    num_classes, bins = positive_histograms.shape
    if negative_histograms.shape != (num_classes, bins):
        raise ValueError("Positive and negative histogram shapes differ")
    if probabilities.shape != (num_classes, *target.shape):
        raise ValueError("Probability tensor and target mask shapes differ")
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("Model produced non-finite probabilities")
    if probabilities.min() < -1e-7 or probabilities.max() > 1.0 + 1e-7:
        raise ValueError("Model probabilities are outside [0, 1]")

    for class_id in range(num_classes):
        scores = np.clip(probabilities[class_id], 0.0, 1.0)
        bin_ids = np.minimum((scores * bins).astype(np.int64), bins - 1)
        positives = target == class_id
        positive_histograms[class_id] += np.bincount(
            bin_ids[positives], minlength=bins
        ).astype(np.int64)
        negative_histograms[class_id] += np.bincount(
            bin_ids[~positives], minlength=bins
        ).astype(np.int64)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    function = getattr(np, "trapezoid", None) or np.trapz
    return float(function(y, x))


def curves_from_histograms(
    positive_histogram: np.ndarray, negative_histogram: np.ndarray
) -> Dict[str, Any]:
    """Construct approximate ROC/PR curves from fixed score bins."""
    positive = np.asarray(positive_histogram, dtype=np.int64)
    negative = np.asarray(negative_histogram, dtype=np.int64)
    if positive.ndim != 1 or positive.shape != negative.shape:
        raise ValueError("Curve histograms must be equal-length vectors")
    bins = positive.size
    if bins < 2 or np.any(positive < 0) or np.any(negative < 0):
        raise ValueError("Curve histograms are invalid")
    positives = int(positive.sum())
    negatives = int(negative.sum())

    cumulative_tp = np.concatenate(([0], np.cumsum(positive[::-1], dtype=np.int64)))
    cumulative_fp = np.concatenate(([0], np.cumsum(negative[::-1], dtype=np.int64)))
    thresholds = np.concatenate(
        ([1.0 + 1.0 / bins], np.arange(bins - 1, -1, -1, dtype=np.float64) / bins)
    )
    tpr = (
        cumulative_tp.astype(np.float64) / positives
        if positives
        else np.full(cumulative_tp.shape, np.nan)
    )
    fpr = (
        cumulative_fp.astype(np.float64) / negatives
        if negatives
        else np.full(cumulative_fp.shape, np.nan)
    )
    predicted_positive = cumulative_tp + cumulative_fp
    precision = np.divide(
        cumulative_tp,
        predicted_positive,
        out=np.ones(cumulative_tp.shape, dtype=np.float64),
        where=predicted_positive != 0,
    )
    recall = tpr.copy()
    roc_auc = _trapezoid(tpr, fpr) if positives and negatives else float("nan")
    average_precision = (
        float(np.sum(np.diff(recall) * precision[1:])) if positives else float("nan")
    )
    pr_auc = _trapezoid(precision, recall) if positives else float("nan")
    return {
        "thresholds": thresholds,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "precision": precision,
        "recall": recall,
        "cumulative_true_positive": cumulative_tp,
        "cumulative_false_positive": cumulative_fp,
        "positive_pixels": positives,
        "negative_pixels": negatives,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "pr_auc": pr_auc,
    }


def _state_shape(state: Mapping[str, Any], key: str) -> Optional[Tuple[int, ...]]:
    value = state.get(key)
    shape = getattr(value, "shape", None)
    return tuple(int(item) for item in shape) if shape is not None else None


def infer_model_config_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Infer structural parameters that are uniquely encoded in a state dict."""
    keys = set(state)
    if any(".branch1." in key for key in keys) and "outc.weight" in keys:
        architecture = (
            "multiscale_attention_unet_pyramid"
            if any(key.startswith("pyramid_context.") for key in keys)
            else "multiscale_attention_unet"
        )
        input_shape = _state_shape(state, "inc.double_conv.0.weight")
        output_shape = _state_shape(state, "outc.weight")
        deep_supervision = any(key.startswith("ds1.") for key in keys)
    elif any(key.startswith("att1.") for key in keys):
        architecture = "attention_unet"
        input_shape = _state_shape(state, "inc.double_conv.0.weight")
        output_shape = _state_shape(state, "outc.conv.weight")
        deep_supervision = False
    elif any(key.startswith("ds1.") for key in keys):
        architecture = "deep_supervision_unet"
        input_shape = _state_shape(state, "inc.double_conv.0.weight")
        output_shape = _state_shape(state, "outc.conv.weight")
        deep_supervision = True
    elif "outc.conv.weight" in keys and "inc.double_conv.0.weight" in keys:
        architecture = "unet"
        input_shape = _state_shape(state, "inc.double_conv.0.weight")
        output_shape = _state_shape(state, "outc.conv.weight")
        deep_supervision = False
    else:
        raise ValueError(
            "Checkpoint architecture is not one of the repository's supported U-Net states"
        )
    if not input_shape or not output_shape:
        raise ValueError("Checkpoint is missing structural convolution weights")
    return {
        "architecture": architecture,
        "n_channels": input_shape[1],
        "num_classes": output_shape[0],
        "base_features": input_shape[0],
        "bilinear": not any(key.startswith("up1.up.") for key in keys),
        "deep_supervision": deep_supervision,
        "source": "inferred_from_checkpoint_tensor_shapes_and_keys",
    }


def _normalize_architecture(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "multiscaleattentionunet": "multiscale_attention_unet",
        "multi_scale_attention_unet": "multiscale_attention_unet",
        "multiscaleattentionunetpyramid": "multiscale_attention_unet_pyramid",
        "multi_scale_attention_unet_pyramid": "multiscale_attention_unet_pyramid",
        "attentionunet": "attention_unet",
        "deep_supervisionunet": "deep_supervision_unet",
        "plain_unet": "unet",
        "u_net": "unet",
    }
    return aliases.get(normalized, normalized)


def _plain_checkpoint_config(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    nested = getattr(value, "config", None)
    return nested if isinstance(nested, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _recorded_normalization(checkpoint: Mapping[str, Any]) -> Optional[Any]:
    config = _plain_checkpoint_config(checkpoint.get("config"))
    candidates = (
        checkpoint.get("input_normalization"),
        _nested(checkpoint, "resolved_model_config", "input_normalization"),
        _nested(checkpoint, "data_config", "input_normalization"),
        _nested(config, "data_config", "input_normalization"),
        _nested(config, "model", "input_normalization"),
    )
    return next((value for value in candidates if value is not None), None)


def _normalization_matches(value: Any) -> bool:
    if isinstance(value, Mapping):
        try:
            return (
                str(value.get("name")) == EXPECTED_INPUT_NORMALIZATION["name"]
                and int(value.get("input_channels")) == 1
                and str(value.get("source_dtype")) == "uint8"
                and float(value.get("scale_divisor")) == 255.0
                and [float(item) for item in value.get("mean", [])] == [0.5]
                and [float(item) for item in value.get("std", [])] == [0.5]
                and [float(item) for item in value.get("output_range", [])]
                == [-1.0, 1.0]
            )
        except (TypeError, ValueError):
            return False
    value = str(value)
    normalized = value.lower().replace(" ", "")
    accepted = {
        NORMALIZATION_ID.lower(),
        NORMALIZATION_FORMULA.lower().replace(" ", ""),
        "[-1,1]",
        "minus_one_to_one",
        "mean=0.5,std=0.5",
    }
    return normalized in accepted


def _torch_load_trusted(path: Path, device: str) -> Mapping[str, Any]:
    return load_weights_only_checkpoint(path, map_location=device)


def validate_selected_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    role = checkpoint.get("checkpoint_role")
    if role != CHECKPOINT_ROLE:
        raise ValueError(
            f"Refusing checkpoint role {role!r}; expected {CHECKPOINT_ROLE!r}"
        )
    if checkpoint.get("best_selection_epoch") is None:
        raise ValueError("Validation-selected checkpoint has no selected epoch")
    if checkpoint.get("selection_metric_name") != SELECTION_METRIC_NAME:
        raise ValueError(
            "Validation-selected checkpoint uses the wrong selection metric: "
            f"{checkpoint.get('selection_metric_name')!r}"
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no model_state_dict")


def validate_checkpoint_normalization(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Require identical structured normalization records at both checkpoint levels."""
    top_level = checkpoint.get("input_normalization")
    resolved = _nested(checkpoint, "resolved_config", "input_normalization")
    for location, value in (("top-level", top_level), ("resolved_config", resolved)):
        if not isinstance(value, Mapping) or not _normalization_matches(value):
            raise ValueError(
                f"Checkpoint {location} normalization does not match the locked "
                f"{NORMALIZATION_ID} contract"
            )
        if _json_value(dict(value)) != EXPECTED_INPUT_NORMALIZATION:
            raise ValueError(
                f"Checkpoint {location} normalization contains unrecognised drift"
            )
    if dict(top_level) != dict(resolved):
        raise ValueError("Checkpoint normalization records disagree with each other")
    return dict(EXPECTED_INPUT_NORMALIZATION)


def validate_checkpoint_target_provenance(
    checkpoint: Mapping[str, Any], development_attestation: Mapping[str, Any]
) -> Dict[str, Any]:
    """Authenticate only the 79 development masks recorded by training."""
    provenance = checkpoint.get("target_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Checkpoint has no structured target_provenance record")
    resolved_provenance = _nested(checkpoint, "resolved_config", "target")
    if not isinstance(resolved_provenance, Mapping) or dict(resolved_provenance) != dict(
        provenance
    ):
        raise ValueError("Checkpoint target-provenance records disagree")
    if provenance.get("target_source") != "lossless_png_masks":
        raise ValueError(
            "Checkpoint target source is not the authoritative lossless PNG corpus"
        )
    try:
        recorded_count = int(provenance.get("mask_count"))
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint target provenance has no valid mask_count") from error
    expected_count = int(development_attestation["mask_count"])
    if recorded_count != expected_count:
        raise ValueError(
            "Checkpoint development target mask count conflicts with the locked "
            f"train+validation subset: {recorded_count} != {expected_count}"
        )
    recorded_hash = provenance.get("mask_aggregate_sha256")
    expected_hash = development_attestation["mask_aggregate_sha256"]
    if recorded_hash != expected_hash:
        raise ValueError(
            "Checkpoint development target-mask hash conflicts with the locked "
            f"train+validation subset: {recorded_hash!r} != {expected_hash!r}"
        )
    if provenance.get("mask_aggregate_sha256_algorithm") != development_attestation.get(
        "mask_aggregate_sha256_algorithm"
    ):
        raise ValueError("Checkpoint target-mask hash algorithm conflicts with evaluator")
    try:
        recorded_values = sorted(int(value) for value in provenance["validated_source_values"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint target provenance has no valid source-value record"
        ) from error
    if recorded_values != [0, 1, 255]:
        raise ValueError(
            "Checkpoint target source values conflict with the locked protocol: "
            f"{recorded_values!r} != [0, 1, 255]"
        )
    if provenance.get("evaluation_mode") != "train_validation_only":
        raise ValueError("Checkpoint target provenance is not validation-only")
    if provenance.get("held_out_dataset_constructed") is not False:
        raise ValueError("Checkpoint target provenance says held-out data was constructed")
    if provenance.get("annotations_role") != "image_index_and_metadata_only":
        raise ValueError("Checkpoint allowed COCO polygons to act as target evidence")
    if provenance.get("canonical_value_mapping") != {
        "0": "0 (disconnected_pore)",
        "1": "1 (connected_pore)",
        "255": "2 (mineral) in three-class mode; ignore_index in two-class mode",
    }:
        raise ValueError("Checkpoint canonical target-value mapping drifted")
    return {
        "verification_status": "matched_checkpoint_development_target_provenance",
        "attested_scope": "train_plus_validation_only",
        "held_out_test_included": False,
        "target_source": "lossless_png_masks",
        "mask_count": recorded_count,
        "mask_aggregate_sha256": recorded_hash,
        "mask_aggregate_sha256_algorithm": provenance[
            "mask_aggregate_sha256_algorithm"
        ],
        "validated_source_values": recorded_values,
    }


def _provenance_attestation_fields(record: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Project an evaluator attestation to the fields persisted by training."""
    return {
        "scope": record["scope"],
        "split_names": list(record["split_names"]),
        f"{prefix}_count": int(record[f"{prefix}_count"]),
        f"{prefix}_aggregate_sha256": record[f"{prefix}_aggregate_sha256"],
        f"{prefix}_aggregate_sha256_algorithm": record[
            f"{prefix}_aggregate_sha256_algorithm"
        ],
        "file_name_list_sha256": record["file_name_list_sha256"],
    }


def validate_checkpoint_input_provenance(
    checkpoint: Mapping[str, Any],
    development_attestations: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Authenticate checkpoint dev79/train74 image bytes without exposing paths."""
    provenance = checkpoint.get("input_provenance")
    resolved = _nested(checkpoint, "resolved_config", "input")
    if not isinstance(provenance, Mapping) or not isinstance(resolved, Mapping):
        raise ValueError("Checkpoint has no structured development input provenance")
    if dict(provenance) != dict(resolved):
        raise ValueError("Checkpoint input-provenance records disagree")
    expected_keys = {
        "input_source",
        "scope",
        "split_names",
        "image_count",
        "image_aggregate_sha256",
        "image_aggregate_sha256_algorithm",
        "file_name_list_sha256",
        "training_subset",
        "held_out_bytes_read",
        "held_out_scope",
    }
    if set(provenance) != expected_keys:
        raise ValueError(
            "Checkpoint input provenance contains missing or unrecognised fields: "
            f"{sorted(set(provenance) ^ expected_keys)}"
        )
    if provenance.get("input_source") != "indexed_source_images":
        raise ValueError("Checkpoint input source is not indexed_source_images")
    development = development_attestations["development_train_plus_validation"]
    expected_development = _provenance_attestation_fields(development, "image")
    observed_development = {
        key: provenance.get(key) for key in expected_development
    }
    if observed_development != expected_development:
        raise ValueError("Checkpoint development input attestation does not match live dev79")
    training = development_attestations["training_only"]
    expected_training = _provenance_attestation_fields(training, "image")
    if provenance.get("training_subset") != expected_training:
        raise ValueError("Checkpoint training input attestation does not match live train74")
    if provenance.get("held_out_bytes_read") != 0:
        raise ValueError("Checkpoint input provenance says held-out bytes were read")
    if provenance.get("held_out_scope") != (
        "not_read_or_hashed_by_validation_only_trainer"
    ):
        raise ValueError("Checkpoint held-out input scope is not validation-only")
    return {
        "verification_status": "matched_checkpoint_development_input_provenance",
        "attested_scope": "train_plus_validation_only",
        "held_out_test_included": False,
        "input_source": "indexed_source_images",
        **expected_development,
        "training_subset": expected_training,
        "held_out_bytes_read": 0,
    }


def validate_checkpoint_training_seed(
    checkpoint: Mapping[str, Any],
    verified_lock: Mapping[str, Any],
    *,
    candidate: str,
    architecture_role: str,
) -> Dict[str, Any]:
    """Match the selected retraining RNG contract to one prespecified screen seed."""
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("Checkpoint has no resolved configuration for seed validation")
    augmentation = resolved.get("augmentation")
    execution = resolved.get("scientific_execution_contract")
    if not isinstance(augmentation, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("Checkpoint lacks augmentation/scientific execution seed records")
    scientific_augmentation = execution.get("augmentation")
    if not isinstance(scientific_augmentation, Mapping):
        raise ValueError("Checkpoint scientific execution contract lacks augmentation")
    if dict(augmentation) != dict(scientific_augmentation):
        raise ValueError("Checkpoint augmentation records disagree")
    try:
        seed = int(augmentation["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Checkpoint has no valid training augmentation seed") from error
    if seed not in SCREEN_SEEDS:
        raise ValueError(f"Checkpoint training seed {seed} is not prespecified")
    loader = augmentation.get("data_loader")
    if not isinstance(loader, Mapping):
        raise ValueError("Checkpoint augmentation lacks data-loader RNG provenance")
    shuffle_seed = loader.get("shuffle_generator_seed")
    sampler_seed = loader.get("distributed_sampler_seed")
    if (shuffle_seed, sampler_seed) not in ((seed, None), (None, seed)):
        raise ValueError("Checkpoint loader seed does not match its augmentation seed")

    cells = _nested(verified_lock, "screen_selection_provenance", "screen_cells")
    matching = [
        cell
        for cell in cells or []
        if cell.get("candidate") == candidate and int(cell.get("seed", -1)) == seed
    ]
    if len(matching) != 1:
        raise ValueError("Selected lock has no unique candidate/training-seed screen cell")
    screen_execution = matching[0].get("scientific_execution_contract")
    if not isinstance(screen_execution, Mapping):
        raise ValueError("Selected lock cell has no scientific execution contract")
    current_non_model = {key: value for key, value in execution.items() if key != "model"}
    screen_non_model = {
        key: value for key, value in screen_execution.items() if key != "model"
    }
    if current_non_model != screen_non_model:
        raise ValueError("Winner retraining changed a frozen non-model scientific setting")
    if architecture_role == "primary_multiscale" and execution.get("model") != (
        screen_execution.get("model")
    ):
        raise ValueError("Primary winner model contract differs from its screen cell")
    if architecture_role == "plain_unet_comparator":
        current_model = execution.get("model")
        protocol = PROSPECTIVE_METHOD_PROTOCOLS[candidate]
        if not isinstance(current_model, Mapping) or (
            current_model.get("architecture_resolved") != "plain_unet"
            or current_model.get("input_channels") != protocol["model_input_channels"]
            or current_model.get("output_classes") != protocol["model_output_classes"]
            or current_model.get("deep_supervision") is not False
            or current_model.get("dropout_execution")
            != {"status": "unused_by_resolved_architecture", "probability": None}
        ):
            raise ValueError("Plain U-Net comparator changed more than architecture role")
    return {
        "verification_status": "matched_prespecified_training_seed_and_execution_contract",
        "training_seed": seed,
        "prespecified_seeds": list(SCREEN_SEEDS),
        "candidate": candidate,
        "selected_architecture_role": architecture_role,
        "matching_screen_array_index": matching[0].get("array_index"),
        "loader_seed_mode": (
            "shuffle_generator" if shuffle_seed == seed else "distributed_sampler"
        ),
        "evaluator_inference_seed": LOCKED_EVALUATOR_INFERENCE_SEED,
        "evaluator_rng_is_distinct": seed != LOCKED_EVALUATOR_INFERENCE_SEED,
    }


def split_files_from_index(
    split_ids: Mapping[str, Sequence[int]], image_by_id: Mapping[int, Mapping[str, Any]]
) -> Dict[str, List[str]]:
    return {
        split_name: [str(image_by_id[int(image_id)]["file_name"]) for image_id in image_ids]
        for split_name, image_ids in split_ids.items()
    }


def validate_checkpoint_split_isolation(
    checkpoint: Mapping[str, Any],
    *,
    manifest_sha256: str,
    split_ids: Mapping[str, Sequence[int]],
    split_files: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """Require an exact validation-only split record before held-out access."""
    if manifest_sha256 != LOCKED_SPLIT_MANIFEST_SHA256:
        raise ValueError(
            "Split-manifest SHA-256 drifted from the locked protocol: "
            f"{manifest_sha256} != {LOCKED_SPLIT_MANIFEST_SHA256}"
        )
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("Checkpoint has no structured resolved_config")
    evaluation = resolved.get("evaluation")
    data_split = resolved.get("data_split")
    if not isinstance(evaluation, Mapping) or not isinstance(data_split, Mapping):
        raise ValueError("Checkpoint lacks evaluation/data-split isolation records")
    if evaluation.get("mode") != "validation_only":
        raise ValueError("Checkpoint evaluation mode is not validation_only")
    if evaluation.get("held_out_dataset_constructed") is not False:
        raise ValueError("Checkpoint says the held-out dataset was constructed")
    if int(evaluation.get("held_out_evaluation_count", -1)) != 0:
        raise ValueError("Checkpoint says held-out evaluation already occurred")
    if data_split.get("validation_only") is not True:
        raise ValueError("Checkpoint data split is not marked validation_only=true")
    if data_split.get("held_out_dataset_constructed") is not False:
        raise ValueError("Checkpoint split record says held-out data was constructed")
    if int(data_split.get("held_out_evaluation_count", -1)) != 0:
        raise ValueError("Checkpoint split record has a nonzero held-out count")
    if data_split.get("manifest_source") != "explicit_manifest":
        raise ValueError("Checkpoint did not use the explicit locked split manifest")
    if data_split.get("manifest_repo_relative_identifier") != (
        "config/confirmatory_splits.json"
    ):
        raise ValueError("Checkpoint used the wrong split-manifest identifier")
    if data_split.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Checkpoint split-manifest SHA-256 does not match the live lock")
    if data_split.get("annotation_index_repo_relative_identifier") != (
        "results/step3_coco_dataset/pore_annotations.json"
    ):
        raise ValueError("Checkpoint used the wrong canonical annotation index")
    if data_split.get("annotation_index_sha256") != LOCKED_ANNOTATION_INDEX_SHA256:
        raise ValueError("Checkpoint annotation-index SHA-256 does not match the live lock")

    partitions = data_split.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("Checkpoint split record has no exact partitions")
    public_partitions: Dict[str, Dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        partition = partitions.get(split_name)
        if not isinstance(partition, Mapping):
            raise ValueError(f"Checkpoint lacks the {split_name} partition record")
        expected_ids = [int(value) for value in split_ids[split_name]]
        expected_files = [str(value) for value in split_files[split_name]]
        if partition.get("image_ids") != expected_ids:
            raise ValueError(f"Checkpoint {split_name} image IDs do not match manifest")
        if partition.get("image_files") != expected_files:
            raise ValueError(f"Checkpoint {split_name} image files do not match COCO/manifest")
        if int(partition.get("image_count", -1)) != len(expected_ids):
            raise ValueError(f"Checkpoint {split_name} image count is inconsistent")
        public_partitions[split_name] = {
            "image_ids": expected_ids,
            "image_files": expected_files,
            "image_count": len(expected_ids),
        }
    expected_assignment_hash = hashlib.sha256(
        json.dumps(
            public_partitions, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if data_split.get("partition_assignment_sha256") != expected_assignment_hash:
        raise ValueError("Checkpoint partition-assignment hash does not match IDs/files")
    return {
        "verification_status": "matched_validation_only_checkpoint_split",
        "manifest_sha256": manifest_sha256,
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_evaluation_count_before_evaluator": 0,
        "partitions": public_partitions,
    }


def load_verified_selected_method_lock(lock_path: Path) -> Tuple[Dict[str, Any], str, str]:
    """Rebuild the complete 15-cell lock and return public-safe identity."""
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock_identifier = _repository_relative(lock_path)
    with lock_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    verified = verify_selected_method_lock_document(document, PROJECT_ROOT)
    if verified.get("schema_version") != SELECTED_METHOD_LOCK_SCHEMA_VERSION:
        raise ValueError("Selected-method lock has the wrong schema version")
    provenance = verified.get("screen_selection_provenance")
    cells = provenance.get("screen_cells") if isinstance(provenance, Mapping) else None
    if not isinstance(cells, list) or len(cells) != SCREEN_CELL_COUNT:
        raise ValueError("Selected-method lock lacks the complete 15-cell provenance")
    expected_pairs = [
        (candidate, seed)
        for candidate in SCREEN_CANDIDATE_ORDER
        for seed in SCREEN_SEEDS
    ]
    observed_pairs = [(cell.get("candidate"), cell.get("seed")) for cell in cells]
    if observed_pairs != expected_pairs:
        raise ValueError("Selected-method lock has the wrong candidate/seed matrix")
    return verified, sha256_file(lock_path), lock_identifier


def validate_checkpoint_selected_method_lock(
    checkpoint: Mapping[str, Any],
    verified_lock: Mapping[str, Any],
    *,
    lock_sha256: str,
    lock_identifier: str,
) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
    """Require the checkpoint to embed the same deterministic winner and protocol."""
    resolved = checkpoint.get("resolved_config")
    embedded = resolved.get("selected_method_lock") if isinstance(resolved, Mapping) else None
    if not isinstance(embedded, Mapping):
        raise ValueError("Checkpoint has no embedded selected-method lock")
    selected_method = verified_lock.get("selected_method")
    if selected_method not in PROSPECTIVE_METHOD_PROTOCOLS:
        raise ValueError("Verified lock names an unknown selected method")
    protocol = dict(PROSPECTIVE_METHOD_PROTOCOLS[selected_method])
    required_equalities = {
        "schema_version": verified_lock.get("schema_version"),
        "selected_method": selected_method,
        "lock_file_repo_relative_identifier": lock_identifier,
        "lock_file_sha256": lock_sha256,
        "resolved_protocol": verified_lock.get("resolved_protocol"),
        "screen_selection_provenance": verified_lock.get(
            "screen_selection_provenance"
        ),
    }
    mismatches = {
        key: {"checkpoint": embedded.get(key), "verified_lock": expected}
        for key, expected in required_equalities.items()
        if embedded.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint selected-method lock mismatch: {mismatches}")
    if resolved.get("protocol_candidate_key") != selected_method:
        raise ValueError("Checkpoint candidate does not match the selected-method lock")
    if resolved.get("protocol_run_role") != "selected_winner_retraining":
        raise ValueError(
            "Checkpoint protocol_run_role is not selected_winner_retraining"
        )
    architecture_role = resolved.get("selected_architecture_role")
    if architecture_role not in {"primary_multiscale", "plain_unet_comparator"}:
        raise ValueError(
            "Checkpoint selected_architecture_role must be primary_multiscale or "
            "plain_unet_comparator"
        )
    campaign_id = resolved.get("protocol_campaign_id")
    cell_index = resolved.get("protocol_cell_index")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or len(campaign_id) > 96
        or not campaign_id[0].isalnum()
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in campaign_id
        )
    ):
        raise ValueError("Selected retraining campaign ID is missing or public-path unsafe")
    try:
        selected_retrain_task_index = int(cell_index)
    except (TypeError, ValueError) as error:
        raise ValueError("Selected retraining has no valid array task index") from error
    if selected_retrain_task_index not in range(len(SCREEN_SEEDS)):
        raise ValueError("Selected retraining array task index must be 0, 1, or 2")
    try:
        recorded_training_seed = int(_nested(resolved, "augmentation", "seed"))
    except (TypeError, ValueError) as error:
        raise ValueError("Selected retraining has no valid augmentation seed") from error
    expected_seed = int(SCREEN_SEEDS[selected_retrain_task_index])
    if recorded_training_seed != expected_seed:
        raise ValueError(
            "Selected retraining task/seed mapping mismatch: "
            f"task {selected_retrain_task_index} requires seed {expected_seed}, "
            f"checkpoint records {recorded_training_seed}"
        )
    if verified_lock.get("resolved_protocol") != protocol:
        raise ValueError("Selected-method lock protocol drifted from its prospective family")
    provenance = verified_lock["screen_selection_provenance"]
    if provenance.get("split_manifest_sha256") != LOCKED_SPLIT_MANIFEST_SHA256:
        raise ValueError("Selected-method lock split hash does not match the locked manifest")
    development_hash = LOCKED_TARGET_ATTESTATIONS[
        "development_train_plus_validation"
    ]["mask_aggregate_sha256"]
    if provenance.get("target_mask_aggregate_sha256") != development_hash:
        raise ValueError("Selected-method lock target hash is not the development-only hash")
    development_input_hash = LOCKED_INPUT_ATTESTATIONS[
        "development_train_plus_validation"
    ]["image_aggregate_sha256"]
    if provenance.get("input_image_aggregate_sha256") != development_input_hash:
        raise ValueError("Selected-method lock input hash is not the development-only hash")
    selected_retraining_array = {
        "verification_status": "matched_selected_retraining_array_task_and_seed",
        "campaign_id": campaign_id,
        "array_task_index": selected_retrain_task_index,
        "training_seed": recorded_training_seed,
        "locked_task_to_seed_mapping": {
            str(index): int(seed) for index, seed in enumerate(SCREEN_SEEDS)
        },
    }
    return (
        str(selected_method),
        protocol,
        str(architecture_role),
        selected_retraining_array,
    )


def validate_checkpoint_protocol_fields(
    checkpoint: Mapping[str, Any],
    candidate: str,
    inferred_model: Mapping[str, Any],
    architecture_role: str = "primary_multiscale",
) -> Dict[str, Any]:
    """Match checkpoint runtime/model evidence to one of all five frozen families."""
    if candidate not in PROSPECTIVE_METHOD_PROTOCOLS:
        raise ValueError(f"Unknown protocol candidate: {candidate!r}")
    protocol = dict(PROSPECTIVE_METHOD_PROTOCOLS[candidate])
    if architecture_role not in {"primary_multiscale", "plain_unet_comparator"}:
        raise ValueError(f"Unknown selected architecture role: {architecture_role!r}")
    runtime_protocol = dict(protocol)
    if architecture_role == "plain_unet_comparator":
        runtime_protocol["model_type"] = "plain_unet"
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("Checkpoint has no resolved protocol configuration")
    model = resolved.get("model")
    loss = resolved.get("loss")
    augmentation = resolved.get("augmentation")
    inference = resolved.get("inference")
    if not all(isinstance(item, Mapping) for item in (model, loss, augmentation, inference)):
        raise ValueError("Checkpoint resolved protocol records are incomplete")

    expected_architecture = _normalize_architecture(runtime_protocol["model_type"])
    checks = {
        "checkpoint.model_type": (
            checkpoint.get("model_type"),
            runtime_protocol["model_type"],
        ),
        "checkpoint.loss_type": (checkpoint.get("loss_type"), protocol["loss_type"]),
        "checkpoint.num_classes": (
            checkpoint.get("num_classes"),
            protocol["model_output_classes"],
        ),
        "model.architecture": (
            model.get("architecture"),
            runtime_protocol["model_type"],
        ),
        "model.input_channels": (
            model.get("input_channels"),
            protocol["model_input_channels"],
        ),
        "model.output_classes": (
            model.get("output_classes"),
            protocol["model_output_classes"],
        ),
        "model.dropout_requested": (
            model.get("dropout_requested"),
            protocol["dropout_requested"],
        ),
        "model.deep_supervision": (model.get("deep_supervision"), False),
        "loss.type": (loss.get("type"), protocol["loss_type"]),
        "state.architecture": (
            _normalize_architecture(inferred_model.get("architecture")),
            expected_architecture,
        ),
        "state.input_channels": (
            inferred_model.get("n_channels"),
            protocol["model_input_channels"],
        ),
        "state.output_classes": (
            inferred_model.get("num_classes"),
            protocol["model_output_classes"],
        ),
        "state.deep_supervision": (
            inferred_model.get("deep_supervision"),
            False,
        ),
    }
    data_loader = augmentation.get("data_loader")
    if not isinstance(data_loader, Mapping):
        raise ValueError("Checkpoint lacks resolved data-loader protocol fields")
    for key in (
        "training_patch_size",
        "training_batch_size",
        "evaluation_patch_size",
        "evaluation_batch_size",
    ):
        checks[f"data_loader.{key}"] = (data_loader.get(key), protocol[key])
    mismatches = {
        name: {"checkpoint": actual, "locked": expected}
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint/model protocol mismatch for {candidate}: {mismatches}")

    conditional = candidate in CONDITIONAL_CANDIDATES
    if conditional:
        conditional_requirements = {
            "mode": "uint8_pore_gate_then_conditional_c0_c1",
            "raw_uint8_pore_rule": "intensity < 100",
            "raw_uint8_mineral_rule": "intensity >= 100",
            "pore_threshold_uint8": CONDITIONAL_PORE_THRESHOLD_UINT8,
            "network_outputs": ["C0", "C1"],
            "composed_outputs": ["C0", "C1", "C2"],
            "model_input_channels": 2,
            "model_input_channel_semantics": [
                "normalized_grayscale",
                "binary_recovered_pore_gate",
            ],
            "mineral_prediction_source": "fixed_raw_intensity_gate",
            "threshold_rule_acknowledged": True,
        }
        bad = {
            key: {"checkpoint": inference.get(key), "locked": value}
            for key, value in conditional_requirements.items()
            if inference.get(key) != value
        }
        if protocol.get("conditional_pore_threshold") != CONDITIONAL_PORE_THRESHOLD_UINT8:
            bad["protocol.conditional_pore_threshold"] = {
                "checkpoint": protocol.get("conditional_pore_threshold"),
                "locked": CONDITIONAL_PORE_THRESHOLD_UINT8,
            }
        if bad:
            raise ValueError(f"Conditional gate/protocol mismatch for {candidate}: {bad}")
    else:
        if inference.get("mode") != "native_model_argmax":
            raise ValueError(f"Native candidate {candidate} has a conditional inference path")
        if inference.get("network_outputs") != 3:
            raise ValueError(f"Native candidate {candidate} does not record three outputs")
        if inference.get("conditional_pore_threshold") is not None:
            raise ValueError(f"Native candidate {candidate} records an illicit threshold")
    if candidate == "C2-FP" and architecture_role == "primary_multiscale" and (
        inferred_model.get("base_features") != 32
        or inferred_model.get("bilinear") is not True
    ):
        raise ValueError("C2-FP pyramid state is not the locked 32-feature bilinear model")
    return {
        "verification_status": "matched_checkpoint_and_selected_protocol",
        "candidate": candidate,
        "selected_architecture_role": architecture_role,
        "conditional_composition": conditional,
        "winner_protocol": protocol,
        "executed_runtime_protocol": runtime_protocol,
    }


def validate_source_code_attestation(
    checkpoint: Mapping[str, Any],
    verified_lock: Mapping[str, Any],
    *,
    repository_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Match screen, checkpoint, and current execution-source checksums."""
    top_level = checkpoint.get("source_code_sha256")
    nested = _nested(checkpoint, "resolved_config", "source_code_sha256")
    locked = _nested(
        verified_lock, "screen_selection_provenance", "source_code_sha256"
    )
    if not all(isinstance(value, Mapping) and value for value in (top_level, nested, locked)):
        raise ValueError("Source-code SHA-256 attestations are missing")
    if dict(top_level) != dict(nested) or dict(top_level) != dict(locked):
        raise ValueError("Screen/checkpoint source-code attestations disagree")
    if set(top_level) != set(EXECUTION_SOURCE_FILES):
        raise ValueError(
            "Execution-source attestation has missing or extra files: "
            f"{sorted(set(top_level) ^ set(EXECUTION_SOURCE_FILES))}"
        )
    evaluator_identifier = _repository_relative(Path(__file__).resolve())
    if evaluator_identifier not in top_level:
        raise ValueError("Execution-source attestation does not freeze this evaluator")
    verified_files: Dict[str, str] = {}
    for identifier, expected_hash in sorted(top_level.items()):
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"Invalid source SHA-256 for {identifier!r}")
        path = _safe_indexed_path(repository_root, identifier)
        if not path.is_file():
            raise FileNotFoundError(f"Attested execution source is missing: {identifier}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Execution source hash drift for {identifier}: "
                f"{observed_hash} != {expected_hash}"
            )
        verified_files[str(identifier)] = observed_hash
    return {
        "verification_status": "matched_screen_checkpoint_and_live_sources",
        "file_count": len(verified_files),
        "files": verified_files,
    }


def _strip_module_prefix(state: Mapping[str, Any]) -> Dict[str, Any]:
    keys = list(state)
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module.") :]: value for key, value in state.items()}
    return dict(state)


def create_model_from_state(
    state: Mapping[str, Any], explicit_architecture: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate the exact local U-Net structure encoded by the checkpoint."""
    inferred = infer_model_config_from_state(state)
    explicit = _normalize_architecture(explicit_architecture)
    if explicit == "improved_unet" and inferred["architecture"] == "attention_unet":
        explicit = "attention_unet"
    if explicit and explicit != inferred["architecture"]:
        raise ValueError(
            f"Requested architecture {explicit!r} conflicts with checkpoint state "
            f"{inferred['architecture']!r}"
        )
    architecture = inferred["architecture"]
    if (inferred["n_channels"], inferred["num_classes"]) not in {(1, 3), (2, 2)}:
        raise ValueError(
            "Confirmatory evaluator supports only native 1-input/3-output or "
            "locked conditional 2-input/2-output checkpoints"
        )
    if architecture in {
        "multiscale_attention_unet",
        "multiscale_attention_unet_pyramid",
    }:
        from src.models.multiscale_attention_unet import (
            MultiScaleAttentionUNet,
            MultiScaleAttentionUNetPyramid,
        )

        model_class = (
            MultiScaleAttentionUNetPyramid
            if architecture == "multiscale_attention_unet_pyramid"
            else MultiScaleAttentionUNet
        )
        model = model_class(
            n_channels=inferred["n_channels"],
            n_classes=inferred["num_classes"],
            bilinear=inferred["bilinear"],
            base_features=inferred["base_features"],
            deep_supervision=inferred["deep_supervision"],
        )
    else:
        from src.models.unet_model import AttentionUNet, DeepSupervisionUNet, UNet

        model_class = {
            "unet": UNet,
            "attention_unet": AttentionUNet,
            "deep_supervision_unet": DeepSupervisionUNet,
        }[architecture]
        model = model_class(
            n_channels=inferred["n_channels"],
            n_classes=inferred["num_classes"],
            bilinear=inferred["bilinear"],
            base_features=inferred["base_features"],
        )
    model.load_state_dict(state, strict=True)
    return model, inferred


def prepare_locked_model_input(image: np.ndarray, candidate: str) -> np.ndarray:
    """Build exact native or conditional channels from one raw uint8 tile."""
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Locked model input requires a single raw uint8 grayscale tile")
    if candidate not in PROSPECTIVE_METHOD_PROTOCOLS:
        raise ValueError(f"Unknown protocol candidate: {candidate!r}")
    normalized = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
    if candidate not in CONDITIONAL_CANDIDATES:
        return normalized[np.newaxis, ...]
    gate = (image < CONDITIONAL_PORE_THRESHOLD_UINT8).astype(np.float32)
    return np.stack((normalized, gate), axis=0)


def compose_locked_probabilities(
    image: np.ndarray, network_probabilities: np.ndarray, candidate: str
) -> np.ndarray:
    """Compose conditional C0/C1 scores with fixed C2 outside the raw gate."""
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Probability composition requires the raw uint8 tile")
    if candidate not in PROSPECTIVE_METHOD_PROTOCOLS:
        raise ValueError(f"Unknown protocol candidate: {candidate!r}")
    probabilities = np.asarray(network_probabilities, dtype=np.float32)
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("Model produced non-finite probabilities")
    expected_outputs = 2 if candidate in CONDITIONAL_CANDIDATES else 3
    if probabilities.shape != (expected_outputs, *image.shape):
        raise ValueError(
            f"Network probabilities have shape {probabilities.shape}; expected "
            f"{(expected_outputs, *image.shape)} for {candidate}"
        )
    if probabilities.min() < -1e-7 or probabilities.max() > 1.0 + 1e-7:
        raise ValueError("Model probabilities are outside [0, 1]")
    if candidate not in CONDITIONAL_CANDIDATES:
        return probabilities

    pore_gate = image < CONDITIONAL_PORE_THRESHOLD_UINT8
    composed = np.zeros((3, *image.shape), dtype=np.float32)
    composed[0, pore_gate] = probabilities[0, pore_gate]
    composed[1, pore_gate] = probabilities[1, pore_gate]
    composed[2, ~pore_gate] = 1.0
    if not np.allclose(composed.sum(axis=0), 1.0, rtol=0.0, atol=2e-6):
        raise ValueError("Composed conditional probabilities do not sum to one")
    return composed


def _assert_no_symlink_components(path: Path, repository_root: Path) -> None:
    """Reject a canonical identifier whose on-disk path traverses a symlink."""
    root = repository_root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as error:
        raise ValueError(f"Canonical locked path escapes the repository: {path}") from error
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(
                f"Canonical locked path contains a symlink component: {cursor}"
            )


def validate_neural_freeze_locked_evaluation_identity(
    verified_freeze: Mapping[str, Any],
    *,
    cell_index: int,
    architecture_role: str,
    repository_root: Path = PROJECT_ROOT,
) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    """Resolve the only checkpoint/output cell approved by one neural freeze."""

    if architecture_role not in ARCHITECTURE_ROLES:
        raise ValueError(f"Unknown selected architecture role: {architecture_role!r}")
    if isinstance(cell_index, bool) or cell_index not in range(len(SCREEN_SEEDS)):
        raise ValueError("Selected retraining cell index must be 0, 1, or 2")
    if not isinstance(verified_freeze, Mapping):
        raise ValueError("Verified neural freeze must be a mapping")
    document = verified_freeze.get("document")
    if not isinstance(document, Mapping):
        raise ValueError("Verified neural freeze lacks its canonical document")
    manifest_id = str(verified_freeze.get("manifest_id", ""))
    scientific_identity = str(
        verified_freeze.get("scientific_identity_sha256", "")
    )
    if (
        document.get("manifest_id") != manifest_id
        or document.get("scientific_identity_sha256") != scientific_identity
        or manifest_id != f"neural-freeze-{scientific_identity[:16]}"
    ):
        raise ValueError("Neural-freeze manifest/scientific identity mismatch")

    retraining = document.get("selected_retraining")
    role_record = retraining.get(architecture_role) if isinstance(retraining, Mapping) else None
    cells = role_record.get("cells") if isinstance(role_record, Mapping) else None
    if not isinstance(cells, list) or len(cells) != len(SCREEN_SEEDS):
        raise ValueError("Neural freeze lacks the complete selected-retraining role")
    cell = cells[cell_index]
    if (
        not isinstance(cell, Mapping)
        or int(cell.get("array_task_index", -1)) != cell_index
        or int(cell.get("training_seed", -1)) != int(SCREEN_SEEDS[cell_index])
        or cell.get("architecture_role") != architecture_role
        or cell.get("campaign_id") != role_record.get("campaign_id")
    ):
        raise ValueError("Neural-freeze role/task/seed record is inconsistent")

    checkpoint_identifier = str(
        cell.get("checkpoint_repo_relative_identifier", "")
    )
    checkpoint_relative = Path(checkpoint_identifier)
    expected_checkpoint_identifier = (
        f"{CANONICAL_SELECTED_RETRAINING_ROOT}/{cell['campaign_id']}/"
        f"cell_{cell_index:02d}/checkpoints/best_model.pth"
    )
    if (
        not checkpoint_identifier
        or checkpoint_relative.is_absolute()
        or "\\" in checkpoint_identifier
        or checkpoint_relative.as_posix() != checkpoint_identifier
        or checkpoint_identifier != expected_checkpoint_identifier
    ):
        raise ValueError("Neural freeze names a non-canonical selected checkpoint")
    checkpoint_sha = str(cell.get("checkpoint_sha256", ""))
    semantic_sha = str(cell.get("checkpoint_state_dict_semantic_sha256", ""))
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in (checkpoint_sha, semantic_sha)
    ):
        raise ValueError("Neural freeze lacks authenticated checkpoint digests")

    root = repository_root.resolve()
    checkpoint_path = root / checkpoint_identifier
    _assert_no_symlink_components(checkpoint_path, root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError("Neural-freeze checkpoint SHA-256 mismatch before loading")

    lock_record = document.get("selected_method_lock")
    lock_identifier = (
        str(lock_record.get("repo_relative_identifier", ""))
        if isinstance(lock_record, Mapping)
        else ""
    )
    lock_path = root / lock_identifier
    if lock_identifier != "config/selected_method_lock.json":
        raise ValueError("Neural freeze names a non-canonical selected-method lock")
    _assert_no_symlink_components(lock_path, root)
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    if sha256_file(lock_path) != lock_record.get("raw_file_sha256"):
        raise ValueError("Neural-freeze selected-method lock SHA-256 mismatch")

    output_identifier = (
        f"{CANONICAL_LOCKED_EVALUATION_ROOT}/{manifest_id}/{architecture_role}/"
        f"cell_{cell_index:02d}"
    )
    output_dir = root / output_identifier
    _assert_no_symlink_components(output_dir, root)
    identity = {
        "verification_status": "matched_content_addressed_neural_freeze_cell",
        "neural_freeze_manifest_id": manifest_id,
        "neural_freeze_manifest_file_sha256": verified_freeze.get(
            "manifest_file_sha256"
        ),
        "neural_freeze_scientific_identity_sha256": scientific_identity,
        "selected_method": document.get("selected_method"),
        "architecture_role": architecture_role,
        "source_campaign_id": cell["campaign_id"],
        "cell_index": int(cell_index),
        "training_seed": int(SCREEN_SEEDS[cell_index]),
        "checkpoint_repo_relative_identifier": checkpoint_identifier,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_dict_semantic_sha256": semantic_sha,
        "output_repo_relative_identifier": output_identifier,
        "reservation_key": f"{manifest_id}:{architecture_role}:cell_{cell_index:02d}",
    }
    return checkpoint_path, lock_path, output_dir, identity


def assert_single_pass_output_available(output_dir: Path) -> Path:
    """Fail if the authenticated freeze/role/cell has ever been reserved."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Evaluation output cell already exists: {output_dir}. "
            "The held-out protocol forbids a second pass or output overwrite."
        )
    return output_dir


def reserve_single_pass_output(
    output_dir: Path,
    checkpoint_sha256: str,
    evaluation_identity: Mapping[str, Any],
) -> Path:
    """Atomically reserve one authenticated cell before any held-out byte is read."""
    assert_single_pass_output_available(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Evaluation output cell already exists: {output_dir}. "
            "The held-out protocol forbids a second pass or output overwrite."
        ) from error
    guard_path = output_dir / "held_out_access_guard.json"
    try:
        with guard_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "status": "reserved_before_held_out_access",
                    "checkpoint_sha256": checkpoint_sha256,
                    "evaluation_identity": dict(evaluation_identity),
                    "held_out_access_pass_limit": 1,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"Held-out access was already reserved for {output_dir}"
        ) from error
    return output_dir


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _format_csv(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.12g}"
    if value is None:
        return ""
    return value


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv(row.get(key)) for key in fieldnames})


def write_metric_tables(
    output_dir: Path,
    aggregate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    per_tile: Sequence[Mapping[str, Any]],
    confusion: np.ndarray,
) -> None:
    interval_map = bootstrap["intervals"]
    aggregate_rows: List[Dict[str, Any]] = [
        {
            "scope": "overall",
            "class_id": "",
            "class_name": "",
            "metric": "accuracy",
            "value": aggregate["accuracy"],
            "ci_lower": interval_map["overall.accuracy"]["lower"],
            "ci_upper": interval_map["overall.accuracy"]["upper"],
        }
    ]
    for scope in ("macro", "weighted", "micro"):
        for metric, value in aggregate[scope].items():
            key = f"{scope}.{metric}"
            aggregate_rows.append(
                {
                    "scope": scope,
                    "class_id": "",
                    "class_name": "",
                    "metric": metric,
                    "value": value,
                    "ci_lower": interval_map[key]["lower"],
                    "ci_upper": interval_map[key]["upper"],
                }
            )
    for metric, value in aggregate["selection_metrics"].items():
        key = f"selection.{metric}"
        aggregate_rows.append(
            {
                "scope": "selection_and_pore_union",
                "class_id": "",
                "class_name": "",
                "metric": metric,
                "value": value,
                "ci_lower": interval_map[key]["lower"],
                "ci_upper": interval_map[key]["upper"],
            }
        )
    for metric in ("accuracy", "agreement"):
        key = f"pore_vs_mineral.{metric}"
        aggregate_rows.append(
            {
                "scope": "merged_pore_vs_mineral",
                "class_id": "",
                "class_name": "",
                "metric": metric,
                "value": aggregate["pore_vs_mineral"][metric],
                "ci_lower": interval_map[key]["lower"],
                "ci_upper": interval_map[key]["upper"],
            }
        )
    for item in aggregate["pore_vs_mineral"]["per_class"]:
        for metric in ("iou", "dice", "precision", "recall", "f1"):
            key = f"merged_class_{item['class_id']}.{metric}"
            aggregate_rows.append(
                {
                    "scope": "merged_pore_vs_mineral_class",
                    "class_id": item["class_id"],
                    "class_name": item["class_name"],
                    "metric": metric,
                    "value": item[metric],
                    "ci_lower": interval_map[key]["lower"],
                    "ci_upper": interval_map[key]["upper"],
                }
            )
    for item in aggregate["per_class"]:
        for metric in ("iou", "dice", "precision", "recall", "f1"):
            key = f"class_{item['class_id']}.{metric}"
            aggregate_rows.append(
                {
                    "scope": "class",
                    "class_id": item["class_id"],
                    "class_name": item["class_name"],
                    "metric": metric,
                    "value": item[metric],
                    "ci_lower": interval_map[key]["lower"],
                    "ci_upper": interval_map[key]["upper"],
                }
            )
    for row in aggregate_rows:
        row.update(
            {
                "bootstrap_unit": bootstrap["sampling_unit"],
                "bootstrap_replicates": bootstrap["replicates"],
                "bootstrap_seed": bootstrap["seed"],
                "confidence": bootstrap["confidence"],
            }
        )
    write_csv(
        output_dir / "aggregate_metrics.csv",
        (
            "scope",
            "class_id",
            "class_name",
            "metric",
            "value",
            "ci_lower",
            "ci_upper",
            "bootstrap_unit",
            "bootstrap_replicates",
            "bootstrap_seed",
            "confidence",
        ),
        aggregate_rows,
    )

    per_tile_rows: List[Dict[str, Any]] = []
    for tile in per_tile:
        metrics = tile["metrics"]
        row: Dict[str, Any] = {
            "image_id": tile["image_id"],
            "file_name": tile["file_name"],
            "height": tile["height"],
            "width": tile["width"],
            "pixels": metrics["total_pixels"],
            "accuracy": metrics["accuracy"],
        }
        for scope in ("macro", "weighted", "micro"):
            for metric, value in metrics[scope].items():
                row[f"{scope}_{metric}"] = value
        for item in metrics["per_class"]:
            prefix = f"class_{item['class_id']}"
            for metric in ("support_pixels", "iou", "dice", "precision", "recall", "f1"):
                row[f"{prefix}_{metric}"] = item[metric]
        for metric, value in metrics["selection_metrics"].items():
            row[f"selection_{metric}"] = value
        row["pore_vs_mineral_accuracy"] = metrics["pore_vs_mineral"]["accuracy"]
        row["pore_vs_mineral_agreement"] = metrics["pore_vs_mineral"]["agreement"]
        for item in metrics["pore_vs_mineral"]["per_class"]:
            prefix = f"merged_class_{item['class_id']}"
            for metric in (
                "support_pixels",
                "iou",
                "dice",
                "precision",
                "recall",
                "f1",
            ):
                row[f"{prefix}_{metric}"] = item[metric]
        gate_diagnostic = tile.get("conditional_gate_reference_diagnostic")
        if isinstance(gate_diagnostic, Mapping):
            for key in (
                "mismatched_pixels",
                "mismatch_rate",
                "agreement_rate",
                "reference_pore_gate_mineral_pixels",
                "reference_c0_gate_mineral_pixels",
                "reference_c1_gate_mineral_pixels",
                "reference_mineral_gate_pore_pixels",
            ):
                row[f"conditional_gate_{key}"] = gate_diagnostic[key]
        per_tile_rows.append(row)
    tile_fields = list(per_tile_rows[0])
    write_csv(output_dir / "per_tile_metrics.csv", tile_fields, per_tile_rows)

    confusion_rows = []
    for true_id in range(3):
        for predicted_id in range(3):
            confusion_rows.append(
                {
                    "true_class_id": true_id,
                    "true_class_name": CLASS_NAMES[true_id],
                    "predicted_class_id": predicted_id,
                    "predicted_class_name": CLASS_NAMES[predicted_id],
                    "pixel_count": int(confusion[true_id, predicted_id]),
                }
            )
    write_csv(
        output_dir / "confusion_matrix.csv",
        (
            "true_class_id",
            "true_class_name",
            "predicted_class_id",
            "predicted_class_name",
            "pixel_count",
        ),
        confusion_rows,
    )

    per_tile_confusion_rows = []
    for tile in per_tile:
        matrix = np.asarray(tile["confusion_matrix"], dtype=np.int64)
        for true_id in range(3):
            for predicted_id in range(3):
                per_tile_confusion_rows.append(
                    {
                        "image_id": tile["image_id"],
                        "file_name": tile["file_name"],
                        "true_class_id": true_id,
                        "true_class_name": CLASS_NAMES[true_id],
                        "predicted_class_id": predicted_id,
                        "predicted_class_name": CLASS_NAMES[predicted_id],
                        "pixel_count": int(matrix[true_id, predicted_id]),
                    }
                )
    write_csv(
        output_dir / "per_tile_confusion.csv",
        (
            "image_id",
            "file_name",
            "true_class_id",
            "true_class_name",
            "predicted_class_id",
            "predicted_class_name",
            "pixel_count",
        ),
        per_tile_confusion_rows,
    )


def write_gate_reference_diagnostics(
    output_dir: Path,
    aggregate: Mapping[str, Any],
    per_tile: Sequence[Mapping[str, Any]],
) -> None:
    """Persist exact fixed-gate mismatch evidence separately from model metrics."""
    fields = (
        "scope",
        "evaluation_ordinal",
        "image_id",
        "file_name",
        "total_pixels",
        "matched_pixels",
        "mismatched_pixels",
        "mismatch_rate",
        "agreement_rate",
        "reference_pore_gate_mineral_pixels",
        "reference_c0_gate_mineral_pixels",
        "reference_c1_gate_mineral_pixels",
        "reference_mineral_gate_pore_pixels",
        "reference_pore_pixels",
        "reference_mineral_pixels",
        "gate_pore_pixels",
        "gate_mineral_pixels",
        "threshold_uint8",
        "threshold_rule",
        "interpretation",
    )

    def row_from_diagnostic(
        diagnostic: Mapping[str, Any],
        *,
        scope: str,
        ordinal: Any = "",
        image_id: Any = "",
        file_name: str = "",
    ) -> Dict[str, Any]:
        return {
            "scope": scope,
            "evaluation_ordinal": ordinal,
            "image_id": image_id,
            "file_name": file_name,
            **{key: diagnostic.get(key) for key in fields[4:]},
        }

    rows = [row_from_diagnostic(aggregate, scope="aggregate_held_out_test")]
    for tile in per_tile:
        diagnostic = tile.get("conditional_gate_reference_diagnostic")
        if not isinstance(diagnostic, Mapping):
            raise ValueError("Conditional tile is missing gate/reference diagnostic")
        rows.append(
            row_from_diagnostic(
                diagnostic,
                scope="held_out_tile",
                ordinal=tile["evaluation_ordinal"],
                image_id=tile["image_id"],
                file_name=tile["file_name"],
            )
        )
    write_csv(output_dir / "conditional_gate_reference_diagnostic.csv", fields, rows)


def write_secondary_2d_diagnostic_tables(
    output_dir: Path,
    aggregate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    per_tile: Sequence[Mapping[str, Any]],
) -> None:
    """Write long aggregate and wide per-tile 2D operational diagnostics."""
    definitions = {
        "c0_area_fraction": "C0 pixels divided by all native-tile pixels",
        "c1_area_fraction": "C1 pixels divided by all native-tile pixels",
        "largest_c1_8_connected_component_c1_fraction": (
            "pixels in largest unfiltered 8-connected C1 component divided by C1 pixels; "
            "undefined when C1 is absent"
        ),
        "c0_8_connected_components_per_megapixel": (
            "unfiltered 8-connected C0 component count per 1,000,000 pixels"
        ),
        "pore_union_boundary_f1_at_2px": (
            "pore-union boundary agreement with fixed 2-pixel Chebyshev tolerance"
        ),
    }
    aggregate_values = _secondary_diagnostic_metric_lookup(aggregate)
    aggregate_rows = []
    for key, value in sorted(aggregate_values.items()):
        metric_name = key.split(".", 1)[0]
        interval = bootstrap["intervals"][key]
        aggregate_rows.append(
            {
                "metric": key,
                "value": value,
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
                "definition": definitions[metric_name],
                "selection_role": "secondary_only_never_used_for_model_selection",
                "interpretation": (
                    "2D operational agreement only; not permeability or 3D connectivity"
                ),
                "bootstrap_unit": bootstrap["sampling_unit"],
                "bootstrap_replicates": bootstrap["replicates"],
                "bootstrap_seed": bootstrap["seed"],
                "confidence": bootstrap["confidence"],
            }
        )
    write_csv(
        output_dir / "secondary_2d_operational_aggregate.csv",
        (
            "metric",
            "value",
            "ci_lower",
            "ci_upper",
            "definition",
            "selection_role",
            "interpretation",
            "bootstrap_unit",
            "bootstrap_replicates",
            "bootstrap_seed",
            "confidence",
        ),
        aggregate_rows,
    )

    tile_rows = []
    for tile in per_tile:
        diagnostic = tile.get("secondary_2d_operational_diagnostics")
        if not isinstance(diagnostic, Mapping):
            raise ValueError("Held-out tile is missing secondary 2D diagnostics")
        row = {
            "evaluation_ordinal": tile["evaluation_ordinal"],
            "image_id": tile["image_id"],
            "file_name": tile["file_name"],
        }
        row.update(_secondary_diagnostic_metric_lookup(diagnostic))
        row.update(
            {
                f"count.{key}": value
                for key, value in diagnostic["sufficient_statistics"].items()
            }
        )
        tile_rows.append(row)
    write_csv(
        output_dir / "secondary_2d_operational_per_tile.csv",
        list(tile_rows[0]),
        tile_rows,
    )


def write_curve_tables(
    output_dir: Path,
    positive_histograms: np.ndarray,
    negative_histograms: np.ndarray,
) -> Dict[str, Any]:
    bins = positive_histograms.shape[1]
    histogram_rows = []
    roc_rows = []
    pr_rows = []
    summary: Dict[str, Any] = {}
    for class_id in range(3):
        for bin_id in range(bins):
            histogram_rows.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "bin_id": bin_id,
                    "score_lower": bin_id / bins,
                    "score_upper": (bin_id + 1) / bins,
                    "positive_pixels": int(positive_histograms[class_id, bin_id]),
                    "negative_pixels": int(negative_histograms[class_id, bin_id]),
                }
            )
        curve = curves_from_histograms(
            positive_histograms[class_id], negative_histograms[class_id]
        )
        summary[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "positive_pixels": curve["positive_pixels"],
            "negative_pixels": curve["negative_pixels"],
            "roc_auc_histogram_approximation": curve["roc_auc"],
            "average_precision_histogram_approximation": curve["average_precision"],
            "pr_auc_histogram_approximation": curve["pr_auc"],
        }
        for index, threshold in enumerate(curve["thresholds"]):
            common = {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "threshold_lower_edge": threshold,
                "cumulative_true_positive": int(curve["cumulative_true_positive"][index]),
                "cumulative_false_positive": int(curve["cumulative_false_positive"][index]),
                "positive_pixels": curve["positive_pixels"],
                "negative_pixels": curve["negative_pixels"],
            }
            roc_rows.append(
                {
                    **common,
                    "false_positive_rate": curve["false_positive_rate"][index],
                    "true_positive_rate": curve["true_positive_rate"][index],
                }
            )
            pr_rows.append(
                {
                    **common,
                    "recall": curve["recall"][index],
                    "precision": curve["precision"][index],
                }
            )
    write_csv(
        output_dir / "probability_histograms.csv",
        (
            "class_id",
            "class_name",
            "bin_id",
            "score_lower",
            "score_upper",
            "positive_pixels",
            "negative_pixels",
        ),
        histogram_rows,
    )
    write_csv(
        output_dir / "roc_curve.csv",
        (
            "class_id",
            "class_name",
            "threshold_lower_edge",
            "false_positive_rate",
            "true_positive_rate",
            "cumulative_true_positive",
            "cumulative_false_positive",
            "positive_pixels",
            "negative_pixels",
        ),
        roc_rows,
    )
    write_csv(
        output_dir / "precision_recall_curve.csv",
        (
            "class_id",
            "class_name",
            "threshold_lower_edge",
            "recall",
            "precision",
            "cumulative_true_positive",
            "cumulative_false_positive",
            "positive_pixels",
            "negative_pixels",
        ),
        pr_rows,
    )
    return summary


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(PUBLICATION_SANS_SERIF_FONTS),
            "mathtext.fontset": "dejavusans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _publication_curve_style(class_id: int, point_count: int) -> Dict[str, Any]:
    """Return the locked colour-plus-shape encoding for a class curve."""
    if class_id not in range(len(PUBLICATION_CLASS_COLORS)):
        raise ValueError(f"Unknown publication class ID: {class_id}")
    if point_count < 2:
        raise ValueError("Publication curves require at least two points")
    color = PUBLICATION_CLASS_COLORS[class_id]
    return {
        "color": color,
        "linestyle": PUBLICATION_CLASS_LINE_STYLES[class_id],
        "linewidth": 2.0,
        "marker": PUBLICATION_CLASS_MARKERS[class_id],
        "markevery": max(1, point_count // 12),
        "markersize": 5.0,
        "markerfacecolor": "white",
        "markeredgecolor": color,
        "markeredgewidth": 1.0,
    }


def plot_confusion_matrix(confusion: np.ndarray, output_dir: Path, tile_count: int) -> None:
    plt = _configure_matplotlib()
    from matplotlib.ticker import PercentFormatter

    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros(confusion.shape, dtype=np.float64),
        where=row_totals != 0,
    )
    fig, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    labels = [CLASS_LABELS[index] for index in range(3)]
    axis.set_xticks(range(3), labels=labels, rotation=20, ha="right")
    axis.set_yticks(range(3), labels=labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Reference class")
    axis.set_title("Held-out test confusion matrix", loc="left", pad=24, fontweight="semibold")
    axis.text(
        0,
        1.025,
        f"Row-normalized; counts aggregated across {tile_count} native 2048×2048 tiles",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
        va="bottom",
    )
    for row in range(3):
        for column in range(3):
            value = normalized[row, column]
            color = "white" if value > 0.52 else "#222222"
            axis.text(
                column,
                row,
                f"{value:.1%}\n{int(confusion[row, column]):,}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Share of reference-class pixels")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"confusion_matrix.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_curves(
    positive_histograms: np.ndarray,
    negative_histograms: np.ndarray,
    output_dir: Path,
    tile_count: int,
) -> None:
    plt = _configure_matplotlib()
    curve_data = [
        curves_from_histograms(positive_histograms[index], negative_histograms[index])
        for index in range(3)
    ]
    curve_bins = int(positive_histograms.shape[1])
    specifications = {
        "precision_recall": {
            "x": "recall",
            "y": "precision",
            "summary": "average_precision",
            "summary_label": "AP",
            "xlabel": "Recall",
            "ylabel": "Precision",
            "title": "One-vs-rest precision–recall curves",
            "legend_location": "lower left",
            "filename": "precision_recall_curves",
        },
        "roc": {
            "x": "false_positive_rate",
            "y": "true_positive_rate",
            "summary": "roc_auc",
            "summary_label": "AUC",
            "xlabel": "False-positive rate",
            "ylabel": "True-positive rate",
            "title": "One-vs-rest ROC curves (supplementary)",
            "legend_location": "lower right",
            "filename": "roc_curves",
        },
    }
    for figure_kind in PUBLICATION_CURVE_ORDER:
        specification = specifications[figure_kind]
        fig, axis = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
        for class_id, curve in enumerate(curve_data):
            x_values = curve[specification["x"]]
            axis.plot(
                x_values,
                curve[specification["y"]],
                **_publication_curve_style(class_id, len(x_values)),
                label=(
                    f"{CLASS_LABELS[class_id]} "
                    f"({specification['summary_label']} "
                    f"{curve[specification['summary']]:.3f})"
                ),
            )
        if figure_kind == "roc":
            axis.plot(
                [0, 1],
                [0, 1],
                color="#666666",
                linewidth=1.0,
                linestyle=":",
                label="Chance",
            )
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel=specification["xlabel"],
            ylabel=specification["ylabel"],
        )
        axis.set_title(
            specification["title"], loc="left", pad=24, fontweight="semibold"
        )
        axis.text(
            0,
            1.025,
            f"Held-out test set; {tile_count} native tiles; "
            f"{curve_bins}-bin score approximation",
            transform=axis.transAxes,
            fontsize=9,
            color="#555555",
            va="bottom",
        )
        axis.grid(True, color="#E2E2E2", linewidth=0.65)
        axis.legend(loc=specification["legend_location"], frameon=False)
        for extension in ("pdf", "png"):
            fig.savefig(
                output_dir / f"{specification['filename']}.{extension}",
                dpi=600 if extension == "png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)


def plot_class_metric_summary(
    aggregate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    output_dir: Path,
    tile_count: int,
) -> None:
    """Plot per-class IoU and Dice with whole-tile bootstrap intervals."""
    plt = _configure_matplotlib()
    from matplotlib.ticker import PercentFormatter

    positions = np.arange(3, dtype=np.float64)
    offsets = {"iou": -0.13, "dice": 0.13}
    styles = {
        "iou": {"label": "IoU", "marker": "o", "open_marker": False},
        "dice": {"label": "Dice", "marker": "s", "open_marker": True},
    }
    fig, axis = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    for class_id in range(3):
        color = PUBLICATION_CLASS_COLORS[class_id]
        for metric in ("iou", "dice"):
            value = float(aggregate["per_class"][class_id][metric])
            interval = bootstrap["intervals"][f"class_{class_id}.{metric}"]
            lower = float(interval["lower"])
            upper = float(interval["upper"])
            x_position = positions[class_id] + offsets[metric]
            axis.errorbar(
                x_position,
                value,
                yerr=np.asarray(
                    [
                        [max(0.0, value - lower)],
                        [max(0.0, upper - value)],
                    ]
                ),
                fmt=styles[metric]["marker"],
                color=color,
                markerfacecolor="white" if styles[metric]["open_marker"] else color,
                markeredgewidth=1.4,
                markersize=7,
                capsize=4,
                elinewidth=1.3,
                linewidth=0,
            )
            axis.annotate(
                f"{styles[metric]['label']} {value:.3f}",
                (x_position, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
    axis.set_xticks(positions, labels=[CLASS_LABELS[index] for index in range(3)])
    for tick_label, color in zip(axis.get_xticklabels(), PUBLICATION_CLASS_COLORS):
        tick_label.set_color(color)
        tick_label.set_fontweight("semibold")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_title("Per-class IoU and Dice", loc="left", pad=24, fontweight="semibold")
    axis.text(
        0,
        1.025,
        f"Held-out test set; {tile_count} native tiles; 95% whole-tile bootstrap intervals",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
        va="bottom",
    )
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"per_class_iou_dice.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_qualitative_triptych(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    image_id: int,
    file_name: str,
    output_dir: Path,
) -> None:
    """Render the outcome-independent, preselected held-out example tile."""
    plt = _configure_matplotlib()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    segmentation_cmap = ListedColormap(PUBLICATION_CLASS_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.8))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].imshow(target, cmap=segmentation_cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    axes[2].imshow(prediction, cmap=segmentation_cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    for axis, title in zip(axes, ("Input", "Lossless reference", "Model prediction")):
        axis.set_title(title, fontsize=11, pad=6)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.7)
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.15, top=0.84, wspace=0.045)
    fig.suptitle(
        "Qualitative segmentation comparison",
        x=0.025,
        y=0.98,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    # Tile identity and the outcome-independent selection rule remain in the
    # authenticated JSON report rather than being embedded in the artwork.
    legend = [
        Patch(facecolor=segmentation_cmap.colors[class_id], edgecolor="#444444", label=CLASS_LABELS[class_id])
        for class_id in range(3)
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"qualitative_triptych.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_locked_inference_runtime(
    device: str, amp_enabled: bool, *, cuda_device_name: Optional[str] = None
) -> Dict[str, Any]:
    """Freeze confirmatory arithmetic before the irreversible held-out pass."""
    if not device.startswith("cuda"):
        raise ValueError(
            "Locked confirmatory inference requires CUDA float16 autocast; "
            f"resolved device was {device!r}"
        )
    if amp_enabled is not True:
        raise ValueError("Locked confirmatory inference requires AMP enabled")
    if not isinstance(cuda_device_name, str) or LOCKED_CUDA_DEVICE_MODEL_TOKEN not in (
        cuda_device_name
    ):
        raise ValueError(
            "Locked confirmatory inference requires an NVIDIA L40S matching the "
            f"validation campaign; resolved device was {cuda_device_name!r}"
        )
    return {
        "verification_status": "matched_locked_inference_precision",
        "precision_protocol": LOCKED_INFERENCE_PRECISION,
        "device_family": "cuda",
        "cuda_device_name": cuda_device_name,
        "locked_device_model_token": LOCKED_CUDA_DEVICE_MODEL_TOKEN,
        "autocast_dtype": "float16",
        "caller_selectable": False,
    }


def validate_locked_evaluator_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """Reject first-pass metric/evidence choices not frozen in evaluator source."""
    checks = {
        "seed": (args.seed, LOCKED_EVALUATOR_INFERENCE_SEED),
        "bootstrap_seed": (args.bootstrap_seed, LOCKED_BOOTSTRAP_SEED),
        "bootstrap_replicates": (
            args.bootstrap_replicates,
            LOCKED_BOOTSTRAP_REPLICATES,
        ),
        "confidence": (args.confidence, LOCKED_CONFIDENCE),
        "curve_bins": (args.curve_bins, LOCKED_CURVE_BINS),
    }
    mismatches = {
        key: {"requested": actual, "locked": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"Confirmatory evaluator parameter drift: {mismatches}")
    return {
        "verification_status": "matched_prospectively_locked_evaluator_parameters",
        "evaluator_inference_seed": LOCKED_EVALUATOR_INFERENCE_SEED,
        "bootstrap_seed": LOCKED_BOOTSTRAP_SEED,
        "bootstrap_replicates": LOCKED_BOOTSTRAP_REPLICATES,
        "confidence": LOCKED_CONFIDENCE,
        "curve_bins": LOCKED_CURVE_BINS,
        "caller_selectable": False,
    }


def _autocast_context(torch: Any, device: str, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    device_type = device.split(":", 1)[0]
    if device_type != "cuda":
        raise ValueError("Locked automatic mixed precision is supported only on CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def preflight_locked_model_output(
    model: Any,
    torch: Any,
    *,
    device: str,
    input_channels: int,
    output_classes: int,
    amp_enabled: bool,
) -> Dict[str, Any]:
    """Run one synthetic full-tile shape/finite check before held-out reservation."""
    synthetic = torch.zeros(
        (1, int(input_channels), *EXPECTED_TILE_SHAPE),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode(), _autocast_context(torch, device, amp_enabled):
        output = model(synthetic)
        logits = output[0] if isinstance(output, (tuple, list)) else output
    expected_shape = (1, int(output_classes), *EXPECTED_TILE_SHAPE)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Synthetic full-tile model output shape {tuple(logits.shape)} != "
            f"{expected_shape}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("Synthetic full-tile model preflight produced non-finite logits")
    output_kind = "deep_supervision_container_main_logits" if isinstance(
        output, (tuple, list)
    ) else "single_logits_tensor"
    del synthetic, output, logits
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "verification_status": "synthetic_full_tile_shape_and_finite_preflight_passed",
        "held_out_bytes_read": 0,
        "input_shape": [1, int(input_channels), *EXPECTED_TILE_SHAPE],
        "output_shape": list(expected_shape),
        "output_kind": output_kind,
    }


def evaluate(args: argparse.Namespace) -> Path:
    evaluator_parameter_attestation = validate_locked_evaluator_parameters(args)
    try:
        import cv2
        import torch
    except ImportError as error:
        raise RuntimeError("Evaluation requires PyTorch and opencv-python") from error

    # Authenticate the complete validation-only neural freeze before any
    # development or held-out corpus path is opened. The freeze, rather than a
    # caller-selected campaign alias, determines the checkpoint and output.
    verified_freeze = load_verified_neural_freeze_manifest(
        args.neural_freeze_id, PROJECT_ROOT
    )
    checkpoint_path, lock_path, output_dir, locked_evaluation_identity = (
        validate_neural_freeze_locked_evaluation_identity(
            verified_freeze,
            cell_index=args.cell_index,
            architecture_role=args.architecture_role,
        )
    )
    assert_single_pass_output_available(output_dir)

    annotation_path = _project_path(args.annotations)
    image_dir = _project_path(args.image_dir)
    mask_dir = _project_path(args.mask_dir)
    manifest_path = _project_path(args.split_manifest)
    for path in (checkpoint_path, annotation_path, manifest_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        _repository_relative(path)
    for directory in (image_dir, mask_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        _repository_relative(directory)

    required_identifiers = {
        "annotations": (
            annotation_path,
            "results/step3_coco_dataset/pore_annotations.json",
        ),
        "images": (image_dir, "results/step3_coco_dataset/images"),
        "lossless masks": (
            mask_dir,
            "results/step2_pore_classification/pore_classifications",
        ),
        "split manifest": (manifest_path, "config/confirmatory_splits.json"),
    }
    for label, (path, expected_identifier) in required_identifiers.items():
        observed_identifier = _repository_relative(path)
        if observed_identifier != expected_identifier:
            raise ValueError(
                f"Locked {label} path drift: {observed_identifier!r} != "
                f"{expected_identifier!r}"
            )
    annotation_sha256 = sha256_file(annotation_path)
    if annotation_sha256 != LOCKED_ANNOTATION_INDEX_SHA256:
        raise ValueError(
            "Canonical COCO image index SHA-256 drifted: "
            f"{annotation_sha256} != {LOCKED_ANNOTATION_INDEX_SHA256}"
        )

    checkpoint_sha = str(locked_evaluation_identity["checkpoint_sha256"])

    checkpoint = _torch_load_trusted(checkpoint_path, "cpu")
    validate_selected_checkpoint(checkpoint)
    recorded_normalization = validate_checkpoint_normalization(checkpoint)
    verified_lock, lock_sha256, lock_identifier = load_verified_selected_method_lock(
        lock_path
    )
    (
        selected_method,
        selected_protocol,
        selected_architecture_role,
        selected_retraining_array,
    ) = validate_checkpoint_selected_method_lock(
        checkpoint,
        verified_lock,
        lock_sha256=lock_sha256,
        lock_identifier=lock_identifier,
    )
    if (
        selected_method != verified_freeze["selected_method"]
        or selected_architecture_role != args.architecture_role
        or int(selected_retraining_array["array_task_index"]) != args.cell_index
        or selected_retraining_array["campaign_id"]
        != locked_evaluation_identity["source_campaign_id"]
    ):
        raise ValueError("Checkpoint does not match its authenticated neural-freeze cell")

    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_sha256 = sha256_file(manifest_path)
    split_ids = resolve_complete_manifest(coco, manifest)
    test_ids = split_ids["test"]
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    relative_name_by_id = {
        image_id: _safe_indexed_path(image_dir, image["file_name"]).relative_to(
            image_dir
        )
        for image_id, image in image_by_id.items()
    }
    split_files = {
        split_name: [relative_name_by_id[image_id].as_posix() for image_id in image_ids]
        for split_name, image_ids in split_ids.items()
    }
    split_attestation = validate_checkpoint_split_isolation(
        checkpoint,
        manifest_sha256=manifest_sha256,
        split_ids=split_ids,
        split_files=split_files,
    )
    state = _strip_module_prefix(checkpoint["model_state_dict"])
    inferred_model = infer_model_config_from_state(state)
    protocol_attestation = validate_checkpoint_protocol_fields(
        checkpoint,
        selected_method,
        inferred_model,
        selected_architecture_role,
    )
    source_attestation = validate_source_code_attestation(checkpoint, verified_lock)
    training_seed_attestation = validate_checkpoint_training_seed(
        checkpoint,
        verified_lock,
        candidate=selected_method,
        architecture_role=selected_architecture_role,
    )
    device = choose_device(args.device)
    cuda_device_name = torch.cuda.get_device_name(torch.device(device))
    precision_attestation = validate_locked_inference_runtime(
        device, args.amp, cuda_device_name=cuda_device_name
    )
    architecture_assertion = (
        "plain_unet" if args.architecture_role == "plain_unet_comparator" else None
    )
    model, model_config = create_model_from_state(state, architecture_assertion)
    model = model.to(device)
    model.eval()
    model_preflight_attestation = preflight_locked_model_output(
        model,
        torch,
        device=device,
        input_channels=int(selected_protocol["model_input_channels"]),
        output_classes=int(selected_protocol["model_output_classes"]),
        amp_enabled=args.amp,
    )

    # Development masks and inputs are authenticated only after every lock and
    # source check, and before any held-out byte may be read.
    development_attestation, development_mask_payloads = (
        attest_development_masks_once(mask_dir, split_files)
    )
    development_attestation["checkpoint_development_attested_scope"] = True
    development_attestation["held_out_test_included"] = False
    checkpoint_target_provenance = validate_checkpoint_target_provenance(
        checkpoint, development_attestation
    )
    development_input_attestations, development_image_payloads = (
        attest_development_inputs_once(image_dir, split_files)
    )
    for record in development_input_attestations.values():
        record["checkpoint_development_attested_scope"] = True
        record["held_out_test_included"] = False
    checkpoint_input_provenance = validate_checkpoint_input_provenance(
        checkpoint, development_input_attestations
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if not args.no_publication_plots:
        _configure_matplotlib()

    # This immutable guard makes a crash or rejection after first test access
    # consume the one allowed pass instead of silently permitting a rerun.
    output_dir = reserve_single_pass_output(
        output_dir, checkpoint_sha, locked_evaluation_identity
    )
    publication_dir = output_dir / "publication"
    if not args.no_publication_plots:
        publication_dir.mkdir(parents=True, exist_ok=True)

    post_reservation_attestations, test_mask_payloads, test_mask_sha256 = (
        attest_held_out_and_full_once(
            mask_dir, split_files, development_mask_payloads
        )
    )
    del development_mask_payloads
    target_attestations = {
        "development_train_plus_validation": development_attestation,
        **post_reservation_attestations,
    }
    post_input_attestations, test_image_payloads, test_image_sha256 = (
        attest_held_out_inputs_and_full_once(
            image_dir, split_files, development_image_payloads
        )
    )
    del development_image_payloads
    input_attestations = {
        **development_input_attestations,
        **post_input_attestations,
    }

    qualitative_image_id = min(
        test_ids,
        key=lambda image_id: (relative_name_by_id[image_id].as_posix(), image_id),
    )
    qualitative_file_name = relative_name_by_id[qualitative_image_id].as_posix()
    qualitative_example = {
        "selection_rule": (
            "lexicographically first file_name in the locked test manifest; "
            "selected before inference without reference to model performance"
        ),
        "image_id": qualitative_image_id,
        "file_name": qualitative_file_name,
        "input_image_sha256": test_image_sha256[qualitative_file_name],
        "target_mask_sha256": test_mask_sha256[qualitative_file_name],
        "post_processing": "none",
    }
    qualitative_arrays: Optional[Dict[str, np.ndarray]] = None

    positive_histograms = np.zeros((3, args.curve_bins), dtype=np.int64)
    negative_histograms = np.zeros((3, args.curve_bins), dtype=np.int64)
    tile_confusions: List[np.ndarray] = []
    gate_reference_diagnostics: List[Dict[str, Any]] = []
    secondary_diagnostics: List[Dict[str, Any]] = []
    per_tile: List[Dict[str, Any]] = []
    expected_network_outputs = int(selected_protocol["model_output_classes"])

    for tile_number, image_id in enumerate(test_ids, start=1):
        image_info = image_by_id[image_id]
        relative_name = relative_name_by_id[image_id]
        file_name = relative_name.as_posix()
        image_path = _safe_indexed_path(image_dir, relative_name)
        try:
            image_payload = test_image_payloads.pop(file_name)
        except KeyError as error:
            raise RuntimeError(
                f"Held-out input payload was missing or reused: {file_name}"
            ) from error
        image_sha256 = test_image_sha256[file_name]
        image = cv2.imdecode(
            np.frombuffer(image_payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if image is None:
            raise ValueError(f"Failed to load test image: {image_path}")
        if image.dtype != np.uint8 or image.ndim != 2 or image.shape != EXPECTED_TILE_SHAPE:
            raise ValueError(
                f"Test image {file_name} is not a raw uint8 2048x2048 tile"
            )
        if (int(image_info["height"]), int(image_info["width"])) != image.shape:
            raise ValueError(f"COCO dimensions disagree with image file for {file_name}")
        try:
            target_payload = test_mask_payloads.pop(file_name)
        except KeyError as error:
            raise RuntimeError(f"Held-out mask payload was missing or reused: {file_name}") from error
        target = load_lossless_target_mask_bytes(
            _safe_indexed_path(mask_dir, file_name), target_payload
        )
        model_input = prepare_locked_model_input(image, selected_method)
        tensor = torch.from_numpy(model_input).unsqueeze(0).to(device)
        with torch.inference_mode(), _autocast_context(torch, device, args.amp):
            output = model(tensor)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            expected_shape = (1, expected_network_outputs, *EXPECTED_TILE_SHAPE)
            if tuple(logits.shape) != expected_shape:
                raise ValueError(
                    f"Unexpected model output shape: {tuple(logits.shape)} != {expected_shape}"
                )
            network_probabilities = (
                torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            )
        probabilities = compose_locked_probabilities(
            image, network_probabilities, selected_method
        )
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        confusion = confusion_from_labels(target, prediction)
        tile_gate_diagnostic = (
            gate_reference_diagnostic(image, target)
            if selected_method in CONDITIONAL_CANDIDATES
            else None
        )
        if tile_gate_diagnostic is not None:
            gate_reference_diagnostics.append(tile_gate_diagnostic)
        tile_secondary_diagnostic = secondary_2d_operational_diagnostics(
            target, prediction
        )
        secondary_diagnostics.append(tile_secondary_diagnostic)
        tile_confusions.append(confusion)
        per_tile.append(
            {
                "evaluation_ordinal": tile_number,
                "image_id": image_id,
                "file_name": file_name,
                "input_image_sha256": image_sha256,
                "target_mask_sha256": test_mask_sha256[file_name],
                "height": image.shape[0],
                "width": image.shape[1],
                "confusion_matrix": confusion,
                "metrics": metrics_from_confusion(confusion),
                "conditional_gate_reference_diagnostic": tile_gate_diagnostic,
                "secondary_2d_operational_diagnostics": tile_secondary_diagnostic,
            }
        )
        if image_id == qualitative_image_id:
            if not args.no_publication_plots:
                qualitative_arrays = {
                    "image": image.copy(),
                    "target": target.copy(),
                    "prediction": prediction.copy(),
                }
        update_probability_histograms(
            positive_histograms, negative_histograms, probabilities, target
        )
        del (
            image_payload,
            target_payload,
            tensor,
            output,
            logits,
            network_probabilities,
            probabilities,
            prediction,
            target,
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"Evaluated held-out tile {tile_number}/{len(test_ids)}: {file_name}", flush=True)

    if test_mask_payloads or test_image_payloads:
        raise RuntimeError(
            "Not every attested held-out input/target was consumed exactly once: "
            f"inputs={sorted(test_image_payloads)}, targets={sorted(test_mask_payloads)}"
        )
    consumed_guard = output_dir / "held_out_access_consumed.json"
    with consumed_guard.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "status": "single_held_out_pass_consumed",
                "checkpoint_sha256": checkpoint_sha,
                "test_tile_count": len(test_ids),
                "test_mask_read_passes": 1,
                "test_input_read_passes": 1,
                "test_inference_passes": 1,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    tile_confusion_array = np.stack(tile_confusions)
    aggregate_confusion = tile_confusion_array.sum(axis=0)
    aggregate = metrics_from_confusion(aggregate_confusion)
    aggregate_gate_diagnostic: Optional[Dict[str, Any]] = None
    if selected_method in CONDITIONAL_CANDIDATES:
        if len(gate_reference_diagnostics) != len(test_ids):
            raise RuntimeError("Conditional gate diagnostics are incomplete")
        aggregate_gate_diagnostic = summarize_gate_reference_confusion(
            np.stack(
                [
                    np.asarray(item["confusion_matrix"], dtype=np.int64)
                    for item in gate_reference_diagnostics
                ]
            ).sum(axis=0)
        )
        aggregate_gate_diagnostic["reference_c0_gate_mineral_pixels"] = int(
            sum(item["reference_c0_gate_mineral_pixels"] for item in gate_reference_diagnostics)
        )
        aggregate_gate_diagnostic["reference_c1_gate_mineral_pixels"] = int(
            sum(item["reference_c1_gate_mineral_pixels"] for item in gate_reference_diagnostics)
        )
    bootstrap = tile_bootstrap_intervals(
        tile_confusion_array,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
    aggregate_secondary_diagnostics = aggregate_secondary_2d_diagnostics(
        secondary_diagnostics
    )
    secondary_bootstrap = bootstrap_secondary_2d_diagnostics(
        secondary_diagnostics,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
    curve_summary = write_curve_tables(
        output_dir, positive_histograms, negative_histograms
    )
    write_metric_tables(output_dir, aggregate, bootstrap, per_tile, aggregate_confusion)
    write_secondary_2d_diagnostic_tables(
        output_dir,
        aggregate_secondary_diagnostics,
        secondary_bootstrap,
        per_tile,
    )
    if aggregate_gate_diagnostic is not None:
        write_gate_reference_diagnostics(
            output_dir, aggregate_gate_diagnostic, per_tile
        )
    if not args.no_publication_plots:
        if qualitative_arrays is None:
            raise RuntimeError("The preselected qualitative test tile was not evaluated")
        plot_confusion_matrix(aggregate_confusion, publication_dir, len(test_ids))
        plot_curves(
            positive_histograms, negative_histograms, publication_dir, len(test_ids)
        )
        plot_class_metric_summary(
            aggregate, bootstrap, publication_dir, len(test_ids)
        )
        plot_qualitative_triptych(
            qualitative_arrays["image"],
            qualitative_arrays["target"],
            qualitative_arrays["prediction"],
            image_id=qualitative_image_id,
            file_name=qualitative_file_name,
            output_dir=publication_dir,
        )
        qualitative_example["publication_files"] = [
            "publication/qualitative_triptych.pdf",
            "publication/qualitative_triptych.png",
        ]
    else:
        qualitative_example["publication_files"] = []

    script_path = Path(__file__).resolve()
    report = {
        "schema_version": "2.0",
        "evaluation_kind": "locked_held_out_confirmatory_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "checkpoint": {
            "path": _repository_relative(checkpoint_path),
            "sha256": checkpoint_sha,
            "role": checkpoint.get("checkpoint_role"),
            "selection_metric_name": checkpoint.get("selection_metric_name"),
            "selection_metric_definition": checkpoint.get("selection_metric_definition"),
            "selected_validation_epoch": checkpoint.get("best_selection_epoch"),
            "selected_validation_components": checkpoint.get("best_selection_components"),
            "validation_only": True,
            "held_out_dataset_constructed": False,
            "held_out_evaluation_count_before_evaluator": 0,
            "development_target_provenance": checkpoint_target_provenance,
            "development_input_provenance": checkpoint_input_provenance,
            "training_seed_attestation": training_seed_attestation,
        },
        "code": {
            "evaluator_path": _repository_relative(script_path),
            "evaluator_sha256": sha256_file(script_path),
            "training_execution_source_attestation": source_attestation,
        },
        "selected_method_lock": {
            "path": lock_identifier,
            "sha256": lock_sha256,
            "schema_version": verified_lock["schema_version"],
            "selected_method": selected_method,
            "selected_architecture_role": selected_architecture_role,
            "selected_retraining_array": selected_retraining_array,
            "resolved_protocol": selected_protocol,
            "complete_screen_cell_count": SCREEN_CELL_COUNT,
            "screen_selection_provenance": verified_lock[
                "screen_selection_provenance"
            ],
        },
        "neural_freeze": {
            "manifest_id": verified_freeze["manifest_id"],
            "manifest_path": verified_freeze[
                "manifest_repo_relative_identifier"
            ],
            "manifest_file_sha256": verified_freeze["manifest_file_sha256"],
            "scientific_identity_sha256": verified_freeze[
                "scientific_identity_sha256"
            ],
            "selected_retraining_checkpoint_sha256": verified_freeze[
                "selected_retraining_checkpoint_sha256"
            ],
        },
        "locked_evaluation_identity": locked_evaluation_identity,
        "data": {
            "annotations_path": _repository_relative(annotation_path),
            "annotations_sha256": annotation_sha256,
            "coco_usage": "image IDs, file names, and dimensions only; polygons rejected as ground truth",
            "image_dir": _repository_relative(image_dir),
            "mask_dir": _repository_relative(mask_dir),
            "split_manifest_path": _repository_relative(manifest_path),
            "split_manifest_sha256": manifest_sha256,
            "manifest_provenance": manifest.get("_provenance"),
            "checkpoint_split_attestation": split_attestation,
            "evaluation_split": "test",
            "test_image_ids": test_ids,
            "test_image_files": split_files["test"],
            "test_tile_count": len(test_ids),
            "native_tile_shape": list(EXPECTED_TILE_SHAPE),
            "evaluated_pixels": aggregate["total_pixels"],
            "target_source": "lossless step-2 single-channel PNG masks",
            "source_to_canonical_mask_value": {"0": 0, "1": 1, "255": 2},
            "allowed_source_mask_values_per_tile": [0, 1, 255],
            "source_mask_value_rule_per_tile": (
                "nonempty subset of the allowed values; a tile need not contain all classes"
            ),
            "polygon_ground_truth_allowed": False,
            "target_attestations": target_attestations,
            "input_attestations": input_attestations,
            "target_scope_statement": (
                "Checkpoint authentication covers only train+validation (79 masks). "
                "Full-corpus and held-out-test attestations were computed solely by "
                "this evaluator after method lock and single-pass reservation."
            ),
            "input_scope_statement": (
                "Checkpoint authentication covers only train+validation input images. "
                "Full-corpus and held-out-test input attestations were computed solely "
                "by this evaluator after method lock and single-pass reservation."
            ),
        },
        "model": {
            **model_config,
            "protocol_attestation": protocol_attestation,
            "synthetic_full_tile_preflight": model_preflight_attestation,
        },
        "inference": {
            "device": device,
            "automatic_mixed_precision": bool(args.amp),
            "precision_attestation": precision_attestation,
            "evaluator_parameter_attestation": evaluator_parameter_attestation,
            "evaluator_inference_seed": args.seed,
            "checkpoint_training_seed": training_seed_attestation["training_seed"],
            "batch_size_tiles": 1,
            "resize": None,
            "input_normalization_id": NORMALIZATION_ID,
            "input_normalization": EXPECTED_INPUT_NORMALIZATION,
            "input_normalization_formula": NORMALIZATION_FORMULA,
            "checkpoint_recorded_normalization": recorded_normalization,
            "normalization_provenance_status": "matched_both_checkpoint_records",
            "selected_method": selected_method,
            "network_input_channels": selected_protocol["model_input_channels"],
            "network_output_classes": selected_protocol["model_output_classes"],
            "conditional_gate": (
                {
                    "source": "raw_uint8_grayscale",
                    "pore_rule": "intensity < 100",
                    "mineral_rule": "intensity >= 100",
                    "second_channel": "binary_raw_uint8_lt_100",
                    "threshold_tuned_on_test": False,
                    "composition": (
                        "C0/C1 network argmax inside gate; C2 outside gate"
                    ),
                }
                if selected_method in CONDITIONAL_CANDIDATES
                else None
            ),
            "conditional_gate_reference_diagnostic": aggregate_gate_diagnostic,
            "decision_rule": (
                "fixed raw-uint8 gate composed with pixelwise C0/C1 argmax"
                if selected_method in CONDITIONAL_CANDIDATES
                else "pixelwise argmax of native three-class softmax probabilities"
            ),
            "post_processing": "none",
            "test_passes": 1,
            "held_out_mask_read_passes": 1,
            "held_out_input_read_passes": 1,
            "output_overwrite_allowed": False,
        },
        "aggregate_confusion_matrix": aggregate_confusion,
        "aggregate_metrics": aggregate,
        "uncertainty": bootstrap,
        "secondary_2d_operational_diagnostics": {
            "selection_role": "secondary_only_never_used_for_model_selection",
            "scientific_interpretation": (
                "2D operational segmentation agreement only; not permeability, flow, "
                "transport, or 3D pore-connectivity evidence"
            ),
            "aggregate": aggregate_secondary_diagnostics,
            "uncertainty": secondary_bootstrap,
        },
        "per_tile": per_tile,
        "curves": {
            "method": "one-vs-rest fixed-width probability histograms",
            "bins": args.curve_bins,
            "score_bin_width": 1.0 / args.curve_bins,
            "raw_probabilities_persisted": False,
            "summary": curve_summary,
        },
        "qualitative_example": qualitative_example,
        "publication_plots_enabled": not args.no_publication_plots,
        "outputs": sorted(
            {
                path.relative_to(output_dir).as_posix()
                for path in output_dir.rglob("*")
                if path.is_file()
            }
            | {"evaluation_summary.json"}
        ),
    }
    report_path = output_dir / "evaluation_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(report), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote locked evaluation to {_repository_relative(output_dir)}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one validation-selected checkpoint on the locked test tiles"
    )
    parser.add_argument(
        "--neural-freeze-id",
        required=True,
        help="Canonical content-addressed neural-freeze manifest identifier",
    )
    parser.add_argument(
        "--architecture-role",
        required=True,
        choices=list(ARCHITECTURE_ROLES),
        help="Frozen primary or plain-U-Net selected-retraining role",
    )
    parser.add_argument(
        "--cell-index",
        required=True,
        type=int,
        choices=range(len(SCREEN_SEEDS)),
        help="Frozen selected-retraining seed cell (0, 1, or 2)",
    )
    parser.add_argument(
        "--annotations",
        default="results/step3_coco_dataset/pore_annotations.json",
        help="COCO image-ID/file-name index; polygons are not used as ground truth",
    )
    parser.add_argument(
        "--image-dir",
        default="results/step3_coco_dataset/images",
        help="Directory containing native COCO image tiles",
    )
    parser.add_argument(
        "--mask-dir",
        default="results/step2_pore_classification/pore_classifications",
        help="Authoritative lossless 0/1/255 target-mask PNGs",
    )
    parser.add_argument(
        "--split-manifest",
        default="config/confirmatory_splits.json",
        help="Complete train/val/test manifest; only test entries are evaluated",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument(
        "--amp",
        action="store_const",
        const=True,
        default=True,
        help="Compatibility flag; CUDA float16 autocast is prospectively locked on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=LOCKED_EVALUATOR_INFERENCE_SEED,
        help="Compatibility assertion; confirmatory inference seed is locked to 0",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=LOCKED_BOOTSTRAP_SEED,
        help="Compatibility assertion; whole-tile bootstrap seed is locked",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=LOCKED_BOOTSTRAP_REPLICATES,
        help="Compatibility assertion; whole-tile bootstrap replicates are locked",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=LOCKED_CONFIDENCE,
        help="Compatibility assertion; bootstrap confidence is locked",
    )
    parser.add_argument(
        "--curve-bins",
        type=int,
        default=LOCKED_CURVE_BINS,
        help="Compatibility assertion; fixed ROC/PR histogram bins are locked",
    )
    parser.add_argument(
        "--no-publication-plots",
        action="store_true",
        help="Write metrics and curve data only; omit publication PDF/PNG assets",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.curve_bins < 128:
        raise SystemExit("--curve-bins must be at least 128")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()

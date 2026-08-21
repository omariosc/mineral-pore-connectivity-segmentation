"""Prospective, task-matched classical C0/C1/C2 comparators.

These implementations are new comparators for the leakage-controlled study.  They
are not claimed to reproduce the eight pipelines in Alwan et al. (2025).  Every
inference function accepts only a clean greyscale tile.  The annotation rings and
target masks are deliberately absent from the inference API.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image


C0_DISCONNECTED = 0
C1_CONNECTED = 1
C2_MINERAL = 2
PORE_THRESHOLD_UINT8 = 100
CLASSICAL_COMPARATOR_IDS = (
    "B0_small_components",
    "B1_marker_watershed",
    "B2_extra_trees",
)

# Candidate cutoffs are fixed before any validation or held-out scoring.  Both
# published and recovered classical workflows use pore-region size as a key
# discriminator; this deliberately compact grid spans those recorded scales.
B0_AREA_CUTOFFS = (25, 50, 100, 200, 300, 500, 1000)
B1_AREA_CUTOFFS = (25, 50, 100, 200, 300, 500, 1000)

B1_MARKER_MIN_DISTANCE_PX = 10
B1_MARKER_THRESHOLD_ABS_PX = 5.0

EXTRA_TREES_FEATURE_NAMES = (
    "gray_0_1",
    "gaussian_sigma_1",
    "gaussian_sigma_3",
    "local_mean_7",
    "local_std_7",
    "local_mean_31",
    "local_std_31",
    "sobel_magnitude",
    "laplacian_of_gaussian_sigma_1",
    "pore_distance_log_scaled",
    "local_pore_fraction_15",
    "component_area_log_scaled",
)

EXTRA_TREES_CONFIG: Dict[str, Any] = {
    "n_estimators": 128,
    "criterion": "gini",
    "max_depth": 24,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
    "bootstrap": False,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 20260821,
}

EXTRA_TREES_NUMERIC_FORMAT = "extra_trees_numeric_npz_v1"
_EXTRA_TREES_NUMERIC_ARRAY_KEYS = (
    "format_version",
    "parameter_json",
    "classes",
    "n_classes",
    "n_outputs",
    "n_features_in",
    "tree_offsets",
    "tree_has_missing",
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "impurity",
    "n_node_samples",
    "weighted_n_node_samples",
    "value",
    "missing_go_to_left",
)
_MAX_NUMERIC_MODEL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class FrozenExtraTreesPredictor:
    """Validated, non-executable numeric representation of fitted trees."""

    parameters: Mapping[str, Any]
    classes: np.ndarray
    n_classes: np.ndarray
    n_outputs: np.ndarray
    n_features_in: np.ndarray
    tree_offsets: np.ndarray
    tree_has_missing: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    impurity: np.ndarray
    n_node_samples: np.ndarray
    weighted_n_node_samples: np.ndarray
    value: np.ndarray
    missing_go_to_left: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Match ExtraTreesClassifier.predict without loading pickle objects."""

        rows = np.asarray(features)
        expected_features = int(self.n_features_in.reshape(-1)[0])
        if rows.ndim != 2 or rows.shape[1] != expected_features:
            raise ValueError(
                "B2 feature matrix must have shape (n, "
                f"{expected_features}), got {rows.shape}"
            )
        if not np.issubdtype(rows.dtype, np.number) or not np.all(np.isfinite(rows)):
            raise ValueError("B2 feature matrix must contain only finite numbers")
        sample_count = int(rows.shape[0])
        if sample_count == 0:
            return np.empty((0,), dtype=np.uint8)

        probability_sum = np.zeros((sample_count, self.classes.size), dtype=np.float64)
        for tree_index in range(self.tree_offsets.size - 1):
            start = int(self.tree_offsets[tree_index])
            end = int(self.tree_offsets[tree_index + 1])
            node = np.zeros(sample_count, dtype=np.int64)
            active = np.ones(sample_count, dtype=bool)
            for _ in range(end - start + 1):
                if not np.any(active):
                    break
                active_indices = np.flatnonzero(active)
                global_nodes = start + node[active_indices]
                left = self.children_left[global_nodes]
                leaf = left == -1
                if np.any(leaf):
                    active[active_indices[leaf]] = False
                branch_indices = active_indices[~leaf]
                if branch_indices.size:
                    branch_nodes = global_nodes[~leaf]
                    feature_index = self.feature[branch_nodes]
                    go_left = (
                        rows[branch_indices, feature_index]
                        <= self.threshold[branch_nodes]
                    )
                    node[branch_indices] = np.where(
                        go_left,
                        self.children_left[branch_nodes],
                        self.children_right[branch_nodes],
                    )
            if np.any(active):
                raise ValueError("B2 numeric tree traversal did not terminate")
            leaf_values = self.value[start + node, 0, :]
            totals = leaf_values.sum(axis=1, keepdims=True)
            if np.any(totals <= 0.0):
                raise ValueError("B2 numeric tree contains an empty leaf distribution")
            probability_sum += leaf_values / totals
        class_indices = np.argmax(probability_sum, axis=1)
        return self.classes[class_indices].astype(np.uint8, copy=False)


def _update_semantic_array_digest(
    digest: Any, label: str, value: Any, *, dtype: str
) -> None:
    """Hash one fitted-model array without serialization/container metadata."""

    array = np.ascontiguousarray(np.asarray(value).astype(dtype, copy=False))
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    digest.update(b"\0")


def extra_trees_semantic_sha256(estimator: Any) -> str:
    """Return a serialization-independent digest of a fitted ExtraTrees model.

    Container bytes can change when an identical fitted estimator is exported
    again. The held-out identity therefore hashes the parameters and complete
    ordered tree state, while the lock separately authenticates the exact inert
    numeric NPZ bytes that are loaded with pickle disabled.
    """

    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError("scikit-learn is required for B2 authentication") from error
    if type(estimator) is not ExtraTreesClassifier:
        raise ValueError("B2 artifact is not exactly an ExtraTreesClassifier")
    if not hasattr(estimator, "estimators_") or not estimator.estimators_:
        raise ValueError("B2 ExtraTrees estimator is not fitted")
    parameters = estimator.get_params(deep=False)
    try:
        parameter_bytes = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("B2 estimator parameters are not canonically serializable") from error

    digest = hashlib.sha256()
    digest.update(b"extra_trees_semantic_state_v1\0")
    digest.update(parameter_bytes)
    digest.update(b"\0")
    _update_semantic_array_digest(
        digest, "classes", getattr(estimator, "classes_", []), dtype="<i8"
    )
    _update_semantic_array_digest(
        digest,
        "n_classes",
        np.atleast_1d(getattr(estimator, "n_classes_", [])),
        dtype="<i8",
    )
    _update_semantic_array_digest(
        digest,
        "n_outputs",
        np.atleast_1d(getattr(estimator, "n_outputs_", [])),
        dtype="<i8",
    )
    _update_semantic_array_digest(
        digest,
        "n_features_in",
        np.atleast_1d(getattr(estimator, "n_features_in_", [])),
        dtype="<i8",
    )
    for index, tree_estimator in enumerate(estimator.estimators_):
        tree = tree_estimator.tree_
        digest.update(f"tree:{index}".encode("ascii"))
        digest.update(b"\0")
        for label, value, dtype in (
            ("children_left", tree.children_left, "<i8"),
            ("children_right", tree.children_right, "<i8"),
            ("feature", tree.feature, "<i8"),
            ("threshold", tree.threshold, "<f8"),
            ("impurity", tree.impurity, "<f8"),
            ("n_node_samples", tree.n_node_samples, "<i8"),
            ("weighted_n_node_samples", tree.weighted_n_node_samples, "<f8"),
            ("value", tree.value, "<f8"),
        ):
            _update_semantic_array_digest(digest, label, value, dtype=dtype)
        if hasattr(tree, "missing_go_to_left"):
            _update_semantic_array_digest(
                digest,
                "missing_go_to_left",
                tree.missing_go_to_left,
                dtype="u1",
            )
    return digest.hexdigest()


def _numeric_model_semantic_sha256(model: FrozenExtraTreesPredictor) -> str:
    """Recompute the fitted-tree digest from the inert numeric representation."""

    parameter_bytes = json.dumps(
        dict(model.parameters),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"extra_trees_semantic_state_v1\0")
    digest.update(parameter_bytes)
    digest.update(b"\0")
    _update_semantic_array_digest(digest, "classes", model.classes, dtype="<i8")
    _update_semantic_array_digest(
        digest, "n_classes", model.n_classes, dtype="<i8"
    )
    _update_semantic_array_digest(
        digest, "n_outputs", model.n_outputs, dtype="<i8"
    )
    _update_semantic_array_digest(
        digest, "n_features_in", model.n_features_in, dtype="<i8"
    )
    for tree_index in range(model.tree_offsets.size - 1):
        start = int(model.tree_offsets[tree_index])
        end = int(model.tree_offsets[tree_index + 1])
        digest.update(f"tree:{tree_index}".encode("ascii"))
        digest.update(b"\0")
        for label, value, dtype in (
            ("children_left", model.children_left[start:end], "<i8"),
            ("children_right", model.children_right[start:end], "<i8"),
            ("feature", model.feature[start:end], "<i8"),
            ("threshold", model.threshold[start:end], "<f8"),
            ("impurity", model.impurity[start:end], "<f8"),
            ("n_node_samples", model.n_node_samples[start:end], "<i8"),
            (
                "weighted_n_node_samples",
                model.weighted_n_node_samples[start:end],
                "<f8",
            ),
            ("value", model.value[start:end], "<f8"),
        ):
            _update_semantic_array_digest(digest, label, value, dtype=dtype)
        if bool(model.tree_has_missing[tree_index]):
            _update_semantic_array_digest(
                digest,
                "missing_go_to_left",
                model.missing_go_to_left[start:end],
                dtype="u1",
            )
    return digest.hexdigest()


def _require_fitted_extra_trees(estimator: Any) -> Tuple[bytes, Sequence[Any]]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError("scikit-learn is required for B2 export") from error
    if type(estimator) is not ExtraTreesClassifier:
        raise ValueError("B2 artifact is not exactly an ExtraTreesClassifier")
    if not hasattr(estimator, "estimators_") or not estimator.estimators_:
        raise ValueError("B2 ExtraTrees estimator is not fitted")
    try:
        parameter_bytes = json.dumps(
            estimator.get_params(deep=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("B2 estimator parameters are not canonically serializable") from error
    return parameter_bytes, tuple(estimator.estimators_)


def save_extra_trees_numeric(estimator: Any, path: Path | str) -> str:
    """Write fitted trees as numeric NPZ arrays with no executable objects."""

    output_path = Path(path)
    if output_path.suffix != ".npz":
        raise ValueError("B2 numeric model path must end in .npz")
    parameter_bytes, estimators = _require_fitted_extra_trees(estimator)
    offsets = [0]
    tree_has_missing = []
    arrays: Dict[str, list[np.ndarray]] = {
        "children_left": [],
        "children_right": [],
        "feature": [],
        "threshold": [],
        "impurity": [],
        "n_node_samples": [],
        "weighted_n_node_samples": [],
        "value": [],
        "missing_go_to_left": [],
    }
    for tree_estimator in estimators:
        tree = tree_estimator.tree_
        node_count = int(tree.node_count)
        offsets.append(offsets[-1] + node_count)
        arrays["children_left"].append(np.asarray(tree.children_left, dtype="<i8"))
        arrays["children_right"].append(np.asarray(tree.children_right, dtype="<i8"))
        arrays["feature"].append(np.asarray(tree.feature, dtype="<i8"))
        arrays["threshold"].append(np.asarray(tree.threshold, dtype="<f8"))
        arrays["impurity"].append(np.asarray(tree.impurity, dtype="<f8"))
        arrays["n_node_samples"].append(
            np.asarray(tree.n_node_samples, dtype="<i8")
        )
        arrays["weighted_n_node_samples"].append(
            np.asarray(tree.weighted_n_node_samples, dtype="<f8")
        )
        arrays["value"].append(np.asarray(tree.value, dtype="<f8"))
        has_missing = hasattr(tree, "missing_go_to_left")
        tree_has_missing.append(has_missing)
        arrays["missing_go_to_left"].append(
            np.asarray(
                tree.missing_go_to_left if has_missing else np.zeros(node_count),
                dtype="u1",
            )
        )

    np.savez_compressed(
        output_path,
        format_version=np.frombuffer(EXTRA_TREES_NUMERIC_FORMAT.encode("ascii"), dtype="u1"),
        parameter_json=np.frombuffer(parameter_bytes, dtype="u1"),
        classes=np.asarray(estimator.classes_, dtype="<i8"),
        n_classes=np.atleast_1d(estimator.n_classes_).astype("<i8", copy=False),
        n_outputs=np.atleast_1d(estimator.n_outputs_).astype("<i8", copy=False),
        n_features_in=np.atleast_1d(estimator.n_features_in_).astype("<i8", copy=False),
        tree_offsets=np.asarray(offsets, dtype="<i8"),
        tree_has_missing=np.asarray(tree_has_missing, dtype="u1"),
        **{key: np.concatenate(value, axis=0) for key, value in arrays.items()},
    )
    loaded = load_extra_trees_numeric(output_path)
    observed = _numeric_model_semantic_sha256(loaded)
    expected = extra_trees_semantic_sha256(estimator)
    if observed != expected:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("B2 numeric export semantic digest mismatch")
    return observed


def _validate_numeric_tree_graph(model: FrozenExtraTreesPredictor) -> None:
    for tree_index in range(model.tree_offsets.size - 1):
        start = int(model.tree_offsets[tree_index])
        end = int(model.tree_offsets[tree_index + 1])
        node_count = end - start
        left = model.children_left[start:end]
        right = model.children_right[start:end]
        feature = model.feature[start:end]
        leaf = left == -1
        if not np.array_equal(leaf, right == -1):
            raise ValueError("B2 numeric tree has one-sided child links")
        if np.any(feature[leaf] != -2):
            raise ValueError("B2 numeric tree leaf feature markers are invalid")
        if np.any((feature[~leaf] < 0) | (feature[~leaf] >= len(EXTRA_TREES_FEATURE_NAMES))):
            raise ValueError("B2 numeric tree feature index is invalid")
        for children in (left[~leaf], right[~leaf]):
            if np.any((children < 0) | (children >= node_count)):
                raise ValueError("B2 numeric tree child index is invalid")
        indegree = np.zeros(node_count, dtype=np.int64)
        np.add.at(indegree, left[~leaf], 1)
        np.add.at(indegree, right[~leaf], 1)
        if indegree[0] != 0 or np.any(indegree[1:] != 1):
            raise ValueError("B2 numeric tree is cyclic, shared, or disconnected")
        seen = np.zeros(node_count, dtype=bool)
        stack = [0]
        while stack:
            node = stack.pop()
            if seen[node]:
                raise ValueError("B2 numeric tree contains a cycle")
            seen[node] = True
            if left[node] != -1:
                stack.extend((int(left[node]), int(right[node])))
        if not np.all(seen):
            raise ValueError("B2 numeric tree contains unreachable nodes")


def load_extra_trees_numeric(path: Path | str) -> FrozenExtraTreesPredictor:
    """Load and validate an inert numeric forest with pickle disabled."""

    model_path = Path(path)
    if model_path.suffix != ".npz" or model_path.is_symlink() or not model_path.is_file():
        raise ValueError("B2 numeric model must be a regular .npz file")
    expected_members = {f"{key}.npy" for key in _EXTRA_TREES_NUMERIC_ARRAY_KEYS}
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            infos = archive.infolist()
            if (
                len(infos) != len(expected_members)
                or {info.filename for info in infos} != expected_members
            ):
                raise ValueError("B2 numeric model member set is invalid")
            if any(info.flag_bits & 0x1 for info in infos):
                raise ValueError("B2 numeric model must not contain encrypted members")
            if sum(info.file_size for info in infos) > _MAX_NUMERIC_MODEL_UNCOMPRESSED_BYTES:
                raise ValueError("B2 numeric model exceeds the uncompressed size limit")
    except zipfile.BadZipFile as error:
        raise ValueError("B2 numeric model is not a valid NPZ archive") from error

    try:
        with np.load(model_path, allow_pickle=False) as archive:
            if set(archive.files) != set(_EXTRA_TREES_NUMERIC_ARRAY_KEYS):
                raise ValueError("B2 numeric model array set is invalid")
            raw = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("B2 numeric model cannot be decoded safely") from error
    if any(value.dtype.hasobject for value in raw.values()):
        raise ValueError("B2 numeric model must not contain object arrays")
    try:
        format_version = raw["format_version"].astype("u1", copy=False).tobytes().decode("ascii")
        parameters = json.loads(
            raw["parameter_json"].astype("u1", copy=False).tobytes().decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("B2 numeric model metadata is invalid") from error
    if format_version != EXTRA_TREES_NUMERIC_FORMAT:
        raise ValueError("B2 numeric model format version is unsupported")
    if not isinstance(parameters, dict) or any(
        parameters.get(key) != expected
        for key, expected in EXTRA_TREES_CONFIG.items()
    ):
        raise ValueError("B2 numeric model estimator configuration drifted")

    model = FrozenExtraTreesPredictor(
        parameters=parameters,
        classes=np.asarray(raw["classes"], dtype="<i8"),
        n_classes=np.asarray(raw["n_classes"], dtype="<i8"),
        n_outputs=np.asarray(raw["n_outputs"], dtype="<i8"),
        n_features_in=np.asarray(raw["n_features_in"], dtype="<i8"),
        tree_offsets=np.asarray(raw["tree_offsets"], dtype="<i8"),
        tree_has_missing=np.asarray(raw["tree_has_missing"], dtype="u1"),
        children_left=np.asarray(raw["children_left"], dtype="<i8"),
        children_right=np.asarray(raw["children_right"], dtype="<i8"),
        feature=np.asarray(raw["feature"], dtype="<i8"),
        threshold=np.asarray(raw["threshold"], dtype="<f8"),
        impurity=np.asarray(raw["impurity"], dtype="<f8"),
        n_node_samples=np.asarray(raw["n_node_samples"], dtype="<i8"),
        weighted_n_node_samples=np.asarray(raw["weighted_n_node_samples"], dtype="<f8"),
        value=np.asarray(raw["value"], dtype="<f8"),
        missing_go_to_left=np.asarray(raw["missing_go_to_left"], dtype="u1"),
    )
    expected_trees = int(EXTRA_TREES_CONFIG["n_estimators"])
    if (
        model.classes.shape != (2,)
        or not np.array_equal(model.classes, np.array([0, 1]))
        or model.n_classes.shape != (1,)
        or int(model.n_classes[0]) != 2
        or model.n_outputs.shape != (1,)
        or int(model.n_outputs[0]) != 1
        or model.n_features_in.shape != (1,)
        or int(model.n_features_in[0]) != len(EXTRA_TREES_FEATURE_NAMES)
        or model.tree_offsets.shape != (expected_trees + 1,)
        or model.tree_has_missing.shape != (expected_trees,)
        or model.tree_offsets[0] != 0
        or np.any(np.diff(model.tree_offsets) <= 0)
    ):
        raise ValueError("B2 numeric model top-level dimensions are invalid")
    node_count = int(model.tree_offsets[-1])
    one_dimensional = (
        model.children_left,
        model.children_right,
        model.feature,
        model.threshold,
        model.impurity,
        model.n_node_samples,
        model.weighted_n_node_samples,
        model.missing_go_to_left,
    )
    if any(value.shape != (node_count,) for value in one_dimensional):
        raise ValueError("B2 numeric model node-array dimensions are inconsistent")
    if model.value.shape != (node_count, 1, 2):
        raise ValueError("B2 numeric model value-array dimensions are invalid")
    if (
        not np.all(np.isfinite(model.threshold))
        or not np.all(np.isfinite(model.impurity))
        or not np.all(np.isfinite(model.weighted_n_node_samples))
        or not np.all(np.isfinite(model.value))
        or np.any(model.value < 0.0)
        or np.any(model.n_node_samples < 0)
        or np.any(model.weighted_n_node_samples < 0.0)
    ):
        raise ValueError("B2 numeric model contains invalid numeric state")
    _validate_numeric_tree_graph(model)
    return model


def extra_trees_numeric_semantic_sha256(model: FrozenExtraTreesPredictor) -> str:
    """Public verifier for the complete inert numeric forest state."""

    return _numeric_model_semantic_sha256(model)


def _require_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional greyscale tile, got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 greyscale tile, got {array.dtype}")
    return array


def load_clean_grayscale(path: Path | str) -> np.ndarray:
    """Load a clean tile and reject colour overlays instead of hiding them.

    Canonical clean PNGs may be stored either as one greyscale channel or as
    three identical RGB channels.  Non-identical colour channels (including a
    yellow annotation ring) are a hard failure.
    """

    source_path = Path(path)
    with Image.open(source_path) as handle:
        source_mode = handle.mode
        source = np.asarray(handle)
    if source_mode == "L" and source.ndim == 2:
        gray = source
    elif source_mode in ("RGB", "RGBA") and source.ndim == 3:
        expected_channels = 3 if source_mode == "RGB" else 4
        if source.shape[2] != expected_channels:
            raise ValueError(
                f"Unexpected {source_mode} array shape {source.shape}: "
                f"{source_path.name}"
            )
        rgb = source[:, :, :3]
        if not (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        ):
            raise ValueError(
                "Input tile contains non-greyscale colour information: "
                f"{source_path.name}"
            )
        if source_mode == "RGBA" and not np.all(source[:, :, 3] == 255):
            raise ValueError(f"Input tile has non-opaque alpha: {source_path.name}")
        gray = rgb[:, :, 0]
    else:
        raise ValueError(
            f"Unsupported clean-image mode/shape {source_mode}/{source.shape}: "
            f"{source_path.name}"
        )
    if gray.dtype != np.uint8:
        raise ValueError(f"Input tile is not uint8: {source_path.name} ({gray.dtype})")
    return np.asarray(gray, dtype=np.uint8)


def fixed_pore_gate(image: np.ndarray) -> np.ndarray:
    """Return the prespecified operational pore gate ``raw uint8 < 100``."""

    return _require_uint8_grayscale(image) < PORE_THRESHOLD_UINT8


def confusion_from_labels(
    target: np.ndarray, prediction: np.ndarray, num_classes: int = 3
) -> np.ndarray:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    if target_array.shape != prediction_array.shape:
        raise ValueError(
            f"Target/prediction shape mismatch: {target_array.shape} != "
            f"{prediction_array.shape}"
        )
    if np.any((target_array < 0) | (target_array >= num_classes)):
        raise ValueError("Target contains values outside the canonical classes")
    if np.any((prediction_array < 0) | (prediction_array >= num_classes)):
        raise ValueError("Prediction contains values outside the canonical classes")
    encoded = (
        target_array.astype(np.int64).ravel() * num_classes
        + prediction_array.astype(np.int64).ravel()
    )
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def iou_from_confusion(confusion: np.ndarray, class_id: int) -> float:
    matrix = np.asarray(confusion, dtype=np.float64)
    intersection = matrix[class_id, class_id]
    union = matrix[class_id, :].sum() + matrix[:, class_id].sum() - intersection
    return float(intersection / union) if union > 0 else 0.0


def balanced_pore_score(iou_c0: float, iou_c1: float) -> float:
    return float(2.0 * iou_c0 * iou_c1 / (iou_c0 + iou_c1 + 1e-8))


def pore_metrics_from_confusion(confusion: np.ndarray) -> Dict[str, float]:
    iou_c0 = iou_from_confusion(confusion, C0_DISCONNECTED)
    iou_c1 = iou_from_confusion(confusion, C1_CONNECTED)
    return {
        "iou_c0": iou_c0,
        "iou_c1": iou_c1,
        "balanced_pore_iou": balanced_pore_score(iou_c0, iou_c1),
    }


def _connected_components(gate: np.ndarray) -> Tuple[np.ndarray, int]:
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - dependency contract
        raise RuntimeError("scipy is required for classical comparators") from error
    labels, count = ndimage.label(
        np.asarray(gate, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8)
    )
    return labels.astype(np.int32, copy=False), int(count)


def b0_component_regions(image: np.ndarray) -> np.ndarray:
    """Return B0's deterministic 8-connected region labels."""

    labels, _ = _connected_components(fixed_pore_gate(image))
    return labels


def prediction_from_regions(
    regions: np.ndarray, gate: np.ndarray, *, area_cutoff_px: int
) -> np.ndarray:
    """Apply the shared strict region-size rule to a fixed-gate partition."""

    if int(area_cutoff_px) <= 0:
        raise ValueError("area_cutoff_px must be positive")
    gate_array = np.asarray(gate, dtype=bool)
    region_array = np.asarray(regions)
    if gate_array.shape != region_array.shape:
        raise ValueError("gate and region-label shapes differ")
    if np.any(region_array < 0):
        raise ValueError("region labels must be non-negative")
    if np.any(gate_array & (region_array == 0)):
        raise ValueError("region partition left fixed-gate pixels unassigned")
    sizes = np.bincount(region_array.astype(np.int64, copy=False).ravel())
    region_is_c0 = sizes < int(area_cutoff_px)
    region_is_c0[0] = False
    result = np.full(gate_array.shape, C2_MINERAL, dtype=np.uint8)
    result[gate_array] = C1_CONNECTED
    result[gate_array & region_is_c0[region_array]] = C0_DISCONNECTED
    return result


def predict_b0_small_components(
    image: np.ndarray, *, area_cutoff_px: int
) -> np.ndarray:
    """B0: exact pore gate plus one label per 8-connected component.

    A gated component is C0 when its area is strictly smaller than the frozen
    cutoff and C1 otherwise.  Pixels outside the fixed gate are C2.
    """

    if int(area_cutoff_px) <= 0:
        raise ValueError("area_cutoff_px must be positive")
    gate = fixed_pore_gate(image)
    labels = b0_component_regions(image)
    return prediction_from_regions(labels, gate, area_cutoff_px=area_cutoff_px)


def _marker_watershed_regions(image: np.ndarray) -> np.ndarray:
    """Partition every fixed-gate pixel with deterministic marker watershed."""

    try:
        from scipy import ndimage
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError as error:  # pragma: no cover - dependency contract
        raise RuntimeError(
            "scipy and scikit-image are required for B1 marker watershed"
        ) from error

    gate = fixed_pore_gate(image)
    if not np.any(gate):
        return np.zeros(gate.shape, dtype=np.int32)

    component_labels, component_count = _connected_components(gate)
    distance = ndimage.distance_transform_edt(gate)
    peaks = peak_local_max(
        distance,
        labels=gate.astype(np.uint8),
        min_distance=B1_MARKER_MIN_DISTANCE_PX,
        threshold_abs=B1_MARKER_THRESHOLD_ABS_PX,
        exclude_border=False,
    )
    markers = np.zeros(gate.shape, dtype=np.int32)
    next_marker = 1
    if len(peaks):
        for row, column in peaks:
            markers[int(row), int(column)] = next_marker
            next_marker += 1

    covered_components = set(
        int(value) for value in component_labels[markers > 0] if int(value) > 0
    )
    missing_components = [
        component_id
        for component_id in range(1, component_count + 1)
        if component_id not in covered_components
    ]
    if missing_components:
        fallback_positions = ndimage.maximum_position(
            distance, labels=component_labels, index=missing_components
        )
        if (
            isinstance(fallback_positions, tuple)
            and len(fallback_positions) == 2
            and all(np.isscalar(value) for value in fallback_positions)
        ):
            fallback_positions = [fallback_positions]
        for row, column in fallback_positions:
            markers[int(row), int(column)] = next_marker
            next_marker += 1

    regions = watershed(
        -distance,
        markers=markers,
        mask=gate,
        connectivity=np.ones((3, 3), dtype=np.uint8),
    ).astype(np.int32, copy=False)
    if np.any(gate & (regions == 0)):
        raise RuntimeError("Marker watershed left fixed-gate pixels unassigned")
    return regions


def b1_watershed_regions(image: np.ndarray) -> np.ndarray:
    """Return B1's deterministic marker-watershed region labels."""

    return _marker_watershed_regions(image)


def predict_b1_marker_watershed(
    image: np.ndarray, *, area_cutoff_px: int
) -> np.ndarray:
    """B1: exact gate, fixed marker watershed, then frozen region-size rule."""

    if int(area_cutoff_px) <= 0:
        raise ValueError("area_cutoff_px must be positive")
    gate = fixed_pore_gate(image)
    regions = b1_watershed_regions(image)
    if not np.any(gate):
        return np.full(gate.shape, C2_MINERAL, dtype=np.uint8)
    return prediction_from_regions(regions, gate, area_cutoff_px=area_cutoff_px)


def _local_mean_and_std(
    image_0_1: np.ndarray, size: int
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy import ndimage

    mean = ndimage.uniform_filter(image_0_1, size=size, mode="reflect")
    mean_square = ndimage.uniform_filter(image_0_1**2, size=size, mode="reflect")
    variance = np.maximum(mean_square - mean**2, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def clean_grayscale_feature_planes(image: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Compute B2 features from clean greyscale and its fixed gate only."""

    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - dependency contract
        raise RuntimeError("scipy is required for B2 feature extraction") from error

    gray_u8 = _require_uint8_grayscale(image)
    gray = gray_u8.astype(np.float32) / 255.0
    gate = gray_u8 < PORE_THRESHOLD_UINT8
    gaussian_1 = ndimage.gaussian_filter(gray, sigma=1.0, mode="reflect")
    gaussian_3 = ndimage.gaussian_filter(gray, sigma=3.0, mode="reflect")
    mean_7, std_7 = _local_mean_and_std(gray, 7)
    mean_31, std_31 = _local_mean_and_std(gray, 31)
    sobel_x = ndimage.sobel(gray, axis=1, mode="reflect")
    sobel_y = ndimage.sobel(gray, axis=0, mode="reflect")
    sobel_magnitude = np.hypot(sobel_x, sobel_y).astype(np.float32)
    laplacian = ndimage.gaussian_laplace(gray, sigma=1.0, mode="reflect").astype(
        np.float32
    )
    distance = ndimage.distance_transform_edt(gate)
    diagonal = max(float(np.hypot(*gray.shape)), 1.0)
    distance_scaled = (
        np.log1p(distance) / np.log1p(diagonal)
    ).astype(np.float32)
    pore_fraction = ndimage.uniform_filter(
        gate.astype(np.float32), size=15, mode="reflect"
    ).astype(np.float32)
    component_labels, component_count = _connected_components(gate)
    component_sizes = np.bincount(
        component_labels.ravel(), minlength=component_count + 1
    )
    component_area = (
        np.log1p(component_sizes[component_labels]) / np.log1p(gray.size)
    ).astype(np.float32)
    component_area[~gate] = 0.0

    planes = (
        gray,
        gaussian_1.astype(np.float32),
        gaussian_3.astype(np.float32),
        mean_7,
        std_7,
        mean_31,
        std_31,
        sobel_magnitude,
        laplacian,
        distance_scaled,
        pore_fraction,
        component_area,
    )
    if len(planes) != len(EXTRA_TREES_FEATURE_NAMES):
        raise RuntimeError("B2 feature-name and feature-plane counts differ")
    return planes


def feature_rows(
    feature_planes: Sequence[np.ndarray], flat_indices: np.ndarray
) -> np.ndarray:
    indices = np.asarray(flat_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("flat_indices must be one-dimensional")
    if not feature_planes:
        raise ValueError("feature_planes cannot be empty")
    shape = np.asarray(feature_planes[0]).shape
    if any(np.asarray(plane).shape != shape for plane in feature_planes):
        raise ValueError("Every B2 feature plane must have the same shape")
    if np.any(indices < 0) or np.any(indices >= int(np.prod(shape))):
        raise ValueError("flat_indices contains an out-of-range pixel")
    return np.column_stack(
        [np.asarray(plane).reshape(-1)[indices] for plane in feature_planes]
    ).astype(np.float32, copy=False)


def deterministic_sample_indices(
    candidate_indices: np.ndarray,
    *,
    limit: int,
    seed: int,
    image_id: int,
    class_id: int,
) -> np.ndarray:
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.ndim != 1:
        raise ValueError("candidate_indices must be one-dimensional")
    if int(limit) <= 0:
        raise ValueError("sample limit must be positive")
    if candidates.size <= int(limit):
        return np.sort(candidates)
    seed_material = f"{int(seed)}:{int(image_id)}:{int(class_id)}".encode("ascii")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = np.random.default_rng(derived_seed)
    chosen = rng.choice(candidates, size=int(limit), replace=False)
    return np.sort(chosen.astype(np.int64, copy=False))


def build_extra_trees(**overrides: Any) -> Any:
    """Construct the single prespecified B2 estimator."""

    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as error:  # pragma: no cover - dependency contract
        raise RuntimeError("scikit-learn is required for B2 ExtraTrees") from error
    config = dict(EXTRA_TREES_CONFIG)
    unknown = sorted(set(overrides) - set(config))
    if unknown:
        raise ValueError(f"Unsupported B2 configuration override(s): {unknown}")
    config.update(overrides)
    return ExtraTreesClassifier(**config)


def predict_b2_extra_trees(
    image: np.ndarray, estimator: Any, *, chunk_size: int = 262_144
) -> np.ndarray:
    """Compose B2 C0/C1 predictions inside the fixed gate and C2 outside."""

    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    gate = fixed_pore_gate(image)
    result = np.full(gate.shape, C2_MINERAL, dtype=np.uint8)
    flat_gate_indices = np.flatnonzero(gate)
    if flat_gate_indices.size == 0:
        return result
    feature_planes = clean_grayscale_feature_planes(image)
    flat_result = result.ravel()
    for start in range(0, flat_gate_indices.size, int(chunk_size)):
        indices = flat_gate_indices[start : start + int(chunk_size)]
        prediction = np.asarray(
            estimator.predict(feature_rows(feature_planes, indices)), dtype=np.uint8
        )
        observed = set(int(value) for value in np.unique(prediction))
        if not observed.issubset({C0_DISCONNECTED, C1_CONNECTED}):
            raise ValueError(
                f"B2 estimator emitted non-pore classes: {sorted(observed)}"
            )
        flat_result[indices] = prediction
    return result


def candidate_mean_group_metrics(
    confusion_by_group: Mapping[str, np.ndarray]
) -> Dict[str, float]:
    if not confusion_by_group:
        raise ValueError("At least one held-out training group is required")
    per_group = candidate_group_metrics(confusion_by_group)
    return {
        key: float(np.mean([record[key] for record in per_group.values()]))
        for key in ("iou_c0", "iou_c1", "balanced_pore_iou")
    }


def candidate_group_metrics(
    confusion_by_group: Mapping[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Return the three selection metrics for every training-series fold."""

    if not confusion_by_group:
        raise ValueError("At least one held-out training group is required")
    return {
        str(group): pore_metrics_from_confusion(confusion_by_group[group])
        for group in sorted(confusion_by_group)
    }


def select_area_cutoff(
    confusion_by_cutoff_and_group: Mapping[int, Mapping[str, np.ndarray]],
) -> Tuple[int, Dict[str, Any]]:
    """Select a cutoff from training-group CV only with deterministic ties."""

    if not confusion_by_cutoff_and_group:
        raise ValueError("No area-cutoff candidates were evaluated")
    group_summaries: Dict[int, Dict[str, Dict[str, float]]] = {
        int(cutoff): candidate_group_metrics(by_group)
        for cutoff, by_group in confusion_by_cutoff_and_group.items()
    }
    summaries: Dict[int, Dict[str, float]] = {
        cutoff: {
            key: float(
                np.mean([record[key] for record in by_group.values()])
            )
            for key in ("iou_c0", "iou_c1", "balanced_pore_iou")
        }
        for cutoff, by_group in group_summaries.items()
    }
    selected = sorted(
        summaries,
        key=lambda cutoff: (
            -summaries[cutoff]["balanced_pore_iou"],
            -min(summaries[cutoff]["iou_c0"], summaries[cutoff]["iou_c1"]),
            int(cutoff),
        ),
    )[0]
    return int(selected), {
        "selection_scope": "canonical_training_groups_only",
        "primary": "mean_group_balanced_pore_iou",
        "tie_break_1": "higher_minimum_of_mean_group_iou_c0_and_iou_c1",
        "tie_break_2": "lower_area_cutoff_px",
        "selected_area_cutoff_px": int(selected),
        "candidate_group_summaries": {
            str(cutoff): group_summaries[cutoff]
            for cutoff in sorted(group_summaries)
        },
        "candidate_summaries": {
            str(cutoff): summaries[cutoff] for cutoff in sorted(summaries)
        },
    }

"""Focused safeguards for the prospective train-only classical comparators."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import fit_classical_comparators as fitter
from scripts.fit_classical_comparators import (
    _source_hashes,
    prepare_train_only_contract,
)
from src.classical.comparators import (
    C0_DISCONNECTED,
    C1_CONNECTED,
    C2_MINERAL,
    EXTRA_TREES_FEATURE_NAMES,
    b1_watershed_regions,
    balanced_pore_score,
    build_extra_trees,
    clean_grayscale_feature_planes,
    confusion_from_labels,
    deterministic_sample_indices,
    extra_trees_semantic_sha256,
    extra_trees_numeric_semantic_sha256,
    fixed_pore_gate,
    load_extra_trees_numeric,
    load_clean_grayscale,
    predict_b0_small_components,
    predict_b1_marker_watershed,
    predict_b2_extra_trees,
    prediction_from_regions,
    select_area_cutoff,
    save_extra_trees_numeric,
)


def test_fixed_pore_gate_is_strictly_below_100() -> None:
    image = np.array([[99, 100, 0, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(
        fixed_pore_gate(image), np.array([[True, False, True, False]])
    )


def test_clean_loader_accepts_only_explicit_grayscale_encodings(tmp_path: Path) -> None:
    grayscale = np.array([[0, 99], [100, 255]], dtype=np.uint8)
    gray_path = tmp_path / "gray.png"
    rgb_path = tmp_path / "rgb.png"
    colour_path = tmp_path / "colour.png"
    palette_path = tmp_path / "palette.png"
    alpha_path = tmp_path / "alpha.png"

    Image.fromarray(grayscale).save(gray_path)
    Image.fromarray(np.repeat(grayscale[:, :, None], 3, axis=2)).save(
        rgb_path
    )
    colour = np.repeat(grayscale[:, :, None], 3, axis=2)
    colour[0, 0] = [255, 255, 0]
    Image.fromarray(colour).save(colour_path)
    Image.fromarray(grayscale).convert("P").save(palette_path)
    rgba = np.dstack(
        [np.repeat(grayscale[:, :, None], 3, axis=2), np.full_like(grayscale, 255)]
    )
    rgba[0, 0, 3] = 0
    Image.fromarray(rgba).save(alpha_path)

    np.testing.assert_array_equal(load_clean_grayscale(gray_path), grayscale)
    np.testing.assert_array_equal(load_clean_grayscale(rgb_path), grayscale)
    with pytest.raises(ValueError, match="non-greyscale colour"):
        load_clean_grayscale(colour_path)
    with pytest.raises(ValueError, match="Unsupported clean-image mode"):
        load_clean_grayscale(palette_path)
    with pytest.raises(ValueError, match="non-opaque alpha"):
        load_clean_grayscale(alpha_path)


def test_b0_uses_strict_component_area_rule_and_c2_outside_gate() -> None:
    image = np.full((5, 7), 255, dtype=np.uint8)
    image[1, 1] = 0
    image[3, 1:4] = 0

    prediction = predict_b0_small_components(image, area_cutoff_px=3)

    assert prediction[1, 1] == C0_DISCONNECTED
    np.testing.assert_array_equal(
        prediction[3, 1:4], np.full(3, C1_CONNECTED, dtype=np.uint8)
    )
    assert np.all(prediction[image >= 100] == C2_MINERAL)


def test_region_composition_fails_if_a_gate_pixel_is_unassigned() -> None:
    gate = np.array([[True, False]], dtype=bool)
    regions = np.zeros(gate.shape, dtype=np.int32)
    with pytest.raises(ValueError, match="unassigned"):
        prediction_from_regions(regions, gate, area_cutoff_px=10)


@pytest.mark.skipif(
    importlib.util.find_spec("skimage") is None,
    reason="scikit-image is not installed in this lightweight local runtime",
)
def test_b1_is_deterministic_and_partitions_the_entire_fixed_gate() -> None:
    image = np.full((33, 33), 255, dtype=np.uint8)
    image[3:15, 3:15] = 0
    image[18:30, 18:30] = 0
    gate = fixed_pore_gate(image)

    first_regions = b1_watershed_regions(image)
    second_regions = b1_watershed_regions(image)
    np.testing.assert_array_equal(first_regions, second_regions)
    assert np.all(first_regions[gate] > 0)
    assert np.all(first_regions[~gate] == 0)

    prediction = predict_b1_marker_watershed(image, area_cutoff_px=200)
    assert set(int(value) for value in np.unique(prediction[gate])).issubset({0, 1})
    assert np.all(prediction[~gate] == C2_MINERAL)


def test_b2_features_are_finite_shape_matched_and_image_only() -> None:
    image = np.arange(9 * 11, dtype=np.uint8).reshape(9, 11)
    planes = clean_grayscale_feature_planes(image)

    assert len(planes) == len(EXTRA_TREES_FEATURE_NAMES)
    for plane in planes:
        assert plane.shape == image.shape
        assert plane.dtype == np.float32
        assert np.all(np.isfinite(plane))


def test_deterministic_sampling_is_reproducible_and_context_specific() -> None:
    candidates = np.arange(1000, dtype=np.int64)
    first = deterministic_sample_indices(
        candidates, limit=40, seed=20260821, image_id=7, class_id=0
    )
    repeat = deterministic_sample_indices(
        candidates, limit=40, seed=20260821, image_id=7, class_id=0
    )
    another_image = deterministic_sample_indices(
        candidates, limit=40, seed=20260821, image_id=8, class_id=0
    )

    np.testing.assert_array_equal(first, repeat)
    assert first.size == 40
    assert np.all(first[:-1] < first[1:])
    assert not np.array_equal(first, another_image)


class _AllDisconnectedEstimator:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], C0_DISCONNECTED, dtype=np.uint8)


class _InvalidMineralEstimator:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], C2_MINERAL, dtype=np.uint8)


def test_b2_composes_only_pore_predictions_inside_the_fixed_gate() -> None:
    image = np.array([[0, 99, 100], [255, 12, 180]], dtype=np.uint8)
    gate = fixed_pore_gate(image)
    prediction = predict_b2_extra_trees(
        image, _AllDisconnectedEstimator(), chunk_size=2
    )

    assert np.all(prediction[gate] == C0_DISCONNECTED)
    assert np.all(prediction[~gate] == C2_MINERAL)
    with pytest.raises(ValueError, match="non-pore classes"):
        predict_b2_extra_trees(image, _InvalidMineralEstimator(), chunk_size=2)


def test_b2_estimator_is_deterministic_and_numeric_export_is_equivalent(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(80, len(EXTRA_TREES_FEATURE_NAMES))).astype(
        np.float32
    )
    target = np.tile(np.array([C0_DISCONNECTED, C1_CONNECTED]), 40)
    first = build_extra_trees().fit(features, target)
    second = build_extra_trees().fit(features, target)
    np.testing.assert_array_equal(first.predict(features), second.predict(features))

    model_path = tmp_path / "model.npz"
    expected_digest = save_extra_trees_numeric(first, model_path)
    reloaded = load_extra_trees_numeric(model_path)
    np.testing.assert_array_equal(first.predict(features), reloaded.predict(features))
    assert extra_trees_numeric_semantic_sha256(reloaded) == expected_digest

    with pytest.raises(ValueError, match="Unsupported B2 configuration"):
        build_extra_trees(not_a_declared_option=True)


def test_b2_semantic_digest_survives_numeric_export_but_detects_model_change(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(41)
    features = rng.normal(size=(96, len(EXTRA_TREES_FEATURE_NAMES))).astype(
        np.float32
    )
    target = np.tile(np.array([C0_DISCONNECTED, C1_CONNECTED]), 48)
    estimator = build_extra_trees().fit(features, target)
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    expected = extra_trees_semantic_sha256(estimator)
    assert save_extra_trees_numeric(estimator, first_path) == expected
    assert save_extra_trees_numeric(estimator, second_path) == expected
    assert extra_trees_numeric_semantic_sha256(
        load_extra_trees_numeric(first_path)
    ) == expected
    assert extra_trees_numeric_semantic_sha256(
        load_extra_trees_numeric(second_path)
    ) == expected

    changed = build_extra_trees().fit(features, 1 - target)
    assert extra_trees_semantic_sha256(changed) != expected


def test_b2_numeric_loader_rejects_object_arrays_without_pickle(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(73)
    features = rng.normal(size=(64, len(EXTRA_TREES_FEATURE_NAMES))).astype(
        np.float32
    )
    target = np.tile(np.array([C0_DISCONNECTED, C1_CONNECTED]), 32)
    valid_path = tmp_path / "valid.npz"
    save_extra_trees_numeric(build_extra_trees().fit(features, target), valid_path)
    with np.load(valid_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["parameter_json"] = np.array(["not executable"], dtype=object)
    malicious_path = tmp_path / "object-array.npz"
    np.savez_compressed(malicious_path, **arrays)

    with pytest.raises(ValueError, match="decoded safely"):
        load_extra_trees_numeric(malicious_path)


def test_b2_numeric_loader_rejects_cyclic_tree_state(tmp_path: Path) -> None:
    rng = np.random.default_rng(79)
    features = rng.normal(size=(64, len(EXTRA_TREES_FEATURE_NAMES))).astype(
        np.float32
    )
    target = np.tile(np.array([C0_DISCONNECTED, C1_CONNECTED]), 32)
    valid_path = tmp_path / "valid.npz"
    save_extra_trees_numeric(build_extra_trees().fit(features, target), valid_path)
    with np.load(valid_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["children_left"][0] = 0
    cyclic_path = tmp_path / "cyclic.npz"
    np.savez_compressed(cyclic_path, **arrays)

    with pytest.raises(ValueError, match="cyclic, shared, or disconnected|cycle"):
        load_extra_trees_numeric(cyclic_path)


def test_b2_sampling_allows_class_absence_per_tile_when_aggregate_has_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((3, 3), dtype=np.uint8)
    targets = {
        1: np.zeros((3, 3), dtype=np.uint8),
        2: np.ones((3, 3), dtype=np.uint8),
    }
    monkeypatch.setattr(
        fitter, "_load_training_tile", lambda _contract, image_id: (image, targets[image_id])
    )
    monkeypatch.setattr(
        fitter,
        "b0_component_regions",
        lambda value: np.ones(value.shape, dtype=np.int32),
    )
    monkeypatch.setattr(
        fitter,
        "b1_watershed_regions",
        lambda value: np.ones(value.shape, dtype=np.int32),
    )
    monkeypatch.setattr(
        fitter,
        "clean_grayscale_feature_planes",
        lambda value: tuple(
            value.astype(np.float32) for _ in EXTRA_TREES_FEATURE_NAMES
        ),
    )
    evidence = fitter.collect_train_only_evidence(
        {
            "train_ids": (1, 2),
            "group_by_id": {1: "training_group", 2: "training_group"},
        }
    )
    records = evidence["b2_training"]["training_group"]
    assert len(records) == 2
    assert {int(value) for record in records for value in np.unique(record[1])} == {
        C0_DISCONNECTED,
        C1_CONNECTED,
    }


def test_confusion_rejects_archived_255_label_defect() -> None:
    canonical = np.array([[0, 1, 2]], dtype=np.uint8)
    confusion = confusion_from_labels(canonical, canonical)
    np.testing.assert_array_equal(confusion, np.eye(3, dtype=np.int64))

    with pytest.raises(ValueError, match="outside the canonical classes"):
        confusion_from_labels(np.array([[0, 1, 255]], dtype=np.uint8), canonical)


def test_area_selection_ties_are_resolved_by_lower_prespecified_cutoff() -> None:
    perfect = np.diag([10, 20, 30]).astype(np.int64)
    selected, record = select_area_cutoff(
        {50: {"group_b": perfect}, 25: {"group_b": perfect}}
    )
    assert selected == 25
    assert record["selected_area_cutoff_px"] == 25


def test_area_selection_records_mean_of_per_group_harmonic_scores() -> None:
    group_a = np.array(
        [[9, 1, 0], [1, 1, 0], [0, 0, 10]], dtype=np.int64
    )
    group_b = np.array(
        [[1, 1, 0], [1, 9, 0], [0, 0, 10]], dtype=np.int64
    )
    _, record = select_area_cutoff({25: {"group_a": group_a, "group_b": group_b}})

    groups = record["candidate_group_summaries"]["25"]
    aggregate = record["candidate_summaries"]["25"]
    expected_mean_harmonic = float(
        np.mean([groups[group]["balanced_pore_iou"] for group in sorted(groups)])
    )
    harmonic_of_mean_ious = balanced_pore_score(
        aggregate["iou_c0"], aggregate["iou_c1"]
    )

    assert aggregate["balanced_pore_iou"] == pytest.approx(expected_mean_harmonic)
    assert aggregate["balanced_pore_iou"] != pytest.approx(harmonic_of_mean_ious)


def test_contract_requires_only_training_image_and_mask_bytes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    train_name = "training_series_tile.png"
    # No validation or test image/mask file is created. Successful preparation
    # therefore proves that their paths are not required or opened.
    Image.fromarray(np.array([[0, 99], [100, 255]], dtype=np.uint8)).save(
        image_dir / train_name
    )
    Image.fromarray(np.array([[0, 1], [255, 255]], dtype=np.uint8)).save(
        mask_dir / train_name
    )

    manifest = {
        "train": [1],
        "val": [2],
        "test": [3],
        "_provenance": {"train_series": ["training_series"]},
    }
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    contract = prepare_train_only_contract(
        split_manifest_path=manifest_path,
        image_dir=image_dir,
        mask_dir=mask_dir,
        enforce_confirmatory_attestations=False,
    )

    assert contract["train_ids"] == (1,)
    assert contract["train_names"] == (train_name,)
    assert set(contract["mask_paths"]) == {1}
    assert contract["input_provenance"]["image_count"] == 1
    assert contract["target_provenance"]["mask_count"] == 1
    assert contract["target_provenance"]["mask_directory"] == "<external>/masks"


def test_train_only_contract_rejects_symlinked_training_files_before_read(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    outside = tmp_path / "outside.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(outside)
    name = "training_series_tile.png"
    (image_dir / name).symlink_to(outside)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(mask_dir / name)
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train": [1],
                "val": [2],
                "test": [3],
                "_provenance": {"train_series": ["training_series"]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="symbolic image/mask files"):
        prepare_train_only_contract(
            split_manifest_path=manifest_path,
            image_dir=image_dir,
            mask_dir=mask_dir,
            enforce_confirmatory_attestations=False,
        )


def test_every_declared_lock_source_has_a_non_null_sha256() -> None:
    hashes = _source_hashes()
    assert hashes
    assert all(
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(set("0123456789abcdef"))
        for value in hashes.values()
    )

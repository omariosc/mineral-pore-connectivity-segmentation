import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.evaluate_confirmatory_checkpoint as evaluator
from scripts.evaluate_confirmatory_checkpoint import (
    EXPECTED_INPUT_NORMALIZATION,
    LOCKED_INPUT_ATTESTATIONS,
    LOCKED_SPLIT_MANIFEST_SHA256,
    LOCKED_TARGET_ATTESTATIONS,
    _normalize_architecture,
    _normalization_matches,
    _recorded_normalization,
    _safe_indexed_path,
    assert_single_pass_output_available,
    aggregate_secondary_2d_diagnostics,
    attest_development_inputs_once,
    attest_development_masks_once,
    attest_held_out_and_full_once,
    attest_held_out_inputs_and_full_once,
    build_parser,
    bootstrap_secondary_2d_diagnostics,
    compose_locked_probabilities,
    confusion_from_labels,
    curves_from_histograms,
    gate_reference_diagnostic,
    infer_model_config_from_state,
    load_lossless_target_mask_bytes,
    load_verified_selected_method_lock,
    mask_corpus_sha256,
    metrics_from_confusion,
    prepare_locked_model_input,
    reserve_single_pass_output,
    resolve_complete_manifest,
    secondary_2d_operational_diagnostics,
    tile_bootstrap_intervals,
    update_probability_histograms,
    validate_checkpoint_input_provenance,
    validate_checkpoint_normalization,
    validate_checkpoint_protocol_fields,
    validate_checkpoint_selected_method_lock,
    validate_checkpoint_split_isolation,
    validate_checkpoint_target_provenance,
    validate_checkpoint_training_seed,
    validate_neural_freeze_locked_evaluation_identity,
    validate_locked_inference_runtime,
    validate_locked_evaluator_parameters,
)
from src.training.data_contract import aggregate_indexed_file_bytes
from src.training.screen_selection import (
    PROSPECTIVE_METHOD_PROTOCOLS,
    SCREEN_CANDIDATE_ORDER,
    SCREEN_SEEDS,
)


def _coco(count=6):
    return {
        "images": [
            {
                "id": image_id,
                "file_name": f"tile_{image_id}.png",
                "width": 2048,
                "height": 2048,
            }
            for image_id in range(1, count + 1)
        ],
        "categories": [
            {"id": 0, "name": "disconnected_pore"},
            {"id": 1, "name": "connected_pore"},
        ],
        "annotations": [],
    }


def _digest(names, payloads):
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(Path(name).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(payloads[name])
        digest.update(b"\0")
    return digest.hexdigest()


def _expectations(split_files, payloads, prefix, include_dev_parts=False):
    count_key = f"{prefix}_count"
    hash_key = f"{prefix}_aggregate_sha256"
    values = {
        "development_train_plus_validation": {
            count_key: len(split_files["train"]) + len(split_files["val"]),
            hash_key: _digest(split_files["train"] + split_files["val"], payloads),
        },
        "held_out_test": {
            count_key: len(split_files["test"]),
            hash_key: _digest(split_files["test"], payloads),
        },
        "full_indexed_corpus": {
            count_key: sum(len(value) for value in split_files.values()),
            hash_key: _digest(sum((list(value) for value in split_files.values()), []), payloads),
        },
    }
    if include_dev_parts:
        values["training_only"] = {
            count_key: len(split_files["train"]),
            hash_key: _digest(split_files["train"], payloads),
        }
        values["validation_only"] = {
            count_key: len(split_files["val"]),
            hash_key: _digest(split_files["val"], payloads),
        }
    return values


def _fake_corpus(tmp_path):
    split_files = {
        "train": ["b.png", "a.png"],
        "val": ["c.png"],
        "test": ["e.png", "d.png"],
    }
    payloads = {name: f"bytes:{name}".encode() for name in sum(split_files.values(), [])}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    return split_files, payloads


def _resolved_protocol_checkpoint(candidate, architecture_role="primary_multiscale"):
    protocol = PROSPECTIVE_METHOD_PROTOCOLS[candidate]
    runtime_model = (
        "plain_unet"
        if architecture_role == "plain_unet_comparator"
        else protocol["model_type"]
    )
    conditional = candidate.startswith("C2-")
    inference = (
        {
            "mode": "uint8_pore_gate_then_conditional_c0_c1",
            "raw_uint8_pore_rule": "intensity < 100",
            "raw_uint8_mineral_rule": "intensity >= 100",
            "pore_threshold_uint8": 100,
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
        if conditional
        else {
            "mode": "native_model_argmax",
            "network_outputs": 3,
            "conditional_pore_threshold": None,
        }
    )
    checkpoint = {
        "model_type": runtime_model,
        "loss_type": protocol["loss_type"],
        "num_classes": protocol["model_output_classes"],
        "resolved_config": {
            "model": {
                "architecture": runtime_model,
                "input_channels": protocol["model_input_channels"],
                "output_classes": protocol["model_output_classes"],
                "dropout_requested": protocol["dropout_requested"],
                "deep_supervision": False,
            },
            "loss": {"type": protocol["loss_type"]},
            "augmentation": {
                "data_loader": {
                    key: protocol[key]
                    for key in (
                        "training_patch_size",
                        "training_batch_size",
                        "evaluation_patch_size",
                        "evaluation_batch_size",
                    )
                }
            },
            "inference": inference,
        },
    }
    inferred = {
        "architecture": (
            "unet" if architecture_role == "plain_unet_comparator" else runtime_model
        ),
        "n_channels": protocol["model_input_channels"],
        "num_classes": protocol["model_output_classes"],
        "base_features": 32,
        "bilinear": True,
        "deep_supervision": False,
    }
    return checkpoint, inferred


def test_manifest_locks_complete_disjoint_test_set():
    resolved = resolve_complete_manifest(
        _coco(), {"train": [1, 2, 3], "val": [4], "test": [5, 6]}
    )
    assert resolved["test"] == [5, 6]
    with pytest.raises(ValueError, match="image leakage"):
        resolve_complete_manifest(
            _coco(), {"train": [1, 2, 3], "val": [4, 5], "test": [5, 6]}
        )
    with pytest.raises(ValueError, match="does not assign every"):
        resolve_complete_manifest(
            _coco(), {"train": [1, 2], "val": [3], "test": [4, 5]}
        )


def test_coco_indexed_paths_cannot_escape_declared_directory(tmp_path):
    assert _safe_indexed_path(tmp_path, "nested/tile.png") == tmp_path / "nested" / "tile.png"
    for unsafe in ("../tile.png", "/tmp/tile.png", "nested\\tile.png", ""):
        with pytest.raises(ValueError, match="Unsafe COCO image file name"):
            _safe_indexed_path(tmp_path, unsafe)


def test_confusion_metrics_include_locked_selection_and_merged_pore_metrics():
    target = np.asarray([[0, 0, 1], [1, 2, 2]], dtype=np.uint8)
    prediction = np.asarray([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    confusion = confusion_from_labels(target, prediction)
    assert confusion.tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    metrics = metrics_from_confusion(confusion)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["selection_metrics"]["c0_c1_harmonic_iou"] == pytest.approx(
        2 * (1 / 3) * (1 / 3) / (2 / 3 + 1e-8)
    )
    assert metrics["selection_metrics"]["pore_union_iou"] == pytest.approx(3 / 5)
    assert metrics["selection_metrics"]["pore_union_agreement"] == pytest.approx(4 / 6)
    assert metrics["pore_vs_mineral"]["confusion_matrix"].tolist() == [[3, 1], [1, 1]]


def test_tile_bootstrap_is_deterministic_and_covers_new_metrics():
    matrices = np.asarray(
        [
            [[8, 1, 1], [1, 7, 2], [0, 1, 9]],
            [[6, 2, 2], [0, 8, 2], [1, 0, 9]],
            [[9, 0, 1], [2, 6, 2], [1, 1, 8]],
        ],
        dtype=np.int64,
    )
    first = tile_bootstrap_intervals(matrices, replicates=200, seed=77)
    second = tile_bootstrap_intervals(matrices, replicates=200, seed=77)
    assert first == second
    assert first["sampling_unit"] == "held-out 2048x2048 tile"
    for key in (
        "selection.c0_c1_harmonic_iou",
        "selection.pore_union_iou",
        "selection.pore_union_agreement",
        "pore_vs_mineral.agreement",
        "merged_class_0.iou",
    ):
        assert key in first["intervals"]


def test_histogram_curves_accumulate_scores_without_raw_probability_storage():
    probabilities = np.asarray(
        [
            [[0.90, 0.60], [0.20, 0.05]],
            [[0.05, 0.30], [0.70, 0.15]],
            [[0.05, 0.10], [0.10, 0.80]],
        ],
        dtype=np.float32,
    )
    target = np.asarray([[0, 0], [1, 2]], dtype=np.uint8)
    positive = np.zeros((3, 16), dtype=np.int64)
    negative = np.zeros((3, 16), dtype=np.int64)
    update_probability_histograms(positive, negative, probabilities, target)
    assert positive.sum(axis=1).tolist() == [2, 1, 1]
    assert negative.sum(axis=1).tolist() == [2, 3, 3]
    curves = curves_from_histograms(positive[0], negative[0])
    assert curves["roc_auc"] == pytest.approx(1.0)
    assert curves["average_precision"] == pytest.approx(1.0)


def test_publication_style_locks_class_encoding_and_pr_as_primary() -> None:
    assert evaluator.PUBLICATION_CLASS_COLORS == (
        "#B33A3A",
        "#2E8B57",
        "#4C78A8",
    )
    assert evaluator.PALETTE == evaluator.PUBLICATION_CLASS_COLORS
    assert evaluator.PUBLICATION_CURVE_ORDER == ("precision_recall", "roc")
    assert tuple(evaluator.CLASS_LABELS) == (0, 1, 2)
    assert [evaluator.CLASS_LABELS[index].split()[0] for index in range(3)] == [
        "C0",
        "C1",
        "C2",
    ]

    styles = [evaluator._publication_curve_style(index, 4097) for index in range(3)]
    assert [style["color"] for style in styles] == list(
        evaluator.PUBLICATION_CLASS_COLORS
    )
    assert len({style["linestyle"] for style in styles}) == 3
    assert len({style["marker"] for style in styles}) == 3
    assert all(style["markerfacecolor"] == "white" for style in styles)
    assert all(style["markevery"] > 0 for style in styles)
    with pytest.raises(ValueError, match="Unknown publication class ID"):
        evaluator._publication_curve_style(3, 4097)
    with pytest.raises(ValueError, match="at least two points"):
        evaluator._publication_curve_style(0, 1)

    plt = evaluator._configure_matplotlib()
    assert list(plt.rcParams["font.family"]) == ["sans-serif"]
    assert list(plt.rcParams["font.sans-serif"][:2]) == ["Arial", "Helvetica"]
    plt.close("all")


def test_model_structure_handles_multiscale_deep_supervision_state():
    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

    state = {
        "inc.double_conv.0.weight": FakeTensor((32, 1, 3, 3)),
        "down2.conv.1.branch1.0.weight": FakeTensor((32, 128, 1, 1)),
        "outc.weight": FakeTensor((3, 32, 1, 1)),
        "ds1.weight": FakeTensor((3, 32, 1, 1)),
    }
    inferred = infer_model_config_from_state(state)
    assert inferred["architecture"] == "multiscale_attention_unet"
    assert inferred["deep_supervision"] is True
    assert inferred["n_channels"] == 1
    assert inferred["num_classes"] == 3


def test_plain_unet_alias_and_pyramid_state_inference():
    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

    assert _normalize_architecture("plain_unet") == "unet"
    assert _normalize_architecture("plain-unet") == "unet"
    state = {
        "inc.double_conv.0.weight": FakeTensor((32, 2, 3, 3)),
        "down2.conv.1.branch1.0.weight": FakeTensor((32, 128, 1, 1)),
        "outc.weight": FakeTensor((2, 32, 1, 1)),
        "pyramid_context.fuse.0.weight": FakeTensor((256, 384, 1, 1)),
    }
    inferred = infer_model_config_from_state(state)
    assert inferred["architecture"] == "multiscale_attention_unet_pyramid"
    assert (inferred["n_channels"], inferred["num_classes"]) == (2, 2)


@pytest.mark.parametrize("candidate", list(PROSPECTIVE_METHOD_PROTOCOLS))
@pytest.mark.parametrize("role", ["primary_multiscale", "plain_unet_comparator"])
def test_all_five_protocol_families_resolve_primary_and_plain_comparator(candidate, role):
    checkpoint, inferred = _resolved_protocol_checkpoint(candidate, role)
    result = validate_checkpoint_protocol_fields(checkpoint, candidate, inferred, role)
    assert result["candidate"] == candidate
    assert result["selected_architecture_role"] == role
    assert result["conditional_composition"] is candidate.startswith("C2-")


def test_conditional_protocol_rejects_wrong_gate_and_wrong_model_shape():
    checkpoint, inferred = _resolved_protocol_checkpoint("C2-F")
    checkpoint["resolved_config"]["inference"]["pore_threshold_uint8"] = 99
    with pytest.raises(ValueError, match="Conditional gate"):
        validate_checkpoint_protocol_fields(checkpoint, "C2-F", inferred)
    checkpoint, inferred = _resolved_protocol_checkpoint("C2-F")
    inferred["n_channels"] = 1
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_checkpoint_protocol_fields(checkpoint, "C2-F", inferred)


def test_conditional_input_and_full_area_composition_are_exact():
    image = np.asarray([[0, 99], [100, 255]], dtype=np.uint8)
    model_input = prepare_locked_model_input(image, "C2-P")
    np.testing.assert_allclose(model_input[0], image.astype(np.float32) / 127.5 - 1.0)
    np.testing.assert_array_equal(model_input[1], [[1.0, 1.0], [0.0, 0.0]])
    network = np.asarray(
        [[[0.8, 0.2], [0.4, 0.9]], [[0.2, 0.8], [0.6, 0.1]]], dtype=np.float32
    )
    composed = compose_locked_probabilities(image, network, "C2-P")
    np.testing.assert_allclose(composed[:, 0, 0], [0.8, 0.2, 0.0])
    np.testing.assert_allclose(composed[:, 0, 1], [0.2, 0.8, 0.0])
    np.testing.assert_allclose(composed[:, 1, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(composed[:, 1, 1], [0.0, 0.0, 1.0])
    assert composed.argmax(axis=0).tolist() == [[0, 1], [2, 2]]


def test_gate_reference_diagnostic_reports_intrinsic_rule_mismatch():
    image = np.asarray([[0, 100], [99, 255]], dtype=np.uint8)
    target = np.asarray([[0, 1], [2, 2]], dtype=np.uint8)
    diagnostic = gate_reference_diagnostic(image, target)
    assert diagnostic["confusion_matrix"].tolist() == [[1, 1], [1, 1]]
    assert diagnostic["mismatched_pixels"] == 2
    assert diagnostic["mismatch_rate"] == pytest.approx(0.5)
    assert diagnostic["reference_c0_gate_mineral_pixels"] == 0
    assert diagnostic["reference_c1_gate_mineral_pixels"] == 1
    assert "not exact label reconstruction" in diagnostic["interpretation"]


def test_secondary_2d_diagnostics_use_locked_operational_definitions():
    pytest.importorskip("cv2")
    target = np.full((8, 8), 2, dtype=np.uint8)
    target[1:3, 1:3] = 0
    target[5, 5] = 0
    target[1:4, 4:7] = 1
    prediction = target.copy()
    diagnostic = secondary_2d_operational_diagnostics(target, prediction)
    assert diagnostic["c0_area_fraction"]["reference"] == pytest.approx(5 / 64)
    assert diagnostic["c0_area_fraction"]["absolute_error"] == 0
    assert diagnostic["c1_area_fraction"]["reference"] == pytest.approx(9 / 64)
    assert diagnostic["largest_c1_8_connected_component_c1_fraction"][
        "reference"
    ] == 1.0
    assert diagnostic["c0_8_connected_components_per_megapixel"][
        "reference"
    ] == pytest.approx(2_000_000 / 64)
    assert diagnostic["pore_union_boundary_f1_at_2px"]["f1"] == 1.0
    assert diagnostic["component_filtering"] == "none"
    assert "not permeability" in diagnostic["scientific_interpretation"]


def test_secondary_2d_aggregate_and_tile_bootstrap_are_deterministic():
    pytest.importorskip("cv2")
    target = np.full((8, 8), 2, dtype=np.uint8)
    target[1:4, 1:4] = 0
    target[4:7, 4:7] = 1
    first_prediction = target.copy()
    second_prediction = target.copy()
    second_prediction[1, 1] = 2
    diagnostics = [
        secondary_2d_operational_diagnostics(target, first_prediction),
        secondary_2d_operational_diagnostics(target, second_prediction),
    ]
    aggregate = aggregate_secondary_2d_diagnostics(diagnostics)
    assert aggregate["tile_count"] == 2
    assert aggregate["c0_area_fraction"]["absolute_error"] == pytest.approx(1 / 128)
    first = bootstrap_secondary_2d_diagnostics(
        diagnostics, replicates=100, seed=19, confidence=0.95
    )
    second = bootstrap_secondary_2d_diagnostics(
        diagnostics, replicates=100, seed=19, confidence=0.95
    )
    assert first == second
    assert "pore_union_boundary_f1_at_2px.f1" in first["intervals"]
    assert (
        "largest_c1_8_connected_component_c1_fraction.absolute_error"
        in first["intervals"]
    )


def test_two_phase_attestation_never_reads_fake_test_before_development_auth(tmp_path):
    split_files, payloads = _fake_corpus(tmp_path)
    expected = _expectations(split_files, payloads, "mask")
    reads = []

    def reader(path):
        reads.append(path.name)
        return path.read_bytes()

    development, cached = attest_development_masks_once(
        tmp_path, split_files, expected_attestations=expected, read_bytes=reader
    )
    assert development["held_out_bytes_read"] == 0
    assert set(reads) == {"a.png", "b.png", "c.png"}
    assert not set(split_files["test"]) & set(reads)
    post, test_payloads, _ = attest_held_out_and_full_once(
        tmp_path,
        split_files,
        cached,
        expected_attestations=expected,
        read_bytes=reader,
    )
    assert post["held_out_test"]["mask_aggregate_sha256"] != development[
        "mask_aggregate_sha256"
    ]
    assert post["full_indexed_corpus"]["mask_aggregate_sha256"] not in {
        development["mask_aggregate_sha256"],
        post["held_out_test"]["mask_aggregate_sha256"],
    }
    assert set(test_payloads) == set(split_files["test"])
    assert all(reads.count(name) == 1 for name in split_files["test"])


def test_development_rejection_path_reads_no_fake_heldout_bytes(tmp_path):
    split_files, payloads = _fake_corpus(tmp_path)
    expected = _expectations(split_files, payloads, "mask")
    expected["development_train_plus_validation"]["mask_aggregate_sha256"] = "0" * 64
    reads = []

    def reader(path):
        reads.append(path.name)
        return path.read_bytes()

    with pytest.raises(ValueError, match="development_train_plus_validation"):
        attest_development_masks_once(
            tmp_path, split_files, expected_attestations=expected, read_bytes=reader
        )
    assert not set(split_files["test"]) & set(reads)


def test_input_attestation_matches_shared_training_hash_implementation(tmp_path):
    split_files, payloads = _fake_corpus(tmp_path)
    expected = _expectations(split_files, payloads, "image", include_dev_parts=True)
    records, cached = attest_development_inputs_once(
        tmp_path, split_files, expected_attestations=expected
    )
    common = aggregate_indexed_file_bytes(
        tmp_path,
        split_files["train"] + split_files["val"],
        scope="development_train_plus_validation",
        split_names=("train", "val"),
    )
    assert records["development_train_plus_validation"]["image_aggregate_sha256"] == common[
        "image_aggregate_sha256"
    ]
    post, test_payloads, _ = attest_held_out_inputs_and_full_once(
        tmp_path, split_files, cached, expected_attestations=expected
    )
    assert post["held_out_test"]["read_passes"] == 1
    assert set(test_payloads) == set(split_files["test"])


def test_live_canonical_development_input_hashes_recompute_without_test_access():
    root = evaluator.PROJECT_ROOT
    annotation_path = root / "results/step3_coco_dataset/pore_annotations.json"
    manifest_path = root / "config/confirmatory_splits.json"
    image_root = root / "results/step3_coco_dataset/images"
    if not annotation_path.is_file() or not manifest_path.is_file() or not image_root.is_dir():
        pytest.skip("Canonical local image corpus is not available")
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_ids = resolve_complete_manifest(coco, manifest)
    by_id = {int(item["id"]): item for item in coco["images"]}
    split_files = {
        split: [str(by_id[image_id]["file_name"]) for image_id in ids]
        for split, ids in split_ids.items()
    }
    reads = []

    def reader(path):
        reads.append(path.relative_to(image_root).as_posix())
        return path.read_bytes()

    records, _ = attest_development_inputs_once(image_root, split_files, read_bytes=reader)
    assert records["training_only"]["image_aggregate_sha256"] == LOCKED_INPUT_ATTESTATIONS[
        "training_only"
    ]["image_aggregate_sha256"]
    assert records["validation_only"]["image_aggregate_sha256"] == LOCKED_INPUT_ATTESTATIONS[
        "validation_only"
    ]["image_aggregate_sha256"]
    assert records["development_train_plus_validation"][
        "image_aggregate_sha256"
    ] == LOCKED_INPUT_ATTESTATIONS["development_train_plus_validation"][
        "image_aggregate_sha256"
    ]
    assert not set(split_files["test"]) & set(reads)


def test_live_canonical_development_mask_hash_recomputes_without_test_access():
    root = evaluator.PROJECT_ROOT
    annotation_path = root / "results/step3_coco_dataset/pore_annotations.json"
    manifest_path = root / "config/confirmatory_splits.json"
    mask_root = root / "results/step2_pore_classification/pore_classifications"
    if not annotation_path.is_file() or not manifest_path.is_file() or not mask_root.is_dir():
        pytest.skip("Canonical local mask corpus is not available")
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_ids = resolve_complete_manifest(coco, manifest)
    by_id = {int(item["id"]): item for item in coco["images"]}
    split_files = {
        split: [str(by_id[image_id]["file_name"]) for image_id in ids]
        for split, ids in split_ids.items()
    }
    reads = []

    def reader(path):
        reads.append(path.relative_to(mask_root).as_posix())
        return path.read_bytes()

    record, _ = attest_development_masks_once(mask_root, split_files, read_bytes=reader)
    assert record["mask_count"] == 79
    assert record["mask_aggregate_sha256"] == LOCKED_TARGET_ATTESTATIONS[
        "development_train_plus_validation"
    ]["mask_aggregate_sha256"]
    assert not set(split_files["test"]) & set(reads)


def test_mask_corpus_hash_is_name_sorted_content_sensitive_and_matches_common_bytes(tmp_path):
    first = tmp_path / "b.png"
    second = tmp_path / "a.png"
    first.write_bytes(b"mask-b")
    second.write_bytes(b"mask-a")
    digest_one = mask_corpus_sha256([first, second], tmp_path)
    digest_two = mask_corpus_sha256([second, first], tmp_path)
    assert digest_one == digest_two
    common = aggregate_indexed_file_bytes(
        tmp_path, ["b.png", "a.png"], scope="test", split_names=("train",)
    )
    assert digest_one == common["image_aggregate_sha256"]
    first.write_bytes(b"changed")
    assert mask_corpus_sha256([first, second], tmp_path) != digest_one


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            np.asarray([[0, 0], [0, 0]], dtype=np.uint8),
            np.asarray([[0, 0], [0, 0]], dtype=np.uint8),
        ),
        (
            np.asarray([[1, 1], [1, 1]], dtype=np.uint8),
            np.asarray([[1, 1], [1, 1]], dtype=np.uint8),
        ),
        (
            np.asarray([[255, 255], [255, 255]], dtype=np.uint8),
            np.asarray([[2, 2], [2, 2]], dtype=np.uint8),
        ),
        (
            np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
            np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
        ),
        (
            np.asarray([[0, 255], [255, 0]], dtype=np.uint8),
            np.asarray([[0, 2], [2, 0]], dtype=np.uint8),
        ),
        (
            np.asarray([[1, 255], [255, 1]], dtype=np.uint8),
            np.asarray([[1, 2], [2, 1]], dtype=np.uint8),
        ),
        (
            np.asarray([[0, 1], [255, 1]], dtype=np.uint8),
            np.asarray([[0, 1], [2, 1]], dtype=np.uint8),
        ),
    ],
)
def test_lossless_target_mask_accepts_valid_class_subsets(
    monkeypatch, source, expected
):
    fake_cv2 = SimpleNamespace(
        IMREAD_UNCHANGED=-1,
        imdecode=lambda payload, mode: source.copy(),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(evaluator, "EXPECTED_TILE_SHAPE", (2, 2))
    observed = load_lossless_target_mask_bytes(Path("tile.png"), b"attested-png")
    np.testing.assert_array_equal(observed, expected)


def test_lossless_target_mask_rejects_any_out_of_contract_value(monkeypatch):
    source = np.asarray([[0, 1], [2, 255]], dtype=np.uint8)
    fake_cv2 = SimpleNamespace(
        IMREAD_UNCHANGED=-1,
        imdecode=lambda payload, mode: source.copy(),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(evaluator, "EXPECTED_TILE_SHAPE", (2, 2))
    with pytest.raises(ValueError, match="nonempty subset"):
        load_lossless_target_mask_bytes(Path("tile.png"), b"attested-png")


def test_checkpoint_normalization_is_required_at_both_structured_levels():
    checkpoint = {
        "input_normalization": dict(EXPECTED_INPUT_NORMALIZATION),
        "resolved_config": {"input_normalization": dict(EXPECTED_INPUT_NORMALIZATION)},
    }
    assert validate_checkpoint_normalization(checkpoint) == EXPECTED_INPUT_NORMALIZATION
    recorded = _recorded_normalization(checkpoint)
    assert isinstance(recorded, dict) and _normalization_matches(recorded)
    checkpoint["resolved_config"]["input_normalization"]["output_range"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="normalization"):
        validate_checkpoint_normalization(checkpoint)


def test_checkpoint_target_provenance_authenticates_dev79_only_and_hides_path():
    development = {
        "mask_count": 79,
        "mask_aggregate_sha256": "a" * 64,
        "mask_aggregate_sha256_algorithm": (
            "sha256 over lexicographically sorted UTF-8 relative filename, NUL, raw file bytes, NUL"
        ),
    }
    provenance = {
        "target_source": "lossless_png_masks",
        "mask_directory": "/private/training/path/not-for-public-output",
        "mask_count": 79,
        "mask_aggregate_sha256": "a" * 64,
        "mask_aggregate_sha256_algorithm": development[
            "mask_aggregate_sha256_algorithm"
        ],
        "validated_source_values": [0, 1, 255],
        "canonical_value_mapping": {
            "0": "0 (disconnected_pore)",
            "1": "1 (connected_pore)",
            "255": "2 (mineral) in three-class mode; ignore_index in two-class mode",
        },
        "annotations_role": "image_index_and_metadata_only",
        "evaluation_mode": "train_validation_only",
        "held_out_dataset_constructed": False,
    }
    checkpoint = {
        "target_provenance": provenance,
        "resolved_config": {"target": dict(provenance)},
    }
    public = validate_checkpoint_target_provenance(checkpoint, development)
    assert public["mask_count"] == 79
    assert public["held_out_test_included"] is False
    assert "mask_directory" not in public
    checkpoint["target_provenance"]["mask_aggregate_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="disagree"):
        validate_checkpoint_target_provenance(checkpoint, development)


def test_checkpoint_input_provenance_requires_dev79_and_train74_not_test():
    algorithm = (
        "sha256 over lexicographically sorted UTF-8 relative filename, NUL, raw file bytes, NUL"
    )
    development_records = {
        "development_train_plus_validation": {
            "scope": "development_train_plus_validation",
            "split_names": ["train", "val"],
            "image_count": 79,
            "image_aggregate_sha256": "d" * 64,
            "image_aggregate_sha256_algorithm": algorithm,
            "file_name_list_sha256": "e" * 64,
        },
        "training_only": {
            "scope": "training_only",
            "split_names": ["train"],
            "image_count": 74,
            "image_aggregate_sha256": "a" * 64,
            "image_aggregate_sha256_algorithm": algorithm,
            "file_name_list_sha256": "b" * 64,
        },
    }
    training = evaluator._provenance_attestation_fields(
        development_records["training_only"], "image"
    )
    provenance = {
        "input_source": "indexed_source_images",
        **evaluator._provenance_attestation_fields(
            development_records["development_train_plus_validation"], "image"
        ),
        "training_subset": training,
        "held_out_bytes_read": 0,
        "held_out_scope": "not_read_or_hashed_by_validation_only_trainer",
    }
    checkpoint = {
        "input_provenance": provenance,
        "resolved_config": {"input": dict(provenance)},
    }
    public = validate_checkpoint_input_provenance(checkpoint, development_records)
    assert public["held_out_test_included"] is False
    checkpoint["resolved_config"]["input"]["held_out_bytes_read"] = 1
    with pytest.raises(ValueError, match="disagree"):
        validate_checkpoint_input_provenance(checkpoint, development_records)


def test_checkpoint_split_isolation_requires_exact_ids_and_files():
    split_ids = {"train": [1, 2], "val": [3], "test": [4]}
    split_files = {key: [f"tile_{value}.png" for value in values] for key, values in split_ids.items()}
    partitions = {
        key: {
            "image_ids": values,
            "image_files": split_files[key],
            "image_count": len(values),
        }
        for key, values in split_ids.items()
    }
    assignment_sha256 = hashlib.sha256(
        json.dumps(partitions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint = {
        "resolved_config": {
            "evaluation": {
                "mode": "validation_only",
                "held_out_dataset_constructed": False,
                "held_out_evaluation_count": 0,
            },
            "data_split": {
                "validation_only": True,
                "held_out_dataset_constructed": False,
                "held_out_evaluation_count": 0,
                "manifest_source": "explicit_manifest",
                "manifest_repo_relative_identifier": "config/confirmatory_splits.json",
                "manifest_sha256": LOCKED_SPLIT_MANIFEST_SHA256,
                "annotation_index_repo_relative_identifier": (
                    "results/step3_coco_dataset/pore_annotations.json"
                ),
                "annotation_index_sha256": evaluator.LOCKED_ANNOTATION_INDEX_SHA256,
                "partition_assignment_sha256": assignment_sha256,
                "partitions": partitions,
            },
        }
    }
    result = validate_checkpoint_split_isolation(
        checkpoint,
        manifest_sha256=LOCKED_SPLIT_MANIFEST_SHA256,
        split_ids=split_ids,
        split_files=split_files,
    )
    assert result["held_out_evaluation_count_before_evaluator"] == 0
    checkpoint["resolved_config"]["data_split"]["partitions"]["test"][
        "image_files"
    ] = ["wrong.png"]
    with pytest.raises(ValueError, match="test image files"):
        validate_checkpoint_split_isolation(
            checkpoint,
            manifest_sha256=LOCKED_SPLIT_MANIFEST_SHA256,
            split_ids=split_ids,
            split_files=split_files,
        )


def _minimal_verified_lock(selected="R3"):
    cells = [
        {"candidate": candidate, "seed": seed}
        for candidate in SCREEN_CANDIDATE_ORDER
        for seed in SCREEN_SEEDS
    ]
    protocol = dict(PROSPECTIVE_METHOD_PROTOCOLS[selected])
    return {
        "schema_version": evaluator.SELECTED_METHOD_LOCK_SCHEMA_VERSION,
        "selected_method": selected,
        "resolved_protocol": protocol,
        "screen_selection_provenance": {
            "screen_cells": cells,
            "split_manifest_sha256": LOCKED_SPLIT_MANIFEST_SHA256,
            "target_mask_aggregate_sha256": LOCKED_TARGET_ATTESTATIONS[
                "development_train_plus_validation"
            ]["mask_aggregate_sha256"],
            "input_image_aggregate_sha256": LOCKED_INPUT_ATTESTATIONS[
                "development_train_plus_validation"
            ]["image_aggregate_sha256"],
        },
    }


def test_selected_lock_rejects_wrong_embedded_lock_and_role():
    verified = _minimal_verified_lock()
    identifier = "config/selected_method_lock.json"
    lock_sha = "f" * 64
    embedded = {
        "schema_version": verified["schema_version"],
        "selected_method": "R3",
        "lock_file_repo_relative_identifier": identifier,
        "lock_file_sha256": lock_sha,
        "resolved_protocol": verified["resolved_protocol"],
        "screen_selection_provenance": verified["screen_selection_provenance"],
    }
    checkpoint = {
        "resolved_config": {
            "selected_method_lock": embedded,
            "protocol_candidate_key": "R3",
            "protocol_run_role": "selected_winner_retraining",
            "selected_architecture_role": "primary_multiscale",
            "protocol_campaign_id": "selected_retrain_campaign_1",
            "protocol_cell_index": 0,
            "augmentation": {"seed": 42},
        }
    }
    result = validate_checkpoint_selected_method_lock(
        checkpoint, verified, lock_sha256=lock_sha, lock_identifier=identifier
    )
    assert result[0] == "R3"
    assert result[3]["array_task_index"] == 0
    assert result[3]["training_seed"] == 42
    checkpoint["resolved_config"]["protocol_run_role"] = "validation_screen_cell"
    with pytest.raises(ValueError, match="protocol_run_role"):
        validate_checkpoint_selected_method_lock(
            checkpoint, verified, lock_sha256=lock_sha, lock_identifier=identifier
        )
    checkpoint["resolved_config"]["protocol_run_role"] = "selected_winner_retraining"
    checkpoint["resolved_config"]["protocol_cell_index"] = 1
    with pytest.raises(ValueError, match="task/seed mapping"):
        validate_checkpoint_selected_method_lock(
            checkpoint, verified, lock_sha256=lock_sha, lock_identifier=identifier
        )
    checkpoint["resolved_config"]["protocol_cell_index"] = 0
    checkpoint["resolved_config"]["selected_method_lock"]["lock_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lock mismatch"):
        validate_checkpoint_selected_method_lock(
            checkpoint, verified, lock_sha256=lock_sha, lock_identifier=identifier
        )


def test_lock_loader_requires_complete_ordered_15_cell_matrix(tmp_path, monkeypatch):
    verified = _minimal_verified_lock()
    verified["screen_selection_provenance"]["screen_cells"].pop()
    lock_path = tmp_path / "incomplete_selected_lock.json"
    lock_path.write_text(json.dumps(verified), encoding="utf-8")
    monkeypatch.setattr(evaluator, "verify_selected_method_lock_document", lambda value, root: value)
    monkeypatch.setattr(evaluator, "_repository_relative", lambda path: path.name)
    with pytest.raises(ValueError, match="complete 15-cell"):
        load_verified_selected_method_lock(lock_path)


def test_training_seed_is_distinct_and_matches_lock_execution_contract():
    seed = 123
    augmentation = {
        "seed": seed,
        "data_loader": {
            "shuffle_generator_seed": seed,
            "distributed_sampler_seed": None,
        },
    }
    model = {"architecture_resolved": "multiscale_attention_unet"}
    execution = {"augmentation": augmentation, "model": model, "loss": {"type": "focal_dice"}}
    checkpoint = {
        "resolved_config": {
            "augmentation": augmentation,
            "scientific_execution_contract": execution,
        }
    }
    cells = []
    for index, (candidate, candidate_seed) in enumerate(
        (pair for candidate in SCREEN_CANDIDATE_ORDER for pair in ((candidate, value) for value in SCREEN_SEEDS))
    ):
        cells.append(
            {
                "candidate": candidate,
                "seed": candidate_seed,
                "array_index": index,
                "scientific_execution_contract": (
                    execution if (candidate, candidate_seed) == ("R3", seed) else {}
                ),
            }
        )
    lock = {"screen_selection_provenance": {"screen_cells": cells}}
    result = validate_checkpoint_training_seed(
        checkpoint,
        lock,
        candidate="R3",
        architecture_role="primary_multiscale",
    )
    assert result["training_seed"] == seed
    assert result["evaluator_rng_is_distinct"] is True
    checkpoint["resolved_config"]["augmentation"]["data_loader"][
        "shuffle_generator_seed"
    ] = 42
    with pytest.raises(ValueError, match="loader seed"):
        validate_checkpoint_training_seed(
            checkpoint,
            lock,
            candidate="R3",
            architecture_role="primary_multiscale",
        )


def _verified_freeze(tmp_path, campaign="selected900", cell=1):
    manifest_id = "neural-freeze-" + "a" * 16
    scientific_identity = "a" * 64
    identifier = (
        "results/patch_training/protocol_runs/selected_winner_retraining/"
        f"{campaign}/cell_{cell:02d}/checkpoints/best_model.pth"
    )
    path = tmp_path / identifier
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original-checkpoint")
    checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    lock_identifier = "config/selected_method_lock.json"
    lock_path = tmp_path / lock_identifier
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"selected_method":"R3"}\n', encoding="utf-8")
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()

    def role_record(role, role_campaign):
        cells = []
        for index, seed in enumerate(SCREEN_SEEDS):
            cell_identifier = (
                "results/patch_training/protocol_runs/selected_winner_retraining/"
                f"{role_campaign}/cell_{index:02d}/checkpoints/best_model.pth"
            )
            cells.append(
                {
                    "array_task_index": index,
                    "training_seed": int(seed),
                    "architecture_role": role,
                    "campaign_id": role_campaign,
                    "checkpoint_repo_relative_identifier": cell_identifier,
                    "checkpoint_sha256": checkpoint_sha if index == cell else "b" * 64,
                    "checkpoint_state_dict_semantic_sha256": "c" * 64,
                }
            )
        return {"campaign_id": role_campaign, "cells": cells}

    document = {
        "manifest_id": manifest_id,
        "scientific_identity_sha256": scientific_identity,
        "selected_method": "R3",
        "selected_method_lock": {
            "repo_relative_identifier": lock_identifier,
            "raw_file_sha256": lock_sha,
        },
        "selected_retraining": {
            "primary_multiscale": role_record("primary_multiscale", campaign),
            "plain_unet_comparator": role_record(
                "plain_unet_comparator", "plain-selected900"
            ),
        },
    }
    verified = {
        "manifest_id": manifest_id,
        "manifest_file_sha256": "d" * 64,
        "scientific_identity_sha256": scientific_identity,
        "selected_method": "R3",
        "document": document,
    }
    return verified, identifier, path


def test_canonical_output_reservation_blocks_repackaged_same_cell(tmp_path):
    verified, _identifier, checkpoint_path = _verified_freeze(tmp_path)
    resolved_checkpoint, _lock, output_dir, identity = (
        validate_neural_freeze_locked_evaluation_identity(
        verified,
        cell_index=1,
        architecture_role="primary_multiscale",
        repository_root=tmp_path,
        )
    )
    assert resolved_checkpoint == checkpoint_path
    assert output_dir == (
        tmp_path
        / "results/confirmatory_evaluation/locked/neural-freeze-aaaaaaaaaaaaaaaa/primary_multiscale/cell_01"
    )
    expected = assert_single_pass_output_available(output_dir)
    first_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    reserved = reserve_single_pass_output(output_dir, first_sha, identity)
    assert reserved == expected
    guard = json.loads((reserved / "held_out_access_guard.json").read_text())
    assert guard["evaluation_identity"] == identity

    # Re-serialising or otherwise replacing the checkpoint changes its file SHA,
    # but cannot create a second reservation for the authenticated cell.
    checkpoint_path.write_bytes(b"re-serialised-equivalent-checkpoint")
    second_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert second_sha != first_sha
    with pytest.raises(FileExistsError, match="forbids a second pass"):
        reserve_single_pass_output(output_dir, second_sha, identity)


def test_campaign_alias_resolves_to_same_freeze_reservation(tmp_path):
    original, _identifier, checkpoint_path = _verified_freeze(
        tmp_path, campaign="selected900"
    )
    _checkpoint, _lock, output_dir, identity = (
        validate_neural_freeze_locked_evaluation_identity(
            original,
            cell_index=1,
            architecture_role="primary_multiscale",
            repository_root=tmp_path,
        )
    )
    reserve_single_pass_output(
        output_dir, hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), identity
    )

    aliased, _alias_identifier, _alias_path = _verified_freeze(
        tmp_path, campaign="copied-selected900"
    )
    _checkpoint, _lock, aliased_output, aliased_identity = (
        validate_neural_freeze_locked_evaluation_identity(
            aliased,
            cell_index=1,
            architecture_role="primary_multiscale",
            repository_root=tmp_path,
        )
    )
    assert aliased_output == output_dir
    assert aliased_identity["reservation_key"] == identity["reservation_key"]
    with pytest.raises(FileExistsError, match="forbids a second pass"):
        reserve_single_pass_output(
            aliased_output,
            aliased_identity["checkpoint_sha256"],
            aliased_identity,
        )


@pytest.mark.parametrize(
    ("identifier_transform", "message"),
    [
        (
            lambda value: value.replace("/cell_01/", "/detour/../cell_01/"),
            "non-canonical selected checkpoint",
        ),
        (
            lambda value: str((Path("/") / value).resolve()),
            "non-canonical selected checkpoint",
        ),
    ],
)
def test_canonical_checkpoint_rejects_traversal_campaign_and_absolute_aliases(
    tmp_path, identifier_transform, message
):
    verified, identifier, _checkpoint_path = _verified_freeze(tmp_path)
    verified["document"]["selected_retraining"]["primary_multiscale"]["cells"][1][
        "checkpoint_repo_relative_identifier"
    ] = identifier_transform(identifier)
    with pytest.raises(ValueError, match=message):
        validate_neural_freeze_locked_evaluation_identity(
            verified,
            cell_index=1,
            architecture_role="primary_multiscale",
            repository_root=tmp_path,
        )


def test_canonical_checkpoint_rejects_symlinked_checkpoint(tmp_path):
    identifier = (
        "results/patch_training/protocol_runs/selected_winner_retraining/"
        "selected900/cell_01/checkpoints/best_model.pth"
    )
    external = tmp_path / "alternate-checkpoint.pth"
    external.write_bytes(b"alternate")
    canonical = tmp_path / identifier
    canonical.parent.mkdir(parents=True)
    canonical.symlink_to(external)
    verified, _identifier, _path = _verified_freeze(
        tmp_path / "fixture", campaign="selected900"
    )
    verified["document"]["selected_retraining"]["primary_multiscale"]["cells"][1][
        "checkpoint_repo_relative_identifier"
    ] = identifier
    verified["document"]["selected_retraining"]["primary_multiscale"]["cells"][1][
        "checkpoint_sha256"
    ] = hashlib.sha256(external.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="symlink component"):
        validate_neural_freeze_locked_evaluation_identity(
            verified,
            cell_index=1,
            architecture_role="primary_multiscale",
            repository_root=tmp_path,
        )


def test_cli_requires_neural_freeze_role_and_cell_and_locks_amp_on():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--neural-freeze-id",
            "neural-freeze-aaaaaaaaaaaaaaaa",
            "--architecture-role",
            "primary_multiscale",
            "--cell-index",
            "1",
        ]
    )
    assert args.amp is True
    assert validate_locked_evaluator_parameters(args)["caller_selectable"] is False
    args.curve_bins = 512
    with pytest.raises(ValueError, match="parameter drift"):
        validate_locked_evaluator_parameters(args)
    with pytest.raises(SystemExit):
        parser.parse_args(["--neural-freeze-id", "neural-freeze-aaaaaaaaaaaaaaaa"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--neural-freeze-id",
                "neural-freeze-aaaaaaaaaaaaaaaa",
                "--architecture-role",
                "primary_multiscale",
                "--cell-index",
                "1",
                "--output-dir",
                "alternate-output-root",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "model.pth",
                "--selected-method-lock",
                "lock.json",
            ]
        )


def test_inference_precision_is_fixed_to_l40s_cuda_float16():
    record = validate_locked_inference_runtime(
        "cuda:0", True, cuda_device_name="NVIDIA L40S"
    )
    assert record["autocast_dtype"] == "float16"
    assert record["caller_selectable"] is False
    with pytest.raises(ValueError, match="L40S"):
        validate_locked_inference_runtime(
            "cuda:0", True, cuda_device_name="NVIDIA A100-SXM4-80GB"
        )
    with pytest.raises(ValueError, match="requires AMP"):
        validate_locked_inference_runtime(
            "cuda:0", False, cuda_device_name="NVIDIA L40S"
        )

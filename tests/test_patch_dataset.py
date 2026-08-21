import hashlib
import json

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.training.patch_dataset import (
    PatchDataset,
    create_patch_data_loaders,
    deterministic_axis_positions,
    resolve_split_manifest,
    seed_patch_dataloader_worker,
    validate_lossless_mask_directory,
)
from src.training.data_contract import aggregate_indexed_file_bytes


def test_deterministic_grid_preserves_683_and_supports_native_tiles():
    assert deterministic_axis_positions(2048, 683) == (0, 683, 1365)
    assert deterministic_axis_positions(2048, 1024) == (0, 1024)
    assert deterministic_axis_positions(2048, 2048) == (0,)
    with pytest.raises(ValueError, match="exceeds"):
        deterministic_axis_positions(2048, 2049)


def test_indexed_input_aggregate_is_content_sensitive_and_subset_isolated(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "train.png").write_bytes(b"train-v1")
    (root / "val.png").write_bytes(b"val-v1")
    (root / "held-out.png").write_bytes(b"test-v1")

    first = aggregate_indexed_file_bytes(
        root,
        ["val.png", "train.png"],
        scope="development_train_plus_validation",
        split_names=("train", "val"),
    )
    (root / "held-out.png").write_bytes(b"test-v2-never-read")
    held_out_changed = aggregate_indexed_file_bytes(
        root,
        ["train.png", "val.png"],
        scope="development_train_plus_validation",
        split_names=("train", "val"),
    )
    assert held_out_changed == first

    (root / "train.png").write_bytes(b"train-v2")
    train_changed = aggregate_indexed_file_bytes(
        root,
        ["train.png", "val.png"],
        scope="development_train_plus_validation",
        split_names=("train", "val"),
    )
    assert train_changed["image_aggregate_sha256"] != first[
        "image_aggregate_sha256"
    ]


def _coco_data(image_count=6):
    return {
        "images": [
            {
                "id": image_id,
                "file_name": f"source_{image_id}.png",
                "width": 6,
                "height": 6,
            }
            for image_id in range(1, image_count + 1)
        ],
        "categories": [
            {"id": 0, "name": "disconnected_pore"},
            {"id": 1, "name": "connected_pore"},
        ],
        "annotations": [],
    }


def test_manifest_produces_disjoint_whole_image_datasets():
    coco_data = _coco_data()
    manifest = {"train": [1, 2, 3], "val": [4], "test": [5, 6]}

    train_loader, val_loader, test_loader = create_patch_data_loaders(
        coco_data,
        image_dir="unused",
        batch_size=2,
        patch_size=2,
        split_manifest=manifest,
        augmentation_config={
            "patch_size": 2,
            "augmentation": {"enabled": False, "strength": "none"},
        },
        num_classes=3,
        num_workers=0,
        pin_memory=False,
        return_test=True,
    )

    train_ids = set(train_loader.dataset.image_ids)
    val_ids = set(val_loader.dataset.image_ids)
    test_ids = set(test_loader.dataset.image_ids)
    assert train_ids == {1, 2, 3}
    assert val_ids == {4}
    assert test_ids == {5, 6}
    assert train_ids.isdisjoint(val_ids | test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_loader.dataset) == 3 * 9
    assert len(val_loader.dataset) == 1 * 9
    assert len(test_loader.dataset) == 2 * 9


def test_loader_uses_training_grid_but_native_validation_tiles():
    coco_data = _coco_data(image_count=3)
    train_loader, val_loader, test_loader = create_patch_data_loaders(
        coco_data,
        image_dir="unused",
        batch_size=2,
        evaluation_batch_size=1,
        patch_size=2,
        evaluation_patch_size=6,
        split_manifest={"train": [1], "val": [2], "test": [3]},
        augmentation_config={
            "patch_size": 2,
            "augmentation": {"enabled": False, "strength": "none"},
        },
        num_classes=3,
        num_workers=0,
        pin_memory=False,
        return_test=True,
    )

    assert len(train_loader.dataset) == 9
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1
    assert train_loader.dataset.patch_size == 2
    assert val_loader.dataset.patch_size == 6
    assert val_loader.batch_size == 1
    assert val_loader.dataset.patches[0]["start_y"] == 0
    assert val_loader.dataset.patches[0]["start_x"] == 0


def test_manifest_rejects_image_leakage_and_missing_images():
    coco_data = _coco_data(image_count=4)
    with pytest.raises(ValueError, match="image leakage"):
        resolve_split_manifest(
            coco_data,
            {"train": [1, 2], "val": [2], "test": [3, 4]},
        )

    with pytest.raises(ValueError, match="does not assign all"):
        resolve_split_manifest(
            coco_data,
            {"train": [1, 2], "val": [3], "test": []},
        )


def test_manifest_accepts_file_names(tmp_path):
    coco_data = _coco_data(image_count=3)
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train": ["source_1.png"],
                "val": ["source_2.png"],
                "test": ["source_3.png"],
            }
        ),
        encoding="utf-8",
    )
    assert resolve_split_manifest(coco_data, manifest_path) == {
        "train": [1],
        "val": [2],
        "test": [3],
    }


def test_three_class_mask_preserves_mineral_and_maps_category_names(tmp_path):
    image = np.full((6, 6), 127, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "source.png"), image)
    coco_data = {
        "images": [
            {"id": 1, "file_name": "source.png", "width": 6, "height": 6}
        ],
        # This deliberately uses the alternative historical category order.
        "categories": [
            {"id": 10, "name": "mineral"},
            {"id": 20, "name": "connected_pore"},
            {"id": 30, "name": "disconnected_pore"},
        ],
        "annotations": [
            {
                "image_id": 1,
                "category_id": 30,
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
            }
        ],
    }
    dataset = PatchDataset(
        coco_data,
        str(tmp_path),
        patch_size=3,
        training=False,
        num_classes=3,
    )

    _, mask, _ = dataset[0]
    assert mask[0, 0].item() == 0
    assert mask[1, 1].item() == 0
    assert mask[0, 1].item() == 0
    assert mask.max().item() == 2
    assert set(mask.unique().tolist()) == {0, 2}


def test_lossless_mask_is_authoritative_and_maps_255_to_mineral(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    image = np.full((6, 6), 127, dtype=np.uint8)
    source_mask = np.full((6, 6), 255, dtype=np.uint8)
    source_mask[0, 0] = 0
    source_mask[0, 1] = 1
    assert cv2.imwrite(str(image_dir / "source.png"), image)
    assert cv2.imwrite(str(mask_dir / "source.png"), source_mask)
    coco_data = {
        "images": [
            {"id": 1, "file_name": "source.png", "width": 6, "height": 6}
        ],
        "categories": [
            {"id": 0, "name": "disconnected_pore"},
            {"id": 1, "name": "connected_pore"},
        ],
        # This polygon deliberately conflicts with the lossless mask. It must
        # be ignored whenever mask_dir is explicit.
        "annotations": [
            {
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]],
            }
        ],
    }
    dataset = PatchDataset(
        coco_data,
        str(image_dir),
        mask_dir=str(mask_dir),
        patch_size=3,
        training=False,
        num_classes=3,
    )

    _, mask, _ = dataset[0]

    assert mask[0, 0].item() == 0
    assert mask[0, 1].item() == 1
    assert mask[1, 1].item() == 2
    assert set(mask.unique().tolist()) == {0, 1, 2}
    assert dataset.target_source == "lossless_png_masks"
    assert dataset.target_provenance["mask_count"] == 1
    assert len(dataset.target_provenance["mask_aggregate_sha256"]) == 64


def test_lossless_mask_validation_rejects_missing_shape_and_values(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    image = np.zeros((6, 6), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "source.png"), image)
    coco_data = {
        "images": [
            {"id": 1, "file_name": "source.png", "width": 6, "height": 6}
        ],
        "categories": [],
        "annotations": [],
    }

    with pytest.raises(FileNotFoundError, match="lossless mask does not exist"):
        validate_lossless_mask_directory(coco_data, image_dir, mask_dir)

    assert cv2.imwrite(
        str(mask_dir / "source.png"), np.zeros((5, 6), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_lossless_mask_directory(coco_data, image_dir, mask_dir)

    invalid_mask = np.zeros((6, 6), dtype=np.uint8)
    invalid_mask[0, 0] = 2
    assert cv2.imwrite(str(mask_dir / "source.png"), invalid_mask)
    with pytest.raises(ValueError, match=r"invalid values \[2\]"):
        validate_lossless_mask_directory(coco_data, image_dir, mask_dir)


def test_lossless_mask_aggregate_sha_is_order_independent_and_content_sensitive(
    tmp_path,
):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    images = []
    for image_id, file_name in [(1, "b.png"), (2, "a.png")]:
        image = np.full((6, 6), image_id, dtype=np.uint8)
        mask = np.full((6, 6), 255, dtype=np.uint8)
        mask[0, 0] = image_id - 1
        assert cv2.imwrite(str(image_dir / file_name), image)
        assert cv2.imwrite(str(mask_dir / file_name), mask)
        images.append(
            {"id": image_id, "file_name": file_name, "width": 6, "height": 6}
        )
    coco_data = {"images": images, "categories": [], "annotations": []}

    _, first = validate_lossless_mask_directory(coco_data, image_dir, mask_dir)
    reversed_data = {**coco_data, "images": list(reversed(images))}
    _, second = validate_lossless_mask_directory(reversed_data, image_dir, mask_dir)
    assert first["mask_aggregate_sha256"] == second["mask_aggregate_sha256"]

    changed_mask = np.full((6, 6), 255, dtype=np.uint8)
    changed_mask[0, 0] = 0
    assert cv2.imwrite(str(mask_dir / "a.png"), changed_mask)
    _, changed = validate_lossless_mask_directory(coco_data, image_dir, mask_dir)
    assert changed["mask_aggregate_sha256"] != first["mask_aggregate_sha256"]


def test_data_loader_threads_one_validated_mask_provenance_across_splits(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    coco_data = _coco_data(image_count=3)
    for image in coco_data["images"]:
        file_name = image["file_name"]
        assert cv2.imwrite(
            str(image_dir / file_name), np.zeros((6, 6), dtype=np.uint8)
        )
        assert cv2.imwrite(
            str(mask_dir / file_name), np.full((6, 6), 255, dtype=np.uint8)
        )

    train_loader, val_loader, test_loader = create_patch_data_loaders(
        coco_data,
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        batch_size=1,
        patch_size=2,
        split_manifest={"train": [1], "val": [2], "test": [3]},
        augmentation_config={
            "patch_size": 2,
            "augmentation": {"enabled": False, "strength": "none"},
        },
        num_classes=3,
        num_workers=0,
        pin_memory=False,
        return_test=True,
    )

    provenances = [
        loader.dataset.target_provenance
        for loader in (train_loader, val_loader, test_loader)
    ]
    assert {item["target_source"] for item in provenances} == {
        "lossless_png_masks"
    }
    assert {item["mask_count"] for item in provenances} == {3}
    assert len({item["mask_aggregate_sha256"] for item in provenances}) == 1
    _, target, _ = next(iter(test_loader))
    assert set(target.unique().tolist()) == {2}


def test_training_class_statistics_use_only_authoritative_training_masks(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    coco_data = _coco_data(image_count=3)
    masks = {
        1: np.asarray(
            [
                [0, 0, 1, 1, 255, 255],
                [0, 1, 1, 255, 255, 255],
                [0, 1, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255],
            ],
            dtype=np.uint8,
        ),
        # Validation and held-out masks intentionally have very different
        # distributions. They must not affect the returned training constants.
        2: np.zeros((6, 6), dtype=np.uint8),
        3: np.ones((6, 6), dtype=np.uint8),
    }
    for image in coco_data["images"]:
        image_id = image["id"]
        assert cv2.imwrite(
            str(image_dir / image["file_name"]), np.zeros((6, 6), dtype=np.uint8)
        )
        assert cv2.imwrite(str(mask_dir / image["file_name"]), masks[image_id])

    train_loader, val_loader, _ = create_patch_data_loaders(
        coco_data,
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        batch_size=1,
        patch_size=2,
        split_manifest={"train": [1], "val": [2], "test": [3]},
        augmentation_config={
            "patch_size": 2,
            "augmentation": {"enabled": False, "strength": "none"},
        },
        num_classes=3,
        num_workers=0,
        pin_memory=False,
        return_test=True,
    )

    statistics = train_loader.dataset.training_class_statistics()

    assert statistics["counts"] == [4, 5, 27]
    assert statistics["total_pixels"] == 36
    assert statistics["image_ids"] == [1]
    assert statistics["source"] == "authoritative_training_masks_only"
    assert len(statistics["image_ids_sha256"]) == 64
    expected_mask_digest = hashlib.sha256()
    expected_mask_digest.update(b"source_1.png\0")
    expected_mask_digest.update((mask_dir / "source_1.png").read_bytes())
    expected_mask_digest.update(b"\0")
    assert statistics["training_mask_count"] == 1
    assert statistics["training_mask_aggregate_sha256"] == (
        expected_mask_digest.hexdigest()
    )
    assert "mask_aggregate_sha256" not in statistics
    with pytest.raises(RuntimeError, match="training dataset"):
        val_loader.dataset.training_class_statistics()

    conditional_train_loader, _, _ = create_patch_data_loaders(
        coco_data,
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        batch_size=1,
        patch_size=2,
        split_manifest={"train": [1], "val": [2], "test": [3]},
        augmentation_config={
            "patch_size": 2,
            "augmentation": {"enabled": False, "strength": "none"},
        },
        num_classes=2,
        num_workers=0,
        pin_memory=False,
        return_test=True,
    )
    conditional_statistics = (
        conditional_train_loader.dataset.training_class_statistics()
    )
    assert conditional_statistics["counts"] == [4, 5, 27]
    assert len(conditional_statistics["counts"]) == 3
    assert conditional_statistics["training_mask_aggregate_sha256"] == (
        expected_mask_digest.hexdigest()
    )


def test_validation_only_loaders_do_not_validate_or_construct_held_out_masks(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    coco_data = _coco_data(image_count=3)
    for image in coco_data["images"]:
        assert cv2.imwrite(
            str(image_dir / image["file_name"]), np.zeros((6, 6), dtype=np.uint8)
        )
    # Only train and validation masks exist. A validation-only screen must not
    # even require the held-out mask path to exist.
    for image_id in (1, 2):
        mask = np.full((6, 6), 255, dtype=np.uint8)
        mask[0, 0] = 0
        mask[0, 1] = 1
        assert cv2.imwrite(str(mask_dir / f"source_{image_id}.png"), mask)

    loaders = create_patch_data_loaders(
        coco_data,
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        batch_size=1,
        patch_size=2,
        split_manifest={"train": [1], "val": [2], "test": [3]},
        augmentation_config={
            "patch_size": 2,
            "augmentation": {"enabled": False, "strength": "none"},
        },
        num_classes=3,
        num_workers=0,
        pin_memory=False,
        return_test=False,
    )

    assert len(loaders) == 2
    train_loader, val_loader = loaders
    assert train_loader.dataset.split_name == "train"
    assert val_loader.dataset.split_name == "val"
    assert train_loader.dataset.target_provenance["mask_count"] == 2
    with pytest.raises(FileNotFoundError, match="lossless mask does not exist"):
        create_patch_data_loaders(
            coco_data,
            image_dir=str(image_dir),
            mask_dir=str(mask_dir),
            batch_size=1,
            patch_size=2,
            split_manifest={"train": [1], "val": [2], "test": [3]},
            augmentation_config={
                "patch_size": 2,
                "augmentation": {"enabled": False, "strength": "none"},
            },
            num_classes=3,
            num_workers=0,
            pin_memory=False,
            return_test=True,
        )


def test_albumentations_208_multiworker_streams_are_diverse_and_repeatable(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("DISABLE_TRANSFORMS", raising=False)
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    # An asymmetric tile makes each rotate/flip outcome directly observable.
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[:2, :3] = 0
    mask[5:, 6:] = 1
    assert cv2.imwrite(str(image_dir / "source.png"), image)
    assert cv2.imwrite(str(mask_dir / "source.png"), mask)
    coco = {
        "images": [
            {"id": 1, "file_name": "source.png", "width": 8, "height": 8}
        ],
        "categories": [],
        "annotations": [],
    }

    def collect():
        dataset = PatchDataset(
            coco,
            str(image_dir),
            mask_dir=str(mask_dir),
            patch_size=8,
            training=True,
            split_name="train",
            bootstrap_factor=24,
            num_classes=3,
            augmentation_config={
                "patch_size": 8,
                "seed": 91,
                "augmentation": {"enabled": True, "strength": "light"},
            },
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            worker_init_fn=seed_patch_dataloader_worker,
            generator=torch.Generator().manual_seed(12345),
        )
        return [batch[0].clone() for batch in loader]

    first = collect()
    second = collect()
    assert len(first) == len(second) == 24
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    unique_outputs = {tensor.numpy().tobytes() for tensor in first}
    assert len(unique_outputs) >= 2
    # With batch_size=1 and shuffle disabled, the two workers receive the
    # repeated source tile in alternating order. Albumentations 2.0.8's old
    # inherited-RNG bug would make these worker subsequences identical.
    assert any(
        not torch.equal(worker_zero, worker_one)
        for worker_zero, worker_one in zip(first[0::2], first[1::2])
    )

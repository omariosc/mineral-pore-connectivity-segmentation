from types import SimpleNamespace
import warnings

import pytest

pytest.importorskip("torch")
A = pytest.importorskip("albumentations")

from src.training.augmentations import GeologicalAugmentations
from scripts.train_patches import resolve_augmentation_settings


def _transform_summary(augmentor):
    return [
        (item["name"], item["probability"])
        for item in augmentor.resolved_config()["transforms"]
    ]


def test_light_policy_contains_only_connectivity_preserving_symmetries():
    augmentor = GeologicalAugmentations(
        training=True,
        augmentation_strength="light",
        seed=123,
    )
    assert _transform_summary(augmentor) == [
        ("RandomRotate90", 0.5),
        ("HorizontalFlip", 0.5),
        ("VerticalFlip", 0.5),
    ]
    resolved = augmentor.resolved_config()
    assert resolved["seed"] == 123
    assert resolved["albumentations_version"] == A.__version__
    assert augmentor.transform.seed == 123


def test_confirmatory_conservative_cli_name_resolves_to_light(monkeypatch):
    monkeypatch.delenv("DISABLE_TRANSFORMS", raising=False)
    settings = resolve_augmentation_settings(
        SimpleNamespace(
            disable_transforms=False,
            augmentation="conservative",
            use_mixup=False,
            use_cutmix=False,
        )
    )
    assert settings == {
        "enabled": True,
        "strength": "light",
        "use_mixup": False,
        "use_cutmix": False,
    }


def test_morphology_warps_start_at_medium_strength():
    light_names = {
        name
        for name, _probability in _transform_summary(
            GeologicalAugmentations(augmentation_strength="light", seed=42)
        )
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        medium = GeologicalAugmentations(
            augmentation_strength="medium", seed=42
        )
    medium_names = {name for name, _probability in _transform_summary(medium)}
    assert "ElasticTransform" not in light_names
    assert "GridDistortion" not in light_names
    assert {"ElasticTransform", "GridDistortion"} <= medium_names


def test_same_seed_produces_the_same_transform_sequence():
    import numpy as np

    first = GeologicalAugmentations(augmentation_strength="light", seed=2025)
    second = GeologicalAugmentations(augmentation_strength="light", seed=2025)
    image = np.arange(9 * 7, dtype=np.uint8).reshape(9, 7)
    mask = (image % 3).astype(np.uint8)

    for _ in range(12):
        first_image, first_mask = first(image.copy(), mask.copy())
        second_image, second_mask = second(image.copy(), mask.copy())
        assert first_image.equal(second_image)
        assert first_mask.equal(second_mask)


def test_unknown_strength_fails_closed():
    with pytest.raises(ValueError, match="Unsupported augmentation strength"):
        GeologicalAugmentations(augmentation_strength="conservativish", seed=42)

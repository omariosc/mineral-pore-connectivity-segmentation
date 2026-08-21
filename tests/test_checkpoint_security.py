"""Regression tests for fail-closed PyTorch checkpoint loading."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.models.dinov3_unet import DINOv3UNet
from src.training.checkpoint_io import (
    load_weights_only_checkpoint,
    normalize_checkpoint_metadata,
    tensor_state_dict_semantic_sha256,
)
from src.training.two_stage_trainer import TwoStageTrainer


class _EvalMarkerPayload:
    """Payload that would create a file under unrestricted pickle loading."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        expression = (
            "__import__('pathlib').Path(" + repr(str(self.marker)) + ").touch()"
        )
        return eval, (expression,)


class _OpenMarkerPayload:
    """Alternate payload using a direct side-effecting built-in."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return open, (str(self.marker), "w")


@pytest.mark.parametrize("payload_type", [_EvalMarkerPayload, _OpenMarkerPayload])
def test_weights_only_loader_rejects_pickle_side_effects(tmp_path, payload_type):
    marker = tmp_path / "pickle-side-effect"
    checkpoint_path = tmp_path / "malicious.pth"
    torch.save(
        {"model_state_dict": {}, "payload": payload_type(marker)},
        checkpoint_path,
    )

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_weights_only_checkpoint(checkpoint_path, map_location="cpu")

    assert not marker.exists()


def test_weights_only_round_trip_preserves_model_optimizer_and_metadata(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    metadata = normalize_checkpoint_metadata(
        {
            "epoch": 4,
            "class_iou": np.asarray([0.25, 0.75], dtype=np.float64),
            "selection_score": np.float64(0.5),
            "checkpoint_path": Path("checkpoints/best_model.pth"),
            "nested": {"seed": np.int64(42), "shape": (2, 1)},
        }
    )
    checkpoint = {
        **metadata,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint_path = tmp_path / "legitimate.pth"
    torch.save(checkpoint, checkpoint_path)

    loaded = load_weights_only_checkpoint(checkpoint_path, map_location="cpu")
    assert loaded["epoch"] == 4
    assert loaded["class_iou"] == [0.25, 0.75]
    assert loaded["selection_score"] == 0.5
    assert loaded["checkpoint_path"] == "checkpoints/best_model.pth"
    assert loaded["nested"] == {"seed": 42, "shape": (2, 1)}

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
    restored_model.load_state_dict(loaded["model_state_dict"])
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)
    assert restored_optimizer.state_dict()["state"]


def test_tensor_state_semantic_hash_ignores_mapping_order_and_serialization(tmp_path):
    first = {
        "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "counter": torch.tensor(7, dtype=torch.int64),
    }
    reordered = {
        "counter": first["counter"].clone(),
        "weight": first["weight"].clone(),
    }
    expected = tensor_state_dict_semantic_sha256(first)
    assert tensor_state_dict_semantic_sha256(reordered) == expected

    first_path = tmp_path / "first.pth"
    second_path = tmp_path / "second.pth"
    torch.save({"model_state_dict": first, "note": "first"}, first_path)
    torch.save({"note": "different transport metadata", "model_state_dict": reordered}, second_path)
    assert first_path.read_bytes() != second_path.read_bytes()
    assert (
        tensor_state_dict_semantic_sha256(
            load_weights_only_checkpoint(first_path)["model_state_dict"]
        )
        == tensor_state_dict_semantic_sha256(
            load_weights_only_checkpoint(second_path)["model_state_dict"]
        )
        == expected
    )

    changed = {name: value.clone() for name, value in first.items()}
    changed["weight"][0, 0] += 1
    assert tensor_state_dict_semantic_sha256(changed) != expected

    with pytest.raises(TypeError, match="not a tensor"):
        tensor_state_dict_semantic_sha256({"weight": [1.0, 2.0]})


def test_two_stage_checkpoint_round_trip_normalizes_numpy_metrics(tmp_path):
    model = torch.nn.Linear(2, 1)
    trainer = TwoStageTrainer(model, device="cpu", save_dir=tmp_path)
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    trainer._save_checkpoint(
        "two-stage.pt",
        epoch=3,
        metrics={"class_iou": np.asarray([0.2, 0.8]), "score": np.float64(0.4)},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    metrics = trainer._load_checkpoint(tmp_path / "two-stage.pt")

    assert metrics == {"class_iou": [0.2, 0.8], "score": 0.4}
    for name, value in model.state_dict().items():
        assert torch.equal(value, original_state[name])


def test_dinov3_checkpoint_cannot_supply_adjacent_architecture_code(tmp_path):
    marker = tmp_path / "checkpoint-module-imported"
    (tmp_path / "vision_transformer.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "raise RuntimeError('checkpoint-local code executed')\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="No DINOv3 checkpoint"):
        reference = DINOv3UNet(
            pretrained_path=None,
            patch_size=2,
            embed_dim=8,
            depth=2,
            num_heads=2,
        )
    checkpoint_path = tmp_path / "dinov3.pth"
    torch.save({"state_dict": reference.backbone.state_dict()}, checkpoint_path)

    original_sys_path = list(sys.path)
    restored = DINOv3UNet(
        pretrained_path=str(checkpoint_path),
        patch_size=2,
        embed_dim=8,
        depth=2,
        num_heads=2,
    )

    assert sys.path == original_sys_path
    assert not marker.exists()
    for name, value in reference.backbone.state_dict().items():
        assert torch.equal(value, restored.backbone.state_dict()[name])

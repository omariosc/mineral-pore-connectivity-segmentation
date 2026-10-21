"""Selection and data-boundary tests for separately registered revision training."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.train_revision_controls import (
    build_inputs,
    selection_key,
    project_development_metadata,
)


def test_grayscale_control_does_not_receive_gate_but_two_channel_arm_does():
    raw = torch.tensor([[[[-1.0, -0.2, 1.0]]]])
    single = build_inputs(raw, 1, 100)
    double = build_inputs(raw, 2, 100)
    assert torch.equal(single, raw)
    assert double.shape == (1, 2, 1, 3)
    assert double[:, 1].tolist() == [[[1.0, 0.0, 0.0]]]


def test_selection_uses_both_pore_classes_then_union_then_loss():
    a = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    b = np.array([[0, 5, 0], [0, 5, 0], [0, 0, 5]])
    assert selection_key(a, 0.4) > selection_key(b, 0.1)
    assert selection_key(a, 0.2) > selection_key(a, 0.3)


def test_metadata_projection_excludes_test_entries_and_rejects_overlap():
    splits = {
        "train": {"image_ids": [1], "image_files": ["a.png"]},
        "val": {"image_ids": [2], "image_files": ["b.png"]},
        "test": {"image_ids": [3], "image_files": ["secret.png"]},
    }
    result = project_development_metadata(splits)
    assert [x["file_name"] for x in result["images"]] == ["a.png", "b.png"]
    assert result["annotations"] == []
    splits["val"]["image_files"] = ["a.png"]
    with pytest.raises(ValueError):
        project_development_metadata(splits)


def test_registered_training_rejects_disabled_augmentation_environment():
    from scripts.train_revision_controls import validate_training_environment

    with pytest.raises(ValueError):
        validate_training_environment({"DISABLE_TRANSFORMS": "1"})
    validate_training_environment({})
    validate_training_environment({"DISABLE_TRANSFORMS": "0"})


def test_smoke_rejects_nonfinite_gradients_and_zero_updates():
    from scripts.train_revision_controls import qualify_smoke

    parameter = torch.nn.Parameter(torch.ones(1))
    (parameter * float("inf")).sum().backward()
    norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=False)
    with pytest.raises(FloatingPointError):
        qualify_smoke([float(norm), 1.0], 2)
    with pytest.raises(FloatingPointError):
        qualify_smoke([1.0, 1.0], 0)
    qualify_smoke([1.0, 2.0], 2)

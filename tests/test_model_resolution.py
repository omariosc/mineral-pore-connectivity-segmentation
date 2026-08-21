import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.models.multiscale_attention_unet import (
    MultiScaleAttentionUNet,
    MultiScaleAttentionUNetPyramid,
    create_multiscale_attention_unet,
    create_multiscale_attention_unet_pyramid,
)
from src.models.focal_loss import CombinedFocalDiceLoss
from src.models.unet_model import AttentionUNet, UNet, create_model
from src.training.model_factory import create_advanced_model
from src.training.patch_trainer import PatchTrainer


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def test_dinov2_fails_closed_without_calling_torch_hub(monkeypatch):
    called = False

    def forbidden_hub_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.hub.load must not be reached")

    monkeypatch.setattr(torch.hub, "load", forbidden_hub_load)

    with pytest.raises(RuntimeError, match="DINOv2 is disabled.*torch.hub"):
        create_advanced_model("dinov2_small", num_classes=3)

    assert called is False


def test_plain_unet_aliases_ignore_attention_enabled_in_pipeline_config():
    """The comparator name must identify structure, not mutable config flags."""
    for model_type in ("unet", "plain_unet"):
        model = create_model(model_type, num_classes=3)
        assert type(model) is UNet
        assert model.architecture_name == "plain_unet"
        assert model.requested_model_type == model_type
        assert not any(name.startswith("att") for name, _ in model.named_modules())
        assert not any(name.startswith("ds") for name, _ in model.named_modules())


def test_explicit_unet_variants_have_distinct_structure_and_parameter_counts():
    plain = create_model("plain_unet", num_classes=3)
    attention = create_model("attention_unet", num_classes=3)
    multiscale = MultiScaleAttentionUNet(
        n_channels=1,
        n_classes=3,
        bilinear=True,
        base_features=32,
        deep_supervision=False,
    )

    assert type(plain) is UNet
    assert type(attention) is AttentionUNet
    assert type(multiscale) is MultiScaleAttentionUNet
    assert plain.architecture_name == "plain_unet"
    assert attention.architecture_name == "attention_unet"
    assert multiscale.architecture_name == "multiscale_attention_unet"
    counts = {
        _parameter_count(plain),
        _parameter_count(attention),
        _parameter_count(multiscale),
    }
    assert len(counts) == 3


def test_legacy_configured_alias_is_explicit_and_warns():
    with pytest.warns(DeprecationWarning, match="configuration-dependent"):
        model = create_model("legacy_configured_unet", num_classes=3)
    assert type(model) is AttentionUNet
    assert model.architecture_name == "attention_unet"


def test_unknown_unet_variant_fails_closed():
    with pytest.raises(ValueError, match="Unsupported U-Net model type"):
        create_model("unett", num_classes=3)


def test_trainer_records_requested_and_resolved_architecture_and_parameter_count():
    trainer = PatchTrainer.__new__(PatchTrainer)
    trainer.multi_gpu = False
    trainer.model_type = "plain_unet"
    trainer.model = create_model("plain_unet", num_classes=3)
    trainer.num_classes = 3
    trainer.dropout = 0.2
    trainer.freeze_encoder = False

    resolved = trainer._resolved_model_config()
    assert resolved["architecture"] == "plain_unet"
    assert resolved["architecture_requested"] == "plain_unet"
    assert resolved["architecture_resolved"] == "plain_unet"
    assert resolved["implementation_class"] == "UNet"
    assert resolved["parameter_count"] == _parameter_count(trainer.model)
    assert resolved["parameter_count"] > 0


def test_conditional_models_resolve_two_explicit_input_channels():
    plain = create_model("plain_unet", num_classes=2, in_channels=2)
    multiscale = create_multiscale_attention_unet(
        {
            "model": {
                "in_channels": 2,
                "num_classes": 2,
                "base_features": 32,
                "bilinear": True,
                "deep_supervision": False,
            }
        }
    )
    pyramid = create_multiscale_attention_unet_pyramid(
        {
            "model": {
                "in_channels": 2,
                "num_classes": 2,
                "base_features": 32,
                "bilinear": True,
                "deep_supervision": False,
            }
        }
    )
    assert plain.n_channels == 2
    assert plain.n_classes == 2
    assert multiscale.n_channels == 2
    assert multiscale.n_classes == 2
    assert pyramid.n_channels == 2
    assert pyramid.n_classes == 2
    assert pyramid.architecture_name == "multiscale_attention_unet_pyramid"
    assert pyramid.resolved_pyramid_context_config()["dropout"][
        "probability"
    ] == 0.0

    trainer = PatchTrainer.__new__(PatchTrainer)
    trainer.multi_gpu = False
    trainer.model_type = "plain_unet"
    trainer.loss_type = "conditional_pore_focal_dice"
    trainer.model = plain
    trainer.num_classes = 2
    trainer.dropout = 0.2
    trainer.freeze_encoder = False
    resolved = trainer._resolved_model_config()
    assert resolved["input_channels"] == 2
    assert resolved["input_channel_semantics"] == [
        "normalized_grayscale",
        "binary_recovered_pore_gate",
    ]


def test_pyramid_candidate_preserves_reference_path_bitwise_when_context_is_bypassed():
    torch.manual_seed(101)
    reference = MultiScaleAttentionUNet(
        n_channels=2,
        n_classes=2,
        bilinear=True,
        base_features=32,
        deep_supervision=False,
    ).eval()
    torch.manual_seed(101)
    candidate = MultiScaleAttentionUNetPyramid(
        n_channels=2,
        n_classes=2,
        bilinear=True,
        base_features=32,
        deep_supervision=False,
    ).eval()

    candidate_state = candidate.state_dict()
    for name, value in reference.state_dict().items():
        assert torch.equal(value, candidate_state[name]), name

    candidate.pyramid_context = torch.nn.Identity()
    inputs = torch.randn(1, 2, 32, 32)
    with torch.no_grad():
        reference_output = reference(inputs)
        candidate_output = candidate(inputs)
    assert torch.equal(reference_output, candidate_output)


def test_pyramid_candidate_two_channel_output_and_context_gradients():
    torch.manual_seed(202)
    candidate = MultiScaleAttentionUNetPyramid(
        n_channels=2,
        n_classes=2,
        bilinear=True,
        base_features=32,
        deep_supervision=False,
    ).train()
    inputs = torch.randn(1, 2, 32, 32, requires_grad=True)

    output = candidate(inputs)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (1, 2, 32, 32)
    assert torch.isfinite(output).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    context_gradient = candidate.pyramid_context.fuse[0].weight.grad
    assert context_gradient is not None
    assert torch.isfinite(context_gradient).all()
    assert torch.count_nonzero(context_gradient) > 0
    assert candidate.pyramid_context.dropout_probability == 0.0
    assert _parameter_count(candidate) == (
        _parameter_count(
            MultiScaleAttentionUNet(
                n_channels=2,
                n_classes=2,
                bilinear=True,
                base_features=32,
                deep_supervision=False,
            )
        )
        + 131_840
    )


def test_focal_dice_records_actual_gamma_weights_and_smoothing():
    criterion = CombinedFocalDiceLoss(
        class_weights=[3.0, 2.0, 1.0],
        focal_weight=0.4,
        dice_weight=0.6,
        gamma=3.25,
        dice_smooth=1e-5,
    )
    resolved = criterion.resolved_config()
    assert resolved["focal_gamma_actual"] == pytest.approx(3.25)
    assert resolved["component_weights_actual"] == {
        "focal": pytest.approx(0.4),
        "dice": pytest.approx(0.6),
    }
    assert resolved["dice_smooth_actual"] == pytest.approx(1e-5)
    assert resolved["class_weights_actual"] == pytest.approx([3.0, 2.0, 1.0])


def test_trainer_synchronizes_requested_loss_type_into_factory_config(monkeypatch):
    class StubConfig:
        def __init__(self):
            self.values = {"model.loss.type": "stale_yaml_value"}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def update(self, key, value):
            self.values[key] = value

    config = StubConfig()
    monkeypatch.setattr(
        "src.training.patch_trainer.load_config", lambda _path: config
    )
    monkeypatch.setattr(
        PatchTrainer, "_setup_device", lambda _self: torch.device("cpu")
    )
    monkeypatch.setattr(PatchTrainer, "setup_directories", lambda _self: None)

    PatchTrainer(
        use_wandb=False,
        model_type="plain_unet",
        loss_type="focal_dice",
    )

    assert config.values["model.loss.type"] == "focal_dice"


def test_reference_focal_dice_keeps_extreme_fp16_rare_gradient_finite():
    criterion = CombinedFocalDiceLoss(
        class_weights=[3.0, 2.0, 1.0],
        focal_weight=0.5,
        dice_weight=0.5,
        gamma=2.0,
    )
    logits = torch.tensor(
        [
            [
                [[-30.0, 30.0, 0.0]],
                [[30.0, -30.0, 0.0]],
                [[0.0, 0.0, 30.0]],
            ]
        ],
        dtype=torch.float16,
        requires_grad=True,
    )
    targets = torch.tensor([[[0, 1, 2]]])

    loss = criterion(logits, targets)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0, 0, 0, 0]) != 0.0
    assert float(logits.grad[0, 1, 0, 0]) != 0.0


def test_multiscale_deep_supervision_fails_before_optimizer_setup(monkeypatch):
    trainer = PatchTrainer.__new__(PatchTrainer)
    trainer.multi_gpu = False
    trainer.device = torch.device("cpu")
    trainer.model_type = "multiscale_attention_unet"
    trainer.config = {
        "model": {
            "in_channels": 1,
            "num_classes": 3,
            "base_features": 32,
            "bilinear": True,
            "deep_supervision": True,
        }
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="deep_supervision must be false"):
        trainer.create_model_and_optimizer()


def test_source_hash_snapshot_uses_relative_keys_and_null_for_missing_optional_file(
    tmp_path, monkeypatch
):
    train_script = tmp_path / "scripts" / "train_patches.py"
    model_source = tmp_path / "src" / "models" / "unet_model.py"
    train_script.parent.mkdir(parents=True)
    model_source.parent.mkdir(parents=True)
    train_script.write_bytes(b"training-entrypoint\n")
    model_source.write_bytes(b"plain-unet-source\n")

    trainer = PatchTrainer.__new__(PatchTrainer)
    trainer.multi_gpu = False
    trainer.model_type = "plain_unet"
    trainer.model = create_model("plain_unet", num_classes=3)
    monkeypatch.setattr(
        PatchTrainer,
        "_repository_root",
        staticmethod(lambda: Path(tmp_path)),
    )

    source_hashes = trainer._source_code_sha256()
    assert source_hashes["scripts/train_patches.py"] == hashlib.sha256(
        train_script.read_bytes()
    ).hexdigest()
    assert source_hashes["src/models/unet_model.py"] == hashlib.sha256(
        model_source.read_bytes()
    ).hexdigest()
    assert source_hashes["scripts/aire_confirmatory.slurm"] is None
    assert all(not key.startswith("/") for key in source_hashes)
    assert str(tmp_path) not in repr(source_hashes)

    # The snapshot is intentionally frozen even if a file changes mid-run.
    train_script.write_bytes(b"changed-after-run-start\n")
    assert trainer._source_code_sha256() == source_hashes

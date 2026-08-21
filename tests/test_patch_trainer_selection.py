import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.conditional_pore_loss import ConditionalPoreFocalDiceLoss
from src.models.hierarchical_pore_loss import HierarchicalPoreConnectivityLoss
from src.training.patch_trainer import (
    EXECUTION_SOURCE_FILES,
    PatchTrainer,
    SELECTION_METRIC_NAME,
)


def _bare_trainer(tmp_path, *, no_checkpoints):
    trainer = PatchTrainer.__new__(PatchTrainer)
    trainer.num_classes = 3
    trainer.model_type = "test_linear"
    trainer.loss_type = "focal_dice"
    trainer.class_weights = None
    trainer.dropout = 0.0
    trainer.freeze_encoder = False
    trainer.multi_gpu = False
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = None
    trainer.scaler = None
    trainer.config = {}
    trainer.no_checkpoints = no_checkpoints
    trainer.save_every_n_epochs = 10
    trainer.run_name = "selection_test"
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.metrics_dir = tmp_path / "metrics"
    trainer.metrics_dir.mkdir(parents=True)
    trainer.train_losses = []
    trainer.val_losses = []
    trainer.train_metrics = []
    trainer.val_metrics = []
    trainer.all_metrics = {}
    trainer.best_val_loss = float("inf")
    trainer.best_val_metric = float("-inf")
    trainer.best_selection_epoch = None
    trainer.best_selection_components = None
    trainer._best_selection_state_dict = None
    trainer.selection_checkpoint_path = None
    trainer.selection_checkpoint_sha256 = None
    trainer.selection_restore_source = None
    trainer.test_evaluation_count = 0
    trainer.test_metrics = None
    trainer.validation_only = False
    trainer.max_batches = None
    trainer.seed = 42
    trainer.evaluation_patch_size = 2048
    trainer.evaluation_batch_size = 1
    trainer.target_provenance = {
        "target_source": "lossless_png_masks",
        "mask_directory": "masks",
        "mask_count": 3,
        "mask_aggregate_sha256": "a" * 64,
    }
    trainer.input_provenance = {
        "input_source": "indexed_source_images",
        "scope": "development_train_plus_validation",
        "split_names": ["train", "val"],
        "image_count": 2,
        "image_aggregate_sha256": "b" * 64,
        "training_subset": {
            "scope": "training_only",
            "split_names": ["train"],
            "image_count": 1,
            "image_aggregate_sha256": "c" * 64,
        },
        "held_out_bytes_read": 0,
    }
    trainer.protocol_run_role = None
    trainer.protocol_candidate_key = None
    trainer.protocol_campaign_id = None
    trainer.protocol_cell_index = None
    trainer.selected_architecture_role = None
    trainer.selected_method_lock = None
    trainer.smoke_preflight_manifest = None
    return trainer


def test_train_restores_selected_epoch_before_single_test_evaluation(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    trainer.output_dir = tmp_path / "run"
    trainer.output_dir.mkdir()
    trainer.plots_dir = trainer.output_dir / "plots"
    trainer.predictions_dir = trainer.plots_dir
    trainer.visualizations_dir = trainer.plots_dir
    trainer.config_dir = trainer.output_dir / "config"
    trainer.epochs = 3
    trainer.start_epoch = 0
    trainer.epoch_times = []
    trainer.batch_size = 1
    trainer.patch_size = 1
    trainer.checkpoint_interval = 1
    trainer.mixed_precision = False
    trainer.early_stopping_enabled = False
    trainer.patience = 2
    trainer.patience_counter = 0
    trainer.scheduler_step_per_batch = False
    trainer.use_wandb = False
    trainer.wandb_run = None
    trainer.use_mixup = False
    trainer.use_cutmix = False
    trainer.augmentation_config = {
        "augmentation": {"enabled": False, "strength": "none"}
    }
    trainer.split_ids = {"test": [9]}
    trainer.split_files = {"test": ["held_out.png"]}
    trainer.category_id_map = {}
    trainer.all_metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "mean_iou": [],
        "disconnected_pore_iou": [],
        "connected_pore_iou": [],
        "mineral_iou": [],
        "learning_rate": [],
        "epoch_time": [],
    }

    dataset = SimpleNamespace(category_id_map={}, augmentor=None)

    class Loader:
        sampler = SimpleNamespace()

        def __init__(self):
            self.dataset = dataset

        def __len__(self):
            return 1

    train_loader, val_loader, test_loader = Loader(), Loader(), Loader()
    trainer.create_data_loaders = lambda: (train_loader, val_loader, test_loader)
    trainer.create_model_and_optimizer = lambda: None
    trainer._save_config_info = lambda: None
    trainer.save_metrics = lambda epoch: None
    trainer.plot_training_curves = lambda epoch: None
    trainer.plot_confusion_matrix = lambda cm, epoch: None
    trainer.plot_roc_curve = lambda loader, epoch: None
    trainer.test_full_image_prediction = lambda: None
    trainer._generate_training_summary = lambda: None

    def train_epoch(_loader, epoch):
        with torch.no_grad():
            trainer.model.weight.fill_(epoch + 1)
        return 1.0 / (epoch + 1), 50.0, 0.01

    validation_scores = [0.2, 0.8, 0.3]
    phases = []
    test_weights = []

    def validate(_loader, epoch, phase="Val"):
        phases.append(phase)
        if phase == "Test":
            test_weights.append(float(trainer.model.weight.item()))
            score = 0.5
        else:
            score = validation_scores[epoch]
        class_iou = np.asarray([score, score, 0.9])
        additional = {
            "weighted_f1": score,
            "confusion_matrix": np.eye(3, dtype=int) * 100,
        }
        return 1.0 - score, 50.0, float(class_iou.mean()), class_iou, additional

    trainer.train_epoch = train_epoch
    trainer.validate = validate
    trainer.train()

    assert phases == ["Val", "Val", "Val", "Test"]
    assert test_weights == pytest.approx([2.0])
    assert trainer.best_selection_epoch == 2
    assert trainer.test_evaluation_count == 1
    assert trainer.test_metrics["evaluation_model"] == "validation_selected_state"
    assert trainer.test_metrics["selection_metric_name"] == SELECTION_METRIC_NAME
    assert trainer.test_metrics["selected_state_restore_source"] == (
        "captured_validation_selected_state_non_scientific_no_checkpoints"
    )


def test_validation_only_training_never_constructs_or_evaluates_test_loader(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    trainer.validation_only = True
    trainer.output_dir = tmp_path / "run"
    trainer.output_dir.mkdir()
    trainer.plots_dir = trainer.output_dir / "plots"
    trainer.predictions_dir = trainer.plots_dir
    trainer.visualizations_dir = trainer.plots_dir
    trainer.config_dir = trainer.output_dir / "config"
    trainer.epochs = 1
    trainer.start_epoch = 0
    trainer.epoch_times = []
    trainer.batch_size = 1
    trainer.patch_size = 1
    trainer.checkpoint_interval = 1
    trainer.mixed_precision = False
    trainer.early_stopping_enabled = False
    trainer.patience = 2
    trainer.patience_counter = 0
    trainer.scheduler_step_per_batch = False
    trainer.use_wandb = False
    trainer.wandb_run = None
    trainer.use_mixup = False
    trainer.use_cutmix = False
    trainer.augmentation_config = {
        "augmentation": {"enabled": False, "strength": "none"}
    }
    trainer.split_ids = {"test": [9]}
    trainer.split_files = {"test": ["held_out.png"]}
    trainer.category_id_map = {}
    trainer.all_metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "mean_iou": [],
        "disconnected_pore_iou": [],
        "connected_pore_iou": [],
        "mineral_iou": [],
        "learning_rate": [],
        "epoch_time": [],
    }

    dataset = SimpleNamespace(category_id_map={}, augmentor=None)

    class Loader:
        sampler = SimpleNamespace()

        def __init__(self):
            self.dataset = dataset

        def __len__(self):
            return 1

    train_loader, val_loader = Loader(), Loader()
    trainer.create_data_loaders = lambda: (train_loader, val_loader, None)
    trainer.create_model_and_optimizer = lambda: None
    trainer._save_config_info = lambda: None
    trainer.save_metrics = lambda epoch: None
    trainer.plot_training_curves = lambda epoch: None
    trainer.plot_confusion_matrix = lambda cm, epoch: None
    trainer.plot_roc_curve = lambda loader, epoch: None
    trainer.test_full_image_prediction = lambda: pytest.fail(
        "validation-only screen attempted full-image prediction"
    )
    trainer._generate_training_summary = lambda: None
    trainer.train_epoch = lambda loader, epoch: (0.5, 50.0, 0.01)
    phases = []

    def validate(_loader, _epoch, phase="Val"):
        phases.append(phase)
        class_iou = np.asarray([0.25, 0.5, 0.9])
        return (
            0.4,
            50.0,
            float(class_iou.mean()),
            class_iou,
            {"confusion_matrix": np.eye(3, dtype=int) * 10},
        )

    trainer.validate = validate
    trainer.train()

    assert phases == ["Val"]
    assert trainer.test_loader is None
    assert trainer.test_evaluation_count == 0
    assert trainer.test_metrics is None
    selection_record = json.loads(
        (trainer.metrics_dir / "model_selection.json").read_text(encoding="utf-8")
    )
    assert selection_record["evaluation_mode"] == "validation_only"
    assert selection_record["held_out_dataset_constructed"] is False
    assert selection_record["held_out_test_evaluated"] is False


def test_composite_checkpoint_is_distinct_from_lowest_loss_checkpoint(tmp_path):
    trainer = _bare_trainer(tmp_path, no_checkpoints=False)

    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    first = trainer._record_validation_selection(
        0,
        0.4,
        np.asarray([0.2, 0.2, 0.9]),
        {"confusion_matrix": np.eye(3, dtype=int) * 20},
    )
    trainer.save_checkpoint(0, 0.4, selection_improved=first["improved"])

    with torch.no_grad():
        trainer.model.weight.fill_(2.0)
    selected = trainer._record_validation_selection(
        1,
        0.5,
        np.asarray([0.8, 0.8, 0.9]),
        {"confusion_matrix": np.eye(3, dtype=int) * 80},
    )
    trainer.save_checkpoint(1, 0.5, selection_improved=selected["improved"])

    with torch.no_grad():
        trainer.model.weight.fill_(3.0)
    terminal = trainer._record_validation_selection(
        2,
        0.1,
        np.asarray([0.3, 0.3, 0.9]),
        {"confusion_matrix": np.eye(3, dtype=int) * 30},
    )
    trainer.save_checkpoint(2, 0.1, selection_improved=terminal["improved"])

    selected_checkpoint = trainer._torch_load_checkpoint(
        trainer.checkpoint_dir / "best_model.pth", torch.device("cpu")
    )
    lowest_loss_checkpoint = trainer._torch_load_checkpoint(
        trainer.checkpoint_dir / "best_val_loss_model.pth", torch.device("cpu")
    )
    assert selected_checkpoint["checkpoint_role"] == "validation_composite_selection"
    assert selected_checkpoint["best_selection_epoch"] == 2
    assert float(selected_checkpoint["model_state_dict"]["weight"].item()) == 2.0
    assert selected_checkpoint["input_normalization"]["output_range"] == [-1.0, 1.0]
    assert selected_checkpoint["target_provenance"]["target_source"] == (
        "lossless_png_masks"
    )
    assert selected_checkpoint["resolved_config"]["model"][
        "implementation_class"
    ] == "Linear"
    source_hashes = selected_checkpoint["source_code_sha256"]
    assert set(EXECUTION_SOURCE_FILES) <= set(source_hashes)
    assert source_hashes["src/training/patch_trainer.py"] == hashlib.sha256(
        (Path(__file__).parents[1] / "src/training/patch_trainer.py").read_bytes()
    ).hexdigest()
    assert selected_checkpoint["resolved_config"]["source_code_sha256"] == (
        source_hashes
    )
    augmentation = selected_checkpoint["resolved_config"]["augmentation"]
    assert augmentation["seed"] == 42
    assert augmentation["transforms"] == []
    assert augmentation["batch_level"]["application_probability"] == 0.0
    assert augmentation["data_loader"]["shuffle_generator_seed"] == 42
    assert augmentation["data_loader"]["persistent_workers"] is False
    assert all(not key.startswith("/") for key in source_hashes)
    assert lowest_loss_checkpoint["checkpoint_role"] == "lowest_validation_loss"
    assert float(lowest_loss_checkpoint["model_state_dict"]["weight"].item()) == 3.0

    trainer.restore_validation_selected_model()
    assert float(trainer.model.weight.item()) == 2.0
    assert trainer.selection_checkpoint_sha256


def test_validate_accumulates_confusion_counts_for_selection(tmp_path):
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    trainer.epochs = 1
    trainer.max_batches = None
    trainer.scaler = None

    logits = torch.full((1, 3, 2, 2), -5.0)
    logits[0, 0, 0, 0] = 5.0
    logits[0, 0, 0, 1] = 5.0
    logits[0, 2, 1, 0] = 5.0
    logits[0, 1, 1, 1] = 5.0

    class FixedModel(torch.nn.Module):
        def __init__(self, fixed_logits):
            super().__init__()
            self.register_buffer("fixed_logits", fixed_logits)

        def forward(self, images):
            return self.fixed_logits.expand(images.shape[0], -1, -1, -1)

    trainer.model = FixedModel(logits)
    trainer.criterion = torch.nn.CrossEntropyLoss()
    images = torch.zeros((1, 1, 2, 2))
    masks = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    loader = [(images, masks, ["patch"])]

    _, accuracy, mean_iou, class_iou, additional = trainer.validate(loader, 0)

    np.testing.assert_array_equal(
        additional["confusion_matrix"],
        np.asarray([[1, 0, 0], [1, 1, 0], [0, 0, 1]]),
    )
    assert accuracy == pytest.approx(75.0)
    np.testing.assert_allclose(class_iou, [0.5, 0.5, 1.0], atol=1e-7)
    assert mean_iou == pytest.approx(2.0 / 3.0)
    assert additional["weighted_f1"] == pytest.approx(0.75)


def test_train_and_validation_fail_before_selection_on_nonfinite_loss(tmp_path):
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    trainer.epochs = 1
    trainer.max_batches = None
    trainer.log_interval = 100
    trainer.use_mixup = False
    trainer.use_cutmix = False
    trainer.batch_mixing_probability = 0.0
    trainer.gradient_clip_val = None
    trainer.scheduler = None
    trainer.scheduler_step_per_batch = False
    trainer.scaler = None
    trainer.model = torch.nn.Conv2d(1, 3, kernel_size=1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

    class NonFiniteCriterion(torch.nn.Module):
        def forward(self, outputs, targets):
            return outputs.sum() * torch.tensor(float("nan"))

    trainer.criterion = NonFiniteCriterion()
    loader = [
        (
            torch.zeros((1, 1, 2, 2)),
            torch.zeros((1, 2, 2), dtype=torch.long),
            ["tile"],
        )
    ]

    with pytest.raises(FloatingPointError, match="non-finite loss in train"):
        trainer.train_epoch(loader, 0)
    with pytest.raises(FloatingPointError, match="non-finite loss in Val"):
        trainer.validate(loader, 0)
    assert trainer.best_selection_epoch is None
    with pytest.raises(FloatingPointError, match="checkpoint metrics must be finite"):
        trainer._checkpoint_payload(0, float("nan"), "validation_screen")


def test_conditional_train_and_native_validation_use_exact_gate_and_fair_metrics(
    tmp_path,
):
    """Exercise both loops so conditional input/composition cannot be misplaced."""
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    trainer.num_classes = 2
    trainer.loss_type = "conditional_pore_focal_dice"
    trainer.validation_only = True
    trainer.conditional_pore_threshold = 100
    trainer.recovered_threshold_acknowledged = True
    trainer.model_input_channels = 2
    trainer.epochs = 1
    trainer.max_batches = None
    trainer.log_interval = 100
    trainer.use_mixup = False
    trainer.use_cutmix = False
    trainer.batch_mixing_probability = 0.0
    trainer.gradient_clip_val = None
    trainer.scheduler = None
    trainer.scheduler_step_per_batch = False
    trainer.scaler = None

    class RecordingConditionalModel(torch.nn.Module):
        architecture_name = "plain_unet"
        n_channels = 2
        n_classes = 2

        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Conv2d(2, 2, kernel_size=1)
            self.seen_inputs = []

        def forward(self, inputs):
            self.seen_inputs.append(inputs.detach().cpu().clone())
            return self.classifier(inputs)

    trainer.model = RecordingConditionalModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.criterion = ConditionalPoreFocalDiceLoss([4, 5, 27])
    trainer.training_class_statistics = {
        "source": "authoritative_training_masks_only",
        "split_name": "train",
        "counts": [4, 5, 27],
    }

    raw_uint8 = torch.tensor([[[[0, 99], [100, 255]]]], dtype=torch.float32)
    images = raw_uint8 / 127.5 - 1.0
    masks = torch.tensor([[[0, 1], [-100, -100]]], dtype=torch.long)
    loader = [(images, masks, ["native_tile"])]

    train_loss, _, _ = trainer.train_epoch(loader, 0)
    validation = trainer.validate(loader, 0)
    _, _, _, class_iou, additional = validation

    assert np.isfinite(train_loss)
    assert len(trainer.model.seen_inputs) == 2
    for model_inputs in trainer.model.seen_inputs:
        assert tuple(model_inputs.shape) == (1, 2, 2, 2)
        torch.testing.assert_close(model_inputs[:, :1], images)
        torch.testing.assert_close(
            model_inputs[:, 1:],
            torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
        )

    confusion = additional["confusion_matrix"]
    assert confusion.shape == (3, 3)
    assert int(confusion.sum()) == 4
    assert int(confusion[2, 2]) == 2
    assert len(class_iou) == 3
    assert additional["evaluation_num_classes"] == 3
    assert additional["inference_config"]["pore_threshold_uint8"] == 100
    assert additional["inference_config"]["data_owner_confirmation"] == "pending"
    assert additional["inference_config"]["epoch_roc_artifacts"]["enabled"] is False
    assert trainer._epoch_roc_enabled(9) is False
    # This returns before touching a loader/model or emitting an artifact.
    trainer.plot_roc_curve(None, 9)

    checkpoint = trainer._checkpoint_payload(
        0, validation[0], checkpoint_role="validation_screen"
    )
    resolved = checkpoint["resolved_config"]
    assert resolved["model"]["input_channels"] == 2
    assert resolved["model"]["output_classes"] == 2
    assert resolved["model"]["input_channel_semantics"] == [
        "normalized_grayscale",
        "binary_recovered_pore_gate",
    ]
    assert resolved["inference"]["composed_outputs"] == ["C0", "C1", "C2"]


def test_selection_is_harmonic_c0_c1_iou_and_ignores_weighted_f1(tmp_path):
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    confusion = np.asarray(
        [
            [8, 1, 1],
            [2, 16, 2],
            [1, 2, 67],
        ]
    )

    low_f1 = trainer._selection_metric_components(
        np.asarray([0.2, 0.8, 0.99]),
        {"weighted_f1": 0.0, "confusion_matrix": confusion},
    )
    high_f1 = trainer._selection_metric_components(
        np.asarray([0.2, 0.8, 0.0]),
        {"weighted_f1": 1.0, "confusion_matrix": confusion},
    )

    expected_score = 2 * 0.2 * 0.8 / (0.2 + 0.8 + 1e-8)
    assert low_f1["score"] == pytest.approx(expected_score)
    assert high_f1["score"] == pytest.approx(expected_score)
    assert low_f1["c0_iou"] == pytest.approx(0.2)
    assert low_f1["c1_iou"] == pytest.approx(0.8)
    assert low_f1["pore_union_iou"] == pytest.approx(27 / 33)
    assert "weighted_f1" not in low_f1


def test_exact_harmonic_tie_is_resolved_by_pore_union_iou(tmp_path):
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    lower_union = np.asarray([[3, 1, 2], [1, 3, 2], [2, 2, 84]])
    higher_union = np.asarray([[4, 1, 1], [1, 4, 1], [1, 1, 86]])

    first = trainer._record_validation_selection(
        0,
        0.5,
        np.asarray([0.2, 0.8, 0.9]),
        {"confusion_matrix": lower_union},
    )
    second = trainer._record_validation_selection(
        1,
        0.5,
        np.asarray([0.8, 0.2, 0.9]),
        {"confusion_matrix": higher_union},
    )
    third = trainer._record_validation_selection(
        2,
        0.5,
        np.asarray([0.2, 0.8, 0.9]),
        {"confusion_matrix": lower_union},
    )

    assert first["score"] == pytest.approx(second["score"])
    assert second["pore_union_iou"] > first["pore_union_iou"]
    assert first["improved"] is True
    assert second["improved"] is True
    assert third["improved"] is False
    assert trainer.best_selection_epoch == 2


def test_exact_primary_and_union_tie_is_resolved_by_lower_validation_loss(tmp_path):
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    confusion = np.asarray([[4, 1, 1], [1, 4, 1], [1, 1, 86]])

    first = trainer._record_validation_selection(
        0,
        0.7,
        np.asarray([0.4, 0.6, 0.9]),
        {"confusion_matrix": confusion},
    )
    second = trainer._record_validation_selection(
        1,
        0.5,
        np.asarray([0.4, 0.6, 0.9]),
        {"confusion_matrix": confusion},
    )
    third = trainer._record_validation_selection(
        2,
        0.8,
        np.asarray([0.4, 0.6, 0.9]),
        {"confusion_matrix": confusion},
    )

    assert first["improved"] is True
    assert second["improved"] is True
    assert second["improvement_reason"] == (
        "lower_validation_loss_exact_primary_secondary_tie"
    )
    assert third["improved"] is False
    assert trainer.best_selection_epoch == 2
    assert trainer.best_selection_components["validation_loss"] == pytest.approx(0.5)


def test_hierarchical_training_counts_are_embedded_in_resolved_provenance(tmp_path):
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    counts = [11, 29, 160]
    trainer.loss_type = "hierarchical_pore_connectivity"
    trainer.criterion = HierarchicalPoreConnectivityLoss(counts)
    trainer.training_class_statistics = {
        "source": "authoritative_training_masks_only",
        "split_name": "train",
        "counts": counts,
        "image_ids": [1, 2],
    }

    resolved = trainer._resolved_run_config()["loss"]

    assert resolved["type"] == "hierarchical_pore_connectivity"
    assert resolved["candidate"]["training_class_counts"] == counts
    assert resolved["candidate"]["prediction_path"] == (
        "softmax_probabilities_only_no_argmax_or_numpy"
    )
    assert resolved["training_class_statistics"]["split_name"] == "train"


def test_checkpoint_resolved_config_embeds_repo_relative_split_contract(
    tmp_path, monkeypatch
):
    repository_root = tmp_path / "stage"
    manifest_path = repository_root / "config" / "confirmatory_splits.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"train": [1], "val": [2], "test": [3]}),
        encoding="utf-8",
    )
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    trainer.split_manifest = str(manifest_path)
    trainer.split_ids = {"train": [1], "val": [2], "test": [3]}
    trainer.split_files = {
        "train": ["train.png"],
        "val": ["val.png"],
        "test": ["test.png"],
    }
    trainer.test_loader = None
    trainer.validation_only = True
    monkeypatch.setattr(
        PatchTrainer,
        "_repository_root",
        staticmethod(lambda: repository_root),
    )

    checkpoint = trainer._checkpoint_payload(
        0, 0.5, checkpoint_role="validation_screen"
    )
    resolved = checkpoint["resolved_config"]["data_split"]

    assert resolved["manifest_repo_relative_identifier"] == (
        "config/confirmatory_splits.json"
    )
    assert resolved["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert resolved["partitions"]["train"] == {
        "image_ids": [1],
        "image_files": ["train.png"],
        "image_count": 1,
    }
    assert resolved["allocation_unit"] == "leading_source_identifier_group"
    assert resolved["observation_unit"] == "2048x2048_tile"
    assert resolved["group_membership_map"] == {
        "train": [],
        "val": [],
        "test": [],
    }
    assert resolved["specimen_independence_confirmation"] == (
        "pending_data_owner_confirmation"
    )
    assert resolved["validation_only"] is True
    assert resolved["held_out_dataset_constructed"] is False
    assert str(repository_root) not in repr(resolved)


def test_resume_fails_closed_when_checkpoint_uses_old_selection_metric(tmp_path):
    trainer = _bare_trainer(tmp_path, no_checkpoints=True)
    checkpoint_path = tmp_path / "old_selection.pth"
    torch.save(
        {
            "best_selection_epoch": 4,
            "selection_metric_name": "validation_pore_iou_weighted_f1",
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="different selection metric"):
        trainer.load_checkpoint(checkpoint_path)

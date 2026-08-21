"""
Patch-based U-Net trainer optimized for speed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import cv2
from pathlib import Path
import json
import gzip
import time
import random
import hashlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


SELECTION_METRIC_NAME = "validation_c0_c1_iou_harmonic_mean"
SELECTION_METRIC_DEFINITION = (
    "harmonic_mean(C0 IoU, C1 IoU) = 2 * C0_IoU * C1_IoU / "
    "(C0_IoU + C1_IoU + 1e-8); no mineral-frequency-weighted term"
)
SELECTION_TIEBREAKER_NAME = "validation_pore_union_iou"
SELECTION_TIEBREAKER_DEFINITION = (
    "IoU after merging C0 and C1 into one pore class; used only when the "
    "C0/C1 harmonic selection scores are exactly equal"
)
SELECTION_TERTIARY_TIEBREAKER_NAME = "lower_validation_loss"
SELECTION_TERTIARY_TIEBREAKER_DEFINITION = (
    "lower validation loss, used only when both the C0/C1 harmonic score and "
    "pore-union IoU are exactly equal"
)

# Try to import CUDA optimization modules
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - CUDA optimizations will be limited")

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - JIT compilation disabled")

# W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Install with: pip install wandb")

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet_model import create_model
from src.models.combined_loss import create_loss_function
from src.models.multiscale_attention_unet import (
    create_multiscale_attention_unet,
    create_multiscale_attention_unet_pyramid,
)
from src.models.boundary_aware_loss import create_boundary_aware_loss
from src.models.topological_loss import create_topological_loss
from src.models.hierarchical_pore_loss import (
    create_hierarchical_pore_connectivity_loss,
)
from src.models.conditional_pore_loss import (
    create_conditional_pore_focal_dice_loss,
)
from src.training.patch_dataset import (
    CANONICAL_CLASS_NAMES,
    INPUT_NORMALIZATION,
    PatchPredictor,
    create_patch_data_loaders,
    create_deterministic_image_splits,
    resolve_split_manifest,
)
from src.training.data_contract import (
    CONFIRMATORY_ANNOTATION_SHA256,
    CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS,
    CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS,
    CONFIRMATORY_SPLIT_MANIFEST_SHA256,
    aggregate_indexed_file_bytes,
)
from src.training.checkpoint_io import (
    load_weights_only_checkpoint,
    normalize_checkpoint_metadata,
)
from src.training.screen_selection import (
    EXECUTION_SOURCE_FILES,
    PROSPECTIVE_METHOD_PROTOCOLS,
    SELECTED_METHOD_LOCK_SCHEMA_VERSION,
    source_code_sha256,
    verify_selected_method_lock_document,
    verify_smoke_preflight_manifest_document,
)
try:
    from src.training.model_factory import create_advanced_model, create_advanced_loss
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("Advanced models not available")
try:
    from src.training.visualization import ImprovedPatchPredictor
    IMPROVED_PREDICTOR_AVAILABLE = True
except ImportError:
    IMPROVED_PREDICTOR_AVAILABLE = False
    print("ImprovedPatchPredictor not available")
from config.config_loader import load_config


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # For custom objects, try to convert their __dict__
        return convert_to_json_serializable(obj.__dict__)
    else:
        # For anything else, convert to string
        return str(obj)


class PatchTrainer:
    """Fast U-Net trainer using image patches."""
    
    def __init__(self, config_path: Optional[str] = None, multi_gpu: bool = False,
                 checkpoint_interval: int = 5, log_interval: int = 10,
                 resume_path: Optional[str] = None, use_wandb: bool = True,
                 wandb_project: str = "mineral-pore-segmentation",
                 wandb_name: Optional[str] = None, batch_size: Optional[int] = None,
                 model_type: Optional[str] = None, loss_type: Optional[str] = None,
                 num_classes: Optional[int] = None, class_weights: Optional[List[float]] = None,
                 learning_rate: Optional[float] = None, weight_decay: float = 0.0001,
                 workers: int = 4, save_predictions: bool = False,
                 experiment_name: Optional[str] = None, early_stopping: bool = False,
                 early_stopping_patience: int = 5, accumulate_grad_batches: int = 1,
                 mixed_precision: bool = False, gradient_clip_val: Optional[float] = None,
                 save_every_n_epochs: int = 2, patch_size: int = 683,
                 evaluation_patch_size: int = 2048,
                 evaluation_batch_size: int = 1,
                 no_checkpoints: bool = False, overlay_only: bool = False,
                 overwrite_plots: bool = False, max_batches: Optional[int] = None,
                 optimizer_type: str = "adamw", momentum: float = 0.9,
                 scheduler_type: str = "onecycle", augmentation_strength: str = "strong",
                 augmentations_enabled: bool = True, use_mixup: bool = False,
                 mixup_alpha: float = 0.2, use_cutmix: bool = False,
                 cutmix_alpha: float = 1.0, split_manifest: Optional[str] = None,
                 val_split: float = 0.1, test_split: float = 0.1,
                 seed: int = 42, annotations_path: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 mask_dir: Optional[str] = None,
                 requested_cli_arguments: Optional[Dict] = None,
                 focal_gamma: float = 2.0, tversky_alpha: float = 0.7,
                 tversky_beta: float = 0.3, dropout: float = 0.2,
                 freeze_encoder: bool = False,
                 validation_only: bool = False,
                 conditional_pore_threshold: Optional[int] = None,
                 recovered_threshold_acknowledged: bool = False,
                 selected_method_lock: Optional[str] = None,
                 protocol_candidate_key: Optional[str] = None,
                 protocol_run_role: Optional[str] = None,
                 protocol_campaign_id: Optional[str] = None,
                 protocol_cell_index: Optional[int] = None,
                 selected_architecture_role: Optional[str] = None,
                 smoke_preflight_manifest: Optional[str] = None):
        """Initialize patch trainer with multi-GPU support."""
        self.config = load_config(config_path or "config/pipeline_config.yaml")
        self.multi_gpu = multi_gpu
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.resume_path = resume_path
        self.seed = int(seed)
        self.optimizer_type = optimizer_type.lower()
        self.momentum = momentum
        self.scheduler_type = scheduler_type.lower()
        self.split_manifest = split_manifest
        self.val_split = val_split
        self.test_split = test_split
        self.annotations_path = Path(annotations_path) if annotations_path else None
        self.image_dir = Path(image_dir) if image_dir else Path("results/step3_coco_dataset/images")
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.loaded_annotations_path: Optional[Path] = None
        self.target_provenance: Dict = {
            'target_source': (
                'lossless_png_masks' if self.mask_dir is not None
                else 'coco_polygon_rasterization'
            ),
            'mask_directory': str(self.mask_dir) if self.mask_dir is not None else None,
            'mask_count': None,
            'mask_aggregate_sha256': None,
        }
        self.input_provenance: Dict = {
            'input_source': 'indexed_source_images',
            'scope': 'development_train_plus_validation',
            'image_count': None,
            'image_aggregate_sha256': None,
            'held_out_bytes_read': 0,
        }
        self.split_ids: Dict[str, List[int]] = {}
        self.split_files: Dict[str, List[str]] = {}
        self.category_id_map: Dict[int, int] = {}
        self.test_metrics: Optional[Dict] = None
        self.test_evaluation_count = 0
        self.validation_only = bool(validation_only)
        self.selected_method_lock_path = (
            Path(selected_method_lock) if selected_method_lock else None
        )
        self.selected_method_lock: Optional[Dict] = None
        self.selected_method_key: Optional[str] = None
        self.protocol_candidate_key = protocol_candidate_key
        self.protocol_run_role = protocol_run_role
        self.protocol_campaign_id = protocol_campaign_id
        self.protocol_cell_index = (
            int(protocol_cell_index)
            if protocol_cell_index is not None else None
        )
        self.selected_architecture_role = selected_architecture_role
        self.smoke_preflight_manifest_path = (
            Path(smoke_preflight_manifest) if smoke_preflight_manifest else None
        )
        self.smoke_preflight_manifest: Optional[Dict] = None
        if (
            self.protocol_candidate_key is not None
            and self.protocol_candidate_key not in PROSPECTIVE_METHOD_PROTOCOLS
        ):
            raise ValueError(
                "protocol_candidate_key must be one of "
                + ", ".join(PROSPECTIVE_METHOD_PROTOCOLS)
            )
        allowed_run_roles = {
            None,
            'validation_screen_cell',
            'validation_smoke_cell',
            'selected_winner_retraining',
        }
        if self.protocol_run_role not in allowed_run_roles:
            raise ValueError(
                "protocol_run_role must be validation_screen_cell, "
                "validation_smoke_cell, selected_winner_retraining, or omitted"
            )
        if self.protocol_run_role in {
            'validation_screen_cell',
            'validation_smoke_cell',
            'selected_winner_retraining',
        }:
            if not self.protocol_campaign_id or self.protocol_cell_index is None:
                raise ValueError(
                    "protocol array cells require campaign ID and cell index"
                )
            if (
                len(self.protocol_campaign_id) > 96
                or not self.protocol_campaign_id[0].isalnum()
                or any(
                    character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-'
                    for character in self.protocol_campaign_id
                )
            ):
                raise ValueError("protocol campaign ID is not public-path safe")
            maximum_index = 14 if (
                self.protocol_run_role == 'validation_screen_cell'
            ) else 2
            if not 0 <= self.protocol_cell_index <= maximum_index:
                raise ValueError(
                    f"protocol cell index must be in 0..{maximum_index}"
                )
        elif (
            self.protocol_campaign_id is not None
            or self.protocol_cell_index is not None
        ):
            raise ValueError(
                "protocol campaign/cell fields are reserved for array cells"
            )
        if self.protocol_run_role is not None:
            if (
                not os.environ.get('SLURM_JOB_ID')
                or os.environ.get('SLURM_ARRAY_JOB_ID')
                != self.protocol_campaign_id
                or os.environ.get('SLURM_ARRAY_TASK_ID')
                != str(self.protocol_cell_index)
            ):
                raise ValueError(
                    'protocol training requires the matching active Slurm array cell'
                )
            if os.environ.get('BOOTSTRAP_FACTOR') != '1':
                raise ValueError('protocol training requires BOOTSTRAP_FACTOR=1')
        if self.protocol_candidate_key is not None and self.protocol_run_role is None:
            raise ValueError(
                "protocol candidate key requires an explicit protocol run role"
            )
        if self.protocol_run_role == 'validation_screen_cell':
            if self.smoke_preflight_manifest_path is None:
                raise ValueError(
                    'validation screen requires an authenticated smoke-preflight manifest'
                )
        elif self.smoke_preflight_manifest_path is not None:
            raise ValueError(
                'smoke-preflight manifest is valid only for validation screen cells'
            )
        if self.selected_architecture_role not in {
            None, 'primary_multiscale', 'plain_unet_comparator'
        }:
            raise ValueError(
                "selected_architecture_role must be primary_multiscale, "
                "plain_unet_comparator, or omitted"
            )
        self.conditional_pore_threshold = (
            int(conditional_pore_threshold)
            if conditional_pore_threshold is not None
            else None
        )
        self.recovered_threshold_acknowledged = bool(
            recovered_threshold_acknowledged
        )
        self.training_class_statistics: Optional[Dict] = None
        self.requested_cli_arguments = requested_cli_arguments or {}
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.dropout = dropout
        self.freeze_encoder = freeze_encoder
        self.config.update('focal_gamma', self.focal_gamma)
        # The focal factory reads this nested key. Keep the CLI/runtime value
        # synchronized so provenance and the executed objective cannot diverge.
        self.config.update('model.loss.gamma', self.focal_gamma)
        self.config.update('tversky_alpha', self.tversky_alpha)
        self.config.update('tversky_beta', self.tversky_beta)
        self.config.update('dropout', self.dropout)
        self.config.update('freeze_encoder', self.freeze_encoder)
        self._set_random_seed()
        
        # Setup distributed training if multi-GPU
        if self.multi_gpu:
            self._setup_distributed()
        
        self.device = self._setup_device()
        if self.protocol_run_role is not None:
            if (
                self.device.type != 'cuda'
                or not torch.cuda.is_available()
                or 'L40S' not in torch.cuda.get_device_name(self.device)
            ):
                raise ValueError(
                    'protocol training requires an allocated NVIDIA L40S GPU'
                )
        
        # W&B configuration - set before setup_directories
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name or experiment_name
        self.wandb_run = None
        self.model_type = model_type or 'improved_unet'
        self.loss_type = loss_type or 'focal_dice'
        # Loss factories consume the nested YAML key. Keep the requested
        # runtime objective and the factory input identical.
        self.config.update('model.loss.type', self.loss_type)
        if (
            self.model_type == 'multiscale_attention_unet_pyramid'
            and self.loss_type != 'conditional_pore_focal_dice'
        ):
            raise ValueError(
                "multiscale_attention_unet_pyramid is restricted to the "
                "conditional_pore_focal_dice validation-screen candidate"
            )
        self.num_classes = num_classes if num_classes is not None else self.config.get('model.out_channels', 3)
        # Keep factories that read the ConfigLoader synchronized with explicit
        # CLI overrides, then record the instantiated model fields below.
        self.config.update('model.num_classes', self.num_classes)
        self.config.update('model.out_channels', self.num_classes)
        if self.loss_type == 'conditional_pore_focal_dice':
            if self.num_classes != 2:
                raise ValueError(
                    "conditional_pore_focal_dice requires --num-classes 2"
                )
            if not self.validation_only:
                raise ValueError(
                    "conditional gated training is validation-only; held-out "
                    "evaluation belongs exclusively to the locked evaluator"
                )
            if self.model_type not in {
                'multiscale_attention_unet',
                'multiscale_attention_unet_pyramid',
                'plain_unet',
                'unet',
            }:
                raise ValueError(
                    "conditional gated candidate requires multiscale_attention_unet "
                    "or plain_unet"
                )
            if self.conditional_pore_threshold != 100:
                raise ValueError(
                    "conditional gated candidate is prospectively fixed to the "
                    "recovered raw-uint8 pore rule intensity < 100"
                )
            if not self.recovered_threshold_acknowledged:
                raise ValueError(
                    "conditional gated candidate requires explicit acknowledgement "
                    "of the recovered threshold rule"
                )
        else:
            if (
                self.conditional_pore_threshold is not None
                or self.recovered_threshold_acknowledged
            ):
                raise ValueError(
                    "conditional_pore_threshold is only valid with "
                    "conditional_pore_focal_dice"
                )
            if self.loss_type == 'hierarchical_pore_connectivity':
                if self.num_classes != 3:
                    raise ValueError(
                        "hierarchical_pore_connectivity requires --num-classes 3"
                    )
                if not self.validation_only:
                    raise ValueError(
                        "hierarchical_pore_connectivity training is validation-only; "
                        "held-out evaluation belongs exclusively to the locked evaluator"
                    )
        self.model_input_channels = (
            2 if self.loss_type == 'conditional_pore_focal_dice' else 1
        )
        self.config.update('model.in_channels', self.model_input_channels)
        # H3/C2 derive their only class balance from authoritative training-mask
        # counts.  Do not make the inherited YAML weights look requested or
        # executable when the candidate criterion deliberately ignores them.
        if class_weights is None and self.loss_type in {
            'hierarchical_pore_connectivity',
            'conditional_pore_focal_dice',
        }:
            self.class_weights = None
        else:
            self.class_weights = (
                class_weights
                if class_weights is not None
                else self.config.get('model.loss.class_weights')
            )
        if self.class_weights is not None and len(self.class_weights) != self.num_classes:
            if class_weights is None and len(self.class_weights) > self.num_classes:
                self.class_weights = list(self.class_weights[:self.num_classes])
            else:
                raise ValueError(
                    f"Expected {self.num_classes} class weights, got {len(self.class_weights)}"
                )
        
        # Training parameters - optimized for patches
        self.epochs = self.config.get('model.epochs', 20)  # Fewer epochs needed
        self.learning_rate = learning_rate if learning_rate is not None else self.config.get('model.learning_rate', 0.001)
        self.weight_decay = weight_decay
        self.workers = workers
        self.save_predictions = save_predictions
        self.early_stopping_enabled = early_stopping
        self.accumulate_grad_batches = accumulate_grad_batches
        self.mixed_precision = mixed_precision
        self.gradient_clip_val = gradient_clip_val
        self.save_every_n_epochs = save_every_n_epochs
        
        # Ablation mode settings
        self.no_checkpoints = no_checkpoints
        self.overlay_only = overlay_only
        self.overwrite_plots = overwrite_plots
        self.max_batches = max_batches  # Limit batches per epoch for testing
        
        # Get batch size from config or override, default to 16
        if batch_size is not None:
            base_batch_size = batch_size
        else:
            base_batch_size = self.config.get('gpu_processing.batch_size', 16)
        self.batch_size = base_batch_size if not multi_gpu else base_batch_size * int(os.environ.get('WORLD_SIZE', 1))
        self.patch_size = patch_size  # Use the passed patch_size parameter
        self.evaluation_patch_size = int(evaluation_patch_size)
        self.evaluation_batch_size = int(evaluation_batch_size)
        if self.evaluation_patch_size <= 0 or self.evaluation_batch_size <= 0:
            raise ValueError("evaluation patch and batch sizes must be positive")
        if self.model_type == 'multiscale_attention_unet_pyramid':
            if (
                self.patch_size != 2048
                or self.evaluation_patch_size != 2048
                or self.batch_size != 1
                or self.evaluation_batch_size != 1
            ):
                raise ValueError(
                    "multiscale_attention_unet_pyramid is locked to 2048-pixel "
                    "train/validation tiles with batch size 1"
                )
            if not self.mixed_precision:
                raise ValueError(
                    "multiscale_attention_unet_pyramid requires AMP for the "
                    "prospective full-tile screen"
                )
            if float(self.dropout) != 0.0:
                raise ValueError(
                    "multiscale_attention_unet_pyramid is prospectively locked "
                    "to --dropout 0.0"
                )

        if self.selected_method_lock_path is not None and not self.validation_only:
            raise ValueError(
                "selected-winner retraining must remain validation-only; only the "
                "locked evaluator may construct the held-out dataset"
            )
        if self.protocol_run_role in {
            'validation_screen_cell',
            'validation_smoke_cell',
            'selected_winner_retraining',
        } and not self.validation_only:
            raise ValueError("all protocol training roles must be validation-only")
        if self.protocol_run_role == 'selected_winner_retraining':
            if self.selected_method_lock_path is None:
                raise ValueError(
                    "selected winner retraining requires a selected-method lock"
                )
            if self.selected_architecture_role is None:
                raise ValueError(
                    "selected winner retraining requires an architecture role"
                )
        elif self.selected_method_lock_path is not None:
            raise ValueError(
                "selected-method lock is valid only for selected_winner_retraining"
            )
        elif self.selected_architecture_role is not None:
            raise ValueError(
                "selected architecture role is valid only for winner retraining"
            )
        runtime_protocol = self._current_selected_method_protocol()
        if self.protocol_candidate_key is not None:
            expected_protocol = dict(
                PROSPECTIVE_METHOD_PROTOCOLS[self.protocol_candidate_key]
            )
            if self.selected_architecture_role == 'plain_unet_comparator':
                expected_protocol['model_type'] = 'plain_unet'
            if runtime_protocol != expected_protocol:
                raise ValueError(
                    "runtime configuration does not match protocol candidate "
                    f"{self.protocol_candidate_key}: runtime={runtime_protocol}, "
                    f"expected={expected_protocol}"
                )
        if self.loss_type in {
            'hierarchical_pore_connectivity',
            'conditional_pore_focal_dice',
        } and self.protocol_candidate_key is None:
            raise ValueError(
                "prospective validation candidates require --protocol-candidate-key"
            )
        if self.selected_method_lock_path is not None and (
            self.protocol_candidate_key is None
        ):
            raise ValueError(
                "selected-method lock requires --protocol-candidate-key"
            )
        if self.selected_method_lock_path is not None:
            self.selected_method_lock = self._resolve_selected_method_lock(
                self.selected_method_lock_path
            )
            self.selected_method_key = self.selected_method_lock[
                'selected_method'
            ]
            if self.protocol_candidate_key != self.selected_method_key:
                raise ValueError(
                    "protocol candidate does not match the deterministic selected "
                    f"method ({self.protocol_candidate_key} != "
                    f"{self.selected_method_key})"
                )
        if self.smoke_preflight_manifest_path is not None:
            self.smoke_preflight_manifest = (
                self._resolve_smoke_preflight_manifest(
                    self.smoke_preflight_manifest_path
                )
            )
        
        # Augmentation configuration
        self.augmentation_config = {
            'patch_size': self.patch_size,
            'seed': self.seed,
            'augmentation': {
                'enabled': bool(augmentations_enabled),
                'strength': augmentation_strength,
                'mixup_alpha': mixup_alpha,
                'cutmix_alpha': cutmix_alpha,
                'use_advanced': bool(use_mixup or use_cutmix)
            }
        }
        self.use_mixup = bool(use_mixup and augmentations_enabled)
        self.use_cutmix = bool(use_cutmix and augmentations_enabled)
        if self.loss_type == 'conditional_pore_focal_dice' and (
            self.use_mixup or self.use_cutmix
        ):
            raise ValueError(
                "conditional threshold-gated candidates forbid MixUp and CutMix"
            )
        self.batch_mixing_probability = 0.5
        self.mixup_prob = 0.5  # Probability of applying mixup vs cutmix
        
        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.epoch_times = []
        
        # Early stopping
        self.patience = early_stopping_patience
        self.patience_counter = 0
        if self.protocol_run_role is not None:
            forbidden_environment = {
                name: os.environ[name]
                for name in ('QUICK_TEST', 'SINGLE_BATCH')
                if name in os.environ
            }
            if forbidden_environment:
                raise ValueError(
                    'protocol runs forbid debug environment switches: '
                    f'{forbidden_environment}'
                )
            if os.environ.get('DISABLE_AMP', '0') != '0':
                raise ValueError('protocol runs forbid disabling CUDA AMP')
            if os.environ.get('DISABLE_TRANSFORMS', '0') != '0':
                raise ValueError(
                    'protocol runs forbid disabling the frozen light transforms'
                )
            fixed_settings = {
                'workers': (self.workers, 8),
                'optimizer': (self.optimizer_type, 'adamw'),
                'scheduler': (self.scheduler_type, 'cosine'),
                'learning_rate': (float(self.learning_rate), 5e-4),
                'weight_decay': (float(self.weight_decay), 1e-4),
                'early_stopping': (self.early_stopping_enabled, True),
                'early_stopping_patience': (self.patience, 10),
                'mixed_precision': (self.mixed_precision, True),
                'gradient_clip_val': (self.gradient_clip_val, 1.0),
                'accumulate_grad_batches': (self.accumulate_grad_batches, 1),
                'augmentations_enabled': (bool(augmentations_enabled), True),
                'augmentation_strength': (augmentation_strength, 'light'),
                'mixup': (self.use_mixup, False),
                'cutmix': (self.use_cutmix, False),
            }
            mismatches = {
                key: actual
                for key, (actual, expected) in fixed_settings.items()
                if actual != expected
            }
            if mismatches:
                raise ValueError(
                    'protocol scientific settings differ from the frozen '
                    f'contract: {mismatches}'
                )
            if self.protocol_run_role in {
                'validation_screen_cell', 'selected_winner_retraining'
            } and (self.no_checkpoints or self.max_batches is not None):
                raise ValueError(
                    'scientific screen/retraining forbids disabled checkpoints '
                    'and bounded debug batches'
                )
            if self.protocol_run_role == 'validation_smoke_cell' and (
                self.no_checkpoints or self.max_batches != 2
            ):
                raise ValueError(
                    'smoke cells require checkpoints and exactly two batches'
                )
        self.setup_directories()
        # Model selection is independent of early stopping: every run retains
        # the state with the highest documented validation composite. Starting
        # at -inf guarantees that the first finite validation result is usable,
        # including zero-valued smoke/debug runs.
        self.best_val_metric = float('-inf')
        self.best_selection_epoch: Optional[int] = None  # One-based epoch.
        self.best_selection_components: Optional[Dict] = None
        self._best_selection_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.selection_checkpoint_path: Optional[Path] = None
        self.selection_checkpoint_sha256: Optional[str] = None
        self.selection_restore_source: Optional[str] = None
        
        # Metrics storage
        self.all_metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'mean_iou': [],
            'disconnected_pore_iou': [],
            'connected_pore_iou': [],
            'mineral_iou': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Setup CUDA optimizations
        self._setup_cuda_optimizations()
        
        # Configuration is written at the start of train(), after CLI epoch and
        # data-source overrides have been applied.

    def _set_random_seed(self):
        """Seed Python, NumPy, and PyTorch for a reproducible rerun."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _sha256(path: Optional[Path]) -> Optional[str]:
        """Return a source-file checksum for run provenance."""
        if path is None or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _repository_root() -> Path:
        """Return the checkout/staging root without serialising it to results."""
        return Path(__file__).resolve().parents[2]

    def _current_selected_method_protocol(self) -> Dict:
        """Return the runtime fields covered by the prospective method lock."""
        return {
            'model_type': str(self.model_type),
            'loss_type': str(self.loss_type),
            'model_input_channels': int(self.model_input_channels),
            'model_output_classes': int(self.num_classes),
            'training_patch_size': int(self.patch_size),
            'training_batch_size': int(self.batch_size),
            'evaluation_patch_size': int(self.evaluation_patch_size),
            'evaluation_batch_size': int(self.evaluation_batch_size),
            'conditional_pore_threshold': self.conditional_pore_threshold,
            'dropout_requested': float(self.dropout),
            'mixed_precision_requested': bool(self.mixed_precision),
        }

    def _resolve_selected_method_lock(self, lock_value: Path) -> Dict:
        """Validate a post-screen lock before validation-only winner retraining."""
        repository_root = self._repository_root().resolve()
        lock_path = Path(lock_value)
        if not lock_path.is_absolute():
            lock_path = repository_root / lock_path
        lock_path = lock_path.resolve()
        try:
            lock_identifier = lock_path.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "selected-method lock must resolve inside the repository/staging root"
            ) from error
        if not lock_path.is_file():
            raise FileNotFoundError(
                f"selected-method lock does not exist: {lock_identifier}"
            )

        with lock_path.open('r', encoding='utf-8') as handle:
            document = json.load(handle)
        document = verify_selected_method_lock_document(
            document, repository_root
        )
        schema_version = document.get('schema_version')
        if schema_version != SELECTED_METHOD_LOCK_SCHEMA_VERSION:
            raise ValueError(
                "selected-method lock schema_version must be "
                f"{SELECTED_METHOD_LOCK_SCHEMA_VERSION}, got {schema_version!r}"
            )
        selected_method = document.get('selected_method')
        if selected_method not in PROSPECTIVE_METHOD_PROTOCOLS:
            raise ValueError(
                "selected-method lock candidate must be one of "
                + ", ".join(PROSPECTIVE_METHOD_PROTOCOLS)
            )
        locked_protocol = document.get('resolved_protocol')
        if not isinstance(locked_protocol, dict):
            raise ValueError(
                "selected-method lock requires a resolved_protocol object"
            )

        prospective_protocol = PROSPECTIVE_METHOD_PROTOCOLS[selected_method]
        runtime_protocol = self._current_selected_method_protocol()
        locked_mismatches = {
            key: {
                'locked': locked_protocol.get(key, '<missing>'),
                'prospective': expected_value,
            }
            for key, expected_value in prospective_protocol.items()
            if locked_protocol.get(key, '<missing>') != expected_value
        }
        if locked_mismatches:
            raise ValueError(
                "selected-method lock does not match the prespecified candidate "
                f"protocol: {locked_mismatches}"
            )
        expected_runtime_protocol = dict(prospective_protocol)
        if self.selected_architecture_role == 'plain_unet_comparator':
            expected_runtime_protocol['model_type'] = 'plain_unet'
        runtime_mismatches = {
            key: {
                'runtime': runtime_protocol[key],
                'locked': expected_value,
            }
            for key, expected_value in expected_runtime_protocol.items()
            if runtime_protocol[key] != expected_value
        }
        if runtime_mismatches:
            raise ValueError(
                "runtime configuration does not match selected-method lock: "
                f"{runtime_mismatches}"
            )

        screen_source_hashes = document['screen_selection_provenance'][
            'source_code_sha256'
        ]
        current_source_hashes = self._source_code_sha256()
        if current_source_hashes != screen_source_hashes:
            raise ValueError(
                "current execution source hashes do not match the frozen screen"
            )

        lock_sha256 = self._sha256(lock_path)
        if lock_sha256 is None:
            raise RuntimeError("selected-method lock checksum could not be computed")
        return {
            'schema_version': int(schema_version),
            'selected_method': str(selected_method),
            'lock_file_repo_relative_identifier': lock_identifier,
            'lock_file_sha256': lock_sha256,
            'resolved_protocol': dict(prospective_protocol),
            'screen_selection_provenance': document[
                'screen_selection_provenance'
            ],
        }

    def _resolve_smoke_preflight_manifest(self, manifest_value: Path) -> Dict:
        """Authenticate the complete non-scientific L40S smoke campaign."""
        repository_root = self._repository_root().resolve()
        path = Path(manifest_value)
        if not path.is_absolute():
            path = repository_root / path
        path = path.resolve()
        try:
            identifier = path.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise ValueError(
                'smoke-preflight manifest must resolve inside the staging root'
            ) from error
        if not path.is_file():
            raise FileNotFoundError(
                f'smoke-preflight manifest does not exist: {identifier}'
            )
        with path.open('r', encoding='utf-8') as handle:
            document = json.load(handle)
        verified = verify_smoke_preflight_manifest_document(
            document, repository_root
        )
        provenance = verified['smoke_campaign_provenance']
        if provenance['source_code_sha256'] != self._source_code_sha256():
            raise ValueError(
                'smoke-preflight source hashes do not match current execution code'
            )
        manifest_sha256 = self._sha256(path)
        if manifest_sha256 is None:
            raise RuntimeError('smoke-preflight manifest checksum is unavailable')
        return {
            'repo_relative_identifier': identifier,
            'sha256': manifest_sha256,
            'campaign_id': provenance['campaign_id'],
            'provenance': provenance,
        }

    def _verify_selected_lock_data_contract(self) -> None:
        """Check split and target hashes after validation-only loaders exist."""
        lock = getattr(self, 'selected_method_lock', None)
        if lock is None:
            return
        provenance = lock['screen_selection_provenance']
        resolved_split = self._resolved_data_split_config()
        if resolved_split != provenance['resolved_data_split']:
            raise RuntimeError(
                "selected retraining resolved split does not match the frozen screen"
            )
        if self.target_provenance != provenance['target_provenance']:
            raise RuntimeError(
                "selected retraining target provenance does not match the frozen screen"
            )
        if self.input_provenance != provenance['input_provenance']:
            raise RuntimeError(
                "selected retraining input provenance does not match the frozen screen"
            )

    def _verify_smoke_preflight_data_contract(self) -> None:
        reference = getattr(self, 'smoke_preflight_manifest', None)
        if reference is None:
            return
        provenance = reference['provenance']
        split = self._resolved_data_split_config()
        if split != provenance['resolved_data_split']:
            raise RuntimeError('screen split differs from smoke preflight')
        if self.target_provenance != provenance['target_provenance']:
            raise RuntimeError('screen targets differ from smoke preflight')
        if self.input_provenance != provenance['input_provenance']:
            raise RuntimeError('screen input provenance differs from smoke preflight')

    def _compute_development_input_provenance(self) -> Dict:
        """Hash train+validation image bytes without opening held-out inputs."""
        train_names = [str(value) for value in self.split_files.get('train', [])]
        development_names = train_names + [
            str(value) for value in self.split_files.get('val', [])
        ]
        train_record = aggregate_indexed_file_bytes(
            self.image_dir,
            train_names,
            scope='training_only',
            split_names=('train',),
        )
        development_record = aggregate_indexed_file_bytes(
            self.image_dir,
            development_names,
            scope='development_train_plus_validation',
            split_names=('train', 'val'),
        )
        self.input_provenance = {
            'input_source': 'indexed_source_images',
            **development_record,
            'training_subset': train_record,
            'held_out_bytes_read': 0,
            'held_out_scope': 'not_read_or_hashed_by_validation_only_trainer',
        }
        if self.protocol_run_role is not None:
            expected_train = CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS['train']
            expected_development = (
                CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS[
                    'train_plus_validation'
                ]
            )
            if (
                train_record['image_count'] != expected_train['image_count']
                or train_record['image_aggregate_sha256']
                != expected_train['image_aggregate_sha256']
                or development_record['image_count']
                != expected_development['image_count']
                or development_record['image_aggregate_sha256']
                != expected_development['image_aggregate_sha256']
            ):
                raise RuntimeError(
                    'protocol input-image bytes do not match the canonical '
                    'train74/development79 attestations'
                )
            expected_target = CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS[
                'train_plus_validation'
            ]
            if (
                self.target_provenance.get('mask_count')
                != expected_target['mask_count']
                or self.target_provenance.get('mask_aggregate_sha256')
                != expected_target['mask_aggregate_sha256']
            ):
                raise RuntimeError(
                    'protocol targets do not match the canonical development79 '
                    'mask attestation'
                )
            resolved_split = self._resolved_data_split_config()
            if (
                resolved_split.get('manifest_sha256')
                != CONFIRMATORY_SPLIT_MANIFEST_SHA256
                or resolved_split.get('annotation_index_sha256')
                != CONFIRMATORY_ANNOTATION_SHA256
                or resolved_split.get(
                    'annotation_index_repo_relative_identifier'
                ) != 'results/step3_coco_dataset/pore_annotations.json'
            ):
                raise RuntimeError(
                    'protocol annotation index or frozen split manifest does '
                    'not match the canonical attestation'
                )
        return dict(self.input_provenance)

    def _verify_selected_lock_execution_contract(self) -> None:
        """Keep winner retraining identical, except for the named comparator model."""
        lock = getattr(self, 'selected_method_lock', None)
        if lock is None:
            return
        matching_cells = [
            cell
            for cell in lock['screen_selection_provenance']['screen_cells']
            if cell['candidate'] == self.protocol_candidate_key
            and int(cell['seed']) == int(self.seed)
        ]
        if len(matching_cells) != 1:
            raise RuntimeError(
                "selected lock has no unique matching candidate/seed screen cell"
            )
        screen_contract = matching_cells[0]['scientific_execution_contract']
        current_contract = self._resolved_scientific_execution_contract()
        screen_non_model = {
            key: value for key, value in screen_contract.items()
            if key != 'model'
        }
        current_non_model = {
            key: value for key, value in current_contract.items()
            if key != 'model'
        }
        if current_non_model != screen_non_model:
            raise RuntimeError(
                "selected retraining changed a non-architecture scientific setting"
            )
        current_model = current_contract['model']
        if self.selected_architecture_role == 'primary_multiscale':
            if current_model != screen_contract['model']:
                raise RuntimeError(
                    "selected primary model does not match its frozen screen model"
                )
        elif self.selected_architecture_role == 'plain_unet_comparator':
            expected_protocol = PROSPECTIVE_METHOD_PROTOCOLS[
                self.protocol_candidate_key
            ]
            if (
                current_model.get('architecture_resolved') != 'plain_unet'
                or current_model.get('input_channels')
                != expected_protocol['model_input_channels']
                or current_model.get('output_classes')
                != expected_protocol['model_output_classes']
                or current_model.get('deep_supervision') is not False
                or current_model.get('dropout_execution') != {
                    'status': 'unused_by_resolved_architecture',
                    'probability': None,
                }
            ):
                raise RuntimeError(
                    "plain comparator changed more than the frozen architecture role"
                )
        else:
            raise RuntimeError("selected retraining architecture role is unresolved")

    def _source_code_sha256(self) -> Dict[str, Optional[str]]:
        """Snapshot execution-critical source hashes using repo-relative keys."""
        cached = getattr(self, 'source_code_sha256', None)
        if cached is not None:
            return dict(cached)

        self.source_code_sha256 = source_code_sha256(self._repository_root())
        return dict(self.source_code_sha256)

    def _actual_loss_class_weights(self) -> Optional[List[float]]:
        """Read class weights from the instantiated criterion when exposed."""
        weights = getattr(getattr(self, 'criterion', None), 'class_weights', None)
        if weights is None:
            return None
        if isinstance(weights, torch.Tensor):
            return [float(value) for value in weights.detach().cpu().flatten().tolist()]
        try:
            return [float(value) for value in weights]
        except TypeError:
            return None

    def _resolved_loss_config(self) -> Dict:
        """Describe the instantiated objective and any training-only constants."""
        criterion = getattr(self, 'criterion', None)
        resolved = {
            'type': getattr(self, 'loss_type', None),
            'implementation_class': (
                type(criterion).__name__ if criterion is not None else None
            ),
            'class_weights_requested': getattr(self, 'class_weights', None),
            'class_weights_actual': self._actual_loss_class_weights(),
        }
        if criterion is not None and hasattr(criterion, 'resolved_config'):
            candidate_config = criterion.resolved_config()
            if not isinstance(candidate_config, dict):
                raise TypeError("criterion.resolved_config() must return a dictionary")
            resolved['candidate'] = candidate_config
        training_statistics = getattr(self, 'training_class_statistics', None)
        if training_statistics is not None:
            resolved['training_class_statistics'] = dict(training_statistics)
        return resolved

    def _resolved_inference_config(self) -> Dict:
        """Record the exact validation decision rule for native or gated runs."""
        threshold = getattr(self, 'conditional_pore_threshold', None)
        if getattr(self, 'loss_type', None) == 'conditional_pore_focal_dice':
            if threshold is None:
                raise RuntimeError(
                    "conditional inference provenance requires an explicit threshold"
                )
            return {
                'mode': 'uint8_pore_gate_then_conditional_c0_c1',
                'raw_uint8_pore_rule': f'intensity < {threshold}',
                'raw_uint8_mineral_rule': f'intensity >= {threshold}',
                'pore_threshold_uint8': int(threshold),
                'normalized_pore_threshold': (2.0 * int(threshold) / 255.0) - 1.0,
                'network_outputs': ['C0', 'C1'],
                'composed_outputs': ['C0', 'C1', 'C2'],
                'model_input_channels': 2,
                'model_input_channel_semantics': [
                    'normalized_grayscale',
                    'binary_recovered_pore_gate',
                ],
                'mineral_prediction_source': 'fixed_raw_intensity_gate',
                'threshold_evidence': (
                    'recovered_step2_code_and_quantified_training_mask_agreement'
                ),
                'gate_relation_to_authoritative_targets': (
                    'prespecified_operational_approximation_not_bit_exact_reconstruction'
                ),
                'training_only_gate_disagreement_audit': {
                    'scope': 'train74_only',
                    'total_pixels': 310378496,
                    'disagreeing_pixels': 14054,
                    'disagreement_fraction': 0.0000452802,
                    'target_c2_restored_overlay_trace_pixels_gate_as_pore': 12980,
                    'target_c1_pixels_with_clean_uint8_greater_equal_100': 1074,
                    'c0_factorization': 'exact_in_training_audit',
                    'used_for_threshold_tuning': False,
                },
                'threshold_rule_acknowledged': bool(
                    getattr(self, 'recovered_threshold_acknowledged', False)
                ),
                'data_owner_confirmation': 'pending',
                'epoch_roc_artifacts': {
                    'enabled': False,
                    'reason': (
                        'conditional probabilities are not full three-class '
                        'composed predictions; final ROC/PR is reserved for the '
                        'locked composed evaluator after method freeze'
                    ),
                },
            }
        return {
            'mode': 'native_model_argmax',
            'network_outputs': int(getattr(self, 'num_classes', 0)),
            'conditional_pore_threshold': None,
        }

    def _resolved_model_config(self) -> Dict:
        """Describe the instantiated model rather than only requested flags."""
        if not hasattr(self, 'model'):
            return {
                'architecture': getattr(self, 'model_type', None),
                'architecture_requested': getattr(self, 'model_type', None),
                'architecture_resolved': None,
                'implementation_class': None,
            }
        base_model = self.model.module if self.multi_gpu else self.model
        parameter_count = sum(parameter.numel() for parameter in base_model.parameters())
        resolved = {
            'architecture': getattr(self, 'model_type', None),
            'architecture_requested': getattr(self, 'model_type', None),
            'architecture_resolved': getattr(
                base_model,
                'architecture_name',
                type(base_model).__name__,
            ),
            'implementation_class': type(base_model).__name__,
            'input_channels': getattr(base_model, 'n_channels', 1),
            'input_channel_semantics': (
                ['normalized_grayscale', 'binary_recovered_pore_gate']
                if getattr(self, 'loss_type', None)
                == 'conditional_pore_focal_dice'
                else ['normalized_grayscale']
            ),
            'output_classes': getattr(
                base_model,
                'n_classes',
                getattr(self, 'num_classes', None),
            ),
            'bilinear': getattr(base_model, 'bilinear', None),
            'base_features': getattr(base_model, 'base_features', None),
            'deep_supervision': getattr(base_model, 'deep_supervision', False),
            'dropout_requested': getattr(self, 'dropout', None),
            'freeze_encoder': getattr(self, 'freeze_encoder', None),
            'parameter_count': int(parameter_count),
            'output': 'logits',
        }
        if hasattr(base_model, 'resolved_pyramid_context_config'):
            resolved['pyramid_context'] = (
                base_model.resolved_pyramid_context_config()
            )
            resolved['dropout_execution'] = {
                'status': 'executed_in_pyramid_context',
                'probability': float(
                    resolved['pyramid_context']['dropout']['probability']
                ),
            }
        else:
            resolved['dropout_execution'] = {
                'status': 'unused_by_resolved_architecture',
                'probability': None,
            }
        return resolved

    def _resolved_augmentation_config(self) -> Dict:
        """Describe the actual per-patch and optional batch-level transforms."""
        requested = getattr(self, 'augmentation_config', {})
        requested_augmentation = requested.get('augmentation', {})
        run_seed = int(getattr(self, 'seed', requested.get('seed', 42)))
        use_mixup = bool(getattr(self, 'use_mixup', False))
        use_cutmix = bool(getattr(self, 'use_cutmix', False))
        train_loader = getattr(self, 'train_loader', None)
        dataset = getattr(train_loader, 'dataset', None)
        provenance = getattr(dataset, 'augmentation_provenance', None)
        if provenance is None:
            provenance = {
                'enabled': bool(requested_augmentation.get('enabled', False)),
                'disable_reason': 'data_loader_not_created',
                'training': True,
                'strength': requested_augmentation.get('strength'),
                'seed': run_seed,
                'albumentations_version': None,
                'transforms': [],
            }
        resolved = dict(provenance)
        resolved['batch_level'] = {
            'application_probability': (
                getattr(self, 'batch_mixing_probability', 0.5)
                if use_mixup or use_cutmix
                else 0.0
            ),
            'mixup_enabled': use_mixup,
            'mixup_alpha': requested_augmentation.get('mixup_alpha'),
            'cutmix_enabled': use_cutmix,
            'cutmix_alpha': requested_augmentation.get('cutmix_alpha'),
            'mixup_choice_probability_when_both_enabled': (
                getattr(self, 'mixup_prob', 0.5)
                if use_mixup and use_cutmix
                else None
            ),
        }
        multi_gpu = bool(getattr(self, 'multi_gpu', False))
        resolved['data_loader'] = {
            'shuffle_generator_seed': None if multi_gpu else run_seed,
            'distributed_sampler_seed': run_seed if multi_gpu else None,
            'num_workers': getattr(self, 'workers', None),
            'persistent_workers': False,
            'drop_last': False,
            'training_patch_size': getattr(self, 'patch_size', None),
            'evaluation_patch_size': getattr(
                self, 'evaluation_patch_size', None
            ),
            'training_batch_size': getattr(self, 'batch_size', None),
            'evaluation_batch_size': getattr(
                self, 'evaluation_batch_size', None
            ),
            'worker_rng_strategy': (
                'top_level_picklable_worker_init_reseeds_training_compose'
            ),
            'worker_init_function': (
                'src.training.patch_dataset.seed_patch_dataloader_worker'
            ),
            'effective_worker_seed_formula': (
                '(augmentation_seed + (torch.initial_seed() % 2**32)) % 2**32'
            ),
            'python_numpy_torch_seeded': True,
            'albumentations_compose_reseeded_for_training': True,
            'albumentations_2_0_8_worker_rng_mitigation': True,
            'validation_and_test_transforms': 'disabled_deterministic',
        }
        return resolved

    def _resolved_scientific_execution_contract(self) -> Dict:
        """Attest fairness-critical settings from live trainer objects."""
        scheduler = getattr(self, 'scheduler', None)
        optimizer = getattr(self, 'optimizer', None)
        train_loader = getattr(self, 'train_loader', None)
        train_dataset = getattr(train_loader, 'dataset', None)
        return {
            'epochs_planned': int(self.epochs),
            'early_stopping': {
                'enabled': bool(self.early_stopping_enabled),
                'patience': int(self.patience),
                'selection_metric': SELECTION_METRIC_NAME,
            },
            'optimizer': {
                'requested': self.optimizer_type,
                'implementation_class': (
                    type(optimizer).__name__ if optimizer is not None else None
                ),
                'configured_learning_rate': float(self.learning_rate),
                'actual_initial_learning_rate': (
                    float(optimizer.defaults['lr'])
                    if optimizer is not None else None
                ),
                'configured_weight_decay': float(self.weight_decay),
                'actual_weight_decay': (
                    float(optimizer.defaults['weight_decay'])
                    if optimizer is not None else None
                ),
            },
            'scheduler': {
                'requested': self.scheduler_type,
                'implementation_class': (
                    type(scheduler).__name__ if scheduler is not None else None
                ),
                't_max': (
                    int(scheduler.T_max)
                    if scheduler is not None and hasattr(scheduler, 'T_max')
                    else None
                ),
                'step_unit': (
                    'optimizer_step'
                    if getattr(self, 'scheduler_step_per_batch', False)
                    else 'epoch'
                ),
            },
            'loss': self._resolved_loss_config(),
            'model': self._resolved_model_config(),
            'augmentation': self._resolved_augmentation_config(),
            'bootstrap_factor': int(
                getattr(train_dataset, 'bootstrap_factor', -1)
            ),
            'workers': int(self.workers),
            'gradient_clip_val': (
                float(self.gradient_clip_val)
                if self.gradient_clip_val is not None else None
            ),
            'mixed_precision_requested': bool(self.mixed_precision),
            'mixed_precision_actual': getattr(self, 'scaler', None) is not None,
            'accumulate_grad_batches': int(self.accumulate_grad_batches),
            'batch_mixup_enabled': bool(self.use_mixup),
            'batch_cutmix_enabled': bool(self.use_cutmix),
        }

    def _resolved_data_split_config(self) -> Dict:
        """Return the exact partition contract embedded in every checkpoint."""
        manifest_value = getattr(self, 'split_manifest', None)
        manifest_identifier = None
        manifest_sha256 = None
        manifest_source = 'generated_from_seed_and_fractions'
        group_membership_map = {'train': [], 'val': [], 'test': []}
        if manifest_value is not None:
            repository_root = self._repository_root().resolve()
            manifest_path = Path(manifest_value)
            if not manifest_path.is_absolute():
                manifest_path = repository_root / manifest_path
            manifest_path = manifest_path.resolve()
            try:
                manifest_identifier = manifest_path.relative_to(
                    repository_root
                ).as_posix()
            except ValueError as error:
                raise RuntimeError(
                    "split manifest must resolve inside the repository/staging root"
                ) from error
            manifest_sha256 = self._sha256(manifest_path)
            if manifest_sha256 is None:
                raise FileNotFoundError(
                    f"resolved split manifest does not exist: {manifest_identifier}"
                )
            manifest_source = 'explicit_manifest'
            with manifest_path.open('r', encoding='utf-8') as handle:
                manifest_document = json.load(handle)
            manifest_provenance = manifest_document.get('_provenance', {})
            group_membership_map = {
                'train': list(manifest_provenance.get('train_series', [])),
                'val': list(manifest_provenance.get('validation_series', [])),
                'test': list(manifest_provenance.get('test_series', [])),
            }

        split_ids = getattr(self, 'split_ids', {}) or {}
        split_files = getattr(self, 'split_files', {}) or {}
        partitions = {}
        for split_name in ('train', 'val', 'test'):
            ids = [int(value) for value in split_ids.get(split_name, [])]
            files = [str(value) for value in split_files.get(split_name, [])]
            if len(ids) != len(files):
                raise RuntimeError(
                    f"resolved {split_name} image IDs/files have different lengths"
                )
            partitions[split_name] = {
                'image_ids': ids,
                'image_files': files,
                'image_count': len(ids),
            }

        annotation_identifier = None
        annotation_sha256 = None
        annotation_path = getattr(self, 'loaded_annotations_path', None)
        if annotation_path is not None:
            annotation_path = Path(annotation_path).resolve()
            try:
                annotation_identifier = annotation_path.relative_to(
                    self._repository_root().resolve()
                ).as_posix()
            except ValueError as error:
                raise RuntimeError(
                    'annotation index must resolve inside the repository/staging root'
                ) from error
            annotation_sha256 = self._sha256(annotation_path)
        assignment_payload = json.dumps(
            partitions, sort_keys=True, separators=(',', ':')
        ).encode('utf-8')

        return {
            'manifest_source': manifest_source,
            'manifest_repo_relative_identifier': manifest_identifier,
            'manifest_sha256': manifest_sha256,
            'annotation_index_repo_relative_identifier': annotation_identifier,
            'annotation_index_sha256': annotation_sha256,
            'partition_assignment_sha256': hashlib.sha256(
                assignment_payload
            ).hexdigest(),
            'allocation_unit': 'leading_source_identifier_group',
            'observation_unit': '2048x2048_tile',
            'group_membership_map': group_membership_map,
            'specimen_independence_confirmation': 'pending_data_owner_confirmation',
            'group_semantics': (
                'filename_derived_acquisition_series_kept_wholly_within_one_partition'
            ),
            'partitions': partitions,
            'validation_only': bool(getattr(self, 'validation_only', False)),
            'held_out_dataset_constructed': bool(
                getattr(self, 'test_loader', None) is not None
            ),
            'held_out_evaluation_count': int(
                getattr(self, 'test_evaluation_count', 0)
            ),
        }

    def _resolved_run_config(self) -> Dict:
        """Return execution-critical fields embedded in every checkpoint."""
        return {
            'model': self._resolved_model_config(),
            'evaluation': {
                'mode': (
                    'validation_only'
                    if getattr(self, 'validation_only', False)
                    else 'confirmatory_with_held_out_test'
                ),
                'held_out_dataset_constructed': bool(
                    getattr(self, 'test_loader', None) is not None
                ),
                'held_out_evaluation_count': int(
                    getattr(self, 'test_evaluation_count', 0)
                ),
            },
            'input_normalization': dict(INPUT_NORMALIZATION),
            'input': dict(getattr(self, 'input_provenance', {})),
            'target': dict(getattr(self, 'target_provenance', {})),
            'data_split': self._resolved_data_split_config(),
            'source_code_sha256': self._source_code_sha256(),
            'augmentation': self._resolved_augmentation_config(),
            'loss': self._resolved_loss_config(),
            'inference': self._resolved_inference_config(),
            'selected_method_lock': (
                dict(self.selected_method_lock)
                if getattr(self, 'selected_method_lock', None) is not None
                else None
            ),
            'smoke_preflight_manifest': (
                {
                    key: self.smoke_preflight_manifest[key]
                    for key in (
                        'repo_relative_identifier', 'sha256', 'campaign_id'
                    )
                }
                if getattr(self, 'smoke_preflight_manifest', None) is not None
                else None
            ),
            'protocol_candidate_key': getattr(
                self, 'protocol_candidate_key', None
            ),
            'protocol_run_role': getattr(self, 'protocol_run_role', None),
            'protocol_campaign_id': getattr(
                self, 'protocol_campaign_id', None
            ),
            'protocol_cell_index': getattr(self, 'protocol_cell_index', None),
            'selected_architecture_role': getattr(
                self, 'selected_architecture_role', None
            ),
            'scientific_execution_contract': (
                self._resolved_scientific_execution_contract()
                if getattr(self, 'protocol_candidate_key', None) is not None
                else None
            ),
        }
    
    def _setup_distributed(self):
        """Setup distributed training."""
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(backend='nccl', init_method='env://')
        
        # Set device for this process
        torch.cuda.set_device(self.local_rank)
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set NCCL environment variables for optimization
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Always use tree algorithm
        if 'NCCL_IB_DISABLE' not in os.environ:
            os.environ['NCCL_IB_DISABLE'] = '0'  # Enable InfiniBand if available
        
        print(f"Rank {self.rank}/{self.world_size} initialized on GPU {self.local_rank}")
    
    def _setup_device(self):
        """Setup device with GPU optimization."""
        if self.multi_gpu:
            device = torch.device(f"cuda:{self.local_rank}")
            print(f"Process {self.rank} using GPU: {torch.cuda.get_device_name(self.local_rank)}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
            # Enable CUDA optimizations for single GPU
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Memory optimization settings
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        return device
    
    def _setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations."""
        if self.device.type != 'cuda':
            return
        
        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Set PyTorch memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Enable TF32 for Ampere GPUs (compute capability >= 8.0)
        if torch.cuda.get_device_capability()[0] >= 8:
            print("Ampere GPU detected - enabling TF32 operations")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # CUDA stream for async operations
        self.cuda_stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
    
    def setup_directories(self):
        """Create timestamped directories for this run."""
        protocol_role = getattr(self, 'protocol_run_role', None)
        immutable_protocol_cell = protocol_role in {
            'validation_screen_cell',
            'validation_smoke_cell',
            'selected_winner_retraining',
        }
        if immutable_protocol_cell:
            run_name = (
                f"{protocol_role}_{self.protocol_campaign_id}_"
                f"cell_{self.protocol_cell_index:02d}"
            )
            output_dir = (
                Path("results/patch_training/protocol_runs")
                / protocol_role
                / self.protocol_campaign_id
                / f"cell_{self.protocol_cell_index:02d}"
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
            if self.wandb_name:
                run_name = f"{run_name}_{self.wandb_name}"
            output_dir = Path("results/patch_training") / run_name
        self.run_name = run_name
        self.output_dir = output_dir
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plots_dir = self.output_dir / "plots"
        self.predictions_dir = self.output_dir / "plots"  # Use plots dir for predictions
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "plots"  # Use plots dir for visualizations
        self.config_dir = self.output_dir / "config"
        
        # Only create directories on rank 0
        if not self.multi_gpu or self.rank == 0:
            if immutable_protocol_cell:
                self.output_dir.parent.mkdir(parents=True, exist_ok=True)
                self.output_dir.mkdir(exist_ok=False)
            else:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            for dir_path in [self.checkpoint_dir, self.plots_dir,
                             self.metrics_dir, self.config_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def _save_config_info(self):
        """Save all configuration and system information for reproducibility."""
        import platform
        import subprocess
        import torch
        import yaml
        
        config_info = {
            'timestamp': datetime.now().isoformat(),
            'run_directory': str(self.output_dir),
            
            # Training configuration
            'training_config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'patch_size': self.patch_size,
                'training_patch_size': self.patch_size,
                'evaluation_patch_size': self.evaluation_patch_size,
                'training_batch_size': self.batch_size,
                'evaluation_batch_size': self.evaluation_batch_size,
                'num_workers': self.workers,
                'seed': self.seed,
                'max_batches_debug_limit': self.max_batches,
                'multi_gpu': self.multi_gpu,
                'world_size': self.world_size if self.multi_gpu and hasattr(self, 'world_size') else 1,
                'checkpoint_interval_legacy_argument': self.checkpoint_interval,
                'save_every_n_epochs_actual': self.save_every_n_epochs,
                'log_interval': self.log_interval,
                'mixed_precision_requested': self.mixed_precision,
                'mixed_precision_actual': getattr(self, 'scaler', None) is not None,
                'gradient_clip_val': self.gradient_clip_val,
                'gradient_accumulation_batches': self.accumulate_grad_batches,
                'early_stopping_patience': self.patience,
                'early_stopping_enabled': self.early_stopping_enabled,
                'resume_from': self.resume_path,
                'evaluation_mode': (
                    'validation_only'
                    if self.validation_only
                    else 'confirmatory_with_held_out_test'
                ),
                'held_out_dataset_constructed': bool(
                    getattr(self, 'test_loader', None) is not None
                ),
                'model_selection_metric': SELECTION_METRIC_NAME,
                'model_selection_metric_definition': SELECTION_METRIC_DEFINITION,
                'model_selection_tie_breaker': SELECTION_TIEBREAKER_NAME,
                'model_selection_tie_breaker_definition': (
                    SELECTION_TIEBREAKER_DEFINITION
                ),
                'model_selection_tertiary_tie_breaker': (
                    SELECTION_TERTIARY_TIEBREAKER_NAME
                ),
                'model_selection_tertiary_tie_breaker_definition': (
                    SELECTION_TERTIARY_TIEBREAKER_DEFINITION
                ),
                'selected_method_lock': (
                    dict(self.selected_method_lock)
                    if self.selected_method_lock is not None else None
                ),
            },
            
            # Model configuration
            'model_config': {
                **self._resolved_model_config(),
                'class_ids': CANONICAL_CLASS_NAMES if self.num_classes == 3 else {
                    0: 'disconnected_pore', 1: 'connected_pore'
                },
            },

            'source_code_sha256': self._source_code_sha256(),

            'input_normalization': dict(INPUT_NORMALIZATION),
            'input_provenance': dict(
                getattr(self, 'input_provenance', {})
            ),
            
            # Augmentation configuration
            'augmentation_config': {
                **self.augmentation_config,
                'resolved': self._resolved_augmentation_config(),
                'use_mixup': self.use_mixup,
                'use_cutmix': self.use_cutmix,
                'batch_mixing_probability': self.batch_mixing_probability,
                'mixup_choice_probability_when_both_enabled': self.mixup_prob,
            },
            
            # Loss configuration
            'loss_config': {
                **self._resolved_loss_config(),
                'focal_gamma': self.focal_gamma,
                'tversky_alpha': self.tversky_alpha,
                'tversky_beta': self.tversky_beta,
            },
            
            # Optimizer configuration
            'optimizer_config': {
                'type': type(self.optimizer).__name__ if hasattr(self, 'optimizer') else self.optimizer_type,
                'configured_base_learning_rate': self.learning_rate,
                'distributed_scaled_learning_rate': self.learning_rate * (
                    self.world_size if self.multi_gpu else 1
                ),
                'weight_decay': self.weight_decay,
                'momentum': self.momentum if self.optimizer_type in {'sgd', 'rmsprop'} else None,
                'scheduler': (
                    type(self.scheduler).__name__ if getattr(self, 'scheduler', None) is not None else None
                ),
                'scheduler_requested': self.scheduler_type,
            },

            # Data provenance and leakage controls
            'data_config': {
                'annotations_path': str(self.loaded_annotations_path) if self.loaded_annotations_path else None,
                'annotations_sha256': self._sha256(self.loaded_annotations_path),
                'image_dir': str(self.image_dir),
                'input_provenance': dict(self.input_provenance),
                'target_source': self.target_provenance.get('target_source'),
                'mask_dir': (
                    str(self.mask_dir) if self.mask_dir is not None else None
                ),
                'mask_count': self.target_provenance.get('mask_count'),
                'mask_aggregate_sha256': self.target_provenance.get(
                    'mask_aggregate_sha256'
                ),
                'target_provenance': self.target_provenance,
                'split_manifest': self.split_manifest,
                'split_manifest_sha256': self._sha256(Path(self.split_manifest)) if self.split_manifest else None,
                'generated_val_fraction': self.val_split,
                'generated_test_fraction': self.test_split,
                'split_image_ids': self.split_ids,
                'split_image_files': self.split_files,
                'source_to_canonical_category_id': self.category_id_map,
                'split_unit': 'COCO image',
                'mineral_label_policy': (
                    'source_255_mapped_to_class_2'
                    if self.mask_dir is not None and self.num_classes == 3
                    else 'source_255_mapped_to_ignore_index'
                    if self.mask_dir is not None
                    else 'class_2_for_unannotated_pixels'
                    if self.num_classes == 3
                    else 'ignore_unannotated_pixels'
                ),
            },

            # Retain the invocation separately from the resolved settings above.
            # The resolved sections are authoritative for what the trainer used.
            'requested_cli_arguments': self.requested_cli_arguments,
            
            # System information
            'system_info': {
                'hostname': platform.node(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
                'cudnn_benchmark': torch.backends.cudnn.benchmark if torch.cuda.is_available() else False,
                'tf32_matmul': torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            
            # Environment variables
            'environment': {
                'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID', 'N/A'),
                'SLURM_JOB_NAME': os.environ.get('SLURM_JOB_NAME', 'N/A'),
                'SLURM_NODELIST': os.environ.get('SLURM_NODELIST', 'N/A'),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A'),
            },
            
            # Original config file
            'pipeline_config': self.config.config if hasattr(self.config, 'config') else {},
        }
        
        # Save as YAML for readability
        with open(self.config_dir / 'training_config.yaml', 'w') as f:
            yaml.dump(config_info, f, default_flow_style=False, sort_keys=False)
        
        # Save as JSON for programmatic access
        try:
            with open(self.config_dir / 'training_config.json', 'w') as f:
                json.dump(convert_to_json_serializable(config_info), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save JSON config: {e}")
            # Try to save a minimal version
            try:
                minimal_config = {
                    'timestamp': config_info.get('timestamp'),
                    'run_directory': str(config_info.get('run_directory')),
                    'training_config': {
                        'epochs': config_info.get('training_config', {}).get('epochs'),
                        'batch_size': config_info.get('training_config', {}).get('batch_size'),
                        'learning_rate': config_info.get('training_config', {}).get('learning_rate'),
                    }
                }
                with open(self.config_dir / 'training_config_minimal.json', 'w') as f:
                    json.dump(minimal_config, f, indent=2)
            except:
                pass
        
        # Copy original config file if it exists
        if self.config.get('config_path'):
            import shutil
            shutil.copy2(self.config['config_path'], self.config_dir / 'original_pipeline_config.yaml')
        
        # Save pip freeze for exact package versions
        try:
            pip_freeze = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
            with open(self.config_dir / 'pip_freeze.txt', 'w') as f:
                f.write(pip_freeze)
        except:
            pass
        
        # Save conda environment if available
        try:
            conda_export = subprocess.check_output(['conda', 'env', 'export']).decode('utf-8')
            with open(self.config_dir / 'conda_environment.yaml', 'w') as f:
                f.write(conda_export)
        except:
            pass
        
        print(f"\nConfiguration saved to: {self.config_dir}")
    
    def load_coco_data(self):
        """Load one authoritative COCO file and record its provenance."""
        coco_dir = Path("results/step3_coco_dataset")
        if self.annotations_path is not None:
            candidates = [self.annotations_path]
        elif self.num_classes == 2:
            candidates = [
                coco_dir / "pore_annotations.json",
                coco_dir / "annotations.json.gz",
                coco_dir / "annotations.json",
            ]
        else:
            # A three-class polygon file is preferred. The pore-only file is
            # also valid because PatchDataset explicitly fills all unannotated
            # pixels with the canonical mineral class (2).
            candidates = [
                coco_dir / "annotations.json",
                coco_dir / "annotations.json.gz",
                coco_dir / "pore_annotations.json",
            ]

        annotation_file = next((path for path in candidates if path.exists()), None)
        if annotation_file is None:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"No COCO annotations found. Searched: {searched}")

        print(f"Loading annotations from {annotation_file}...")
        if annotation_file.suffix == ".gz":
            with gzip.open(annotation_file, "rt", encoding="utf-8") as handle:
                train_data = json.load(handle)
        else:
            with annotation_file.open("r", encoding="utf-8") as handle:
                train_data = json.load(handle)
        self.loaded_annotations_path = annotation_file
        return train_data
    
    def _merge_coco_chunks(self, chunk_files):
        """Merge multiple COCO chunk files."""
        merged_data = None
        
        for i, chunk_file in enumerate(chunk_files):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            if i == 0:
                merged_data = chunk_data
            else:
                merged_data['images'].extend(chunk_data['images'])
                merged_data['annotations'].extend(chunk_data['annotations'])
        
        return merged_data
    
    def create_data_loaders(self):
        """Create train/val/test loaders from mutually exclusive images."""
        # Load COCO data
        coco_data = self.load_coco_data()

        manifest_source = self.split_manifest
        if manifest_source is None:
            annotation_dir = self.loaded_annotations_path.parent
            manifest_candidates = [
                annotation_dir / "pore_splits.json",
                annotation_dir / "splits.json",
            ]
            existing_manifest = next((path for path in manifest_candidates if path.exists()), None)
            manifest_source = str(existing_manifest) if existing_manifest else None

        if manifest_source is not None:
            self.split_ids = resolve_split_manifest(coco_data, manifest_source)
            self.split_manifest = str(manifest_source)
        else:
            self.split_ids = create_deterministic_image_splits(
                coco_data,
                val_split=self.val_split,
                test_split=self.test_split,
                seed=self.seed,
            )
            self.split_manifest = None

        filename_by_id = {
            int(image['id']): str(image['file_name']) for image in coco_data['images']
        }
        self.split_files = {
            split_name: [filename_by_id[image_id] for image_id in image_ids]
            for split_name, image_ids in self.split_ids.items()
        }
        
        # Create patch data loaders
        return_test = not self.validation_only
        if self.multi_gpu:
            # For distributed training, each process gets a subset
            loaders = create_patch_data_loaders(
                coco_data, str(self.image_dir),
                batch_size=self.batch_size // self.world_size,  # Divide batch size by world size
                split_manifest=self.split_ids,
                split_seed=self.seed,
                distributed=True,
                rank=self.rank,
                world_size=self.world_size,
                augmentation_config=self.augmentation_config,
                bootstrap_factor=int(os.environ.get('BOOTSTRAP_FACTOR', '3')),  # Configurable bootstrap factor
                patch_size=self.patch_size,
                evaluation_patch_size=self.evaluation_patch_size,
                evaluation_batch_size=self.evaluation_batch_size,
                num_classes=self.num_classes,
                num_workers=self.workers,
                return_test=return_test,
                mask_dir=self.mask_dir,
            )
        else:
            loaders = create_patch_data_loaders(
                coco_data, str(self.image_dir),
                batch_size=self.batch_size,
                split_manifest=self.split_ids,
                split_seed=self.seed,
                augmentation_config=self.augmentation_config,
                bootstrap_factor=int(os.environ.get('BOOTSTRAP_FACTOR', '3')),  # Configurable bootstrap factor
                patch_size=self.patch_size,
                evaluation_patch_size=self.evaluation_patch_size,
                evaluation_batch_size=self.evaluation_batch_size,
                num_classes=self.num_classes,
                num_workers=self.workers,
                return_test=return_test,
                mask_dir=self.mask_dir,
            )

        if self.validation_only:
            train_loader, val_loader = loaders
            test_loader = None
        else:
            train_loader, val_loader, test_loader = loaders

        self.target_provenance = dict(train_loader.dataset.target_provenance)
        self.target_provenance['annotations_role'] = (
            'image_index_and_metadata_only'
            if self.mask_dir is not None
            else 'segmentation_targets'
        )
        self.target_provenance['evaluation_mode'] = (
            'train_validation_only' if self.validation_only else 'confirmatory_with_test'
        )
        self.target_provenance['held_out_dataset_constructed'] = bool(
            test_loader is not None
        )
        return train_loader, val_loader, test_loader
    
    def create_model_and_optimizer(self):
        """Create model optimized for patches with multi-GPU support."""
        # Clear GPU cache before creating model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Create model using specified model type
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
            print(f"[GPU Memory Before model creation] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
        # Create model based on type
        if self.model_type == 'multiscale_attention_unet':
            self.model = create_multiscale_attention_unet(self.config)
        elif self.model_type == 'multiscale_attention_unet_pyramid':
            self.model = create_multiscale_attention_unet_pyramid(self.config)
        elif self.model_type in ['segformer_b0', 'segformer_b1', 'segformer_b2', 
                                  'dinov2_small', 'dinov2_base', 'dinov2_large'] and ADVANCED_MODELS_AVAILABLE:
            # Use advanced models from model_factory
            self.model = create_advanced_model(
                self.model_type,
                num_classes=self.num_classes,
                dropout=self.config.get('dropout', 0.1),
                freeze_backbone=self.config.get('freeze_encoder', False)
            )
        else:
            self.model = create_model(
                self.model_type,
                num_classes=self.num_classes,
                in_channels=self.model_input_channels,
            )
        if self.model_type in {
            'multiscale_attention_unet',
            'multiscale_attention_unet_pyramid',
        } and bool(getattr(self.model, 'deep_supervision', False)):
            raise RuntimeError(
                "multiscale deep_supervision must be false because PatchTrainer "
                "losses require one logits tensor"
            )
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
            print(f"[GPU Memory After model creation] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
        self.model = self.model.to(self.device)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
            print(f"[GPU Memory After model to device] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
        
        # Apply mixed precision only if CUDA is available and not in test mode
        # Temporarily disable mixed precision due to dtype issues
        use_amp = self.mixed_precision and torch.cuda.is_available() and os.environ.get('DISABLE_AMP', '0') != '1'
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Wrap model for distributed training
        if self.multi_gpu:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                           output_device=self.local_rank,
                           find_unused_parameters=False)
        
        # Loss function with class weights from config
        # Add class weights to config if provided
        if self.class_weights:
            # Store original values to restore later
            self.config.update('model.loss.class_weights', self.class_weights)
            if not self.multi_gpu or self.rank == 0:
                print(f"Using class weights: {self.class_weights}")
        
        # Use specified loss type
        if self.num_classes == 2 and self.loss_type == 'conditional_pore_focal_dice':
            statistics = getattr(self, 'training_class_statistics', None)
            if not statistics or statistics.get('source') != (
                'authoritative_training_masks_only'
            ):
                raise RuntimeError(
                    "conditional_pore_focal_dice requires prospectively recorded "
                    "authoritative training-mask class counts"
                )
            self.criterion = create_conditional_pore_focal_dice_loss(
                training_class_counts=statistics['counts']
            ).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print(
                    "Using conditional C0/C1 focal plus macro-Dice loss; "
                    "mineral targets use ignore_index=-100"
                )
        elif self.num_classes == 2:
            # Use binary pore loss for 2-class problem
            from ..models.binary_pore_loss import create_binary_pore_loss
            self.criterion = create_binary_pore_loss(self.config).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print(f"Using binary pore loss function for {self.num_classes} classes")
        elif self.loss_type == 'hierarchical_pore_connectivity':
            statistics = getattr(self, 'training_class_statistics', None)
            if not statistics or statistics.get('source') != (
                'authoritative_training_masks_only'
            ):
                raise RuntimeError(
                    "hierarchical_pore_connectivity requires prospectively "
                    "recorded authoritative training-mask class counts"
                )
            self.criterion = create_hierarchical_pore_connectivity_loss(
                training_class_counts=statistics['counts'],
                component_weights=(1.0, 1.0, 1.0),
            ).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print(
                    "Using hierarchical pore-connectivity loss with equal "
                    "normalized region, pore-union, and conditional components"
                )
        elif self.loss_type == 'focal_dice':
            try:
                from ..models.focal_loss import create_advanced_loss_function
                self.criterion = create_advanced_loss_function(self.config).to(self.device)
                if not self.multi_gpu or self.rank == 0:
                    print("Using advanced loss function (focal_dice)")
            except ImportError:
                # Fall back to original loss
                self.criterion = create_loss_function(self.config).to(self.device)
                if not self.multi_gpu or self.rank == 0:
                    print("Using standard loss function (fallback from focal_dice)")
        elif self.loss_type == 'asymmetric':
            # Use asymmetric loss for better handling of disconnected pores
            from ..models.asymmetric_loss import create_asymmetric_loss
            self.criterion = create_asymmetric_loss(self.config).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print("Using asymmetric loss function (heavily penalizes FN for class 0)")
        elif self.loss_type == 'mineral_aware':
            # Use mineral-aware loss that penalizes based on patch statistics
            from ..models.mineral_aware_loss import create_mineral_aware_loss
            self.criterion = create_mineral_aware_loss(self.config).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print("Using mineral-aware loss function")
        elif self.loss_type == 'sparse_pore':
            # Use sparse-aware loss that heavily penalizes false positives
            from ..models.sparse_pore_loss import create_sparse_pore_loss
            self.criterion = create_sparse_pore_loss(self.config).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print("Using sparse pore loss function (heavy FP penalty)")
        elif self.loss_type == 'boundary_aware':
            # Use boundary-aware loss for better pore-mineral separation
            self.criterion = create_boundary_aware_loss(self.config).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print("Using boundary-aware loss function")
        elif self.loss_type == 'topological':
            # Use topological consistency loss for preserving connectivity
            self.criterion = create_topological_loss(self.config).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print("Using topological consistency loss function")
        elif self.loss_type in ['focal_tversky', 'lovasz', 'focal_lovasz', 'focal_dice'] and ADVANCED_MODELS_AVAILABLE:
            # Use advanced loss functions from model_factory
            self.criterion = create_advanced_loss(
                self.loss_type,
                num_classes=self.num_classes,
                class_weights=self.class_weights,
                focal_gamma=self.config.get('focal_gamma', 2.0),
                tversky_alpha=self.config.get('tversky_alpha', 0.7),
                tversky_beta=self.config.get('tversky_beta', 0.3)
            ).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print(f"Using advanced {self.loss_type} loss function")
        else:
            # Use standard combined loss
            self.criterion = create_loss_function(self.config).to(self.device)
            if not self.multi_gpu or self.rank == 0:
                print("Using standard combined loss function")
        
        effective_lr = self.learning_rate * (self.world_size if self.multi_gpu else 1)
        optimizer_kwargs = {
            "lr": effective_lr,
            "weight_decay": self.weight_decay,
        }
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_kwargs)
        elif self.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                **optimizer_kwargs,
                fused=torch.cuda.is_available(),
            )
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), **optimizer_kwargs, momentum=self.momentum
            )
        elif self.optimizer_type == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), **optimizer_kwargs, momentum=self.momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        self.scheduler_step_per_batch = self.scheduler_type == "onecycle"
        if self.scheduler_type == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=effective_lr,
                epochs=self.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,
                anneal_strategy="cos",
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.epochs)
            )
        elif self.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.epochs // 3), gamma=0.1
            )
        elif self.scheduler_type == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        elif self.scheduler_type == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
        
        # Load checkpoint if resuming
        if self.resume_path:
            self.load_checkpoint(self.resume_path)

        # Count parameters after the complete model/optimizer setup.
        base_model = self.model.module if self.multi_gpu else self.model
        total_params = sum(p.numel() for p in base_model.parameters())
        if not self.multi_gpu or self.rank == 0:
            print(f"Model parameters: {total_params:,}")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
                print(f"[GPU Memory After full model setup] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")

    def _prepare_model_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Add the exact recovered pore-gate channel for conditional models."""
        if self.loss_type != 'conditional_pore_focal_dice':
            return images
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                "conditional model input preparation requires one grayscale channel"
            )
        threshold = self.conditional_pore_threshold
        if threshold is None or not self.recovered_threshold_acknowledged:
            raise RuntimeError(
                "conditional inputs require an acknowledged recovered threshold rule"
            )
        normalized_threshold = (2.0 * int(threshold) / 255.0) - 1.0
        gate = (images < normalized_threshold).to(dtype=images.dtype)
        return torch.cat((images, gate), dim=1)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch on patches with mixed precision."""
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        batch_times = []
        
        # Check for test mode (only run one batch)
        test_mode = os.environ.get('QUICK_TEST', '0') == '1'
        single_batch = os.environ.get('SINGLE_BATCH', '0') == '1'
        if test_mode and (not self.multi_gpu or self.rank == 0):
            print("\n⚡ QUICK TEST MODE: Running only 1 batch")
        if single_batch and (not self.multi_gpu or self.rank == 0):
            print("\n⚡ SINGLE BATCH MODE: Running only 1 batch then stopping")
        
        # Setup progress bar on rank 0 only
        if not self.multi_gpu or self.rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        else:
            pbar = train_loader
        
        for batch_idx, batch_data in enumerate(pbar):
            # Check if we've reached max_batches limit (for testing)
            if self.max_batches is not None and batch_idx >= self.max_batches:
                if not self.multi_gpu or self.rank == 0:
                    print(f"Reached max_batches limit ({self.max_batches}), stopping epoch early")
                break
            
            batch_start = time.time()
            
            # Unpack batch data
            images, masks, _ = batch_data
            
            # Debug GPU memory for first batch
            if batch_idx == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
                print(f"[GPU Memory Start of epoch {epoch+1}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
            
            # Always ensure tensors are on the correct device
            if images.device != self.device:
                images = images.to(self.device, non_blocking=True)
            if masks.device != self.device:
                masks = masks.to(self.device, non_blocking=True)
            
            # Debug after data transfer
            if batch_idx == 0:
                print(f"Batch shape: {images.shape}, dtype: {images.dtype}")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
                    print(f"[GPU Memory After data to device] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
            
            # Apply mixup or cutmix with some probability
            if (
                self.model.training
                and (self.use_mixup or self.use_cutmix)
                and np.random.rand() < self.batch_mixing_probability
            ):
                if self.use_mixup and self.use_cutmix:
                    # Randomly choose between mixup and cutmix
                    use_mixup = np.random.rand() < self.mixup_prob
                else:
                    use_mixup = self.use_mixup
                
                if use_mixup:
                    # Apply mixup
                    images, masks, lam = self.batch_augmentor.mixup(images, masks)
                else:
                    # Apply cutmix
                    images, masks, lam = self.batch_augmentor.cutmix(images, masks)
                
                # Adjust loss computation for mixed samples
                mixed_loss = True
            else:
                mixed_loss = False
                lam = None
            
                # Zero gradients (set_to_none is more memory efficient)
            self.optimizer.zero_grad(set_to_none=True)
            model_inputs = self._prepare_model_inputs(images)
            
            # Mixed precision forward pass
            if self.device.type == 'cuda' and self.scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model(model_inputs)
                    if batch_idx == 0 and torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
                        print(f"[GPU Memory After forward pass] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
                    if mixed_loss and lam is not None:
                        # For mixed samples, we already have mixed masks
                        loss = self.criterion(outputs, masks)
                    else:
                        loss = self.criterion(outputs, masks)
            else:
                # No mixed precision
                outputs = self.model(model_inputs)
                if mixed_loss and lam is not None:
                    loss = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)

            if not torch.isfinite(outputs).all():
                raise FloatingPointError(
                    f"non-finite model output in train epoch {epoch + 1}, "
                    f"batch {batch_idx}"
                )
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f"non-finite loss in train epoch {epoch + 1}, batch {batch_idx}"
                )
            
            # Backward pass with gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if batch_idx == 0 and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
                    print(f"[GPU Memory After backward pass] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # No mixed precision
                loss.backward()
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)
                self.optimizer.step()
            
            # OneCycleLR advances per optimizer step. Other schedulers advance
            # once per epoch in train().
            if self.scheduler is not None and self.scheduler_step_per_batch:
                self.scheduler.step()
            
            # Calculate accuracy
            with torch.no_grad():
                _, predicted = outputs.max(1)
                if self.num_classes == 2:
                    # For 2-class, exclude ignore pixels from accuracy
                    valid_mask = masks != -100
                    total += valid_mask.sum().item()
                    correct += (predicted.eq(masks) & valid_mask).sum().item()
                else:
                    # For 3-class, calculate normally
                    total += masks.numel()
                    correct += predicted.eq(masks).sum().item()
            
            # Update metrics
            epoch_loss += loss.item()
            accuracy = 100. * correct / total
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Log every N batches
            if batch_idx % self.log_interval == 0 and (not self.multi_gpu or self.rank == 0):
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{accuracy:.2f}%',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                        'batch_time': f'{batch_time:.3f}s'
                    })
            
            # Break after one batch in test mode or single batch mode
            if test_mode or single_batch:
                if not self.multi_gpu or self.rank == 0:
                    mode_name = "Single batch" if single_batch else "Test"
                    print(f"\n⚡ {mode_name} batch complete: loss={loss.item():.4f}, acc={accuracy:.2f}%")
                break
        
        # Synchronize metrics across GPUs
        if self.multi_gpu:
            # Reduce metrics across all processes
            metrics_tensor = torch.tensor([epoch_loss, correct, total], device=self.device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = metrics_tensor[0].item()
            correct = metrics_tensor[1].item()
            total = metrics_tensor[2].item()
        
        avg_loss = epoch_loss / len(train_loader) / (self.world_size if self.multi_gpu else 1)
        avg_acc = 100. * correct / total
        avg_batch_time = np.mean(batch_times)
        
        return avg_loss, avg_acc, avg_batch_time
    
    def validate(self, val_loader, epoch, phase: str = "Val"):
        """Evaluate a non-training loader with comprehensive metrics."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        # Check for test mode
        test_mode = os.environ.get('QUICK_TEST', '0') == '1'
        single_batch = os.environ.get('SINGLE_BATCH', '0') == '1'
        conditional_gate = self.loss_type == 'conditional_pore_focal_dice'
        if conditional_gate:
            threshold = getattr(self, 'conditional_pore_threshold', None)
            if threshold is None or not getattr(
                self, 'recovered_threshold_acknowledged', False
            ):
                raise RuntimeError(
                    "conditional validation requires an acknowledged recovered "
                    "raw-uint8 threshold rule"
                )
            metric_num_classes = 3
            normalized_gate_threshold = (2.0 * int(threshold) / 255.0) - 1.0
        else:
            metric_num_classes = self.num_classes
            normalized_gate_threshold = None
        
        # For IoU calculation - based on num_classes
        intersection = torch.zeros(metric_num_classes).to(self.device)
        union = torch.zeros(metric_num_classes).to(self.device)
        
        # Accumulate the confusion matrix directly. In distributed runs this is
        # reduced across ranks, so model selection is based on the complete
        # validation split rather than rank 0's shard.
        confusion_counts = torch.zeros(
            (metric_num_classes, metric_num_classes),
            dtype=torch.long,
            device=self.device,
        )
        
        # Setup progress bar on rank 0 only
        if not self.multi_gpu or self.rank == 0:
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [{phase}]")
        else:
            pbar = val_loader
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                # Check if we've reached max_batches limit (for testing)
                if self.max_batches is not None and batch_idx >= self.max_batches:
                    if not self.multi_gpu or self.rank == 0:
                        print(f"Reached max_batches limit ({self.max_batches}), stopping validation early")
                    break
                
                # Unpack batch data
                images, masks, patch_ids = batch_data
                
                # Always ensure tensors are on the correct device
                if images.device != self.device:
                    images = images.to(self.device, non_blocking=True)
                if masks.device != self.device:
                    masks = masks.to(self.device, non_blocking=True)
                model_inputs = self._prepare_model_inputs(images)
                
                # Mixed precision inference
                if self.device.type == 'cuda' and self.scaler is not None:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = self.model(model_inputs)
                        loss = self.criterion(outputs, masks)
                else:
                    # No mixed precision
                    outputs = self.model(model_inputs)
                    loss = self.criterion(outputs, masks)

                if not torch.isfinite(outputs).all():
                    raise FloatingPointError(
                        f"non-finite model output in {phase} epoch {epoch + 1}, "
                        f"batch {batch_idx}"
                    )
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        f"non-finite loss in {phase} epoch {epoch + 1}, batch {batch_idx}"
                    )
                
                _, predicted = outputs.max(1)
                if conditional_gate:
                    # Compose a fair three-class validation prediction. The
                    # two-output network decides C0/C1 only below the recovered
                    # raw-intensity gate; all other pixels become C2.
                    pore_gate = images[:, 0] < normalized_gate_threshold
                    metric_predictions = torch.where(
                        pore_gate,
                        predicted,
                        torch.full_like(predicted, 2),
                    )
                    metric_targets = masks.clone()
                    metric_targets[metric_targets == -100] = 2
                    total += metric_targets.numel()
                    correct += metric_predictions.eq(metric_targets).sum().item()
                elif self.num_classes == 2:
                    # For 2-class, exclude ignore pixels from accuracy
                    valid_mask = masks != -100
                    total += valid_mask.sum().item()
                    correct += (predicted.eq(masks) & valid_mask).sum().item()
                    metric_predictions = predicted
                    metric_targets = masks
                else:
                    # For 3-class, calculate normally
                    total += masks.numel()
                    correct += predicted.eq(masks).sum().item()
                    metric_predictions = predicted
                    metric_targets = masks
                
                # Accumulate predictions for the confusion matrix.
                if self.num_classes == 2 and not conditional_gate:
                    valid_targets = metric_targets[
                        metric_targets != -100
                    ].reshape(-1)
                    valid_predictions = metric_predictions[
                        metric_targets != -100
                    ].reshape(-1)
                else:
                    valid_targets = metric_targets.reshape(-1)
                    valid_predictions = metric_predictions.reshape(-1)
                encoded_pairs = (
                    valid_targets.long() * metric_num_classes
                    + valid_predictions.long()
                )
                confusion_counts += torch.bincount(
                    encoded_pairs,
                    minlength=metric_num_classes * metric_num_classes,
                ).reshape(metric_num_classes, metric_num_classes)
                
                # Calculate IoU for all classes, handling ignore pixels for 2-class case
                if self.num_classes == 2 and not conditional_gate:
                    # For 2-class, filter out ignore pixels (-100)
                    valid_mask = masks != -100
                    valid_predicted = predicted[valid_mask]
                    valid_masks = masks[valid_mask]
                    
                    for cls in range(self.num_classes):
                        pred_cls = (valid_predicted == cls)
                        target_cls = (valid_masks == cls)
                        intersection[cls] += (pred_cls & target_cls).sum().item()
                        union[cls] += (pred_cls | target_cls).sum().item()
                else:
                    # Native or composed three-class prediction.
                    for cls in range(metric_num_classes):
                        pred_cls = (metric_predictions == cls)
                        target_cls = (metric_targets == cls)
                        intersection[cls] += (pred_cls & target_cls).sum().item()
                        union[cls] += (pred_cls | target_cls).sum().item()
                
                val_loss += loss.item()
                
                # Break after one batch in test mode or single batch mode
                if (test_mode or single_batch) and batch_idx == 0:
                    if not self.multi_gpu or self.rank == 0:
                        mode_name = "Single batch" if single_batch else "Test"
                        print(f"\n⚡ {mode_name} validation batch complete: loss={loss.item():.4f}")
                    break
        
        # Synchronize metrics across GPUs
        if self.multi_gpu:
            # Reduce metrics
            metrics_tensor = torch.tensor([val_loss, correct, total], device=self.device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            val_loss = metrics_tensor[0].item()
            correct = metrics_tensor[1].item()
            total = metrics_tensor[2].item()
            
            # Reduce IoU metrics
            dist.all_reduce(intersection, op=dist.ReduceOp.SUM)
            dist.all_reduce(union, op=dist.ReduceOp.SUM)
            dist.all_reduce(confusion_counts, op=dist.ReduceOp.SUM)
        
        avg_loss = val_loss / len(val_loader) / (self.world_size if self.multi_gpu else 1)
        avg_acc = 100. * correct / total
        
        # Calculate mean IoU
        iou = intersection / (union + 1e-8)
        mean_iou = iou.mean().item()
        
        # Calculate identical additional metrics on every rank. This keeps the
        # validation-selection decision synchronized in DDP jobs.
        cm = confusion_counts.detach().cpu().numpy()
        additional_metrics = {}
        if cm.sum() > 0:
            
            # Per-class metrics
            if metric_num_classes == 2:
                class_names = ['disconnected_pore', 'connected_pore']
            else:
                class_names = ['disconnected_pore', 'connected_pore', 'mineral']
            per_class_metrics = {}
            
            # Calculate metrics for each class
            for i, class_name in enumerate(class_names):
                # True positives, false positives, false negatives, true negatives
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                # Per-class metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                specificity = tn / (tn + fp + 1e-8)
                
                per_class_metrics[class_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'specificity': specificity,
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tn': int(tn)
                }
            
            # Overall metrics
            overall_precision = np.mean([per_class_metrics[cn]['precision'] for cn in class_names])
            overall_recall = np.mean([per_class_metrics[cn]['recall'] for cn in class_names])
            overall_f1 = np.mean([per_class_metrics[cn]['f1_score'] for cn in class_names])
            
            # Weighted metrics (by class frequency)
            class_weights = cm.sum(axis=1) / cm.sum()
            weighted_precision = sum(per_class_metrics[cn]['precision'] * class_weights[i] 
                                   for i, cn in enumerate(class_names))
            weighted_recall = sum(per_class_metrics[cn]['recall'] * class_weights[i] 
                                for i, cn in enumerate(class_names))
            weighted_f1 = sum(per_class_metrics[cn]['f1_score'] * class_weights[i] 
                            for i, cn in enumerate(class_names))
            
            additional_metrics = {
                'confusion_matrix': cm,
                'per_class_metrics': per_class_metrics,
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1': overall_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'class_distribution': class_weights,
                'evaluation_num_classes': metric_num_classes,
                'inference_config': self._resolved_inference_config(),
            }
        else:
            additional_metrics = {
                'confusion_matrix': cm,
                'per_class_metrics': {},
                'overall_precision': 0.0,
                'overall_recall': 0.0,
                'overall_f1': 0.0,
                'weighted_precision': 0.0,
                'weighted_recall': 0.0,
                'weighted_f1': 0.0,
                'class_distribution': np.zeros(metric_num_classes, dtype=float),
                'evaluation_num_classes': metric_num_classes,
                'inference_config': self._resolved_inference_config(),
            }
        
        return avg_loss, avg_acc, mean_iou, iou.cpu().numpy(), additional_metrics
    
    def _selection_metric_components(self, class_iou, additional_metrics):
        """Return the pore-only validation score and deterministic tie-break."""
        class_iou = np.asarray(class_iou, dtype=float).reshape(-1)
        if class_iou.size < 2:
            raise ValueError("Model selection requires IoU values for both pore classes")
        c0_iou = float(class_iou[0])
        c1_iou = float(class_iou[1])
        if (
            not np.isfinite([c0_iou, c1_iou]).all()
            or not 0.0 <= c0_iou <= 1.0
            or not 0.0 <= c1_iou <= 1.0
        ):
            raise ValueError(
                "Validation pore IoU values must be finite and in [0, 1]: "
                f"C0={c0_iou}, C1={c1_iou}"
            )

        score = 2.0 * c0_iou * c1_iou / (c0_iou + c1_iou + 1e-8)

        confusion = np.asarray(
            (additional_metrics or {}).get('confusion_matrix'), dtype=float
        )
        if confusion.ndim != 2 or confusion.shape[0] < 2 or confusion.shape[1] < 2:
            raise ValueError(
                "Pore-union tie-break requires a validation confusion matrix"
            )
        if not np.isfinite(confusion).all() or (confusion < 0).any():
            raise ValueError("Validation confusion matrix must be finite and non-negative")
        if confusion.shape[0] == 2 and confusion.shape[1] == 2:
            pore_union = float(confusion[:2, :2].sum())
            pore_union_iou = 1.0 if pore_union > 0 else 0.0
        else:
            if confusion.shape[0] < 3 or confusion.shape[1] < 3:
                raise ValueError(
                    "Three-class pore-union tie-break requires a 3x3 confusion matrix"
                )
            true_positive = float(confusion[:2, :2].sum())
            false_positive = float(confusion[2:, :2].sum())
            false_negative = float(confusion[:2, 2:].sum())
            pore_union = true_positive + false_positive + false_negative
            if pore_union <= 0:
                raise ValueError(
                    "Validation partition has no pore-union pixels or predictions"
                )
            pore_union_iou = true_positive / pore_union

        return {
            'name': SELECTION_METRIC_NAME,
            'definition': SELECTION_METRIC_DEFINITION,
            'score': score,
            'c0_iou': c0_iou,
            'c1_iou': c1_iou,
            'tie_breaker_name': SELECTION_TIEBREAKER_NAME,
            'tie_breaker_definition': SELECTION_TIEBREAKER_DEFINITION,
            'tie_breaker_score': pore_union_iou,
            'pore_union_iou': pore_union_iou,
        }

    def _record_validation_selection(self, epoch, val_loss, class_iou, additional_metrics):
        """Capture the best validation-selected model state in CPU memory."""
        selection = self._selection_metric_components(class_iou, additional_metrics)
        selection['epoch'] = int(epoch) + 1
        selection['validation_loss'] = float(val_loss)
        if not np.isfinite(selection['validation_loss']):
            raise ValueError("Validation loss used for selection must be finite")
        selection['tertiary_tie_breaker_name'] = (
            SELECTION_TERTIARY_TIEBREAKER_NAME
        )
        selection['tertiary_tie_breaker_definition'] = (
            SELECTION_TERTIARY_TIEBREAKER_DEFINITION
        )
        best_tie_breaker = float(
            (self.best_selection_components or {}).get(
                'tie_breaker_score', float('-inf')
            )
        )
        best_validation_loss = float(
            (self.best_selection_components or {}).get(
                'validation_loss', float('inf')
            )
        )
        selection_key = (
            selection['score'],
            selection['tie_breaker_score'],
            -selection['validation_loss'],
        )
        best_key = (
            self.best_val_metric,
            best_tie_breaker,
            -best_validation_loss,
        )
        selection['selection_key'] = list(selection_key)
        selection['improved'] = selection_key > best_key
        if not selection['improved']:
            selection['improvement_reason'] = 'not_improved'
        elif self.best_selection_components is None:
            selection['improvement_reason'] = 'initial_finite_selection'
        elif selection['score'] > self.best_val_metric:
            selection['improvement_reason'] = 'higher_c0_c1_harmonic_iou'
        elif selection['tie_breaker_score'] > best_tie_breaker:
            selection['improvement_reason'] = 'higher_pore_union_iou_exact_primary_tie'
        else:
            selection['improvement_reason'] = 'lower_validation_loss_exact_primary_secondary_tie'

        if selection['improved']:
            self.best_val_metric = selection['score']
            self.best_selection_epoch = selection['epoch']
            self.best_selection_components = {
                key: value for key, value in selection.items() if key != 'improved'
            }
            base_model = self.model.module if self.multi_gpu else self.model
            self._best_selection_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in base_model.state_dict().items()
            }
        return selection

    def _checkpoint_payload(self, epoch, val_loss, checkpoint_role):
        """Build a self-describing checkpoint for reproducible restoration."""
        finite_metric_values = {'validation_loss_at_epoch': float(val_loss)}
        if self.best_selection_epoch is not None:
            finite_metric_values['best_validation_selection_score'] = float(
                self.best_val_metric
            )
        invalid_metrics = {
            name: value
            for name, value in finite_metric_values.items()
            if not np.isfinite(value)
        }
        if invalid_metrics:
            raise FloatingPointError(
                f"checkpoint metrics must be finite: {invalid_metrics}"
            )
        model_state = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
        metadata = normalize_checkpoint_metadata({
            'epoch': epoch,
            'checkpoint_role': checkpoint_role,
            'validation_loss_at_epoch': float(val_loss),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'selection_metric_name': SELECTION_METRIC_NAME,
            'selection_metric_definition': SELECTION_METRIC_DEFINITION,
            'selection_tie_breaker_name': SELECTION_TIEBREAKER_NAME,
            'selection_tie_breaker_definition': SELECTION_TIEBREAKER_DEFINITION,
            'selection_tertiary_tie_breaker_name': (
                SELECTION_TERTIARY_TIEBREAKER_NAME
            ),
            'selection_tertiary_tie_breaker_definition': (
                SELECTION_TERTIARY_TIEBREAKER_DEFINITION
            ),
            'best_selection_epoch': self.best_selection_epoch,
            'best_selection_components': self.best_selection_components,
            'selection_checkpoint_path': (
                str(self.selection_checkpoint_path)
                if self.selection_checkpoint_path is not None else None
            ),
            'selection_checkpoint_sha256': self.selection_checkpoint_sha256,
            'model_type': getattr(self, 'model_type', None),
            'num_classes': getattr(self, 'num_classes', None),
            'loss_type': getattr(self, 'loss_type', None),
            'input_normalization': dict(INPUT_NORMALIZATION),
            'input_provenance': dict(
                getattr(self, 'input_provenance', {})
            ),
            'target_provenance': dict(getattr(self, 'target_provenance', {})),
            'source_code_sha256': self._source_code_sha256(),
            'resolved_config': self._resolved_run_config(),
            'all_metrics': self.all_metrics,
            # Persist the ConfigLoader's data rather than pickling its class.
            # This preserves its path/config metadata while keeping the file
            # compatible with strict weights-only loading.
            'config': convert_to_json_serializable(self.config),
        })
        metadata.update({
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict()
                if self.scheduler is not None else None
            ),
            'scaler_state_dict': (
                self.scaler.state_dict() if self.scaler is not None else None
            ),
        })
        return metadata

    @staticmethod
    def _torch_load_checkpoint(checkpoint_path, map_location):
        """Load a local checkpoint without permitting arbitrary pickle code."""
        return load_weights_only_checkpoint(
            checkpoint_path,
            map_location=map_location,
        )

    def _load_verified_selection_checkpoint(
        self, checkpoint_path: Path, expected_sha256: str
    ) -> Dict:
        """Read and authenticate the checkpoint eligible for reporting/testing."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"validation-selected checkpoint is missing: {checkpoint_path}"
            )
        if not expected_sha256:
            raise RuntimeError(
                "validation-selected checkpoint has no recorded SHA-256"
            )
        actual_sha256 = self._sha256(checkpoint_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "validation-selected checkpoint SHA-256 mismatch: "
                f"{actual_sha256} != {expected_sha256}"
            )
        checkpoint = self._torch_load_checkpoint(
            checkpoint_path, map_location=self.device
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("validation-selected checkpoint is not a mapping")
        if checkpoint.get('checkpoint_role') != 'validation_composite_selection':
            raise ValueError(
                "checkpoint role is not validation_composite_selection"
            )
        if checkpoint.get('selection_metric_name') != SELECTION_METRIC_NAME:
            raise ValueError(
                "checkpoint was not selected with the configured validation metric"
            )
        checkpoint_selection_epoch = checkpoint.get('best_selection_epoch')
        if checkpoint_selection_epoch is None:
            raise ValueError("checkpoint has no selected validation epoch")
        if (
            self.best_selection_epoch is not None
            and checkpoint_selection_epoch != self.best_selection_epoch
        ):
            raise ValueError(
                "checkpoint selection epoch does not match the recorded decision "
                f"({checkpoint_selection_epoch} != {self.best_selection_epoch})"
            )
        if 'model_state_dict' not in checkpoint:
            raise ValueError("checkpoint has no model_state_dict")
        return checkpoint

    def save_checkpoint(self, epoch, val_loss, is_scheduled=False,
                        selection_improved=False):
        """Save training, validation-selected, and lowest-loss checkpoints."""
        val_loss_improved = val_loss < self.best_val_loss
        if val_loss_improved:
            self.best_val_loss = float(val_loss)

        # --no-checkpoints still tracks both validation optima and retains the
        # selected state in memory for the one post-training test evaluation.
        if self.no_checkpoints:
            return

        # Only rank 0 writes shared checkpoint artifacts.
        if self.multi_gpu and self.rank != 0:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # best_model.pth is the validation-composite selected model used for
        # held-out testing. Lowest validation loss is intentionally separate.
        if selection_improved:
            selection_path = self.checkpoint_dir / 'best_model.pth'
            self.selection_checkpoint_path = selection_path
            # A checkpoint cannot contain its own checksum. Clear any checksum
            # from a previous best epoch before overwriting the selected file.
            self.selection_checkpoint_sha256 = None
            selection_checkpoint = self._checkpoint_payload(
                epoch, val_loss, checkpoint_role='validation_composite_selection'
            )
            try:
                torch.save(selection_checkpoint, selection_path)
                selection_sha256 = self._sha256(selection_path)
                if selection_sha256 is None:
                    raise RuntimeError(
                        "validation-selected checkpoint checksum is unavailable"
                    )
                self._load_verified_selection_checkpoint(
                    selection_path, selection_sha256
                )
            except Exception as error:
                self.selection_checkpoint_sha256 = None
                raise RuntimeError(
                    "scientific run cannot continue because the validation-selected "
                    f"checkpoint was not written and verified: {error}"
                ) from error
            self.selection_checkpoint_sha256 = selection_sha256
            print(
                "\nNew validation-selected model saved and verified! "
                f"{SELECTION_METRIC_NAME}: {self.best_val_metric:.6f} "
                f"(epoch {self.best_selection_epoch})"
            )

        checkpoint = self._checkpoint_payload(
            epoch, val_loss, checkpoint_role='latest_training_state'
        )
        try:
            torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pth')
        except Exception as error:
            print(f"\nWarning: Could not save latest checkpoint: {error}")

        if is_scheduled or (epoch + 1) % self.save_every_n_epochs == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
            except Exception as error:
                print(f"\nWarning: Could not save checkpoint: {error}")

            latest_path = self.checkpoint_dir / 'latest.pth'
            try:
                torch.save(checkpoint, latest_path)
                print(f"Latest checkpoint updated: {latest_path}")
            except Exception as error:
                print(f"\nWarning: Could not save latest.pth: {error}")

        if val_loss_improved:
            val_loss_path = self.checkpoint_dir / 'best_val_loss_model.pth'
            val_loss_checkpoint = self._checkpoint_payload(
                epoch, val_loss, checkpoint_role='lowest_validation_loss'
            )
            try:
                torch.save(val_loss_checkpoint, val_loss_path)
                print(f"\nNew lowest-loss model saved! Val loss: {val_loss:.4f}")
            except Exception as error:
                print(f"\nWarning: Could not save lowest-loss model: {error}")

    def restore_validation_selected_model(self):
        """Restore the selected state before touching the held-out test split."""
        if self.multi_gpu:
            dist.barrier()

        selected_state = None
        if self.no_checkpoints:
            if self._best_selection_state_dict is not None:
                selected_state = self._best_selection_state_dict
                self.selection_restore_source = (
                    'captured_validation_selected_state_non_scientific_no_checkpoints'
                )
        else:
            checkpoint_path = (
                Path(self.selection_checkpoint_path)
                if self.selection_checkpoint_path is not None
                else self.checkpoint_dir / 'best_model.pth'
            )
            try:
                checkpoint = self._load_verified_selection_checkpoint(
                    checkpoint_path, self.selection_checkpoint_sha256
                )
            except Exception as error:
                raise RuntimeError(
                    "scientific run cannot restore an authenticated "
                    f"validation-selected checkpoint: {error}"
                ) from error
            self.best_selection_epoch = checkpoint['best_selection_epoch']
            self.best_val_metric = checkpoint.get(
                'best_val_metric', self.best_val_metric
            )
            self.best_selection_components = checkpoint.get(
                'best_selection_components', self.best_selection_components
            )
            selected_state = checkpoint['model_state_dict']
            self.selection_checkpoint_path = checkpoint_path
            self.selection_restore_source = 'verified_validation_selected_checkpoint'

        if selected_state is None or self.best_selection_epoch is None:
            raise RuntimeError(
                "No validation-selected model state is available. At least one "
                "finite validation epoch is required; in-memory restoration is "
                "restricted to explicit --no-checkpoints debug runs."
            )

        base_model = self.model.module if self.multi_gpu else self.model
        base_model.load_state_dict(selected_state)
        return self.selection_restore_source

    def _write_model_selection_record(self, test_evaluated=False):
        """Persist the selection decision separately from training checkpoints."""
        if self.multi_gpu and self.rank != 0:
            return
        debug_limited = (
            self.max_batches is not None
            or os.environ.get('QUICK_TEST', '0') == '1'
            or os.environ.get('SINGLE_BATCH', '0') == '1'
        )
        run_role = getattr(self, 'protocol_run_role', None) or (
            'legacy_or_reference_run'
        )
        scientific_result_eligible = (
            self.validation_only
            and run_role == 'validation_screen_cell'
            and getattr(self, 'protocol_candidate_key', None) is not None
            and not self.no_checkpoints
            and not debug_limited
            and self.selection_restore_source
            == 'verified_validation_selected_checkpoint'
            and self.selection_checkpoint_sha256 is not None
        )
        selected_checkpoint_identifier = None
        if self.selection_checkpoint_path is not None:
            checkpoint_path = Path(self.selection_checkpoint_path).resolve()
            if getattr(self, 'protocol_candidate_key', None) is not None:
                try:
                    selected_checkpoint_identifier = checkpoint_path.relative_to(
                        self._repository_root().resolve()
                    ).as_posix()
                except ValueError as error:
                    raise RuntimeError(
                        "scientific screen checkpoint must remain inside the "
                        "repository/staging root"
                    ) from error
            else:
                selected_checkpoint_identifier = str(self.selection_checkpoint_path)
        record = {
            'screen_result_schema_version': 1,
            'outcome_status': 'success',
            'run_role': run_role,
            'protocol_campaign_id': getattr(
                self, 'protocol_campaign_id', None
            ),
            'protocol_cell_index': getattr(self, 'protocol_cell_index', None),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
            'selected_architecture_role': getattr(
                self, 'selected_architecture_role', None
            ),
            'protocol_candidate_key': getattr(
                self, 'protocol_candidate_key', None
            ),
            'seed': int(self.seed),
            'runtime_protocol': (
                self._current_selected_method_protocol()
                if getattr(self, 'protocol_candidate_key', None) is not None
                else None
            ),
            'scientific_execution_contract': (
                self._resolved_scientific_execution_contract()
                if getattr(self, 'protocol_candidate_key', None) is not None
                else None
            ),
            'successful_cell': bool(
                scientific_result_eligible
                and run_role == 'validation_screen_cell'
            ),
            'successful_smoke': bool(
                run_role == 'validation_smoke_cell'
                and self.validation_only
                and self.max_batches == 2
                and not self.no_checkpoints
                and self.selection_restore_source
                == 'verified_validation_selected_checkpoint'
                and self.selection_checkpoint_sha256 is not None
                and self.test_evaluation_count == 0
                and getattr(self, 'test_loader', None) is None
            ),
            'selection_metric_name': SELECTION_METRIC_NAME,
            'selection_metric_definition': SELECTION_METRIC_DEFINITION,
            'selection_tie_breaker_name': SELECTION_TIEBREAKER_NAME,
            'selection_tie_breaker_definition': SELECTION_TIEBREAKER_DEFINITION,
            'selection_tertiary_tie_breaker_name': (
                SELECTION_TERTIARY_TIEBREAKER_NAME
            ),
            'selection_tertiary_tie_breaker_definition': (
                SELECTION_TERTIARY_TIEBREAKER_DEFINITION
            ),
            'best_selection_epoch': self.best_selection_epoch,
            'best_selection_score': self.best_val_metric,
            'best_selection_components': self.best_selection_components,
            'selected_checkpoint_path': (
                str(self.selection_checkpoint_path)
                if self.selection_checkpoint_path is not None else None
            ),
            'selected_checkpoint_repo_relative_identifier': (
                selected_checkpoint_identifier
            ),
            'selected_checkpoint_sha256': self.selection_checkpoint_sha256,
            'restore_source': self.selection_restore_source,
            'checkpoints_disabled': self.no_checkpoints,
            'scientific_result_eligible': scientific_result_eligible,
            'debug_limited': debug_limited,
            'max_batches': self.max_batches,
            'validation_only': self.validation_only,
            'held_out_test_evaluated': bool(test_evaluated),
            'held_out_test_evaluation_count': self.test_evaluation_count,
            'evaluation_mode': (
                'validation_only'
                if self.validation_only
                else 'confirmatory_with_held_out_test'
            ),
            'held_out_dataset_constructed': bool(
                getattr(self, 'test_loader', None) is not None
            ),
            'source_code_sha256': self._source_code_sha256(),
            'data_split': self._resolved_data_split_config(),
            'target_provenance': dict(self.target_provenance),
            'input_provenance': dict(self.input_provenance),
            'smoke_preflight_manifest': (
                {
                    key: self.smoke_preflight_manifest[key]
                    for key in (
                        'repo_relative_identifier', 'sha256', 'campaign_id'
                    )
                }
                if getattr(self, 'smoke_preflight_manifest', None) is not None
                else None
            ),
            'execution_environment': {
                'cuda_available': bool(torch.cuda.is_available()),
                'device_type': str(getattr(self, 'device', torch.device('cpu')).type),
                'gpu_name': (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None
                ),
            },
            'selected_method_lock': (
                dict(self.selected_method_lock)
                if getattr(self, 'selected_method_lock', None) is not None
                else None
            ),
        }
        # This file is the immutable scheduler-cell outcome.  Never overwrite
        # it: a second attempt must use a new campaign rather than replacing an
        # unfavourable or failed result.
        with (self.metrics_dir / 'model_selection.json').open(
            'x', encoding='utf-8'
        ) as handle:
            json.dump(convert_to_json_serializable(record), handle, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = self._torch_load_checkpoint(checkpoint_path, self.device)
        recorded_selection_metric = checkpoint.get('selection_metric_name')
        if (
            checkpoint.get('best_selection_epoch') is not None
            and recorded_selection_metric != SELECTION_METRIC_NAME
        ):
            raise ValueError(
                "Cannot resume a validation-selected run under a different "
                f"selection metric: {recorded_selection_metric!r} != "
                f"{SELECTION_METRIC_NAME!r}"
            )
        
        # Load model state
        if self.multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_state = checkpoint.get('scheduler_state_dict')
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load metrics
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_metric = checkpoint.get('best_val_metric', float('-inf'))
        self.best_selection_epoch = checkpoint.get('best_selection_epoch')
        self.best_selection_components = checkpoint.get('best_selection_components')
        recorded_selection_path = checkpoint.get('selection_checkpoint_path')
        if recorded_selection_path:
            self.selection_checkpoint_path = Path(recorded_selection_path)
        elif (checkpoint_path.parent / 'best_model.pth').is_file():
            self.selection_checkpoint_path = checkpoint_path.parent / 'best_model.pth'
        self.selection_checkpoint_sha256 = checkpoint.get('selection_checkpoint_sha256')

        # When the supplied file is itself the selected checkpoint, authenticate
        # that exact file even though a checkpoint cannot embed its own digest.
        if checkpoint.get('checkpoint_role') == 'validation_composite_selection':
            self.selection_checkpoint_path = checkpoint_path
            self.selection_checkpoint_sha256 = self._sha256(checkpoint_path)
            self._best_selection_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in checkpoint['model_state_dict'].items()
            }
        self.all_metrics = checkpoint.get('all_metrics', self.all_metrics)
        # Migrate historical metric names that mislabeled class 0 as
        # "background" and class 1 as the only "pore" class.
        if 'disconnected_pore_iou' not in self.all_metrics:
            self.all_metrics['disconnected_pore_iou'] = self.all_metrics.pop('background_iou', [])
        if 'connected_pore_iou' not in self.all_metrics:
            self.all_metrics['connected_pore_iou'] = self.all_metrics.pop('pore_iou', [])
        self.all_metrics.setdefault('mineral_iou', [None] * len(self.all_metrics.get('epoch', [])))
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def save_metrics(self, epoch):
        """Save detailed metrics to JSON and CSV files."""
        if self.multi_gpu and self.rank != 0:
            return
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        # Save metrics as JSON
        metrics_json = {
            'epoch': epoch + 1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': convert_to_serializable(self.train_metrics),
            'val_metrics': convert_to_serializable(self.val_metrics),
            'best_val_loss': self.best_val_loss,
            'best_selection_metric': self.best_val_metric,
            'selection_metric_name': SELECTION_METRIC_NAME,
            'selection_metric_definition': SELECTION_METRIC_DEFINITION,
            'selection_tie_breaker_name': SELECTION_TIEBREAKER_NAME,
            'selection_tie_breaker_definition': SELECTION_TIEBREAKER_DEFINITION,
            'best_selection_epoch': self.best_selection_epoch,
            'best_selection_components': self.best_selection_components,
            'training_time': sum(self.epoch_times)
        }
        
        with open(self.metrics_dir / 'training_metrics.json', 'w') as f:
            json.dump(convert_to_json_serializable(metrics_json), f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(self.all_metrics)
        df.to_csv(self.metrics_dir / 'training_metrics.csv', index=False)
    
    def plot_training_curves(self, epoch):
        """Generate and save training curve visualizations."""
        if self.multi_gpu and self.rank != 0:
            return
        
        # Create plots directory if it doesn't exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax = axes[0, 1]
        train_acc = [m['accuracy'] for m in self.train_metrics]
        val_acc = [m['accuracy'] for m in self.val_metrics]
        ax.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # IoU curves
        ax = axes[0, 2]
        mean_ious = [m['mean_iou'] for m in self.val_metrics]
        pore_ious = [m['class_iou'][1] for m in self.val_metrics]
        bg_ious = [m['class_iou'][0] for m in self.val_metrics]
        ax.plot(epochs, mean_ious, 'g-', label='Mean IoU', linewidth=2)
        ax.plot(epochs, pore_ious, 'r-', label='Pore IoU', linewidth=2)
        ax.plot(epochs, bg_ious, 'b-', label='Background IoU', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.set_title('Intersection over Union')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 0]
        lrs = [self.optimizer.param_groups[0]['lr']] * len(epochs)
        ax.plot(epochs, lrs, 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # F1 Score
        if self.val_metrics and 'f1_score' in self.val_metrics[-1]:
            ax = axes[1, 1]
            f1_scores = [m.get('f1_score', 0) for m in self.val_metrics]
            ax.plot(epochs, f1_scores, 'purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.set_title('F1 Score (Pore Class)')
            ax.grid(True, alpha=0.3)
        
        # Epoch times
        if self.epoch_times:
            ax = axes[1, 2]
            ax.bar(range(1, len(self.epoch_times) + 1), self.epoch_times, color='skyblue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Time per Epoch')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save training curves - overwrite if requested
        if self.overwrite_plots:
            plt.savefig(self.plots_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.plots_dir / f'training_curves_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
            plt.savefig(self.plots_dir / 'latest_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, epoch):
        """Plot and save confusion matrix."""
        if self.multi_gpu and self.rank != 0:
            return
        
        # Create plots directory if it doesn't exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        class_names = ['Disconnected', 'Connected', 'Minerals'][: cm.shape[0]]
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save confusion matrix - overwrite if requested
        if self.overwrite_plots:
            plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=150)
        else:
            plt.savefig(self.visualizations_dir / f'confusion_matrix_epoch_{epoch+1}.png', dpi=150)
        plt.close()
    
    def _epoch_roc_enabled(self, epoch: int) -> bool:
        return (
            (int(epoch) + 1) % 10 == 0
            and self.loss_type != 'conditional_pore_focal_dice'
        )

    def plot_roc_curve(self, val_loader, epoch):
        """Generate and save ROC curve."""
        if self.multi_gpu and self.rank != 0:
            return
        if self.loss_type == 'conditional_pore_focal_dice':
            # Conditional probabilities are defined only inside the recovered
            # pore gate. Final full-image ROC/PR belongs to the locked composed
            # evaluator after the method is frozen.
            return
        
        # Skip ROC curve for multiclass problems
        if self.num_classes > 2:
            return
        
        self.model.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(self._prepare_model_inputs(images))
                    probs = torch.softmax(outputs, dim=1)
                
                # For binary classification (2 classes)
                # Filter out ignore index (-100) for 2-class problems
                mask_cpu = masks.cpu().numpy().flatten()
                valid_mask = mask_cpu != -100
                all_targets.extend(mask_cpu[valid_mask])
                batch_probs = probs[:, 1].cpu().numpy().flatten()
                all_probs.extend(batch_probs[valid_mask])
        
        # Calculate ROC curve only if we have both classes
        unique_targets = np.unique(all_targets)
        if len(unique_targets) < 2:
            print(f"Skipping ROC curve - only one class present in validation set: {unique_targets}")
            return
        
        try:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(all_targets, all_probs)
            roc_auc = auc(fpr, tpr)
        except ValueError as e:
            print(f"Skipping ROC curve due to error: {e}")
            return
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Epoch {epoch+1}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f'roc_curve_epoch_{epoch+1}.png', dpi=150)
        plt.close()
    
    def test_full_image_prediction(self):
        """Test full image prediction using patch stitching."""
        if self.multi_gpu and self.rank != 0:
            return
        
        print("\nTesting full image prediction...")
        
        # Create plots directory for overlay images if needed
        if self.overlay_only:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load a test image
        # Use appropriate dataset directory based on num_classes
        if self.num_classes == 2:
            test_images = list(Path("results/step3_coco_dataset/images").glob("*.png"))[:3]  # Test on 3 images
        else:
            test_images = list(Path("results/step3_coco_dataset/images").glob("*.png"))[:3]  # Test on 3 images
        
        # Get base model for prediction
        base_model = self.model.module if self.multi_gpu else self.model
        # Use ImprovedPatchPredictor for multi-class predictions
        if IMPROVED_PREDICTOR_AVAILABLE:
            predictor = ImprovedPatchPredictor(base_model, self.device, self.patch_size, num_classes=self.num_classes)
        else:
            predictor = PatchPredictor(base_model, self.device, self.patch_size)
        
        for img_path in test_images:
            print(f"Processing {img_path.name}...")
            
            start_time = time.time()
            
            if IMPROVED_PREDICTOR_AVAILABLE:
                # ImprovedPatchPredictor returns (class_prediction, class_probabilities)
                class_prediction, class_probabilities = predictor.predict_full_image(str(img_path))
                pred_time = time.time() - start_time
                
                print(f"Prediction stats - Min: {class_prediction.min()}, Max: {class_prediction.max()}, Unique: {np.unique(class_prediction)}")
                
                # Create color visualization for overlay
                h, w = class_prediction.shape
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                color_mask[class_prediction == 0] = [0, 0, 255]  # Red for disconnected pores
                color_mask[class_prediction == 1] = [0, 255, 0]  # Green for connected pores
                # Class 2 (background/minerals) remains black - not visualized
                
                # Create overlay with original
                original = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                overlay = cv2.addWeighted(original, 0.5, color_mask, 0.5, 0)
                
                if self.overlay_only:
                    # Only save overlay in plots directory
                    overlay_path = self.plots_dir / f"{img_path.stem}_overlay.png"
                    cv2.imwrite(str(overlay_path), overlay)
                    save_path = overlay_path
                else:
                    # Save all visualizations
                    # Save class prediction as grayscale (0, 1, 2)
                    save_path = self.predictions_dir / f"{img_path.stem}_prediction.png"
                    pred_scaled = (class_prediction * 127).astype(np.uint8)
                    cv2.imwrite(str(save_path), pred_scaled)
                    
                    # Also save raw prediction for debugging
                    raw_save_path = self.predictions_dir / f"{img_path.stem}_raw.npy"
                    np.save(str(raw_save_path), class_prediction)
                    
                    # Save color visualization
                    vis_path = self.predictions_dir / f"{img_path.stem}_visualization.png"
                    cv2.imwrite(str(vis_path), color_mask)
                    
                    # Save grayscale visualization
                    gray_vis = np.zeros_like(class_prediction, dtype=np.uint8)
                    gray_vis[class_prediction == 0] = 85     # Dark gray for disconnected
                    gray_vis[class_prediction == 1] = 170    # Light gray for connected
                    gray_path = self.predictions_dir / f"{img_path.stem}_gray.png"
                    cv2.imwrite(str(gray_path), gray_vis)
                    
                    # Save overlay
                    overlay_path = self.predictions_dir / f"{img_path.stem}_overlay.png"
                    cv2.imwrite(str(overlay_path), overlay)
                
            else:
                # Old predictor - single class probability
                prediction = predictor.predict_full_image(str(img_path))
                pred_time = time.time() - start_time
                
                # Save prediction
                pred_binary = (prediction > 0.5).astype(np.uint8) * 255
                save_path = self.predictions_dir / f"{img_path.stem}_prediction.png"
                cv2.imwrite(str(save_path), pred_binary)
                
                # Create visualization
                original = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                pred_colored = cv2.applyColorMap(pred_binary, cv2.COLORMAP_JET)
                combined = cv2.addWeighted(original, 0.7, pred_colored, 0.3, 0)
                
                vis_path = self.predictions_dir / f"{img_path.stem}_visualization.png"
                cv2.imwrite(str(vis_path), combined)
            
            print(f"  - Predicted in {pred_time:.2f}s")
            print(f"  - Saved to {save_path}")
    
    def train(self, num_epochs: int = None):
        """Main training loop with comprehensive logging."""
        if num_epochs:
            self.epochs = num_epochs
        if self.protocol_run_role == 'validation_smoke_cell':
            if int(self.epochs) != 1:
                raise ValueError('protocol smoke cells are fixed to one epoch')
        elif self.protocol_run_role in {
            'validation_screen_cell', 'selected_winner_retraining'
        }:
            if int(self.epochs) != 30:
                raise ValueError(
                    'scientific screen and winner retraining are fixed to 30 epochs'
                )
        
        # Clear memory before creating data loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        self.train_loader = train_loader  # Store for scheduler
        self.test_loader = test_loader
        self.category_id_map = dict(train_loader.dataset.category_id_map)
        if not getattr(self, 'input_provenance', {}).get(
            'image_aggregate_sha256'
        ):
            self._compute_development_input_provenance()
        self._verify_smoke_preflight_data_contract()
        self._verify_selected_lock_data_contract()
        self.batch_augmentor = getattr(train_loader.dataset, 'augmentor', None)
        if (self.use_mixup or self.use_cutmix) and self.batch_augmentor is None:
            raise RuntimeError("MixUp/CutMix requested but no training augmentor is available")

        if self.loss_type in {
            'hierarchical_pore_connectivity',
            'conditional_pore_focal_dice',
        }:
            # This candidate's conditional C0/C1 balance is derived once from
            # authoritative training masks. The dataset method refuses val/test
            # partitions and legacy polygon targets, so leakage fails closed.
            self.training_class_statistics = (
                train_loader.dataset.training_class_statistics()
            )

        # Create model
        self.create_model_and_optimizer()
        self._verify_selected_lock_execution_contract()

        # Save the resolved runtime configuration, including the exact data
        # files, image IDs, and instantiated model/loss/optimizer classes,
        # before model fitting begins.
        if not self.multi_gpu or self.rank == 0:
            self._save_config_info()
        
        # Initialize W&B
        if self.use_wandb and (not self.multi_gpu or self.rank == 0):
            run_name = self.wandb_name or f"pore_seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=run_name,
                config={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "patch_size": self.patch_size,
                    "training_patch_size": self.patch_size,
                    "evaluation_patch_size": self.evaluation_patch_size,
                    "training_batch_size": self.batch_size,
                    "evaluation_batch_size": self.evaluation_batch_size,
                    "seed": self.seed,
                    "evaluation_mode": (
                        "validation_only"
                        if self.validation_only
                        else "confirmatory_with_held_out_test"
                    ),
                    "held_out_dataset_constructed": bool(test_loader is not None),
                    "model": self.model_type,
                    "num_classes": self.num_classes,
                    "loss": self.loss_type,
                    "class_weights_requested": self.class_weights,
                    "class_weights_actual": self._actual_loss_class_weights(),
                    "resolved_loss_config": self._resolved_loss_config(),
                    "focal_gamma": self.focal_gamma,
                    "tversky_alpha": self.tversky_alpha,
                    "tversky_beta": self.tversky_beta,
                    "optimizer": self.optimizer_type,
                    "scheduler": self.scheduler_type,
                    "annotations_path": str(self.loaded_annotations_path),
                    "image_dir": str(self.image_dir),
                    "target_source": self.target_provenance.get('target_source'),
                    "mask_dir": str(self.mask_dir) if self.mask_dir is not None else None,
                    "mask_aggregate_sha256": self.target_provenance.get(
                        'mask_aggregate_sha256'
                    ),
                    "input_normalization": dict(INPUT_NORMALIZATION),
                    "resolved_model_config": self._resolved_model_config(),
                    "split_manifest": self.split_manifest,
                    "split_image_ids": self.split_ids,
                    "split_image_files": self.split_files,
                    "augmentations_enabled": self.augmentation_config['augmentation']['enabled'],
                    "augmentation_strength": self.augmentation_config['augmentation']['strength'],
                    "use_mixup": self.use_mixup,
                    "use_cutmix": self.use_cutmix,
                    "mixup_alpha": self.augmentation_config['augmentation']['mixup_alpha'],
                    "cutmix_alpha": self.augmentation_config['augmentation']['cutmix_alpha'],
                    "multi_gpu": self.multi_gpu,
                    "world_size": self.world_size if self.multi_gpu else 1,
                    "mixed_precision_requested": self.mixed_precision,
                    "mixed_precision_actual": self.scaler is not None,
                    "gradient_clipping": self.gradient_clip_val,
                    "early_stopping_patience": self.patience,
                    "model_selection_metric": SELECTION_METRIC_NAME,
                    "model_selection_metric_definition": SELECTION_METRIC_DEFINITION,
                    "model_selection_tie_breaker": SELECTION_TIEBREAKER_NAME,
                    "model_selection_tie_breaker_definition": (
                        SELECTION_TIEBREAKER_DEFINITION
                    ),
                    "early_stopping_metric": SELECTION_METRIC_NAME,
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                    "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
            )
        
        # Print training setup
        if not self.multi_gpu or self.rank == 0:
            print(f"\nStarting patch-based training")
            print(f"Device: {self.device}")
            print(f"Multi-GPU: {self.multi_gpu} (World Size: {self.world_size if self.multi_gpu else 1})")
            print(f"Batch size: {self.batch_size} patches total")
            print(f"Training patch size: {self.patch_size}x{self.patch_size}")
            print(
                "Validation/test tile size: "
                f"{self.evaluation_patch_size}x{self.evaluation_patch_size}; "
                f"batch size {self.evaluation_batch_size}"
            )
            print(f"Epochs: {self.epochs}")
            print(f"Train batches: {len(train_loader)}")
            if self.use_wandb:
                print(f"W&B tracking enabled: {self.wandb_project}/{run_name}")
            print(f"Val batches: {len(val_loader)}")
            if self.validation_only:
                if test_loader is not None:
                    raise RuntimeError(
                        "validation-only mode constructed a held-out test loader"
                    )
                print("Test batches (held out): LOCKED; loader not constructed")
            else:
                print(f"Test batches (held out): {len(test_loader)}")
            print(f"Checkpoint interval: every {self.save_every_n_epochs} epochs")
            print(f"Mixed precision: {'Enabled' if self.scaler is not None else 'Disabled'}")
            print("-" * 50)
        
        # Training loop
        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()
            
            # Set epoch for distributed sampler
            if self.multi_gpu and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_loss, train_acc, avg_batch_time = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_results = self.validate(val_loader, epoch)
            val_loss, val_acc, mean_iou, class_iou = val_results[:4]
            additional_metrics = val_results[4] if len(val_results) > 4 else {}
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append({'accuracy': train_acc, 'avg_batch_time': avg_batch_time})
            
            val_metrics_dict = {
                'accuracy': val_acc,
                'mean_iou': mean_iou,
                'class_iou': class_iou
            }
            val_metrics_dict.update(additional_metrics)
            self.val_metrics.append(val_metrics_dict)
            
            # Update comprehensive metrics
            self.all_metrics['epoch'].append(epoch + 1)
            self.all_metrics['train_loss'].append(train_loss)
            self.all_metrics['val_loss'].append(val_loss)
            self.all_metrics['train_acc'].append(train_acc)
            self.all_metrics['val_acc'].append(val_acc)
            self.all_metrics['mean_iou'].append(mean_iou)
            self.all_metrics['disconnected_pore_iou'].append(class_iou[0])
            self.all_metrics['connected_pore_iou'].append(class_iou[1])
            self.all_metrics['mineral_iou'].append(
                class_iou[2] if len(class_iou) > 2 else None
            )
            self.all_metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.all_metrics['epoch_time'].append(epoch_time)

            # Select the model using validation data only. The state is captured
            # before any checkpoint write and before the held-out test loader is
            # ever evaluated.
            selection = self._record_validation_selection(
                epoch, val_loss, class_iou, additional_metrics
            )
            combined_metric = selection['score']
            selection_improved = selection['improved']
            should_stop = False
            if selection_improved:
                self.patience_counter = 0
                if not self.multi_gpu or self.rank == 0:
                    print(
                        "New best validation selection metric: "
                        f"C0 IoU={selection['c0_iou']:.4f}, "
                        f"C1 IoU={selection['c1_iou']:.4f}, "
                        f"harmonic mean={combined_metric:.4f}, "
                        f"pore-union IoU tie-break="
                        f"{selection['pore_union_iou']:.4f}"
                    )
            elif self.early_stopping_enabled:
                self.patience_counter += 1
                should_stop = self.patience_counter >= self.patience
                if not self.multi_gpu or self.rank == 0:
                    print(
                        f"No validation-selection improvement. Patience: "
                        f"{self.patience_counter}/{self.patience}"
                    )
            
            # Log to W&B
            if self.use_wandb and (not self.multi_gpu or self.rank == 0):
                wandb_log = {
                    # Basic info
                    "epoch": epoch + 1,
                    
                    # Training metrics
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/batch_time": avg_batch_time,
                    "train/batches_per_second": 1.0 / avg_batch_time if avg_batch_time > 0 else 0,
                    
                    # Validation metrics
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/mean_iou": mean_iou,
                    "val/disconnected_pore_iou": class_iou[0],  # Class 0
                    "val/connected_pore_iou": class_iou[1],      # Class 1
                    # Only log mineral IoU for 3-class
                    **({f"val/mineral_iou": class_iou[2]} if len(class_iou) > 2 else {}),
                    "val/pore_combined_iou": (class_iou[0] + class_iou[1]) / 2,  # Combined pore IoU
                    "val/dice_score": 2 * mean_iou / (1 + mean_iou),  # Convert IoU to Dice
                    
                    # Learning dynamics
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch_time": epoch_time,
                    "samples_per_second": len(train_loader.dataset) / epoch_time,
                    
                    # Early stopping tracking
                    "early_stopping/patience_counter": self.patience_counter,
                    "early_stopping/best_combined_metric": self.best_val_metric,
                    "early_stopping/current_combined_metric": combined_metric,
                    
                    # GPU metrics (if available)
                    "system/gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    "system/gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
                }
                
                # Add additional metrics if available
                if additional_metrics:
                    # Overall metrics
                    wandb_log.update({
                        "val/overall_precision": additional_metrics.get('overall_precision', 0),
                        "val/overall_recall": additional_metrics.get('overall_recall', 0),
                        "val/overall_f1_score": additional_metrics.get('overall_f1', 0),
                        "val/weighted_precision": additional_metrics.get('weighted_precision', 0),
                        "val/weighted_recall": additional_metrics.get('weighted_recall', 0),
                        "val/weighted_f1_score": additional_metrics.get('weighted_f1', 0),
                    })
                    
                    # Per-class metrics
                    if 'per_class_metrics' in additional_metrics:
                        for class_name, metrics in additional_metrics['per_class_metrics'].items():
                            wandb_log.update({
                                f"val/{class_name}/accuracy": metrics['accuracy'],
                                f"val/{class_name}/precision": metrics['precision'],
                                f"val/{class_name}/recall": metrics['recall'],
                                f"val/{class_name}/f1_score": metrics['f1_score'],
                                f"val/{class_name}/specificity": metrics['specificity'],
                                f"val/{class_name}/tp": metrics['tp'],
                                f"val/{class_name}/fp": metrics['fp'],
                                f"val/{class_name}/fn": metrics['fn'],
                                f"val/{class_name}/tn": metrics['tn'],
                            })
                    
                    # Class distribution
                    if 'class_distribution' in additional_metrics:
                        class_dist = additional_metrics['class_distribution']
                        wandb_log.update({
                            "val/class_distribution/disconnected_pore": class_dist[0],
                            "val/class_distribution/connected_pore": class_dist[1],
                        })
                        if len(class_dist) > 2:
                            wandb_log["val/class_distribution/mineral"] = class_dist[2]
                    
                    # Log confusion matrix as a wandb table
                    if 'confusion_matrix' in additional_metrics:
                        cm = additional_metrics['confusion_matrix']
                        # Create a wandb Table for confusion matrix
                        if cm.shape[0] == 2:
                            cm_table = wandb.Table(
                                columns=["Actual\\Predicted", "Disconnected", "Connected"],
                                data=[
                                    ["Disconnected", int(cm[0,0]), int(cm[0,1])],
                                    ["Connected", int(cm[1,0]), int(cm[1,1])]
                                ]
                            )
                        else:
                            cm_table = wandb.Table(
                                columns=["Actual\\Predicted", "Disconnected", "Connected", "Mineral"],
                                data=[
                                    ["Disconnected", int(cm[0,0]), int(cm[0,1]), int(cm[0,2])],
                                    ["Connected", int(cm[1,0]), int(cm[1,1]), int(cm[1,2])],
                                    ["Mineral", int(cm[2,0]), int(cm[2,1]), int(cm[2,2])]
                                ]
                            )
                        wandb_log["val/confusion_matrix"] = cm_table
                
                # Add augmentation usage metrics
                wandb_log.update({
                    "augmentation/strength": self.augmentation_config['augmentation']['strength'],
                    "augmentation/mixup_enabled": self.use_mixup,
                    "augmentation/cutmix_enabled": self.use_cutmix,
                })
                
                # Log gradient norms for monitoring training stability
                total_grad_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                wandb_log["train/gradient_norm"] = total_grad_norm
                
                wandb.log(wandb_log, step=epoch)
            
            # Save checkpoint
            self.save_checkpoint(
                epoch,
                val_loss,
                selection_improved=selection_improved,
            )
            
            # Generate visualizations and save metrics
            if not self.multi_gpu or self.rank == 0:
                # Save metrics
                self.save_metrics(epoch)
                
                # Plot training curves
                self.plot_training_curves(epoch)
                
                # Plot confusion matrix
                if 'confusion_matrix' in additional_metrics:
                    self.plot_confusion_matrix(additional_metrics['confusion_matrix'], epoch)
                
                # Generate ROC curve every 10 epochs
                if self._epoch_roc_enabled(epoch):
                    self.plot_roc_curve(val_loader, epoch)
                
                # Print epoch summary
                print(f"\nEpoch {epoch+1}/{self.epochs} - Time: {epoch_time:.1f}s (avg batch: {avg_batch_time:.3f}s)")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Mean IoU: {mean_iou:.4f}")
                if len(class_iou) == 2:
                    print(f"Class IoU - Disconnected: {class_iou[0]:.4f}, Connected: {class_iou[1]:.4f}")
                else:
                    print(f"Class IoU - Disconnected: {class_iou[0]:.4f}, Connected: {class_iou[1]:.4f}, Mineral: {class_iou[2]:.4f}")
                if 'f1_score' in additional_metrics:
                    print(f"F1 Score: {additional_metrics['f1_score']:.4f}, Precision: {additional_metrics['precision']:.4f}, Recall: {additional_metrics['recall']:.4f}")
                print("-" * 50)

            if self.scheduler is not None and not self.scheduler_step_per_batch:
                self.scheduler.step()

            if should_stop:
                if not self.multi_gpu or self.rank == 0:
                    print(
                        f"\nEarly stopping triggered after {self.patience} "
                        "epochs without validation-selection improvement."
                    )
                    print(
                        f"Best {SELECTION_METRIC_NAME}: {self.best_val_metric:.4f} "
                        f"at epoch {self.best_selection_epoch}"
                    )
                break
        
        # Persist the terminal optimization state before replacing model weights
        # with the validation-selected state. This keeps resume checkpoints
        # truthful while ensuring all reported test results use selected weights.
        final_epoch = max(0, len(self.train_losses) - 1)
        final_val_loss = self.val_losses[-1] if self.val_losses else float('inf')
        if not self.multi_gpu or self.rank == 0:
            print("\nSaving final training-state checkpoint...")
        self.save_checkpoint(final_epoch, final_val_loss, is_scheduled=True)

        restore_source = self.restore_validation_selected_model()
        if not self.multi_gpu or self.rank == 0:
            print(
                "Reloaded validation-selected model "
                f"from {restore_source} (epoch {self.best_selection_epoch})."
            )
        if self.validation_only:
            if test_loader is not None:
                raise RuntimeError(
                    "validation-only mode must not construct a held-out test loader"
                )
            if self.test_evaluation_count != 0 or self.test_metrics is not None:
                raise RuntimeError(
                    "validation-only run detected held-out evaluation state"
                )
            if not self.multi_gpu or self.rank == 0:
                print(
                    "\nValidation-only training completed; held-out test "
                    "dataset was not constructed or evaluated."
                )
                self.save_metrics(final_epoch)
                self._generate_training_summary()
                if self.use_wandb and self.wandb_run:
                    wandb.finish()
                self._write_model_selection_record(test_evaluated=False)
                print(f"\nResults saved to: {self.output_dir}")
            return

        # The test split remains untouched during fitting and model selection.
        # Evaluate it exactly once, only after restoring the selected state.
        if test_loader is None:
            raise RuntimeError("Confirmatory mode requires a held-out test loader")
        if self.test_evaluation_count != 0:
            raise RuntimeError("Held-out test evaluation was requested more than once")
        self.test_evaluation_count += 1
        test_results = self.validate(test_loader, final_epoch, phase="Test")
        if not self.multi_gpu or self.rank == 0:
            test_loss, test_acc, test_mean_iou, test_class_iou = test_results[:4]
            test_additional = test_results[4] if len(test_results) > 4 else {}
            debug_limited = (
                self.max_batches is not None
                or os.environ.get('QUICK_TEST', '0') == '1'
                or os.environ.get('SINGLE_BATCH', '0') == '1'
            )
            self.test_metrics = {
                'evaluation_model': 'validation_selected_state',
                'evaluated_once_after_training': True,
                'evaluation_count': self.test_evaluation_count,
                'evaluation_scope': 'debug_subset' if debug_limited else 'full_test_split',
                'selection_metric_name': SELECTION_METRIC_NAME,
                'selection_metric_definition': SELECTION_METRIC_DEFINITION,
                'selection_tie_breaker_name': SELECTION_TIEBREAKER_NAME,
                'selection_tie_breaker_definition': (
                    SELECTION_TIEBREAKER_DEFINITION
                ),
                'selection_tertiary_tie_breaker_name': (
                    SELECTION_TERTIARY_TIEBREAKER_NAME
                ),
                'selection_tertiary_tie_breaker_definition': (
                    SELECTION_TERTIARY_TIEBREAKER_DEFINITION
                ),
                'selected_validation_epoch': self.best_selection_epoch,
                'selected_validation_score': self.best_val_metric,
                'selected_validation_components': self.best_selection_components,
                'selected_checkpoint_path': (
                    str(self.selection_checkpoint_path)
                    if self.selection_checkpoint_path is not None else None
                ),
                'selected_checkpoint_sha256': self.selection_checkpoint_sha256,
                'selected_state_restore_source': self.selection_restore_source,
                'resolved_model_config': self._resolved_model_config(),
                'source_code_sha256': self._source_code_sha256(),
                'resolved_augmentation_config': self._resolved_augmentation_config(),
                'data_split': self._resolved_data_split_config(),
                'selected_method_lock': (
                    dict(self.selected_method_lock)
                    if self.selected_method_lock is not None else None
                ),
                'input_normalization': dict(INPUT_NORMALIZATION),
                'target_provenance': self.target_provenance,
                'test_image_ids': self.split_ids.get('test', []),
                'test_image_files': self.split_files.get('test', []),
                'loss': test_loss,
                'accuracy': test_acc,
                'mean_iou': test_mean_iou,
                'class_iou': test_class_iou,
                **test_additional,
            }
            with (self.metrics_dir / 'test_metrics.json').open('w', encoding='utf-8') as handle:
                json.dump(convert_to_json_serializable(self.test_metrics), handle, indent=2)
            self._write_model_selection_record(test_evaluated=True)

        # Final tasks
        if not self.multi_gpu or self.rank == 0:
            print("\nTraining completed!")
            
            # Save final metrics
            self.save_metrics(final_epoch)
            
            # Test full image prediction
            self.test_full_image_prediction()
            
            # Generate final summary
            self._generate_training_summary()
            
            # Close W&B run
            if self.use_wandb and self.wandb_run:
                wandb.finish()
            
            print(f"\nResults saved to: {self.output_dir}")
            
            # Create a quick access symlink to latest run
            latest_link = Path("results/patch_training/latest")
            try:
                if latest_link.exists() or latest_link.is_symlink():
                    latest_link.unlink()
                latest_link.symlink_to(self.output_dir.absolute())
            except (OSError, FileExistsError) as e:
                print(f"Warning: Could not create symlink to latest run: {e}")
    
    def _generate_training_summary(self):
        """Generate a comprehensive training summary."""
        # Calculate additional statistics
        epochs_trained = len(self.train_losses)
        early_stopped = epochs_trained < self.epochs
        best_pore_iou_epoch = max(range(len(self.val_metrics)), 
                                  key=lambda i: self.val_metrics[i]['class_iou'][1]) + 1
        
        summary = {
            'run_info': {
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_dir),
                'wandb_run': self.wandb_name if self.use_wandb else None,
                'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
            },
            'training_config': {
                'epochs_planned': self.epochs,
                'epochs_trained': epochs_trained,
                'early_stopped': early_stopped,
                'early_stop_reason': 'patience_exceeded' if early_stopped else 'completed',
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'patch_size': self.patch_size,
                'training_patch_size': self.patch_size,
                'evaluation_patch_size': self.evaluation_patch_size,
                'training_batch_size': self.batch_size,
                'evaluation_batch_size': self.evaluation_batch_size,
                'seed': self.seed,
                'evaluation_mode': (
                    'validation_only'
                    if self.validation_only
                    else 'confirmatory_with_held_out_test'
                ),
                'held_out_dataset_constructed': bool(
                    getattr(self, 'test_loader', None) is not None
                ),
                'max_batches_debug_limit': self.max_batches,
                'multi_gpu': self.multi_gpu,
                'world_size': self.world_size if self.multi_gpu else 1,
                'model_type': self.model_type,
                'resolved_model_config': self._resolved_model_config(),
                'input_normalization': dict(INPUT_NORMALIZATION),
                'loss_type': self.loss_type,
                'class_weights_requested': self.class_weights,
                'class_weights_actual': self._actual_loss_class_weights(),
                'resolved_loss_config': self._resolved_loss_config(),
                'focal_gamma': self.focal_gamma,
                'tversky_alpha': self.tversky_alpha,
                'tversky_beta': self.tversky_beta,
                'optimizer': self.optimizer_type,
                'scheduler': self.scheduler_type,
                'annotations_path': str(self.loaded_annotations_path),
                'image_dir': str(self.image_dir),
                'target_provenance': self.target_provenance,
                'split_manifest': self.split_manifest,
                'split_image_ids': self.split_ids,
                'split_image_files': self.split_files,
                'source_to_canonical_category_id': self.category_id_map,
                'augmentations_enabled': self.augmentation_config['augmentation']['enabled'],
                'augmentation_strength': self.augmentation_config['augmentation']['strength'],
                'resolved_augmentation_config': self._resolved_augmentation_config(),
                'use_mixup': self.use_mixup,
                'use_cutmix': self.use_cutmix,
            },
            'final_metrics': {
                'best_val_loss': float(self.best_val_loss),
                'best_selection_metric': float(self.best_val_metric),
                'selection_metric_name': SELECTION_METRIC_NAME,
                'selection_metric_definition': SELECTION_METRIC_DEFINITION,
                'selection_tie_breaker_name': SELECTION_TIEBREAKER_NAME,
                'selection_tie_breaker_definition': (
                    SELECTION_TIEBREAKER_DEFINITION
                ),
                'selection_tertiary_tie_breaker_name': (
                    SELECTION_TERTIARY_TIEBREAKER_NAME
                ),
                'selection_tertiary_tie_breaker_definition': (
                    SELECTION_TERTIARY_TIEBREAKER_DEFINITION
                ),
                'best_selection_epoch': self.best_selection_epoch,
                'best_selection_components': self.best_selection_components,
                'selected_checkpoint_path': (
                    str(self.selection_checkpoint_path)
                    if self.selection_checkpoint_path is not None else None
                ),
                'selected_checkpoint_sha256': self.selection_checkpoint_sha256,
                'selected_state_restore_source': self.selection_restore_source,
                'best_connected_pore_iou': float(max(
                    metrics['class_iou'][1] for metrics in self.val_metrics
                )) if self.val_metrics else 0,
                'final_train_loss': float(self.train_losses[-1]) if self.train_losses else 0,
                'final_val_loss': float(self.val_losses[-1]) if self.val_losses else 0,
                'final_train_acc': float(self.train_metrics[-1]['accuracy']) if self.train_metrics else 0,
                'final_val_acc': float(self.val_metrics[-1]['accuracy']) if self.val_metrics else 0,
                'final_mean_iou': float(self.val_metrics[-1]['mean_iou']) if self.val_metrics else 0,
                'final_disconnected_pore_iou': float(self.val_metrics[-1]['class_iou'][0]) if self.val_metrics else 0,
                'final_connected_pore_iou': float(self.val_metrics[-1]['class_iou'][1]) if self.val_metrics else 0,
                'final_mean_pore_iou': float(np.mean(
                    self.val_metrics[-1]['class_iou'][:2]
                )) if self.val_metrics else 0,
                'final_mineral_iou': float(self.val_metrics[-1]['class_iou'][2]) if self.val_metrics and len(self.val_metrics[-1]['class_iou']) > 2 else None,
                'total_training_time': sum(self.epoch_times),
                'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
            },
            'test_metrics': self.test_metrics,
            'best_epochs': {
                'best_loss_epoch': self.val_losses.index(min(self.val_losses)) + 1 if self.val_losses else 0,
                'best_selection_epoch': self.best_selection_epoch,
                'best_pore_iou_epoch': best_pore_iou_epoch,
            },
            'training_history': {
                'train_losses': [float(x) for x in self.train_losses],
                'val_losses': [float(x) for x in self.val_losses],
                'learning_rates': [float(lr) for lr in self.all_metrics['learning_rate'][:epochs_trained]],
            }
        }
        
        # Add final confusion matrix if available
        if self.val_metrics and 'confusion_matrix' in self.val_metrics[-1]:
            cm = self.val_metrics[-1]['confusion_matrix']
            evaluation_num_classes = int(
                self.val_metrics[-1].get('evaluation_num_classes', cm.shape[0])
            )
            if cm.shape != (evaluation_num_classes, evaluation_num_classes):
                raise RuntimeError(
                    "final confusion shape conflicts with evaluation_num_classes"
                )
            summary['final_metrics']['confusion_matrix'] = {
                'class_order': [
                    CANONICAL_CLASS_NAMES[index]
                    for index in range(evaluation_num_classes)
                ],
                'counts': cm.tolist(),
            }
        
        # Save summary as JSON
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(convert_to_json_serializable(summary), f, indent=2)
        
        # Save summary as readable text
        with open(self.output_dir / 'training_summary.txt', 'w') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Run Directory: {self.output_dir}\n")
            f.write(f"Timestamp: {summary['run_info']['timestamp']}\n")
            f.write(f"SLURM Job ID: {summary['run_info']['slurm_job_id']}\n\n")
            
            f.write("Training Configuration:\n")
            f.write(f"  Epochs: {epochs_trained}/{self.epochs}")
            if early_stopped:
                f.write(" (early stopped)")
            f.write("\n")
            f.write(f"  Batch Size: {self.batch_size}\n")
            f.write(f"  Learning Rate: {self.learning_rate}\n")
            f.write(f"  Model/Loss: {self.model_type}/{self.loss_type}\n")
            f.write(f"  Optimizer/Scheduler: {self.optimizer_type}/{self.scheduler_type}\n")
            f.write(f"  Split Manifest: {self.split_manifest or 'generated'}\n")
            f.write(f"  Augmentation: {self.augmentation_config['augmentation']['strength']}\n")
            f.write(f"  Mixup/CutMix: {self.use_mixup}/{self.use_cutmix}\n\n")
            
            f.write("Final Metrics:\n")
            f.write(f"  Best Validation Loss: {summary['final_metrics']['best_val_loss']:.4f} (Epoch {summary['best_epochs']['best_loss_epoch']})\n")
            f.write(
                f"  Best Selection Metric: "
                f"{summary['final_metrics']['best_selection_metric']:.4f} "
                f"(Epoch {summary['best_epochs']['best_selection_epoch']})\n"
            )
            f.write(f"  Best Connected-Pore IoU: {summary['final_metrics']['best_connected_pore_iou']:.4f} (Epoch {summary['best_epochs']['best_pore_iou_epoch']})\n")
            f.write(f"  Final Validation Accuracy: {summary['final_metrics']['final_val_acc']:.2f}%\n")
            f.write(f"  Final Mean IoU: {summary['final_metrics']['final_mean_iou']:.4f}\n")
            f.write(f"  Final Disconnected-Pore IoU: {summary['final_metrics']['final_disconnected_pore_iou']:.4f}\n")
            f.write(f"  Final Connected-Pore IoU: {summary['final_metrics']['final_connected_pore_iou']:.4f}\n")
            f.write(f"  Final Mean Pore IoU: {summary['final_metrics']['final_mean_pore_iou']:.4f}\n")
            if summary['final_metrics']['final_mineral_iou'] is not None:
                f.write(f"  Final Mineral IoU: {summary['final_metrics']['final_mineral_iou']:.4f}\n")
            f.write(f"  Total Training Time: {summary['final_metrics']['total_training_time']/60:.1f} minutes\n")
            f.write(f"  Average Epoch Time: {summary['final_metrics']['avg_epoch_time']:.1f} seconds\n")
            if self.test_metrics:
                f.write("\nHeld-out Test Metrics (evaluated once after training):\n")
                f.write(f"  Mean IoU: {self.test_metrics['mean_iou']:.4f}\n")
                f.write(f"  Class IoU: {self.test_metrics['class_iou']}\n")
        
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Run saved to: {self.output_dir}")
        print(f"Best validation loss: {summary['final_metrics']['best_val_loss']:.4f} (Epoch {summary['best_epochs']['best_loss_epoch']})")
        print(
            f"Best selection metric: {summary['final_metrics']['best_selection_metric']:.4f} "
            f"(Epoch {summary['best_epochs']['best_selection_epoch']})"
        )
        print(f"Best connected-pore IoU: {summary['final_metrics']['best_connected_pore_iou']:.4f} (Epoch {summary['best_epochs']['best_pore_iou_epoch']})")
        print(f"Final validation accuracy: {summary['final_metrics']['final_val_acc']:.2f}%")
        print(f"Final mean IoU: {summary['final_metrics']['final_mean_iou']:.4f}")
        print(f"Total training time: {summary['final_metrics']['total_training_time']/60:.1f} minutes")
        if early_stopped:
            print(f"Training stopped early at epoch {epochs_trained}/{self.epochs}")
        print("=" * 70)


def main():
    """Main training function."""
    trainer = PatchTrainer()
    trainer.train(num_epochs=10)  # Train for 10 epochs


if __name__ == "__main__":
    main()

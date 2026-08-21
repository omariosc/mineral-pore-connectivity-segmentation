#!/usr/bin/env python3
"""Patch/full-tile segmentation training with explicit resolved provenance."""

import sys
from pathlib import Path
import argparse
import os

# Add project root to path. Heavy ML imports are intentionally delayed until
# after argument and credential validation so rejected CLI inputs cannot trigger
# model/runtime initialization.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def resolve_augmentation_settings(args):
    """Translate legacy CLI names to the augmentation pipeline actually implemented."""
    strength_by_strategy = {
        'none': 'none',
        'minimal': 'light',
        'conservative': 'light',
        'targeted': 'medium',
        'class_balanced': 'medium',
        'class_aware': 'medium',
        'smart_crop': 'medium',
        'full': 'strong',
        'heavy': 'strong',
        'copy_paste': 'strong',
        'mixup': 'strong',
        'cutmix': 'strong',
        'mixup_cutmix': 'strong',
    }
    disabled_by_environment = os.environ.get('DISABLE_TRANSFORMS', '0') == '1'
    enabled = not args.disable_transforms and not disabled_by_environment and args.augmentation != 'none'
    use_mixup = enabled and (args.use_mixup or args.augmentation in {'mixup', 'mixup_cutmix'})
    use_cutmix = enabled and (args.use_cutmix or args.augmentation in {'cutmix', 'mixup_cutmix'})
    return {
        'enabled': enabled,
        'strength': strength_by_strategy[args.augmentation],
        'use_mixup': use_mixup,
        'use_cutmix': use_cutmix,
    }


def main():
    """Train using patches for speed."""
    parser = argparse.ArgumentParser(description='Train U-Net model with patches')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--checkpoint-interval', '--checkpoint_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--log-interval', '--log_interval', type=int, default=10, help='Log metrics every N batches')
    parser.add_argument('--multi-gpu', '--multi_gpu', action='store_true', help='Use multi-GPU training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--model-type', '--model_type', type=str, default='improved_unet', 
                       choices=['unet', 'plain_unet', 'legacy_configured_unet',
                               'improved_unet', 'multiscale_unet', 'multiscale_attention_unet',
                               'multiscale_attention_unet_pyramid',
                               'unet_plusplus', 'deeplabv3', 'transunet', 'edge_aware_unet',
                               'segformer_b0', 'segformer_b1', 'segformer_b2',
                               'dinov3', 'dinov3_vits16',
                               'yolov8', 'yolov8_s', 'yolov8_m', 'yolov8_l', 'yolov8_x', 'yolov8_seg'], 
                       help='Model type to use')
    parser.add_argument('--loss-type', '--loss_type', type=str, default='focal_dice', 
                       choices=['combined', 'focal', 'focal_dice', 'tversky', 'asymmetric', 'binary_pore', 'mineral_aware', 
                               'sparse_pore', 'boundary_aware', 'topological',
                               'hierarchical_pore_connectivity',
                               'conditional_pore_focal_dice',
                               'focal_tversky', 'adaptive_focal_tversky', 'unified_focal',
                               'lovasz', 'weighted_lovasz', 'focal_lovasz', 'asymmetric_lovasz',
                               'boundary', 'active_boundary', 'hausdorff', 'contour', 
                               'topological_boundary', 'multi_task', 'yolov8'], 
                       help='Loss function type')
    parser.add_argument('--num-classes', '--num_classes', type=int, default=3, help='Number of output classes')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=None, help='Batch size (overrides config file)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save-predictions', '--save_predictions', action='store_true', help='Save predictions after training')
    parser.add_argument('--experiment-name', '--experiment_name', type=str, default=None, help='Experiment name for tracking')
    parser.add_argument('--weight-decay', '--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--early-stopping', '--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--early-stopping-patience', '--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--accumulate-grad-batches', '--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed-precision', '--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient-clip-val', '--gradient_clip_val', type=float, default=None, help='Gradient clipping value')
    parser.add_argument('--save-every-n-epochs', '--save_every_n_epochs', type=int, default=5, help='Save checkpoints every N epochs')
    parser.add_argument('--disable-transforms', action='store_true', help='Disable augmentations')
    parser.add_argument('--class-weights', '--class_weights', nargs='+', type=float, default=None, 
                       help='Class weights for loss function (e.g., --class-weights 50 10 1)')
    
    # New parameters for advanced models
    parser.add_argument('--encoder', type=str, default='resnet34', 
                       help='Encoder backbone (for UNet variants)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights for encoder')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder weights (for foundation models)')
    parser.add_argument('--two-stage', action='store_true',
                       help='Use two-stage training (binary then 3-class)')
    parser.add_argument('--stage1-epochs', type=int, default=15,
                       help='Epochs for stage 1 (binary) training')
    parser.add_argument('--stage2-epochs', type=int, default=35,
                       help='Epochs for stage 2 (3-class) training')
    
    # Loss-specific parameters
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Gamma parameter for focal losses')
    parser.add_argument('--tversky-alpha', type=float, default=0.7,
                       help='Alpha parameter for Tversky loss')
    parser.add_argument('--tversky-beta', type=float, default=0.3,
                       help='Beta parameter for Tversky loss')
    
    # Augmentation parameters
    parser.add_argument('--augmentation', type=str, default='full',
                       choices=['none', 'minimal', 'conservative', 'targeted', 'full',
                               'class_balanced', 'mixup', 'cutmix', 'mixup_cutmix', 'heavy',
                               'copy_paste', 'class_aware', 'smart_crop'],
                       help='Augmentation strategy')
    parser.add_argument('--use-mixup', action='store_true',
                       help='Use MixUp augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='MixUp alpha parameter')
    parser.add_argument('--use-cutmix', action='store_true',
                       help='Use CutMix augmentation')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0,
                       help='CutMix alpha parameter')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                       help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'exponential', 'onecycle', 'none'],
                       help='Learning rate scheduler')
    
    # Hyperparameter tuning
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Validation
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save model (for testing)')
    parser.add_argument('--patch-size', type=int, default=683,
                       help='Patch size for training')
    parser.add_argument(
        '--evaluation-patch-size',
        type=int,
        default=2048,
        help='Native validation/test tile size (confirmatory default: 2048)',
    )
    parser.add_argument(
        '--evaluation-batch-size',
        type=int,
        default=1,
        help='Validation/test batch size (confirmatory default: 1)',
    )
    parser.add_argument('--annotations-path', type=str, default=None,
                       help='Authoritative COCO annotation JSON (default: auto-detect under results/step3_coco_dataset)')
    parser.add_argument('--image-dir', type=str, default='results/step3_coco_dataset/images',
                       help='Directory containing COCO image files')
    parser.add_argument(
        '--mask-dir',
        type=str,
        default=None,
        help=(
            'Directory containing authoritative lossless masks with COCO image '
            'file names. Omit only to use legacy COCO polygon rasterization.'
        ),
    )
    parser.add_argument('--split-manifest', type=str, default=None,
                       help='JSON with disjoint train/val/test image IDs or file names (default: auto-detect)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation image fraction if no manifest exists')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Held-out test image fraction if no manifest exists')
    
    # Experiment tracking is opt-in so unattended cluster jobs never prompt for
    # credentials or publish run metadata unexpectedly.
    parser.add_argument('--wandb', action='store_true',
                       help=(
                           'Enable Weights & Biases tracking (disabled by default; '
                           'requires WANDB_API_KEY in the process environment)'
                       ))
    
    # Additional missing arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                       help='Number of warmup epochs')
    parser.add_argument('--hard-mining-ratio', type=float, default=None,
                       help='Ratio for online hard example mining')
    parser.add_argument('--curriculum-learning', action='store_true',
                       help='Use curriculum learning strategy')
    parser.add_argument('--post-process', type=str, default=None,
                       choices=['crf', 'morphology', 'graph_cut', 'guided_filter', 'threshold', None],
                       help='Post-processing method')
    parser.add_argument('--post-process-threshold', action='store_true',
                       help='Apply threshold post-processing (Config 1: C1→C2>120, C2→C0<30, no C1→C0)')
    parser.add_argument('--boundary-weight', type=float, default=1.0,
                       help='Weight for boundary loss')
    parser.add_argument('--yolo-model', type=str, default='yolov8s-seg',
                       help='YOLO model variant')
    parser.add_argument('--use-active-boundary', action='store_true',
                       help='Use active boundary learning')
    parser.add_argument('--use-hausdorff', action='store_true',
                       help='Use Hausdorff distance loss')
    parser.add_argument('--use-contour-loss', action='store_true',
                       help='Use contour-aware loss')
    parser.add_argument('--use-topological-boundary', action='store_true',
                       help='Use topological boundary loss')
    parser.add_argument('--use-graph-cut', action='store_true',
                       help='Use graph cut refinement')
    parser.add_argument('--use-guided-filter', action='store_true',
                       help='Use guided filter refinement')
    parser.add_argument('--multi-task', action='store_true',
                       help='Use multi-task learning')
    
    # Ablation mode arguments for minimal storage
    parser.add_argument('--ablation-mode', action='store_true',
                       help='Enable ablation mode: no checkpoints, minimal visualizations')
    parser.add_argument('--no-checkpoints', action='store_true',
                       help='Disable checkpoint saving completely')
    parser.add_argument('--overlay-only', action='store_true',
                       help='Only save overlay visualizations in plots/')
    parser.add_argument('--overwrite-plots', action='store_true',
                       help='Overwrite plots instead of creating new ones each epoch')
    
    # Testing/debugging arguments
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches per epoch (for testing)')
    parser.add_argument(
        '--validation-only',
        action='store_true',
        help=(
            'Fit and select on train/validation only; do not construct, read, '
            'or evaluate the held-out test dataset'
        ),
    )
    parser.add_argument(
        '--conditional-pore-threshold',
        type=int,
        default=None,
        help=(
            'Recovered raw-uint8 pore gate for the conditional C0/C1 '
            'validation-only candidate; the prespecified value is 100'
        ),
    )
    parser.add_argument(
        '--acknowledge-recovered-threshold-rule',
        action='store_true',
        help=(
            'Acknowledge that the operational gate is recovered from step-2 '
            'code and quantified train-only agreement, not bit-exact target '
            'reconstruction; data-owner confirmation remains pending'
        ),
    )
    parser.add_argument(
        '--selected-method-lock',
        type=str,
        default=None,
        help=(
            'Repository-relative JSON lock created only after the validation '
            'screen; required for validation-only selected-winner retraining. '
            'Only the separate locked evaluator may read held-out data.'
        ),
    )
    parser.add_argument(
        '--protocol-candidate-key',
        choices=['R3', 'H3', 'C2-P', 'C2-F', 'C2-FP'],
        default=None,
        help=(
            'Prespecified validation-screen candidate identity. Required for '
            'screen cells and selected-winner retraining.'
        ),
    )
    parser.add_argument(
        '--protocol-run-role',
        choices=[
            'validation_screen_cell',
            'validation_smoke_cell',
            'selected_winner_retraining',
        ],
        default=None,
    )
    parser.add_argument('--protocol-campaign-id', default=None)
    parser.add_argument('--protocol-cell-index', type=int, default=None)
    parser.add_argument(
        '--selected-architecture-role',
        choices=['primary_multiscale', 'plain_unet_comparator'],
        default=None,
    )
    parser.add_argument(
        '--smoke-preflight-manifest',
        default=None,
        help=(
            'Authenticated three-cell L40S smoke manifest; required only by '
            'scientific validation-screen cells'
        ),
    )
    
    args = parser.parse_args()

    if args.accumulate_grad_batches != 1 or args.gradient_accumulation != 1:
        parser.error(
            "Gradient accumulation is not implemented by PatchTrainer; use 1 so run metadata remains truthful."
        )
    if args.wandb and not os.environ.get('WANDB_API_KEY', '').strip():
        parser.error(
            "--wandb requires WANDB_API_KEY in the process environment; "
            "credentials are never accepted as command-line arguments"
        )

    from src.training.patch_trainer import PatchTrainer
    import torch
    
    # Check if running with torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.multi_gpu = True
    
    # Set environment variables
    if args.disable_transforms:
        os.environ['DISABLE_TRANSFORMS'] = '1'
    augmentation_settings = resolve_augmentation_settings(args)
    
    # Check if QUICK_TEST mode is enabled
    quick_test = os.environ.get('QUICK_TEST', '0') == '1'
    if quick_test:
        # Override batch size to 1 for fastest possible execution
        args.batch_size = 1
        print("⚡ QUICK TEST MODE: Setting batch_size=1 for fastest execution")
    
    print("\n" + "="*60)
    print("Patch-Based Training Configuration")
    print("="*60)
    print(f"Model: {args.model_type}")
    print(f"Loss Type: {args.loss_type}")
    print(f"Classes: {args.num_classes}")
    print(f"Batch Size: {args.batch_size if args.batch_size else 'from config'}")
    print(f"Learning Rate: {args.lr if args.lr else 'from config'}")
    print(f"Transforms: {'Disabled' if args.disable_transforms else 'Enabled'}")
    print(f"Image split manifest: {args.split_manifest or 'auto-detect/generate'}")
    print(
        f"Target source: {args.mask_dir or 'legacy COCO polygon rasterization'}"
    )
    if args.multi_gpu:
        print(f"Multi-GPU: {os.environ.get('WORLD_SIZE', 'N/A')} GPUs")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("="*60 + "\n")
    
    # Print environment variables
    print("\n[Environment Settings]")
    print(f"DISABLE_TRANSFORMS: {os.environ.get('DISABLE_TRANSFORMS', '0')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"\n[GPU Info]")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    
    print("=" * 50)
    
    # Handle ablation mode
    if args.ablation_mode:
        args.no_checkpoints = True
        args.overlay_only = True
        args.overwrite_plots = True
    if args.no_save:
        args.no_checkpoints = True

    requested_cli_arguments = vars(args).copy()
    
    # Create trainer with all configuration
    trainer = PatchTrainer(
        use_wandb=args.wandb,
        multi_gpu=args.multi_gpu,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        resume_path=args.resume,
        batch_size=args.batch_size,
        model_type=args.model_type,
        loss_type=args.loss_type,
        num_classes=args.num_classes,
        class_weights=args.class_weights,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        save_predictions=args.save_predictions,
        experiment_name=args.experiment_name,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        accumulate_grad_batches=args.accumulate_grad_batches,
        mixed_precision=args.mixed_precision,
        gradient_clip_val=args.gradient_clip_val,
        save_every_n_epochs=args.save_every_n_epochs,
        patch_size=args.patch_size,
        evaluation_patch_size=args.evaluation_patch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        no_checkpoints=args.no_checkpoints,
        overlay_only=args.overlay_only,
        overwrite_plots=args.overwrite_plots,
        max_batches=args.max_batches,
        optimizer_type=args.optimizer,
        momentum=args.momentum,
        scheduler_type=args.scheduler,
        augmentation_strength=augmentation_settings['strength'],
        augmentations_enabled=augmentation_settings['enabled'],
        use_mixup=augmentation_settings['use_mixup'],
        mixup_alpha=args.mixup_alpha,
        use_cutmix=augmentation_settings['use_cutmix'],
        cutmix_alpha=args.cutmix_alpha,
        split_manifest=args.split_manifest,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        annotations_path=args.annotations_path,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        requested_cli_arguments=requested_cli_arguments,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        validation_only=args.validation_only,
        conditional_pore_threshold=args.conditional_pore_threshold,
        recovered_threshold_acknowledged=(
            args.acknowledge_recovered_threshold_rule
        ),
        selected_method_lock=args.selected_method_lock,
        protocol_candidate_key=args.protocol_candidate_key,
        protocol_run_role=args.protocol_run_role,
        protocol_campaign_id=args.protocol_campaign_id,
        protocol_cell_index=args.protocol_cell_index,
        selected_architecture_role=args.selected_architecture_role,
        smoke_preflight_manifest=args.smoke_preflight_manifest,
    )
    
    # Train for specified epochs
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "=" * 50)
    print("✅ Patch training completed!")
    print("Check results/patch_training/ for outputs")
    print("=" * 50)


if __name__ == "__main__":
    main()

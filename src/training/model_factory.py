"""
Model and Loss Factory for all supported architectures
Centralizes model and loss creation for the training pipeline
"""

import torch
import torch.nn as nn
import warnings
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))


def create_advanced_model(model_type, num_classes=3, **kwargs):
    """
    Create model based on type string.
    
    Args:
        model_type: Model architecture name
        num_classes: Number of output classes
        **kwargs: Additional model-specific parameters
    
    Returns:
        Model instance
    """
    
    # Standard U-Net variants
    if model_type in [
        'unet',
        'plain_unet',
        'legacy_configured_unet',
        'improved_unet',
        'multiscale_attention_unet',
    ]:
        from src.models.unet_model import create_model
        from src.models.multiscale_attention_unet import create_multiscale_attention_unet
        
        if model_type == 'multiscale_attention_unet':
            # Use config-based creation
            config = kwargs.get('config', {})
            return create_multiscale_attention_unet(config)
        else:
            return create_model(model_type, num_classes=num_classes)
    
    # SegFormer variants
    elif model_type.startswith('segformer'):
        try:
            from src.models.segformer import get_segformer_model
            variant = model_type.split('_')[1] if '_' in model_type else 'b0'
            return get_segformer_model(variant=variant, num_classes=num_classes, 
                                      pretrained=kwargs.get('pretrained', True))
        except ImportError as e:
            warnings.warn(f"SegFormer not available: {e}. Using fallback UNet.")
            from src.models.unet_model import create_model
            return create_model('improved_unet', num_classes=num_classes)
    
    # The historical DINOv2 path used torch.hub with a mutable repository ref.
    # It is deliberately unavailable rather than silently downloading code or
    # substituting a mock network that would invalidate the requested model.
    elif model_type.startswith('dinov2'):
        from src.models.dinov2_unet import DINOV2_DISABLED_MESSAGE

        raise RuntimeError(DINOV2_DISABLED_MESSAGE)

    # DINOv3 requires an explicitly supplied local checkpoint.
    elif model_type in ['dinov3', 'dinov3_vits16']:
        from src.models.dinov3_unet import create_dinov3_unet

        return create_dinov3_unet(num_classes=num_classes, **kwargs)
    
    # YOLOv8 (would need separate implementation)
    elif model_type == 'yolov8':
        warnings.warn("YOLOv8 not yet implemented. Using UNet instead.")
        from src.models.unet_model import create_model
        return create_model('improved_unet', num_classes=num_classes)
    
    # SMP models with different encoders
    elif model_type.startswith('unet_'):
        try:
            import segmentation_models_pytorch as smp
            encoder = model_type.replace('unet_', '')
            
            return smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet' if kwargs.get('pretrained', True) else None,
                in_channels=1,
                classes=num_classes,
                activation=None
            )
        except Exception as e:
            warnings.warn(f"Could not create SMP model: {e}")
            from src.models.unet_model import create_model
            return create_model('improved_unet', num_classes=num_classes)
    
    else:
        warnings.warn(f"Unknown model type: {model_type}. Using default improved_unet.")
        from src.models.unet_model import create_model
        return create_model('improved_unet', num_classes=num_classes)


def create_advanced_loss(loss_type, num_classes=3, class_weights=None, **kwargs):
    """
    Create loss function based on type string.
    
    Args:
        loss_type: Loss function name
        num_classes: Number of classes
        class_weights: Optional class weights
        **kwargs: Additional loss-specific parameters
    
    Returns:
        Loss function instance
    """
    
    # Default class weights for our problem
    if class_weights is None:
        class_weights = [3.0, 2.0, 1.0]
    
    # Convert to tensor if needed
    if not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Standard losses
    if loss_type == 'combined':
        from src.models.combined_loss import create_loss_function
        return create_loss_function('combined', num_classes=num_classes, 
                                  class_weights=class_weights)
    
    elif loss_type == 'focal_dice':
        from src.models.combined_loss import create_loss_function
        return create_loss_function('focal_dice', num_classes=num_classes,
                                  class_weights=class_weights)
    
    elif loss_type == 'asymmetric':
        from src.models.asymmetric_loss import create_asymmetric_loss
        return create_asymmetric_loss(num_classes=num_classes, 
                                     class_weights=class_weights)
    
    elif loss_type == 'boundary_aware':
        from src.models.boundary_aware_loss import create_boundary_aware_loss
        return create_boundary_aware_loss(num_classes=num_classes,
                                        class_weights=class_weights)
    
    # Advanced boundary losses
    elif loss_type in ['boundary', 'active_boundary', 'hausdorff', 'contour', 'topological_boundary']:
        try:
            from src.losses.boundary_loss import create_boundary_loss
            return create_boundary_loss(
                loss_type=loss_type.replace('_boundary', '') if '_boundary' in loss_type else loss_type,
                num_classes=num_classes,
                **kwargs
            )
        except ImportError:
            warnings.warn(f"Boundary loss {loss_type} not available, using focal_tversky")
            loss_type = 'focal_tversky'
    
    # YOLOv8 Loss  
    elif loss_type == 'yolov8':
        try:
            from src.models.yolov8_seg import YOLOv8Loss
            return YOLOv8Loss(
                num_classes=num_classes,
                class_weights=class_weights,
                use_focal=kwargs.get('use_focal', True),
                focal_gamma=kwargs.get('focal_gamma', 2.0)
            )
        except ImportError:
            warnings.warn("YOLOv8Loss not available, using focal_tversky")
            loss_type = 'focal_tversky'
    
    elif loss_type == 'topological':
        from src.models.topological_loss import create_topological_loss
        return create_topological_loss(num_classes=num_classes,
                                      class_weights=class_weights)
    
    elif loss_type == 'sparse_pore':
        from src.models.sparse_pore_loss import create_sparse_pore_loss
        return create_sparse_pore_loss(num_classes=num_classes,
                                      class_weights=class_weights)
    
    elif loss_type == 'mineral_aware':
        from src.models.mineral_aware_loss import create_mineral_aware_loss
        return create_mineral_aware_loss(num_classes=num_classes,
                                       class_weights=class_weights)
    
    elif loss_type == 'binary_pore':
        from src.models.binary_pore_loss import create_binary_pore_loss
        return create_binary_pore_loss(num_classes=num_classes,
                                     class_weights=class_weights)
    
    # Focal Tversky variants
    elif loss_type == 'focal_tversky':
        from src.losses.focal_tversky import FocalTverskyLoss
        return FocalTverskyLoss(
            alpha=kwargs.get('tversky_alpha', 0.7),
            beta=kwargs.get('tversky_beta', 0.3),
            gamma=kwargs.get('focal_gamma', 2.0),
            num_classes=num_classes,
            class_weights=class_weights.tolist() if isinstance(class_weights, torch.Tensor) else class_weights
        )
    
    elif loss_type == 'adaptive_focal_tversky':
        from src.losses.focal_tversky import AdaptiveFocalTverskyLoss
        return AdaptiveFocalTverskyLoss(
            base_alpha=kwargs.get('tversky_alpha', 0.5),
            base_beta=kwargs.get('tversky_beta', 0.5),
            gamma=kwargs.get('focal_gamma', 2.0),
            num_classes=num_classes,
            class_weights=class_weights.tolist() if isinstance(class_weights, torch.Tensor) else class_weights
        )
    
    elif loss_type == 'unified_focal':
        from src.losses.focal_tversky import UnifiedFocalLoss
        return UnifiedFocalLoss(
            focal_weight=0.5,
            tversky_weight=0.5,
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            tversky_alpha=kwargs.get('tversky_alpha', 0.7),
            tversky_beta=kwargs.get('tversky_beta', 0.3),
            tversky_gamma=kwargs.get('focal_gamma', 2.0),
            num_classes=num_classes,
            class_weights=class_weights.tolist() if isinstance(class_weights, torch.Tensor) else class_weights
        )
    
    # Lovász variants
    elif loss_type == 'lovasz':
        from src.losses.lovasz_loss import LovaszSoftmaxLoss
        return LovaszSoftmaxLoss(
            classes='present',
            per_image=False,
            class_weights=class_weights.tolist() if isinstance(class_weights, torch.Tensor) else class_weights
        )
    
    elif loss_type == 'weighted_lovasz':
        from src.losses.lovasz_loss import WeightedLovaszSoftmaxLoss
        return WeightedLovaszSoftmaxLoss(
            lovasz_weight=0.5,
            ce_weight=0.5,
            classes='present',
            class_weights=class_weights.tolist() if isinstance(class_weights, torch.Tensor) else class_weights
        )
    
    elif loss_type == 'focal_lovasz':
        from src.losses.lovasz_loss import FocalLovaszLoss
        return FocalLovaszLoss(
            focal_weight=0.3,
            lovasz_weight=0.7,
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            class_weights=class_weights.tolist() if isinstance(class_weights, torch.Tensor) else class_weights
        )
    
    elif loss_type == 'asymmetric_lovasz':
        from src.losses.lovasz_loss import AsymmetricLovaszLoss
        return AsymmetricLovaszLoss(
            fp_weight=1.0,
            fn_weights=[20.0, 5.0, 1.0],  # Higher penalty for minority class FN
            classes='present'
        )
    
    # Focal loss
    elif loss_type == 'focal':
        from src.models.focal_loss import create_focal_loss
        return create_focal_loss(
            gamma=kwargs.get('focal_gamma', 2.0),
            num_classes=num_classes,
            class_weights=class_weights
        )
    
    else:
        warnings.warn(f"Unknown loss type: {loss_type}. Using default focal_dice.")
        from src.models.combined_loss import create_loss_function
        return create_loss_function('focal_dice', num_classes=num_classes,
                                  class_weights=class_weights)


def create_optimizer(model, optimizer_type='adamw', lr=0.001, weight_decay=0.0001, **kwargs):
    """
    Create optimizer based on type.
    
    Args:
        model: Model to optimize
        optimizer_type: Optimizer name
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer-specific parameters
    
    Returns:
        Optimizer instance
    """
    
    # Get trainable parameters
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    elif optimizer_type == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', True)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, 
                             weight_decay=weight_decay, nesterov=nesterov)
    
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    
    else:
        warnings.warn(f"Unknown optimizer: {optimizer_type}. Using AdamW.")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, scheduler_type='cosine', epochs=50, **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Scheduler name
        epochs: Total epochs for training
        **kwargs: Additional scheduler-specific parameters
    
    Returns:
        Scheduler instance or None
    """
    
    if scheduler_type == 'cosine':
        T_0 = kwargs.get('T_0', 10)
        T_mult = kwargs.get('T_mult', 2)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult
        )
    
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'onecycle':
        max_lr = kwargs.get('max_lr', optimizer.param_groups[0]['lr'])
        steps_per_epoch = kwargs.get('steps_per_epoch', 100)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, 
            steps_per_epoch=steps_per_epoch
        )
    
    elif scheduler_type == 'none':
        return None
    
    else:
        warnings.warn(f"Unknown scheduler: {scheduler_type}. Using cosine.")
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )


# Test implementation
if __name__ == "__main__":
    print("Testing Model and Loss Factory...")
    
    # Test model creation
    print("\n1. Testing model creation:")
    models_to_test = [
        'unet', 'improved_unet', 'multiscale_attention_unet',
        'segformer_b0', 'dinov2_small', 'dinov2_hybrid'
    ]
    
    for model_type in models_to_test:
        try:
            model = create_advanced_model(model_type, num_classes=3)
            print(f"   ✓ {model_type}: Created successfully")
        except Exception as e:
            print(f"   ✗ {model_type}: {e}")
    
    # Test loss creation
    print("\n2. Testing loss creation:")
    losses_to_test = [
        'focal_tversky', 'lovasz', 'focal_lovasz', 
        'asymmetric_lovasz', 'unified_focal'
    ]
    
    for loss_type in losses_to_test:
        try:
            loss = create_advanced_loss(loss_type, num_classes=3)
            print(f"   ✓ {loss_type}: Created successfully")
        except Exception as e:
            print(f"   ✗ {loss_type}: {e}")
    
    print("\nFactory functions tested successfully!")

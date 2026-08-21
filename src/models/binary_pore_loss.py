"""
Binary loss function for pore segmentation (disconnected vs connected).
Heavily weights disconnected pores since they are much rarer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BinaryPoreFocalLoss(nn.Module):
    """
    Focal loss adapted for binary pore classification.
    Penalizes both false negatives AND false positives appropriately.
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,  # Per-class weights
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Per-class weights tensor [weight_class0, weight_class1]
                   If None, uses [0.6, 0.4] to slightly favor disconnected pores
            gamma: Focusing parameter
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        if alpha is None:
            # Default: slightly favor disconnected pores but not too extreme
            self.alpha = torch.tensor([0.6, 0.4])
        else:
            self.alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, 2, H, W) logits
            targets: (N, H, W) class indices (0 or 1, -100 for ignore)
        """
        # Get softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get cross entropy loss with ignore_index
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=-100)
        
        # Create valid mask (non-ignored pixels)
        valid_mask = targets != -100
        
        # If no valid pixels, return 0
        if not valid_mask.any():
            if self.reduction == 'mean':
                return torch.tensor(0.0, device=inputs.device)
            elif self.reduction == 'sum':
                return torch.tensor(0.0, device=inputs.device)
            else:
                return torch.zeros_like(ce_loss)
        
        # Get class probabilities only for valid pixels
        valid_targets = targets.clone()
        valid_targets[~valid_mask] = 0  # Set ignored pixels to 0 temporarily for gather
        p_t = torch.gather(p, 1, valid_targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply focal term (ce_loss already has 0 for ignored pixels)
        loss = focal_term * ce_loss
        
        # Apply per-class alpha weighting
        # Move alpha to same device as inputs
        alpha = self.alpha.to(inputs.device)
        
        # Create alpha tensor for all pixels
        alpha_t_full = torch.zeros_like(targets, dtype=torch.float32)
        
        # For valid pixels, select alpha based on target class
        if valid_mask.any():
            valid_targets_flat = targets[valid_mask]
            alpha_t = alpha[valid_targets_flat]  # This gives us a 1D tensor
            alpha_t_full[valid_mask] = alpha_t
        
        loss = alpha_t_full * loss
        
        # Reduction
        if self.reduction == 'mean':
            # Mean over valid pixels only
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BinaryPoreDiceLoss(nn.Module):
    """Dice loss for binary pore segmentation."""
    
    def __init__(self, smooth: float = 1e-6, class_weight: float = 5.0):
        """
        Args:
            smooth: Smoothing factor
            class_weight: Weight for disconnected pores (class 0)
        """
        super().__init__()
        self.smooth = smooth
        self.class_weight = class_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (N, 2, H, W) logits
            targets: (N, H, W) class indices
        """
        # Filter out ignore pixels
        valid_mask = targets != -100
        
        # If no valid pixels, return 0 loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device)
        
        # Convert to one-hot (only for valid pixels)
        valid_targets = targets[valid_mask]
        targets_one_hot = F.one_hot(valid_targets, 2).float()
        
        # Apply softmax
        predictions = F.softmax(predictions, dim=1)
        
        # Calculate dice loss for each class
        dice_loss = 0
        
        # Extract valid predictions
        valid_predictions = predictions.permute(0, 2, 3, 1)[valid_mask]  # Shape: (N_valid, 2)
        
        # Class 0 (disconnected) - heavily weighted
        pred_0 = valid_predictions[:, 0]
        target_0 = targets_one_hot[:, 0]
        intersection_0 = (pred_0 * target_0).sum()
        union_0 = pred_0.sum() + target_0.sum()
        dice_0 = 1 - (2 * intersection_0 + self.smooth) / (union_0 + self.smooth)
        dice_loss += dice_0 * self.class_weight
        
        # Class 1 (connected)
        pred_1 = valid_predictions[:, 1]
        target_1 = targets_one_hot[:, 1]
        intersection_1 = (pred_1 * target_1).sum()
        union_1 = pred_1.sum() + target_1.sum()
        dice_1 = 1 - (2 * intersection_1 + self.smooth) / (union_1 + self.smooth)
        dice_loss += dice_1
        
        return dice_loss / (1 + self.class_weight)  # Normalize


class BinaryPoreCombinedLoss(nn.Module):
    """
    Combined focal + dice loss for binary pore segmentation.
    Balanced to prevent extreme predictions while handling class imbalance.
    """
    
    def __init__(self, 
                 focal_alpha: Optional[torch.Tensor] = None,
                 focal_gamma: float = 2.0,
                 dice_class_weight: float = 5.0,
                 focal_weight: float = 0.7,
                 dice_weight: float = 0.3):
        super().__init__()
        
        self.focal_loss = BinaryPoreFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = BinaryPoreDiceLoss(class_weight=dice_class_weight)
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        # Add regularization to prevent predicting everything as one class
        # Penalize if predictions are too skewed
        probs = F.softmax(inputs, dim=1)
        valid_mask = targets != -100
        
        if valid_mask.any():
            # Calculate mean predicted probability for each class over valid pixels
            mean_probs = []
            for c in range(2):
                class_probs = probs[:, c][valid_mask]
                mean_probs.append(class_probs.mean())
            
            # Penalty increases as predictions become more extreme
            # Ideal would be close to actual class distribution, but we penalize extremes
            regularization = 0.0
            for prob in mean_probs:
                # Penalize if any class gets too high average probability
                if prob > 0.8:  # If predicting >80% of pixels as one class
                    regularization += (prob - 0.8) * 2.0
            
            total_loss = self.focal_weight * focal + self.dice_weight * dice + 0.1 * regularization
        else:
            total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return total_loss


def create_binary_pore_loss(config):
    """Create the binary pore loss function from config."""
    # Get loss parameters from config
    if hasattr(config, 'get'):
        # ConfigLoader object
        class_weights = config.get('model.loss.class_weights', [5.0, 1.0])
        focal_weight = config.get('model.loss.focal_weight', 0.7)
        dice_weight = config.get('model.loss.dice_weight', 0.3)
        gamma = config.get('model.loss.gamma', 2.0)
    else:
        # Dictionary
        loss_config = config.get('model', {}).get('loss', {})
        class_weights = loss_config.get('class_weights', [5.0, 1.0])
        focal_weight = loss_config.get('focal_weight', 0.7)
        dice_weight = loss_config.get('dice_weight', 0.3)
        gamma = loss_config.get('gamma', 2.0)
    
    # Normalize class weights to get alpha values
    # But keep them more balanced to avoid extreme predictions
    total_weight = sum(class_weights)
    normalized_weights = [w / total_weight for w in class_weights]
    
    # Apply softening to prevent extreme weights
    # This ensures we penalize false positives for both classes
    softening_factor = 0.7  # Brings weights closer to 0.5
    focal_alpha = [
        0.5 + (normalized_weights[0] - 0.5) * softening_factor,
        0.5 + (normalized_weights[1] - 0.5) * softening_factor
    ]
    
    return BinaryPoreCombinedLoss(
        focal_alpha=torch.tensor(focal_alpha),
        focal_gamma=gamma,
        dice_class_weight=class_weights[0] / class_weights[1],
        focal_weight=focal_weight,
        dice_weight=dice_weight
    )
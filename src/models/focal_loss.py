"""
Focal loss and other advanced loss functions for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Class weights (C,) tensor
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) logits
            targets: (N, H, W) class indices
        """
        # Keep rare-class log-probabilities and reductions in float32 under AMP.
        # Computing softmax in float16 can round an extreme hard-class
        # probability to zero and erase the focal gradient.
        log_probabilities = F.log_softmax(inputs.float(), dim=1)
        log_p_t = torch.gather(
            log_probabilities, 1, targets.unsqueeze(1)
        ).squeeze(1)
        p_t = log_p_t.exp()
        ce_loss = -log_p_t
        
        # Calculate focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply focal term to cross entropy
        loss = focal_term * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(device=inputs.device, dtype=torch.float32)
            alpha_t = alpha[targets]
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    """
    Tversky loss for better handling of false positives/negatives.
    Allows control over FP/FN trade-off.
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        """
        Args:
            alpha: Weight for false positives (higher = penalize FP more)
            beta: Weight for false negatives (higher = penalize FN more)
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: (N, C, H, W) logits
            targets: (N, H, W) class indices
            class_weights: (C,) tensor of class weights
        """
        num_classes = predictions.shape[1]
        
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax
        predictions = F.softmax(predictions, dim=1)
        
        # Calculate Tversky index for each class
        tversky_loss = 0
        for c in range(num_classes):
            pred_c = predictions[:, c]
            target_c = targets_one_hot[:, c]
            
            # True positives, false positives, false negatives
            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()
            
            # Tversky index
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            
            # Apply class weight if provided
            if class_weights is not None:
                tversky_loss += class_weights[c] * (1 - tversky)
            else:
                tversky_loss += (1 - tversky)
        
        return tversky_loss / num_classes


class CombinedFocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice loss for severe class imbalance.
    """
    
    def __init__(self, class_weights: Optional[list] = None,
                 focal_weight: float = 0.5, dice_weight: float = 0.5,
                 gamma: float = 2.0, dice_smooth: float = 1e-6):
        """
        Args:
            class_weights: List of weights for each class
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            gamma: Focal loss gamma parameter
            dice_smooth: Dice loss smoothing factor
        """
        super().__init__()
        
        # Default class weights for pore segmentation
        if class_weights is None:
            # Much higher weights for rare classes
            class_weights = [10.0, 5.0, 1.0]  # [disconnected, connected, minerals]
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.gamma = float(gamma)
        self.dice_smooth = float(dice_smooth)
        
        # Initialize component losses
        self.focal_loss = FocalLoss(alpha=self.class_weights, gamma=self.gamma)
        
        # Import DiceLoss from combined_loss.py
        from .combined_loss import DiceLoss
        self.dice_loss = DiceLoss(smooth=self.dice_smooth)
    
    def to(self, device):
        """Move to device."""
        self.class_weights = self.class_weights.to(device)
        self.focal_loss.alpha = self.class_weights
        return super().to(device)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: (N, C, H, W) tensor of logits
            targets: (N, H, W) tensor of class indices
        """
        # Calculate individual losses
        focal = self.focal_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets, self.class_weights)
        
        # Combine losses
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return total_loss

    def resolved_config(self):
        """Return the actual instantiated focal-Dice objective parameters."""
        return {
            'implementation_class': type(self).__name__,
            'components': ['focal_cross_entropy', 'weighted_soft_dice'],
            'component_weights_actual': {
                'focal': self.focal_weight,
                'dice': self.dice_weight,
            },
            'focal_gamma_actual': self.gamma,
            'dice_smooth_actual': self.dice_smooth,
            'class_weights_actual': [
                float(value)
                for value in self.class_weights.detach().cpu().tolist()
            ],
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to improve feature separation between classes.
    Particularly useful for distinguishing connected vs disconnected pores.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        """
        Args:
            margin: Margin for contrastive loss
            temperature: Temperature for similarity scaling
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, D) feature vectors
            labels: (N,) class labels
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.t()) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float()
        
        # Mask out diagonal
        mask = mask - torch.eye(mask.shape[0], device=mask.device)
        
        # Compute contrastive loss
        pos_sim = similarity * mask
        neg_sim = similarity * (1 - mask)
        
        # Apply margin to negative pairs
        neg_sim = torch.clamp(self.margin - neg_sim, min=0)
        
        # Average over positive and negative pairs
        loss = (pos_sim.sum() + neg_sim.sum()) / (mask.sum() + (1 - mask).sum())
        
        return loss


def create_advanced_loss_function(config):
    """
    Create advanced loss function based on configuration.
    
    Args:
        config: Configuration object or dictionary
    
    Returns:
        Loss function module
    """
    # Handle both ConfigLoader objects and dictionaries
    if hasattr(config, 'get'):
        # ConfigLoader object
        loss_type = config.get('model.loss.type', 'focal_dice')
        class_weights = config.get('model.loss.class_weights', [10.0, 5.0, 1.0])
        focal_weight = config.get('model.loss.focal_weight', 0.5)
        dice_weight = config.get('model.loss.dice_weight', 0.5)
        gamma = config.get('model.loss.gamma', 2.0)
        alpha = config.get('model.loss.tversky_alpha', 0.7)
        beta = config.get('model.loss.tversky_beta', 0.3)
    else:
        # Dictionary
        loss_config = config.get('model', {}).get('loss', {})
        loss_type = loss_config.get('type', 'focal_dice')
        class_weights = loss_config.get('class_weights', [10.0, 5.0, 1.0])
        focal_weight = loss_config.get('focal_weight', 0.5)
        dice_weight = loss_config.get('dice_weight', 0.5)
        gamma = loss_config.get('gamma', 2.0)
        alpha = loss_config.get('tversky_alpha', 0.7)
        beta = loss_config.get('tversky_beta', 0.3)
    
    if loss_type == 'focal_dice':
        return CombinedFocalDiceLoss(
            class_weights=class_weights,
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            gamma=gamma
        )
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=torch.tensor(class_weights),
            gamma=gamma
        )
    
    elif loss_type == 'tversky':
        return TverskyLoss(alpha=alpha, beta=beta)
    
    else:
        # Fall back to original combined loss
        from .combined_loss import create_loss_function
        return create_loss_function(config)

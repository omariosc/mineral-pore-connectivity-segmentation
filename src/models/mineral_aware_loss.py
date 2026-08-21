"""
Mineral-aware loss function for 2-class pore segmentation.
Incorporates mineral class information to penalize false positives/negatives more accurately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class MineralAwareFocalLoss(nn.Module):
    """
    Focal loss for 3-class segmentation where class 2 is a throwaway background class.
    Classes: 0=disconnected pores, 1=connected pores, 2=background/minerals (throwaway)
    """
    
    def __init__(self, 
                 pore_weight: float = 2.0,
                 background_weight: float = 0.1,  # Low weight for throwaway class
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            pore_weight: Weight for pore classes (0 and 1)
            background_weight: Weight for background class (2) - kept low since it's throwaway
            gamma: Focusing parameter for focal loss
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.class_weights = torch.tensor([pore_weight, pore_weight, background_weight])
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                mineral_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: (N, 3, H, W) logits for 3 classes
            targets: (N, H, W) targets (0=disconnected, 1=connected, 2=background)
            mineral_mask: Not used in this version (kept for compatibility)
        """
        # Move weights to same device as inputs
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # Use weighted cross entropy with reduction='none' to apply focal term
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        
        # Get softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get predicted probabilities for the true class
        p_t = torch.gather(p, 1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal term
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MineralAwareDiceLoss(nn.Module):
    """Dice loss for 3-class segmentation, focusing on pore classes only."""
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mineral_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: (N, 3, H, W) logits
            targets: (N, H, W) class indices (0, 1, or 2)
            mineral_mask: Not used (kept for compatibility)
        """
        # Apply softmax
        predictions = F.softmax(predictions, dim=1)
        
        # Calculate dice loss only for pore classes (0 and 1)
        # Ignore background class (2) in dice calculation
        dice_losses = []
        
        for class_idx in [0, 1]:  # Only pore classes
            pred_class = predictions[:, class_idx]
            target_class = (targets == class_idx).float()
            
            intersection = (pred_class * target_class).sum()
            pred_sum = pred_class.sum()
            target_sum = target_class.sum()
            
            dice_score = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_losses.append(1 - dice_score)
        
        # Average dice loss across pore classes
        return sum(dice_losses) / len(dice_losses)


class MineralAwareCombinedLoss(nn.Module):
    """
    Combined focal + dice loss for 3-class segmentation with throwaway background class.
    """
    
    def __init__(self, 
                 pore_weight: float = 2.0,
                 background_weight: float = 0.1,
                 focal_gamma: float = 2.0,
                 focal_weight: float = 0.7,
                 dice_weight: float = 0.3):
        super().__init__()
        
        self.focal_loss = MineralAwareFocalLoss(
            pore_weight=pore_weight,
            background_weight=background_weight,
            gamma=focal_gamma
        )
        
        self.dice_loss = MineralAwareDiceLoss()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                mineral_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: (N, 3, H, W) model outputs
            targets: (N, H, W) targets (0, 1, or 2)
            mineral_mask: Not used (kept for compatibility)
        """
        focal = self.focal_loss(inputs, targets, mineral_mask)
        dice = self.dice_loss(inputs, targets, mineral_mask)
        
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return total_loss


def create_mineral_aware_loss(config: Dict) -> nn.Module:
    """Create mineral-aware loss function from config."""
    # Get loss parameters
    if hasattr(config, 'get'):
        # ConfigLoader object
        pore_weight = config.get('model.loss.pore_weight', 2.0)
        background_weight = config.get('model.loss.background_weight', 0.1)
        focal_weight = config.get('model.loss.focal_weight', 0.7)
        dice_weight = config.get('model.loss.dice_weight', 0.3)
        gamma = config.get('model.loss.gamma', 2.0)
    else:
        # Dictionary
        loss_config = config.get('model', {}).get('loss', {})
        pore_weight = loss_config.get('pore_weight', 2.0)
        background_weight = loss_config.get('background_weight', 0.1)
        focal_weight = loss_config.get('focal_weight', 0.7)
        dice_weight = loss_config.get('dice_weight', 0.3)
        gamma = loss_config.get('gamma', 2.0)
    
    return MineralAwareCombinedLoss(
        pore_weight=pore_weight,
        background_weight=background_weight,
        focal_gamma=gamma,
        focal_weight=focal_weight,
        dice_weight=dice_weight
    )
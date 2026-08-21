"""
Combined loss function for 3-class segmentation with class weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            predictions: (N, C, H, W) tensor of logits
            targets: (N, H, W) tensor of class indices
            class_weights: (C,) tensor of class weights
        """
        num_classes = predictions.shape[1]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(
            targets, num_classes
        ).permute(0, 3, 1, 2).to(dtype=torch.float32)
        
        # Apply softmax to predictions
        predictions = F.softmax(predictions.float(), dim=1)
        
        # Calculate Dice coefficient for each class
        dice_loss = predictions.new_zeros(())
        for c in range(num_classes):
            pred_c = predictions[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice_c = (2 * intersection + self.smooth) / (union + self.smooth)
            
            # Apply class weight if provided
            if class_weights is not None:
                weight = class_weights[c].to(
                    device=predictions.device, dtype=torch.float32
                )
                dice_loss += weight * (1 - dice_c)
            else:
                dice_loss += (1 - dice_c)
        
        return dice_loss / num_classes


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice loss with class weights.
    Specifically designed for 3-class mineral pore segmentation.
    """
    
    def __init__(self, class_weights: Optional[list] = None, 
                 ce_weight: float = 0.5, dice_weight: float = 0.5):
        """
        Initialize combined loss.
        
        Args:
            class_weights: List of weights for each class [disconnected, connected, mineral]
            ce_weight: Weight for cross-entropy component
            dice_weight: Weight for Dice component
        """
        super().__init__()
        
        # Default class weights if not provided
        if class_weights is None:
            class_weights = [3.0, 2.0, 1.0]  # Higher weight for rare classes
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # Cross-entropy loss with class weights
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Dice loss
        self.dice_loss = DiceLoss()
    
    def to(self, device):
        """Move loss to device."""
        self.class_weights = self.class_weights.to(device)
        self.ce_loss.weight = self.class_weights
        return super().to(device)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: (N, C, H, W) tensor of logits
            targets: (N, H, W) tensor of class indices
        """
        # Ensure class weights are on the same device as predictions
        if self.class_weights.device != predictions.device:
            self.class_weights = self.class_weights.to(predictions.device)
            self.ce_loss.weight = self.class_weights
        
        # Calculate individual losses
        ce_loss = self.ce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets, self.class_weights)
        
        # Combine losses
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss


def create_loss_function(config) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Configuration object or dictionary
    
    Returns:
        Loss function module
    """
    # Handle both ConfigLoader objects and dictionaries
    if hasattr(config, 'get'):
        # ConfigLoader object
        loss_type = config.get('model.loss.type', 'combined')
        class_weights = config.get('model.loss.class_weights', [3.0, 2.0, 1.0])
        ce_weight = config.get('model.loss.ce_weight', 0.5)
        dice_weight = config.get('model.loss.dice_weight', 0.5)
    else:
        # Dictionary
        loss_config = config.get('model', {}).get('loss', {})
        loss_type = loss_config.get('type', 'combined')
        class_weights = loss_config.get('class_weights', [3.0, 2.0, 1.0])
        ce_weight = loss_config.get('ce_weight', 0.5)
        dice_weight = loss_config.get('dice_weight', 0.5)
    
    if loss_type == 'combined':
        return CombinedLoss(
            class_weights=class_weights,
            ce_weight=ce_weight,
            dice_weight=dice_weight
        )
    
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    
    elif loss_type == 'dice':
        return DiceLoss()
    
    elif loss_type == 'focal_dice':
        # Import and use focal_dice from focal_loss module
        from .focal_loss import CombinedFocalDiceLoss
        # Get additional parameters for focal_dice
        if hasattr(config, 'get'):
            focal_weight = config.get('model.loss.focal_weight', 0.5)
            dice_weight = config.get('model.loss.dice_weight', 0.5)
            gamma = config.get('model.loss.gamma', 2.0)
        else:
            loss_config = config.get('model', {}).get('loss', {})
            focal_weight = loss_config.get('focal_weight', 0.5)
            dice_weight = loss_config.get('dice_weight', 0.5)
            gamma = loss_config.get('gamma', 2.0)
        
        return CombinedFocalDiceLoss(
            class_weights=class_weights,
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            gamma=gamma
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

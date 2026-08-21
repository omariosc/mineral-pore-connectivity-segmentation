"""
Asymmetric loss function that penalizes false negatives for disconnected pores (class 0)
and false positives for minerals (class 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss that allows different penalties for FP and FN per class.
    Specifically designed for the mineral pore segmentation task.
    """
    
    def __init__(self, 
                 gamma_pos: float = 2.0,
                 gamma_neg: float = 4.0,
                 alpha_pos: Optional[torch.Tensor] = None,
                 alpha_neg: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            gamma_pos: Focusing parameter for positive samples
            gamma_neg: Focusing parameter for negative samples
            alpha_pos: Per-class weights for positive samples
            alpha_neg: Per-class weights for negative samples
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) logits
            targets: (N, H, W) class indices
        """
        num_classes = inputs.shape[1]
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Calculate losses for positive and negative samples
        pos_loss = -targets_one_hot * torch.log(probs.clamp(min=1e-7))
        neg_loss = -(1 - targets_one_hot) * torch.log((1 - probs).clamp(min=1e-7))
        
        # Apply focusing parameters
        pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)
        neg_loss = neg_loss * (probs ** self.gamma_neg)
        
        # Apply class-specific weights
        if self.alpha_pos is not None:
            alpha_pos = self.alpha_pos.to(inputs.device)
            pos_loss = pos_loss * alpha_pos.view(1, -1, 1, 1)
        if self.alpha_neg is not None:
            alpha_neg = self.alpha_neg.to(inputs.device)
            neg_loss = neg_loss * alpha_neg.view(1, -1, 1, 1)
        
        # Combine losses
        loss = pos_loss + neg_loss
        
        # Sum over classes
        loss = loss.sum(dim=1)
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CustomPoreLoss(nn.Module):
    """
    Custom loss function specifically for mineral pore segmentation.
    Heavily penalizes false negatives for disconnected pores (class 0)
    and false positives for minerals (class 2).
    """
    
    def __init__(self, 
                 fn_weight_class0: float = 20.0,  # Heavy penalty for missing disconnected pores
                 fp_weight_class2: float = 5.0,    # Penalty for falsely predicting minerals
                 base_weights: Optional[list] = None):
        super().__init__()
        
        # Base class weights
        if base_weights is None:
            base_weights = [10.0, 5.0, 1.0]
        
        # Create asymmetric weights
        # For positive samples (when target is this class)
        self.alpha_pos = torch.tensor([
            base_weights[0] * fn_weight_class0,  # Class 0: heavily weight FN
            base_weights[1],                      # Class 1: normal weight
            base_weights[2]                       # Class 2: normal weight
        ])
        
        # For negative samples (when target is NOT this class)
        self.alpha_neg = torch.tensor([
            base_weights[0],                      # Class 0: normal weight for FP
            base_weights[1],                      # Class 1: normal weight
            base_weights[2] * fp_weight_class2    # Class 2: heavily weight FP
        ])
        
        # Create asymmetric focal loss
        self.focal_loss = AsymmetricFocalLoss(
            gamma_pos=2.0,
            gamma_neg=4.0,
            alpha_pos=self.alpha_pos,
            alpha_neg=self.alpha_neg
        )
        
        # Also use dice loss for spatial consistency
        self.dice_loss = DiceLoss(class_weights=torch.tensor(base_weights))
        
        self.focal_weight = 0.7
        self.dice_weight = 0.3
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice


class DiceLoss(nn.Module):
    """Dice loss for segmentation with per-class weights."""
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, smooth: float = 1e-6):
        super().__init__()
        self.class_weights = class_weights
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = predictions.shape[1]
        
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax
        predictions = F.softmax(predictions, dim=1)
        
        # Calculate dice loss for each class
        dice_loss = 0
        for i in range(num_classes):
            pred_i = predictions[:, i]
            target_i = targets_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice_i = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            
            if self.class_weights is not None:
                weight = self.class_weights[i].to(predictions.device) if hasattr(self.class_weights[i], 'to') else self.class_weights[i]
                dice_i = dice_i * weight
            
            dice_loss += dice_i
        
        return dice_loss / num_classes


def create_asymmetric_loss(config):
    """Create the asymmetric loss function from config."""
    return CustomPoreLoss(
        fn_weight_class0=20.0,
        fp_weight_class2=5.0,
        base_weights=[10.0, 5.0, 1.0]
    )
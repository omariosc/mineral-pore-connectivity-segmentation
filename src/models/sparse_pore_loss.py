"""
Sparse-aware loss function for pore segmentation.
Heavily penalizes false positives to prevent over-prediction of pores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class SparsePoreLoss(nn.Module):
    """
    Loss function designed for sparse pore detection.
    Key principle: Most pixels are background, so penalize false positives heavily.
    """
    
    def __init__(self, 
                 false_positive_weight: float = 10.0,  # Heavy penalty for predicting pores on background
                 pore_positive_weight: float = 3.0,    # Reward for correctly detecting pores
                 background_weight: float = 1.0,       # Standard weight for background
                 focal_gamma: float = 2.0,
                 expected_pore_ratio: float = 0.15):   # Expected ratio of pore pixels
        """
        Args:
            false_positive_weight: Penalty multiplier for false positive pore predictions
            pore_positive_weight: Weight for true positive pore predictions
            background_weight: Weight for correct background predictions
            focal_gamma: Focal loss gamma parameter
            expected_pore_ratio: Expected ratio of pore pixels in the dataset
        """
        super().__init__()
        self.false_positive_weight = false_positive_weight
        self.pore_positive_weight = pore_positive_weight
        self.background_weight = background_weight
        self.focal_gamma = focal_gamma
        self.expected_pore_ratio = expected_pore_ratio
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, 3, H, W) logits for 3 classes
            targets: (N, H, W) targets (0=disconnected, 1=connected, 2=background)
        """
        # Handle 2-class dataset with 3-class model
        # For 2-class dataset, -100 is used for ignore pixels which should be treated as background (class 2)
        targets_orig = targets.clone()
        targets = targets.clone()
        targets[targets == -100] = 2  # Convert ignore pixels to background class
        
        # Get predictions
        probs = F.softmax(inputs, dim=1)
        _, predicted = inputs.max(1)
        
        # Calculate cross entropy with reduction='none' and ignore_index for original -100 pixels
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Create weight map based on prediction errors
        weight_map = torch.ones_like(targets, dtype=torch.float32)
        
        # Heavy penalty for false positive pores (predicting 0 or 1 when target is 2)
        false_positive_mask = ((predicted == 0) | (predicted == 1)) & (targets == 2)
        weight_map[false_positive_mask] = self.false_positive_weight
        
        # Reward for true positive pores
        true_positive_mask = ((predicted == 0) & (targets == 0)) | ((predicted == 1) & (targets == 1))
        weight_map[true_positive_mask] = self.pore_positive_weight
        
        # Standard weight for correct background predictions
        true_negative_mask = (predicted == 2) & (targets == 2)
        weight_map[true_negative_mask] = self.background_weight
        
        # Don't apply loss to originally ignored pixels
        valid_mask = targets_orig != -100
        weight_map[~valid_mask] = 0.0
        
        # Apply focal term to focus on hard examples
        # Clamp targets to valid range for gather operation
        targets_clamped = targets.clamp(0, inputs.size(1) - 1)
        p_t = torch.gather(probs, 1, targets_clamped.unsqueeze(1)).squeeze(1)
        focal_term = (1 - p_t) ** self.focal_gamma
        
        # Combine all terms
        loss = weight_map * focal_term * ce_loss
        
        # Additional regularization: penalize if too many pixels are predicted as pores
        pore_predictions = ((predicted == 0) | (predicted == 1)).float()
        pore_predictions = pore_predictions * valid_mask.float()  # Only count valid pixels
        pore_ratio = pore_predictions.sum() / valid_mask.sum().clamp(min=1)
        
        # Add penalty if predicting too many pores
        if pore_ratio > self.expected_pore_ratio:
            ratio_penalty = (pore_ratio - self.expected_pore_ratio) * 10.0
            loss = loss * (1 + ratio_penalty)
        
        # Only average over valid pixels
        if valid_mask.any():
            return loss[valid_mask].mean()
        else:
            return loss.mean()


class SparsePoreCombinedLoss(nn.Module):
    """
    Combined loss with sparse-aware focal loss and dice loss for pores only.
    """
    
    def __init__(self,
                 false_positive_weight: float = 10.0,
                 focal_weight: float = 0.7,
                 dice_weight: float = 0.3):
        super().__init__()
        
        self.focal_loss = SparsePoreLoss(
            false_positive_weight=false_positive_weight,
            pore_positive_weight=3.0,
            background_weight=1.0,
            focal_gamma=2.0,
            expected_pore_ratio=0.15
        )
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-6
        
    def dice_loss_sparse(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Dice loss that only considers pore classes and penalizes over-prediction.
        """
        # Handle 2-class dataset with 3-class model
        targets_orig = targets.clone()
        targets = targets.clone()
        targets[targets == -100] = 2  # Convert ignore pixels to background class
        
        # Apply softmax
        predictions = F.softmax(predictions, dim=1)
        
        dice_losses = []
        
        # Calculate dice only for pore classes
        for class_idx in [0, 1]:
            pred_class = predictions[:, class_idx]
            target_class = (targets == class_idx).float()
            
            # Only consider valid pixels (not originally -100)
            valid_mask = (targets_orig != -100).float()
            pred_class = pred_class * valid_mask
            target_class = target_class * valid_mask
            
            # Add penalty for over-prediction
            intersection = (pred_class * target_class).sum()
            pred_sum = pred_class.sum()
            target_sum = target_class.sum()
            
            # Skip if no target pixels for this class
            if target_sum == 0:
                continue
            
            # Modified dice that penalizes when pred_sum >> target_sum
            if pred_sum > target_sum * 1.5:  # If predicting 50% more pixels than target
                penalty = (pred_sum / (target_sum + 1e-6)).clamp(max=10.0)
                dice_score = (2 * intersection + self.smooth) / ((pred_sum * penalty) + target_sum + self.smooth)
            else:
                dice_score = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            
            dice_losses.append(1 - dice_score)
        
        # Return average of computed dice losses, or 0 if no valid classes
        return sum(dice_losses) / len(dice_losses) if dice_losses else torch.tensor(0.0, device=predictions.device)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined sparse-aware loss."""
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss_sparse(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice


def create_sparse_pore_loss(config: Dict) -> nn.Module:
    """Create sparse pore loss function from config."""
    # Get loss parameters
    if hasattr(config, 'get'):
        # ConfigLoader object
        false_positive_weight = config.get('model.loss.false_positive_weight', 10.0)
        focal_weight = config.get('model.loss.focal_weight', 0.7)
        dice_weight = config.get('model.loss.dice_weight', 0.3)
    else:
        # Dictionary
        loss_config = config.get('model', {}).get('loss', {})
        false_positive_weight = loss_config.get('false_positive_weight', 10.0)
        focal_weight = loss_config.get('focal_weight', 0.7)
        dice_weight = loss_config.get('dice_weight', 0.3)
    
    return SparsePoreCombinedLoss(
        false_positive_weight=false_positive_weight,
        focal_weight=focal_weight,
        dice_weight=dice_weight
    )
"""
Boundary-aware loss function for improved pore segmentation.
Focuses on accurate boundary detection to reduce false positives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from typing import Optional, Dict


class BoundaryAwareLoss(nn.Module):
    """
    Loss function that emphasizes boundary accuracy for pore segmentation.
    
    Key innovations:
    1. Boundary detection using Sobel filters
    2. Distance-weighted loss near boundaries
    3. Separate handling of pore-mineral vs pore-pore boundaries
    """
    
    def __init__(self,
                 boundary_weight: float = 5.0,
                 pore_boundary_weight: float = 10.0,  # Higher weight for pore boundaries
                 distance_threshold: int = 5,  # Pixels within this distance from boundary
                 class_weights: Optional[torch.Tensor] = None,
                 focal_gamma: float = 2.0):
        """
        Args:
            boundary_weight: Weight multiplier for pixels near boundaries
            pore_boundary_weight: Extra weight for pore-mineral boundaries
            distance_threshold: Distance from boundary to apply weighting
            class_weights: Per-class weights for handling imbalance
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.boundary_weight = boundary_weight
        self.pore_boundary_weight = pore_boundary_weight
        self.distance_threshold = distance_threshold
        self.focal_gamma = focal_gamma
        
        # Default class weights if not provided
        if class_weights is None:
            # Higher weights for rare pore classes
            self.class_weights = torch.tensor([5.0, 3.0, 1.0])  # disconnected, connected, mineral
        else:
            self.class_weights = class_weights
            
        # Sobel filters for boundary detection
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
    def detect_boundaries(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Detect boundaries in the ground truth masks using Sobel filters.
        
        Args:
            masks: (N, H, W) ground truth masks
            
        Returns:
            (N, H, W) boundary map
        """
        # Convert to one-hot for boundary detection
        num_classes = 3
        masks_one_hot = F.one_hot(masks.long(), num_classes).permute(0, 3, 1, 2).float()
        
        boundaries = torch.zeros_like(masks, dtype=torch.float32)
        
        for c in range(num_classes):
            class_mask = masks_one_hot[:, c:c+1, :, :]
            
            # Apply Sobel filters
            grad_x = F.conv2d(class_mask, self.sobel_x, padding=1)
            grad_y = F.conv2d(class_mask, self.sobel_y, padding=1)
            
            # Compute gradient magnitude
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
            boundaries += grad_mag.squeeze(1)
        
        # Threshold to get binary boundary map
        boundaries = (boundaries > 0.1).float()
        
        return boundaries
    
    def compute_distance_weights(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        Compute distance-based weights from boundaries.
        
        Args:
            boundaries: (N, H, W) binary boundary map
            
        Returns:
            (N, H, W) weight map
        """
        weights = torch.ones_like(boundaries)
        
        # Process each sample in the batch
        for i in range(boundaries.shape[0]):
            boundary_np = boundaries[i].cpu().numpy()
            
            # Compute distance transform
            if boundary_np.max() > 0:
                distance = ndimage.distance_transform_edt(1 - boundary_np)
                
                # Create weight based on distance
                weight = np.ones_like(distance)
                mask = distance <= self.distance_threshold
                weight[mask] = self.boundary_weight - (distance[mask] / self.distance_threshold) * (self.boundary_weight - 1)
                
                weights[i] = torch.from_numpy(weight).to(boundaries.device)
        
        return weights
    
    def compute_pore_boundary_weights(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute extra weights for pore-mineral boundaries.
        
        Args:
            masks: (N, H, W) ground truth masks
            
        Returns:
            (N, H, W) pore boundary weight map
        """
        weights = torch.ones_like(masks, dtype=torch.float32)
        
        # Identify pore pixels (classes 0 and 1) vs mineral (class 2)
        is_pore = (masks == 0) | (masks == 1)
        is_mineral = (masks == 2)
        
        # Apply dilation to find boundary regions
        for i in range(masks.shape[0]):
            pore_mask = is_pore[i].cpu().numpy().astype(np.uint8)
            mineral_mask = is_mineral[i].cpu().numpy().astype(np.uint8)
            
            # Dilate both masks
            kernel = np.ones((3, 3), np.uint8)
            pore_dilated = ndimage.binary_dilation(pore_mask, kernel)
            mineral_dilated = ndimage.binary_dilation(mineral_mask, kernel)
            
            # Find intersection (boundary region)
            boundary = pore_dilated & mineral_dilated
            
            if boundary.any():
                # Apply extra weight to pore-mineral boundaries
                weight_map = np.ones_like(pore_mask, dtype=np.float32)
                weight_map[boundary] = self.pore_boundary_weight
                weights[i] = torch.from_numpy(weight_map).to(masks.device)
        
        return weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-aware loss.
        
        Args:
            inputs: (N, 3, H, W) model predictions (logits)
            targets: (N, H, W) ground truth masks
        """
        # Handle ignore index
        valid_mask = targets != -100
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 2  # Set to background class
        
        # Detect boundaries
        boundaries = self.detect_boundaries(targets_clean)
        
        # Compute distance-based weights
        distance_weights = self.compute_distance_weights(boundaries)
        
        # Compute pore-mineral boundary weights
        pore_weights = self.compute_pore_boundary_weights(targets_clean)
        
        # Combine weights
        total_weights = distance_weights * pore_weights
        
        # Apply class weights
        class_weight_map = torch.zeros_like(targets_clean, dtype=torch.float32)
        for c in range(3):
            class_weight_map[targets_clean == c] = self.class_weights[c].to(targets_clean.device)
        
        total_weights = total_weights * class_weight_map
        
        # Compute focal loss
        ce_loss = F.cross_entropy(inputs, targets_clean, reduction='none')
        
        # Apply focal term
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets_clean.long(), 3).permute(0, 3, 1, 2)
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Combine all components
        loss = ce_loss * focal_weight * total_weights
        
        # Only average over valid pixels
        if valid_mask.any():
            return loss[valid_mask].mean()
        else:
            return loss.mean()


class BoundaryDiceLoss(nn.Module):
    """
    Combined boundary-aware focal loss with Dice loss.
    """
    
    def __init__(self,
                 boundary_weight: float = 5.0,
                 focal_weight: float = 0.6,
                 dice_weight: float = 0.4,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.boundary_loss = BoundaryAwareLoss(
            boundary_weight=boundary_weight,
            pore_boundary_weight=10.0,
            class_weights=class_weights
        )
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-6
        
    def dice_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Multi-class Dice loss.
        """
        # Handle ignore index
        valid_mask = targets != -100
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 2
        
        predictions = F.softmax(predictions, dim=1)
        
        dice_losses = []
        weights = [5.0, 3.0, 1.0]  # Class weights
        
        for c in range(3):
            pred_c = predictions[:, c]
            target_c = (targets_clean == c).float()
            
            # Apply valid mask
            pred_c = pred_c * valid_mask.float()
            target_c = target_c * valid_mask.float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append((1 - dice) * weights[c])
        
        return sum(dice_losses) / sum(weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined loss."""
        boundary = self.boundary_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * boundary + self.dice_weight * dice


def create_boundary_aware_loss(config: Dict) -> nn.Module:
    """Create boundary-aware loss from config."""
    if hasattr(config, 'get'):
        # ConfigLoader object
        boundary_weight = config.get('model.loss.boundary_weight', 5.0)
        focal_weight = config.get('model.loss.focal_weight', 0.6)
        dice_weight = config.get('model.loss.dice_weight', 0.4)
    else:
        # Dictionary
        loss_config = config.get('model', {}).get('loss', {})
        boundary_weight = loss_config.get('boundary_weight', 5.0)
        focal_weight = loss_config.get('focal_weight', 0.6)
        dice_weight = loss_config.get('dice_weight', 0.4)
    
    # Class weights for handling imbalance
    class_weights = torch.tensor([5.0, 3.0, 1.0])  # disconnected, connected, mineral
    
    return BoundaryDiceLoss(
        boundary_weight=boundary_weight,
        focal_weight=focal_weight,
        dice_weight=dice_weight,
        class_weights=class_weights
    )
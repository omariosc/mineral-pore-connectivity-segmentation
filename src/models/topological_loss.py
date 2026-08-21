"""
Topological consistency loss for pore segmentation.
Preserves connectivity patterns and reduces fragmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage.measure import label
from typing import Optional, Dict, Tuple


class TopologicalConsistencyLoss(nn.Module):
    """
    Loss function that preserves topological properties of pore networks.
    
    Key innovations:
    1. Connected component analysis to preserve connectivity
    2. Persistence homology-inspired features
    3. Size-aware component weighting
    """
    
    def __init__(self,
                 connectivity_weight: float = 2.0,
                 fragmentation_penalty: float = 5.0,
                 min_component_size: int = 10,
                 topology_scale: float = 0.1):
        """
        Args:
            connectivity_weight: Weight for connectivity preservation
            fragmentation_penalty: Penalty for creating fragmented predictions
            min_component_size: Minimum size for valid components
            topology_scale: Scale factor for topological loss component
        """
        super().__init__()
        self.connectivity_weight = connectivity_weight
        self.fragmentation_penalty = fragmentation_penalty
        self.min_component_size = min_component_size
        self.topology_scale = topology_scale
        
    def compute_connected_components(self, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute connected components for a binary mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            labeled: Labeled components
            num_components: Number of components
        """
        labeled, num_components = label(mask, connectivity=2, return_num=True)
        return labeled, num_components
    
    def compute_topology_features(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute topological features for comparison.
        
        Args:
            pred_mask: Predicted binary mask for a class
            gt_mask: Ground truth binary mask for a class
            
        Returns:
            Dictionary of topological features
        """
        features = {}
        
        # Convert to numpy for connected component analysis
        pred_np = pred_mask.cpu().numpy().astype(np.uint8)
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)
        
        # Get connected components
        pred_labeled, pred_num = self.compute_connected_components(pred_np)
        gt_labeled, gt_num = self.compute_connected_components(gt_np)
        
        # Component count difference (fragmentation metric)
        features['component_diff'] = abs(pred_num - gt_num) / max(gt_num, 1)
        
        # Size distribution similarity
        pred_sizes = []
        for i in range(1, pred_num + 1):
            size = np.sum(pred_labeled == i)
            if size >= self.min_component_size:
                pred_sizes.append(size)
        
        gt_sizes = []
        for i in range(1, gt_num + 1):
            size = np.sum(gt_labeled == i)
            if size >= self.min_component_size:
                gt_sizes.append(size)
        
        # Compute size distribution distance
        if gt_sizes:
            if pred_sizes:
                # Normalized size histograms
                max_size = max(max(pred_sizes), max(gt_sizes))
                bins = np.linspace(0, max_size, 20)
                pred_hist, _ = np.histogram(pred_sizes, bins, density=True)
                gt_hist, _ = np.histogram(gt_sizes, bins, density=True)
                
                # Wasserstein-like distance
                features['size_dist_error'] = np.sum(np.abs(pred_hist - gt_hist))
            else:
                features['size_dist_error'] = 1.0  # Maximum error if no components predicted
        else:
            features['size_dist_error'] = 0.0
        
        # Euler characteristic (topology invariant)
        # For 2D: χ = #components - #holes
        # Approximate holes using internal contours
        features['euler_diff'] = 0.0  # Simplified for now
        
        return features
    
    def compute_connectivity_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on connectivity preservation.
        
        Args:
            predictions: (N, 3, H, W) predicted probabilities
            targets: (N, H, W) ground truth masks
            
        Returns:
            Connectivity loss
        """
        batch_size = predictions.shape[0]
        total_loss = 0.0
        
        # Process each sample and each pore class
        for b in range(batch_size):
            for class_idx in [0, 1]:  # Only for pore classes
                # Get binary masks
                pred_probs = predictions[b, class_idx]
                pred_binary = (pred_probs > 0.5).float()
                gt_binary = (targets[b] == class_idx).float()
                
                # Skip if no ground truth pixels for this class
                if gt_binary.sum() == 0:
                    continue
                
                # Compute topological features
                topo_features = self.compute_topology_features(pred_binary, gt_binary)
                
                # Combine features into loss
                connectivity_loss = (
                    self.fragmentation_penalty * topo_features['component_diff'] +
                    self.connectivity_weight * topo_features['size_dist_error']
                )
                
                total_loss += connectivity_loss
        
        return total_loss / (batch_size * 2)  # Average over batch and classes
    
    def compute_smoothness_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss to reduce noise and fragmentation.
        
        Args:
            predictions: (N, 3, H, W) predicted probabilities
            
        Returns:
            Smoothness loss
        """
        # Compute gradients
        dy = torch.abs(predictions[:, :, 1:, :] - predictions[:, :, :-1, :])
        dx = torch.abs(predictions[:, :, :, 1:] - predictions[:, :, :, :-1])
        
        # Focus on pore classes
        pore_dy = dy[:, :2, :, :].mean()
        pore_dx = dx[:, :2, :, :].mean()
        
        return pore_dy + pore_dx
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute topological consistency loss.
        
        Args:
            inputs: (N, 3, H, W) model predictions (logits)
            targets: (N, H, W) ground truth masks
        """
        # Get probabilities
        predictions = F.softmax(inputs, dim=1)
        
        # Handle ignore index
        valid_mask = targets != -100
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 2
        
        # Compute connectivity loss
        connectivity_loss = self.compute_connectivity_loss(predictions, targets_clean)
        
        # Compute smoothness loss
        smoothness_loss = self.compute_smoothness_loss(predictions)
        
        # Combine losses
        total_loss = connectivity_loss + 0.1 * smoothness_loss
        
        return self.topology_scale * total_loss


class TopologicalCombinedLoss(nn.Module):
    """
    Combined loss with boundary awareness and topological consistency.
    """
    
    def __init__(self,
                 ce_weight: float = 0.4,
                 dice_weight: float = 0.3,
                 boundary_weight: float = 0.2,
                 topology_weight: float = 0.1):
        super().__init__()
        
        # Import boundary loss
        from .boundary_aware_loss import BoundaryDiceLoss
        
        self.boundary_dice_loss = BoundaryDiceLoss()
        self.topology_loss = TopologicalConsistencyLoss()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.topology_weight = topology_weight
        
        # Class weights for CE loss
        self.class_weights = torch.tensor([5.0, 3.0, 1.0])
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss with all components.
        
        Args:
            inputs: (N, 3, H, W) model predictions (logits)
            targets: (N, H, W) ground truth masks
        """
        # Handle ignore index
        valid_mask = targets != -100
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 2
        
        # Cross-entropy loss with class weights
        ce_loss = F.cross_entropy(
            inputs, targets_clean,
            weight=self.class_weights.to(inputs.device),
            reduction='mean'
        )
        
        # Dice loss (from boundary_dice_loss)
        predictions = F.softmax(inputs, dim=1)
        dice_loss = self.boundary_dice_loss.dice_loss(inputs, targets)
        
        # Boundary loss (from boundary_dice_loss)
        boundary_loss = self.boundary_dice_loss.boundary_loss(inputs, targets)
        
        # Topology loss
        topology_loss = self.topology_loss(inputs, targets)
        
        # Combine all losses
        total_loss = (
            self.ce_weight * ce_loss +
            self.dice_weight * dice_loss +
            self.boundary_weight * boundary_loss +
            self.topology_weight * topology_loss
        )
        
        return total_loss


def create_topological_loss(config: Dict) -> nn.Module:
    """Create topological loss from config."""
    if hasattr(config, 'get'):
        # ConfigLoader object
        ce_weight = config.get('model.loss.ce_weight', 0.4)
        dice_weight = config.get('model.loss.dice_weight', 0.3)
        boundary_weight = config.get('model.loss.boundary_weight', 0.2)
        topology_weight = config.get('model.loss.topology_weight', 0.1)
    else:
        # Dictionary
        loss_config = config.get('model', {}).get('loss', {})
        ce_weight = loss_config.get('ce_weight', 0.4)
        dice_weight = loss_config.get('dice_weight', 0.3)
        boundary_weight = loss_config.get('boundary_weight', 0.2)
        topology_weight = loss_config.get('topology_weight', 0.1)
    
    return TopologicalCombinedLoss(
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        boundary_weight=boundary_weight,
        topology_weight=topology_weight
    )
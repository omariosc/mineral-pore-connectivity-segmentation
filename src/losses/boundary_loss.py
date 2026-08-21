"""
Advanced boundary-aware loss functions for improved edge detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import cv2


class BoundaryLoss(nn.Module):
    """
    Boundary loss based on distance transform for precise edge localization.
    Paper: "Boundary loss for highly unbalanced segmentation" (MIDL 2019)
    """
    def __init__(self, num_classes=3, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def compute_dtm(self, mask):
        """
        Compute distance transform map for boundary regions.
        """
        batch_size = mask.shape[0]
        dtm = torch.zeros_like(mask, dtype=torch.float32)
        
        for b in range(batch_size):
            mask_np = mask[b].cpu().numpy().astype(np.uint8)
            
            # Compute boundaries using morphological operations
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(mask_np, kernel, iterations=1)
            boundary = mask_np - eroded
            
            # Distance transform
            if boundary.any():
                dist = ndimage.distance_transform_edt(1 - boundary)
                dist = dist / (dist.max() + 1e-5)  # Normalize
                dtm[b] = torch.from_numpy(dist).to(mask.device)
                
        return dtm
    
    def forward(self, logits, targets):
        """
        Compute boundary loss.
        Args:
            logits: [B, C, H, W] predicted logits
            targets: [B, H, W] ground truth labels
        """
        probs = F.softmax(logits, dim=1)
        
        total_loss = 0
        for c in range(self.num_classes):
            target_binary = (targets == c).float()
            dtm = self.compute_dtm(target_binary)
            
            # Weighted by distance to boundary
            boundary_loss = probs[:, c] * dtm
            total_loss += boundary_loss.mean()
            
        return total_loss / self.num_classes


class ActiveBoundaryLoss(nn.Module):
    """
    Active Boundary Loss that focuses on hard boundary pixels.
    Combines boundary detection with hard example mining.
    """
    def __init__(self, num_classes=3, boundary_width=2, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.boundary_width = boundary_width
        self.alpha = alpha  # Balance between boundary and region loss
        
    def get_boundary_mask(self, targets):
        """
        Extract boundary regions using morphological operations.
        """
        batch_size = targets.shape[0]
        boundary_mask = torch.zeros_like(targets, dtype=torch.bool)
        
        for b in range(batch_size):
            target_np = targets[b].cpu().numpy().astype(np.uint8)
            
            # Morphological gradient to find boundaries
            kernel = np.ones((self.boundary_width*2+1, self.boundary_width*2+1), np.uint8)
            dilated = cv2.dilate(target_np, kernel, iterations=1)
            eroded = cv2.erode(target_np, kernel, iterations=1)
            boundary = (dilated != eroded)
            
            boundary_mask[b] = torch.from_numpy(boundary).to(targets.device)
            
        return boundary_mask
    
    def forward(self, logits, targets):
        """
        Compute active boundary loss.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get boundary mask
        boundary_mask = self.get_boundary_mask(targets)
        
        # Separate boundary and region losses
        boundary_loss = ce_loss[boundary_mask].mean() if boundary_mask.any() else 0
        region_loss = ce_loss[~boundary_mask].mean() if (~boundary_mask).any() else 0
        
        # Weighted combination
        total_loss = self.alpha * boundary_loss + (1 - self.alpha) * region_loss
        
        return total_loss


class HausdorffDistanceLoss(nn.Module):
    """
    Hausdorff Distance Loss for boundary-aware segmentation.
    Minimizes the maximum distance between predicted and ground truth boundaries.
    """
    def __init__(self, num_classes=3, percentile=95):
        super().__init__()
        self.num_classes = num_classes
        self.percentile = percentile  # Use percentile HD to reduce noise sensitivity
        
    def compute_hausdorff_distance(self, pred, target):
        """
        Compute Hausdorff distance between prediction and target boundaries.
        """
        # Get boundaries
        pred_boundary = self.extract_boundary(pred)
        target_boundary = self.extract_boundary(target)
        
        if not pred_boundary.any() or not target_boundary.any():
            return torch.tensor(0.0, device=pred.device)
        
        # Compute distance transforms
        pred_dt = self.distance_transform(~pred_boundary)
        target_dt = self.distance_transform(~target_boundary)
        
        # Sample distances at boundary points
        pred_to_target = pred_dt[target_boundary]
        target_to_pred = target_dt[pred_boundary]
        
        # Compute percentile Hausdorff distance
        if len(pred_to_target) > 0 and len(target_to_pred) > 0:
            hd_pred = torch.quantile(pred_to_target.float(), self.percentile/100)
            hd_target = torch.quantile(target_to_pred.float(), self.percentile/100)
            hd = torch.max(hd_pred, hd_target)
        else:
            hd = torch.tensor(0.0, device=pred.device)
            
        return hd
    
    def extract_boundary(self, mask):
        """
        Extract boundary pixels from mask.
        """
        # Use convolution for efficient boundary detection
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)
        
        # Dilate and erode
        dilated = F.conv2d(mask_float, kernel, padding=1) > 0
        eroded = F.conv2d(mask_float, kernel, padding=1) == 9
        
        boundary = dilated.squeeze() & ~eroded.squeeze()
        return boundary
    
    def distance_transform(self, mask):
        """
        Compute distance transform using repeated convolutions.
        """
        mask_float = mask.float()
        dist = torch.zeros_like(mask_float)
        
        # Approximate distance transform with iterative convolutions
        kernel = torch.ones(1, 1, 3, 3, device=mask.device) / 9
        temp = mask_float.unsqueeze(0).unsqueeze(0)
        
        for i in range(20):  # Iterate to propagate distances
            temp = F.conv2d(temp, kernel, padding=1)
            dist = dist + (temp.squeeze() > 0).float()
            
        return dist
    
    def forward(self, logits, targets):
        """
        Compute Hausdorff distance loss.
        """
        preds = torch.argmax(logits, dim=1)
        
        total_loss = 0
        batch_size = logits.shape[0]
        
        for b in range(batch_size):
            for c in range(self.num_classes):
                pred_mask = (preds[b] == c)
                target_mask = (targets[b] == c)
                
                hd = self.compute_hausdorff_distance(pred_mask, target_mask)
                total_loss += hd
                
        return total_loss / (batch_size * self.num_classes)


class ContourAwareLoss(nn.Module):
    """
    Contour-aware loss that explicitly models object contours.
    Particularly effective for connected vs disconnected pore distinction.
    """
    def __init__(self, num_classes=3, contour_weight=2.0, thickness=2):
        super().__init__()
        self.num_classes = num_classes
        self.contour_weight = contour_weight
        self.thickness = thickness
        
    def extract_contours(self, mask):
        """
        Extract contours from segmentation mask.
        """
        batch_size = mask.shape[0]
        contours = torch.zeros_like(mask, dtype=torch.float32)
        
        for b in range(batch_size):
            mask_np = mask[b].cpu().numpy().astype(np.uint8)
            
            # Find contours for each class
            for c in range(self.num_classes):
                class_mask = (mask_np == c).astype(np.uint8) * 255
                
                # Find contours
                contours_list, _ = cv2.findContours(
                    class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Draw contours
                contour_mask = np.zeros_like(class_mask)
                cv2.drawContours(contour_mask, contours_list, -1, 1, self.thickness)
                
                contours[b] += torch.from_numpy(contour_mask).to(mask.device)
                
        return (contours > 0).float()
    
    def forward(self, logits, targets):
        """
        Compute contour-aware loss.
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Extract contours
        contour_mask = self.extract_contours(targets)
        
        # Weight loss by contour importance
        weighted_loss = ce_loss * (1 + self.contour_weight * contour_mask)
        
        return weighted_loss.mean()


class TopologicalBoundaryLoss(nn.Module):
    """
    Topology-preserving boundary loss that maintains connectivity patterns.
    Critical for distinguishing connected vs disconnected pores.
    """
    def __init__(self, num_classes=3, topology_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.topology_weight = topology_weight
        
    def compute_euler_characteristic(self, mask):
        """
        Compute Euler characteristic to measure topology.
        EC = #connected components - #holes
        """
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Connected components
        num_labels, labels = cv2.connectedComponents(mask_np)
        
        # Holes (using inverted mask)
        inverted = 1 - mask_np
        num_holes, _ = cv2.connectedComponents(inverted)
        
        # Euler characteristic
        ec = num_labels - num_holes
        
        return ec
    
    def topology_loss(self, pred, target):
        """
        Penalize topology changes between prediction and target.
        """
        batch_size = pred.shape[0]
        topo_loss = 0
        
        for b in range(batch_size):
            for c in range(self.num_classes):
                pred_mask = (pred[b] == c).float()
                target_mask = (target[b] == c).float()
                
                # Compute Euler characteristics
                pred_ec = self.compute_euler_characteristic(pred_mask)
                target_ec = self.compute_euler_characteristic(target_mask)
                
                # Penalize topology differences
                topo_loss += abs(pred_ec - target_ec)
                
        return topo_loss / (batch_size * self.num_classes)
    
    def forward(self, logits, targets):
        """
        Compute topology-preserving boundary loss.
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets)
        
        # Predicted segmentation
        preds = torch.argmax(logits, dim=1)
        
        # Topology preservation term
        topo_loss = self.topology_loss(preds, targets)
        
        # Combined loss
        total_loss = ce_loss + self.topology_weight * topo_loss
        
        return total_loss


def create_boundary_loss(loss_type='active', **kwargs):
    """
    Factory function to create boundary-aware losses.
    """
    if loss_type == 'boundary':
        return BoundaryLoss(**kwargs)
    elif loss_type == 'active_boundary':
        return ActiveBoundaryLoss(**kwargs)
    elif loss_type == 'hausdorff':
        return HausdorffDistanceLoss(**kwargs)
    elif loss_type == 'contour':
        return ContourAwareLoss(**kwargs)
    elif loss_type == 'topological_boundary':
        return TopologicalBoundaryLoss(**kwargs)
    else:
        raise ValueError(f"Unknown boundary loss type: {loss_type}")
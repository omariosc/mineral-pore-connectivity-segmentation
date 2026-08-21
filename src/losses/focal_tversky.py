"""
Focal Tversky Loss Implementation for Extreme Class Imbalance
Based on: Abraham & Khan (2019) - "A Novel Focal Tversky Loss Function"

Specifically designed for our geological segmentation with 0.59% minority class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for extreme class imbalance in segmentation.
    
    Combines:
    1. Tversky index to control false positives/negatives
    2. Focal mechanism to focus on hard examples
    3. Per-class weighting for extreme imbalance
    
    Args:
        alpha (float): Controls false positives (higher = penalize FP more)
        beta (float): Controls false negatives (higher = penalize FN more)
        gamma (float): Focal parameter (higher = focus more on hard examples)
        num_classes (int): Number of segmentation classes
        class_weights (list): Per-class weights [w0, w1, w2]
        smooth (float): Smoothing factor to avoid division by zero
    """
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, 
                 num_classes=3, class_weights=None, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.smooth = smooth
        
        # Default weights for our specific problem
        # Class 0 (0.59%): weight=100, Class 1 (12.49%): weight=10, Class 2: weight=1
        if class_weights is None:
            class_weights = [100.0, 10.0, 1.0]
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - Model predictions (logits)
            targets: (B, H, W) - Ground truth labels
        
        Returns:
            Weighted focal tversky loss
        """
        # Move weights to same device as inputs
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate per-class Tversky index
        total_loss = 0.0
        
        for class_idx in range(self.num_classes):
            # Get predictions and targets for this class
            pred = inputs[:, class_idx, :, :]
            target = targets_one_hot[:, class_idx, :, :]
            
            # Calculate True Positives, False Positives, False Negatives
            tp = (pred * target).sum(dim=(1, 2))
            fp = (pred * (1 - target)).sum(dim=(1, 2))
            fn = ((1 - pred) * target).sum(dim=(1, 2))
            
            # Tversky index
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            
            # Focal Tversky Loss
            focal_tversky = (1 - tversky) ** self.gamma
            
            # Apply class weight
            weighted_loss = self.class_weights[class_idx] * focal_tversky.mean()
            
            total_loss += weighted_loss
        
        # Normalize by sum of weights
        total_loss = total_loss / self.class_weights.sum()
        
        return total_loss


class AdaptiveFocalTverskyLoss(nn.Module):
    """
    Adaptive version that adjusts alpha/beta based on class distribution in batch.
    Better for extreme imbalance where minority class may not appear in every batch.
    """
    
    def __init__(self, base_alpha=0.5, base_beta=0.5, gamma=2.0,
                 num_classes=3, class_weights=None, smooth=1e-6):
        super(AdaptiveFocalTverskyLoss, self).__init__()
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.smooth = smooth
        
        if class_weights is None:
            class_weights = [100.0, 10.0, 1.0]
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Adaptively adjusts alpha/beta based on class distribution in batch.
        """
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # Calculate class distribution in batch
        class_counts = torch.bincount(targets.view(-1), minlength=self.num_classes).float()
        class_freq = class_counts / class_counts.sum()
        
        # Apply softmax
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        total_loss = 0.0
        
        for class_idx in range(self.num_classes):
            # Adaptive alpha/beta based on class frequency
            # For minority classes: higher alpha (penalize FP more)
            # For majority classes: higher beta (penalize FN more)
            if class_freq[class_idx] < 0.01:  # Very rare class (<1%)
                alpha = 0.7  # High penalty for false positives
                beta = 0.3   # Lower penalty for false negatives
            elif class_freq[class_idx] < 0.2:  # Minority class (<20%)
                alpha = 0.6
                beta = 0.4
            else:  # Majority class
                alpha = 0.3
                beta = 0.7
            
            pred = inputs[:, class_idx, :, :]
            target = targets_one_hot[:, class_idx, :, :]
            
            tp = (pred * target).sum(dim=(1, 2))
            fp = (pred * (1 - target)).sum(dim=(1, 2))
            fn = ((1 - pred) * target).sum(dim=(1, 2))
            
            tversky = (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)
            focal_tversky = (1 - tversky) ** self.gamma
            
            weighted_loss = self.class_weights[class_idx] * focal_tversky.mean()
            total_loss += weighted_loss
        
        total_loss = total_loss / self.class_weights.sum()
        
        return total_loss


class UnifiedFocalLoss(nn.Module):
    """
    Unified loss combining Focal Loss and Focal Tversky Loss.
    Prevents over-suppression while maintaining focus on hard samples.
    """
    
    def __init__(self, focal_weight=0.5, tversky_weight=0.5,
                 focal_gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3,
                 tversky_gamma=2.0, num_classes=3, class_weights=None):
        super(UnifiedFocalLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # Initialize component losses
        self.focal_loss = FocalLoss(gamma=focal_gamma, num_classes=num_classes, 
                                   class_weights=class_weights)
        self.focal_tversky = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta,
                                             gamma=tversky_gamma, num_classes=num_classes,
                                             class_weights=class_weights)
    
    def forward(self, inputs, targets):
        """Combine focal and focal tversky losses."""
        focal = self.focal_loss(inputs, targets)
        tversky = self.focal_tversky(inputs, targets)
        
        return self.focal_weight * focal + self.tversky_weight * tversky


class FocalLoss(nn.Module):
    """
    Standard Focal Loss for comparison.
    """
    
    def __init__(self, gamma=2.0, num_classes=3, class_weights=None, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.smooth = smooth
        
        if class_weights is None:
            class_weights = [100.0, 10.0, 1.0]
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """Standard focal loss implementation."""
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # Apply log_softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Gather probabilities for true classes
        targets = targets.long()
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # Get probability of true class for focal weighting
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights
        weights = self.class_weights[targets].view_as(focal_loss)
        weighted_loss = weights * focal_loss
        
        return weighted_loss.mean()


# Test the implementation
if __name__ == "__main__":
    # Create dummy data matching our problem
    batch_size = 2
    height, width = 256, 256
    num_classes = 3
    
    # Simulate predictions and targets
    inputs = torch.randn(batch_size, num_classes, height, width)
    
    # Create targets with realistic class distribution
    # Class 0: 0.59%, Class 1: 12.49%, Class 2: 86.92%
    targets = torch.ones(batch_size, height, width, dtype=torch.long) * 2  # Mostly class 2
    
    # Add some class 1 pixels (12.49%)
    num_class1 = int(0.1249 * height * width)
    targets.view(-1)[:num_class1] = 1
    
    # Add very few class 0 pixels (0.59%)
    num_class0 = int(0.0059 * height * width)
    targets.view(-1)[:num_class0] = 0
    
    # Shuffle targets
    targets = targets.view(batch_size, -1)
    for i in range(batch_size):
        perm = torch.randperm(height * width)
        targets[i] = targets[i][perm]
    targets = targets.view(batch_size, height, width)
    
    # Test different loss functions
    print("Testing Focal Tversky Loss implementations...")
    
    # Standard Focal Tversky
    loss_fn1 = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=2.0)
    loss1 = loss_fn1(inputs, targets)
    print(f"Focal Tversky Loss: {loss1.item():.4f}")
    
    # Adaptive Focal Tversky
    loss_fn2 = AdaptiveFocalTverskyLoss()
    loss2 = loss_fn2(inputs, targets)
    print(f"Adaptive Focal Tversky Loss: {loss2.item():.4f}")
    
    # Unified Focal Loss
    loss_fn3 = UnifiedFocalLoss()
    loss3 = loss_fn3(inputs, targets)
    print(f"Unified Focal Loss: {loss3.item():.4f}")
    
    print("\nLoss functions implemented successfully!")
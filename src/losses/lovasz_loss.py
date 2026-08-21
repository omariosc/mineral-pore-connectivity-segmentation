"""
Lovász-Softmax Loss: Direct IoU optimization for semantic segmentation
Based on: Berman et al., CVPR 2018 - "The Lovász-Softmax loss"
Optimized for extreme class imbalance in mineral pore segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovász extension w.r.t sorted errors.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present', class_weights=None):
    """
    Multi-class Lovász-Softmax loss.
    
    Args:
        probas: [P, C] class probabilities at each prediction (between 0 and 1)
        labels: [P] ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes in labels, or list of classes
        class_weights: Optional weights for each class
    """
    if probas.numel() == 0:
        return probas * 0.
    
    C = probas.size(1)
    losses = []
    
    # Determine which classes to compute loss for
    if classes == 'all':
        class_to_sum = list(range(C))
    elif classes == 'present':
        class_to_sum = list(labels.unique().cpu().numpy())
    else:
        class_to_sum = classes
    
    for c in class_to_sum:
        # Foreground for class c
        fg = (labels == c).float()
        
        if fg.sum() == 0:
            continue
            
        # Class c scores
        class_pred = probas[:, c]
        
        # Compute errors
        errors = (1 - fg) * class_pred + fg * (1 - class_pred)
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        
        # Compute gradient
        grad = lovasz_grad(fg_sorted)
        
        # Compute loss
        loss = torch.dot(errors_sorted, grad)
        
        # Apply class weight if provided
        if class_weights is not None:
            loss = loss * class_weights[c]
            
        losses.append(loss)
    
    return torch.stack(losses).mean() if losses else probas.sum() * 0.


class LovaszSoftmaxLoss(nn.Module):
    """
    Multi-class Lovász-Softmax loss for semantic segmentation.
    Directly optimizes the IoU metric.
    
    Args:
        classes: 'all', 'present', or list of classes to compute loss
        per_image: Compute loss per image instead of per batch
        ignore_index: Class index to ignore
        class_weights: Weights for each class [w0, w1, w2]
    """
    
    def __init__(self, classes='present', per_image=False, 
                 ignore_index=None, class_weights=None):
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
        
        # Default weights for our problem (Class 0: 0.59%, Class 1: 12.49%, Class 2: 86.92%)
        if class_weights is None:
            class_weights = [10.0, 2.0, 1.0]  # Moderate weights for stability
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] raw logits
            targets: [B, H, W] ground truth labels
        """
        # Move weights to same device
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # Apply softmax
        probas = F.softmax(inputs, dim=1)
        
        # Flatten
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1)
        
        # Filter out ignore index
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            probas = probas[valid]
            targets = targets[valid]
        
        # Compute loss
        if self.per_image:
            # Compute loss per image
            B_flat = B * H * W // probas.shape[0] if probas.shape[0] > 0 else B
            losses = []
            
            for i in range(B):
                start_idx = i * (H * W)
                end_idx = (i + 1) * (H * W)
                
                if end_idx > len(targets):
                    end_idx = len(targets)
                if start_idx >= len(targets):
                    break
                    
                probas_i = probas[start_idx:end_idx]
                targets_i = targets[start_idx:end_idx]
                
                loss_i = lovasz_softmax_flat(probas_i, targets_i, 
                                            classes=self.classes,
                                            class_weights=self.class_weights)
                losses.append(loss_i)
            
            return torch.stack(losses).mean() if losses else probas.sum() * 0.
        else:
            # Compute loss for whole batch
            return lovasz_softmax_flat(probas, targets, 
                                      classes=self.classes,
                                      class_weights=self.class_weights)


class WeightedLovaszSoftmaxLoss(nn.Module):
    """
    Weighted combination of Lovász-Softmax and Cross-Entropy.
    Provides stability while optimizing IoU.
    """
    
    def __init__(self, lovasz_weight=0.5, ce_weight=0.5,
                 classes='present', class_weights=None):
        super(WeightedLovaszSoftmaxLoss, self).__init__()
        
        self.lovasz_weight = lovasz_weight
        self.ce_weight = ce_weight
        
        # Initialize component losses
        self.lovasz = LovaszSoftmaxLoss(classes=classes, class_weights=class_weights)
        
        if class_weights is None:
            class_weights = [10.0, 2.0, 1.0]
        self.ce_weights = torch.tensor(class_weights, dtype=torch.float32)
        
    def forward(self, inputs, targets):
        """Combined loss computation."""
        # Move weights to device
        if self.ce_weights.device != inputs.device:
            self.ce_weights = self.ce_weights.to(inputs.device)
        
        # Lovász loss
        lovasz_loss = self.lovasz(inputs, targets)
        
        # Weighted cross-entropy
        ce_loss = F.cross_entropy(inputs, targets.long(), weight=self.ce_weights)
        
        # Combine
        return self.lovasz_weight * lovasz_loss + self.ce_weight * ce_loss


class FocalLovaszLoss(nn.Module):
    """
    Combination of Focal Loss and Lovász-Softmax Loss.
    Focal component handles class imbalance, Lovász optimizes IoU.
    """
    
    def __init__(self, focal_weight=0.3, lovasz_weight=0.7,
                 focal_gamma=2.0, class_weights=None):
        super(FocalLovaszLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        
        # Component losses
        self.lovasz = LovaszSoftmaxLoss(class_weights=class_weights)
        
        # Focal loss parameters
        self.gamma = focal_gamma
        if class_weights is None:
            class_weights = [10.0, 2.0, 1.0]
        self.focal_weights = torch.tensor(class_weights, dtype=torch.float32)
        
    def focal_loss(self, inputs, targets):
        """Compute focal loss component."""
        if self.focal_weights.device != inputs.device:
            self.focal_weights = self.focal_weights.to(inputs.device)
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        p_t = torch.gather(p, 1, targets.long().unsqueeze(1)).squeeze(1)
        
        # Apply focal term
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        # Apply class weights
        weights = self.focal_weights[targets.long()]
        weighted_loss = weights * focal_loss
        
        return weighted_loss.mean()
    
    def forward(self, inputs, targets):
        """Combined focal and Lovász loss."""
        focal = self.focal_loss(inputs, targets)
        lovasz = self.lovasz(inputs, targets)
        
        return self.focal_weight * focal + self.lovasz_weight * lovasz


class AsymmetricLovaszLoss(nn.Module):
    """
    Asymmetric Lovász loss that penalizes false negatives more for minority classes.
    Specifically designed for extreme imbalance.
    """
    
    def __init__(self, fp_weight=1.0, fn_weights=None, classes='present'):
        super(AsymmetricLovaszLoss, self).__init__()
        
        self.fp_weight = fp_weight
        
        # Higher FN penalty for minority classes
        if fn_weights is None:
            fn_weights = [20.0, 5.0, 1.0]  # High penalty for missing class 0
        self.fn_weights = torch.tensor(fn_weights, dtype=torch.float32)
        self.classes = classes
        
    def forward(self, inputs, targets):
        """Asymmetric Lovász computation."""
        if self.fn_weights.device != inputs.device:
            self.fn_weights = self.fn_weights.to(inputs.device)
        
        probas = F.softmax(inputs, dim=1)
        B, C, H, W = probas.size()
        
        probas_flat = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)
        
        losses = []
        
        # Compute per-class asymmetric loss
        for c in range(C):
            fg = (targets_flat == c).float()
            
            if fg.sum() == 0:
                continue
            
            class_pred = probas_flat[:, c]
            
            # Asymmetric errors
            fn_errors = fg * (1 - class_pred) * self.fn_weights[c]  # False negatives
            fp_errors = (1 - fg) * class_pred * self.fp_weight      # False positives
            
            errors = fn_errors + fp_errors
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            
            grad = lovasz_grad(fg_sorted)
            loss = torch.dot(errors_sorted, grad)
            
            losses.append(loss)
        
        return torch.stack(losses).mean() if losses else inputs.sum() * 0.


# Test implementation
if __name__ == "__main__":
    print("Testing Lovász-Softmax Loss implementations...")
    
    # Create dummy data
    batch_size = 2
    num_classes = 3
    height, width = 256, 256
    
    inputs = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test different variants
    print("\n1. Standard Lovász-Softmax Loss:")
    loss_fn1 = LovaszSoftmaxLoss()
    loss1 = loss_fn1(inputs, targets)
    print(f"   Loss: {loss1.item():.4f}")
    
    print("\n2. Weighted Lovász-Softmax + CE Loss:")
    loss_fn2 = WeightedLovaszSoftmaxLoss()
    loss2 = loss_fn2(inputs, targets)
    print(f"   Loss: {loss2.item():.4f}")
    
    print("\n3. Focal + Lovász Loss:")
    loss_fn3 = FocalLovaszLoss()
    loss3 = loss_fn3(inputs, targets)
    print(f"   Loss: {loss3.item():.4f}")
    
    print("\n4. Asymmetric Lovász Loss:")
    loss_fn4 = AsymmetricLovaszLoss()
    loss4 = loss_fn4(inputs, targets)
    print(f"   Loss: {loss4.item():.4f}")
    
    print("\nAll Lovász loss variants implemented successfully!")
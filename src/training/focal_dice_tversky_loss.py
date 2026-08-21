import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceTverskyLoss(nn.Module):
    """Combined Focal + Dice + Tversky Loss for C7."""
    
    def __init__(self, num_classes=3, class_weights=[20, 5, 1], 
                 focal_alpha=0.25, focal_gamma=2.0,
                 tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] model outputs
            targets: [B, H, W] ground truth labels
        """
        device = logits.device
        self.class_weights = self.class_weights.to(device)
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # 1. Focal Loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        # Apply class weights to focal loss
        weight_map = self.class_weights[targets]
        focal_loss = (focal_loss * weight_map).mean()
        
        # 2. Dice Loss
        dice_loss = 0
        for c in range(self.num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice_c = (2 * intersection + 1e-7) / (union + 1e-7)
            dice_loss += (1 - dice_c) * self.class_weights[c]
        
        dice_loss = dice_loss / self.class_weights.sum()
        
        # 3. Tversky Loss
        tversky_loss = 0
        for c in range(self.num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()
            
            tversky_c = (tp + 1e-7) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1e-7)
            tversky_loss += (1 - tversky_c) * self.class_weights[c]
        
        tversky_loss = tversky_loss / self.class_weights.sum()
        
        # Combine losses
        total_loss = 0.4 * focal_loss + 0.3 * dice_loss + 0.3 * tversky_loss
        
        return total_loss

"""
YOLOv8 Segmentation model for geological pore segmentation.
Adapted from Ultralytics YOLOv8 for 3-class mineral segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class Conv(nn.Module):
    """YOLOv8 Conv layer with activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLOv8 bottleneck block."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """YOLOv8 C2f block for CSPNet."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SegmentationHead(nn.Module):
    """YOLOv8 segmentation head."""
    def __init__(self, in_channels, num_classes, num_masks=32):
        super().__init__()
        self.num_classes = num_classes
        self.num_masks = num_masks
        
        # Proto net for mask coefficients
        self.proto = nn.Sequential(
            Conv(in_channels[0], in_channels[0] // 2, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv(in_channels[0] // 2, in_channels[0] // 2, 3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv(in_channels[0] // 2, num_masks, 1)
        )
        
        # Detection heads for each scale
        self.detect = nn.ModuleList()
        for ch in in_channels:
            self.detect.append(nn.Sequential(
                Conv(ch, ch, 3),
                Conv(ch, ch, 3),
                nn.Conv2d(ch, num_classes + num_masks, 1)
            ))

    def forward(self, features):
        """
        Args:
            features: List of feature maps at different scales
        Returns:
            masks: Segmentation masks [B, num_classes, H, W]
        """
        # Generate prototypes
        proto = self.proto(features[0])
        
        # Generate mask coefficients and class predictions
        outputs = []
        for feat, head in zip(features, self.detect):
            out = head(feat)
            outputs.append(out)
        
        # Combine outputs
        batch_size = features[0].shape[0]
        h, w = features[0].shape[2] * 4, features[0].shape[3] * 4  # Output size
        
        # Process each scale
        all_masks = []
        for out in outputs:
            # Split class and mask predictions
            cls_pred = out[:, :self.num_classes]
            mask_coeff = out[:, self.num_classes:]
            
            # Upsample to match proto size
            mask_coeff = F.interpolate(mask_coeff, size=proto.shape[2:], 
                                      mode='bilinear', align_corners=False)
            
            # Generate masks: [B, num_masks, H, W] @ [B, num_masks, num_classes] -> [B, num_classes, H, W]
            masks = torch.einsum('bmhw,bmc->bchw', proto, mask_coeff.permute(0, 2, 3, 1))
            
            # Apply class predictions as weights
            cls_weight = F.interpolate(cls_pred, size=masks.shape[2:], 
                                      mode='bilinear', align_corners=False)
            masks = masks * torch.sigmoid(cls_weight)
            
            all_masks.append(masks)
        
        # Combine masks from all scales
        final_masks = sum(all_masks) / len(all_masks)
        
        # Upsample to original size
        final_masks = F.interpolate(final_masks, size=(h, w), 
                                   mode='bilinear', align_corners=False)
        
        return final_masks


class YOLOv8Backbone(nn.Module):
    """YOLOv8 backbone network."""
    def __init__(self, channels=[64, 128, 256, 512], depths=[3, 6, 9, 3]):
        super().__init__()
        
        # Stem
        self.stem = Conv(3, channels[0], 3, 2)
        
        # Stages
        self.stages = nn.ModuleList()
        in_channels = channels[0]
        
        for i, (ch, depth) in enumerate(zip(channels, depths)):
            stage = nn.Sequential(
                Conv(in_channels, ch, 3, 2),
                C2f(ch, ch, depth, shortcut=True)
            )
            self.stages.append(stage)
            in_channels = ch
        
        # SPPF
        self.sppf = SPPF(channels[-1], channels[-1])
        
    def forward(self, x):
        features = []
        
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        features[-1] = self.sppf(features[-1])
        
        return features


class YOLOv8Neck(nn.Module):
    """YOLOv8 FPN+PAN neck."""
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        
        # FPN upsampling path
        self.up = nn.ModuleList()
        self.fpn_conv = nn.ModuleList()
        
        for i in range(len(channels) - 1, 0, -1):
            self.up.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.fpn_conv.append(C2f(channels[i] + channels[i-1], channels[i-1], 3))
        
        # PAN downsampling path
        self.down = nn.ModuleList()
        self.pan_conv = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            self.down.append(Conv(channels[i], channels[i], 3, 2))
            self.pan_conv.append(C2f(channels[i] * 2, channels[i+1], 3))
            
    def forward(self, features):
        # FPN upsampling
        fpn_features = [features[-1]]
        
        for i in range(len(features) - 1, 0, -1):
            upsampled = self.up[len(features) - 1 - i](fpn_features[-1])
            concat = torch.cat([upsampled, features[i-1]], dim=1)
            fpn_features.append(self.fpn_conv[len(features) - 1 - i](concat))
        
        fpn_features = fpn_features[::-1]
        
        # PAN downsampling
        pan_features = [fpn_features[0]]
        
        for i in range(len(fpn_features) - 1):
            downsampled = self.down[i](pan_features[-1])
            concat = torch.cat([downsampled, fpn_features[i+1]], dim=1)
            pan_features.append(self.pan_conv[i](concat))
        
        return pan_features


class YOLOv8Segmentation(nn.Module):
    """
    YOLOv8 Segmentation model for geological pore segmentation.
    Combines detection and segmentation for precise pore delineation.
    """
    def __init__(self, 
                 num_classes=3,
                 model_size='s',  # 's', 'm', 'l', 'x'
                 pretrained=False):
        super().__init__()
        
        # Model configurations
        configs = {
            's': {'channels': [64, 128, 256, 512], 'depths': [3, 6, 9, 3]},
            'm': {'channels': [96, 192, 384, 576], 'depths': [3, 6, 9, 3]},
            'l': {'channels': [128, 256, 512, 768], 'depths': [3, 9, 12, 3]},
            'x': {'channels': [160, 320, 640, 960], 'depths': [3, 9, 15, 3]}
        }
        
        config = configs[model_size]
        
        # Build model
        self.backbone = YOLOv8Backbone(config['channels'], config['depths'])
        self.neck = YOLOv8Neck(config['channels'])
        self.head = SegmentationHead(config['channels'], num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            masks: Segmentation masks [B, num_classes, H, W]
        """
        # Extract features
        features = self.backbone(x)
        
        # FPN + PAN
        features = self.neck(features)
        
        # Generate masks
        masks = self.head(features)
        
        return masks


class YOLOv8Loss(nn.Module):
    """
    Loss function for YOLOv8 segmentation.
    Combines classification and mask losses.
    """
    def __init__(self, 
                 num_classes=3,
                 class_weights=None,
                 use_focal=True,
                 focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        
        # Dice loss for segmentation
        self.dice_loss = DiceLoss(num_classes=num_classes)
        
    def forward(self, predictions, targets):
        """
        Compute loss.
        Args:
            predictions: Model outputs [B, num_classes, H, W]
            targets: Ground truth masks [B, H, W]
        """
        batch_size = predictions.shape[0]
        
        # Resize targets to match predictions
        if predictions.shape[2:] != targets.shape[1:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), 
                                   size=predictions.shape[2:], 
                                   mode='nearest').squeeze(1).long()
        
        # Classification loss (Focal Cross Entropy)
        if self.use_focal:
            ce_loss = self.focal_loss(predictions, targets)
        else:
            ce_loss = F.cross_entropy(predictions, targets, 
                                     weight=self.class_weights)
        
        # Segmentation loss (Dice)
        dice_loss = self.dice_loss(predictions, targets)
        
        # Combined loss
        total_loss = ce_loss + dice_loss
        
        return total_loss
    
    def focal_loss(self, inputs, targets):
        """Focal loss for addressing class imbalance."""
        ce_loss = F.cross_entropy(inputs, targets, 
                                 weight=self.class_weights, 
                                 reduction='none')
        
        # Get predicted probabilities
        p = torch.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focal term
        focal_weight = (1 - p_t) ** self.focal_gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, num_classes=3, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Compute Dice loss.
        Args:
            inputs: Predictions [B, C, H, W]
            targets: Ground truth [B, H, W]
        """
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Apply softmax to predictions
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Compute Dice coefficient for each class
        dice_loss = 0
        for c in range(self.num_classes):
            input_c = inputs_soft[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (input_c * target_c).sum()
            union = input_c.sum() + target_c.sum()
            
            dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice_coeff)
            
        return dice_loss / self.num_classes


def create_yolov8_model(model_size='s', num_classes=3, pretrained=False):
    """
    Factory function to create YOLOv8 segmentation model.
    Args:
        model_size: Model size ('s', 'm', 'l', 'x')
        num_classes: Number of segmentation classes
        pretrained: Whether to load pretrained weights
    """
    model = YOLOv8Segmentation(
        num_classes=num_classes,
        model_size=model_size,
        pretrained=pretrained
    )
    
    return model
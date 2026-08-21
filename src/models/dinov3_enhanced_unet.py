import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.models.multiscale_attention_unet import MultiScaleAttentionUNet

class DINOv3EnhancedUNet(nn.Module):
    """MultiScale Attention UNet with DINOv3 feature extraction."""
    
    def __init__(self, n_channels=1, n_classes=3, base_features=32):
        super().__init__()
        
        # DINOv3 backbone for feature extraction
        # Using DINOv2 (as DINOv3 might not be available in timm yet)
        self.dino_backbone = timm.create_model(
            'vit_small_patch14_dinov2',
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=224,   # Will resize patches
            in_chans=3      # Convert grayscale to RGB
        )
        
        # Freeze DINOv3 backbone
        for param in self.dino_backbone.parameters():
            param.requires_grad = False
        
        # Feature dimension from DINOv3 (384 for small model)
        dino_features = 384
        
        # Channel adapter: grayscale to RGB
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        # Feature projection layers
        self.feature_proj = nn.Sequential(
            nn.Linear(dino_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, base_features * 4 * 4),  # Project to spatial features
            nn.ReLU(inplace=True)
        )
        
        # Feature reshaping for UNet input
        self.feature_reshape = nn.Conv2d(base_features, base_features, kernel_size=3, padding=1)
        
        # Main UNet model
        self.unet = MultiScaleAttentionUNet(
            n_channels=base_features,  # Takes features instead of raw input
            n_classes=n_classes,
            base_features=base_features
        )
        
        # Skip connection weight
        self.skip_weight = nn.Parameter(torch.tensor(0.3))
        
    def extract_dino_features(self, x):
        """Extract features using DINOv3 backbone."""
        b, c, h, w = x.shape
        
        # Convert to RGB
        x_rgb = self.channel_adapter(x)
        
        # Process in patches (to handle large images)
        patch_size = 224
        stride = 112  # 50% overlap
        
        features = []
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = x_rgb[:, :, i:i+patch_size, j:j+patch_size]
                
                # Resize if needed
                if patch.shape[-2:] != (224, 224):
                    patch = F.interpolate(patch, size=(224, 224), mode='bilinear', align_corners=False)
                
                # Extract features
                with torch.no_grad():
                    feat = self.dino_backbone(patch)
                features.append(feat)
        
        # Average pool features
        if features:
            features = torch.stack(features, dim=1).mean(dim=1)
        else:
            # Fallback for small images
            x_resized = F.interpolate(x_rgb, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                features = self.dino_backbone(x_resized)
        
        return features
    
    def forward(self, x):
        """Forward pass with DINOv3 feature extraction."""
        b, c, h, w = x.shape
        
        # Extract DINOv3 features
        dino_features = self.extract_dino_features(x)
        
        # Project features
        projected = self.feature_proj(dino_features)
        
        # Reshape to spatial dimensions
        spatial_features = projected.view(b, -1, 4, 4)
        
        # Upsample to match input size
        spatial_features = F.interpolate(
            spatial_features, 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Refine features
        spatial_features = self.feature_reshape(spatial_features)
        
        # Combine with original input (skip connection)
        enhanced_input = spatial_features + self.skip_weight * x
        
        # Pass through UNet
        output = self.unet(enhanced_input)
        
        return output

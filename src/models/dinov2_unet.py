"""
DINOv2/DINOv3 Vision Transformer backbone for U-Net
Leverages pretrained foundation models for superior feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


DINOV2_DISABLED_MESSAGE = (
    "DINOv2 is disabled in the public runtime because the historical "
    "implementation executed mutable repository code through torch.hub. "
    "Use one of the reviewed local U-Net implementations. Re-enabling DINOv2 "
    "requires vendored, revision-pinned source and separately verified weights; "
    "the public code does not download or execute a remote model repository."
)


class DINOv2UNet(nn.Module):
    """
    U-Net with DINOv2 vision transformer backbone.
    Combines powerful pretrained features with U-Net decoder.
    
    Args:
        model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_classes: Number of segmentation classes
        freeze_backbone: Whether to freeze DINOv2 weights
        use_registers: Use register tokens (DINOv2 feature)
    """
    
    def __init__(self, model_name='dinov2_vitb14', num_classes=3, 
                 freeze_backbone=False, use_registers=False):
        super(DINOv2UNet, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Extract model size from name
        if 'vits' in model_name:
            self.model_size = 'small'
        elif 'vitb' in model_name:
            self.model_size = 'base'
        elif 'vitl' in model_name:
            self.model_size = 'large'
        elif 'vitg' in model_name:
            self.model_size = 'giant'
        else:
            self.model_size = 'base'  # default
        
        # Load DINOv2 backbone
        self.backbone = self._load_dinov2(model_name, use_registers)
        
        # Get feature dimensions based on model variant
        self.feature_dims = self._get_feature_dims(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # U-Net decoder with skip connections
        self.decoder = DINOv2Decoder(
            encoder_dims=self.feature_dims,
            num_classes=num_classes
        )
        
    def _load_dinov2(self, model_name, use_registers):
        """Fail closed until a reviewed, local DINOv2 source is supplied."""
        raise RuntimeError(DINOV2_DISABLED_MESSAGE)
    
    def _get_feature_dims(self, model_name):
        """Get feature dimensions for different DINOv2 variants."""
        dim_map = {
            'dinov2_vits14': [384, 384, 384, 384],  # ViT-S/14
            'dinov2_vitb14': [768, 768, 768, 768],  # ViT-B/14
            'dinov2_vitl14': [1024, 1024, 1024, 1024],  # ViT-L/14
            'dinov2_vitg14': [1536, 1536, 1536, 1536],  # ViT-G/14 (giant)
        }
        return dim_map.get(model_name, [768, 768, 768, 768])
    
    def forward(self, x):
        """Forward pass with multi-scale feature extraction."""
        B, C, H, W = x.shape
        
        # Convert grayscale to RGB if needed
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Extract features at multiple scales
        features = self._extract_features(x)
        
        # Decode to segmentation map
        out = self.decoder(features, (H, W))
        
        return out
    
    def _extract_features(self, x):
        """Extract hierarchical features from DINOv2."""
        if hasattr(self.backbone, 'get_intermediate_layers'):
            # Get features from multiple layers
            # DINOv2-small has 11 blocks, base has 12, large has 24, giant has 40
            if self.model_size == 'small':
                # Small model has 11 blocks (indexed 0-10), so use [2, 5, 8, 11]
                features = self.backbone.get_intermediate_layers(x, n=[2, 5, 8, 11])
            elif self.model_size == 'base':
                features = self.backbone.get_intermediate_layers(x, n=[3, 6, 9, 12])
            elif self.model_size == 'large':
                features = self.backbone.get_intermediate_layers(x, n=[6, 12, 18, 24])
            else:  # giant
                features = self.backbone.get_intermediate_layers(x, n=[10, 20, 30, 40])
        else:
            # Fallback for mock model
            features = self.backbone(x)
        
        return features


class DINOv2Decoder(nn.Module):
    """
    U-Net style decoder for DINOv2 features.
    Progressively upsamples and fuses features.
    """
    
    def __init__(self, encoder_dims, num_classes=3, decoder_dims=None):
        super(DINOv2Decoder, self).__init__()
        
        if decoder_dims is None:
            decoder_dims = [512, 256, 128, 64]
        
        # Adjust decoder dims based on encoder
        if encoder_dims[0] > 1024:  # Giant model
            decoder_dims = [1024, 512, 256, 128]
        elif encoder_dims[0] > 768:  # Large model
            decoder_dims = [768, 384, 192, 96]
        elif encoder_dims[0] == 384:  # Small model
            decoder_dims = [256, 128, 64, 32]
        
        # Project encoder features to decoder dims
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(enc_dim, dec_dim, 1)
            for enc_dim, dec_dim in zip(encoder_dims, decoder_dims)
        ])
        
        # Decoder blocks with skip connections
        # Build decoder from deep to shallow
        self.decoder_blocks = nn.ModuleList()
        
        # For DINOv2 small: decoder_dims = [256, 128, 64, 32]
        # We want: 256->128 (skip from 128), 128->64 (skip from 64), 64->32 (skip from 32)
        for i in range(len(decoder_dims)-1):
            in_ch = decoder_dims[i]
            skip_ch = decoder_dims[i+1]  # Skip connection from next level
            out_ch = decoder_dims[i+1]
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], decoder_dims[-1] // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dims[-1] // 2, num_classes, 1)
        )
        
    def forward(self, features, original_size):
        """Decode features to segmentation map."""
        H, W = original_size
        
        # Process transformer features
        processed_features = []
        for i, feat in enumerate(features):
            # Reshape from (B, N, D) to (B, D, H', W')
            if len(feat.shape) == 3:
                B, N, D = feat.shape
                h = w = int(N ** 0.5)
                feat = feat.transpose(1, 2).reshape(B, D, h, w)
            
            # Project to decoder dimension
            feat = self.proj_layers[i](feat)
            processed_features.append(feat)
        
        # Progressive decoding with skip connections
        # Start from first projected features (index 0)
        x = processed_features[0]
        
        # Process through decoder blocks with appropriate skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get skip connection from next level features
            skip = processed_features[i+1] if i+1 < len(processed_features) else None
            x = decoder_block(x, skip)
        
        # Final segmentation map
        x = self.segmentation_head(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


class DecoderBlock(nn.Module):
    """Single decoder block with skip connection."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.use_skip = skip_channels > 0
        
        # Upsampling - make sure in_channels matches what's expected
        # Use Conv2d + Upsample instead of ConvTranspose2d for better stability
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # Skip connection processing
        if self.use_skip:
            self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1)
        
        # Feature fusion
        fusion_channels = out_channels * 2 if self.use_skip else out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        # Upsample
        x = self.upsample(x)
        
        # Add skip connection
        if self.use_skip and skip is not None:
            # Match spatial dimensions
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], 
                                   mode='bilinear', align_corners=False)
            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)
        
        # Fuse features
        x = self.fusion(x)
        
        return x


class MockDINOv2(nn.Module):
    """Mock DINOv2 model for testing when pretrained weights aren't available."""
    
    def __init__(self, model_name):
        super(MockDINOv2, self).__init__()
        
        # Get dimensions based on model name
        dim_map = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536
        }
        self.hidden_dim = dim_map.get(model_name, 768)
        
        # Simple conv layers to simulate feature extraction
        self.features = nn.ModuleList([
            nn.Conv2d(3, self.hidden_dim, 14, stride=14),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        ])
        
    def forward(self, x):
        """Extract mock features."""
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x.flatten(2).transpose(1, 2))
        return features
    
    def get_intermediate_layers(self, x, n):
        """Mock intermediate layer extraction."""
        return self.forward(x)


class HybridDINOv2UNet(nn.Module):
    """
    Hybrid model combining DINOv2 backbone with CNN decoder.
    Best of both worlds: transformer features + CNN efficiency.
    """
    
    def __init__(self, dino_model='dinov2_vits14', cnn_encoder='resnet34',
                 num_classes=3, freeze_dino=True):
        super(HybridDINOv2UNet, self).__init__()
        
        # DINOv2 for global features
        self.dino = DINOv2UNet(dino_model, num_classes, freeze_dino)
        
        # CNN branch for local features
        import segmentation_models_pytorch as smp
        self.cnn = smp.Unet(
            encoder_name=cnn_encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
        
        # Fusion layer
        self.fusion = nn.Conv2d(num_classes * 2, num_classes, 1)
        
    def forward(self, x):
        # Get predictions from both branches
        if x.shape[1] == 1:
            x_rgb = x.repeat(1, 3, 1, 1)
        else:
            x_rgb = x
            
        dino_out = self.dino(x)
        cnn_out = self.cnn(x_rgb)
        
        # Fuse predictions
        fused = torch.cat([dino_out, cnn_out], dim=1)
        out = self.fusion(fused)
        
        return out


# Factory function for easy model creation
def get_dinov2_model(variant='small', num_classes=3, freeze=False, hybrid=False):
    """
    Get DINOv2-based model.
    
    Args:
        variant: Model size ('small', 'base', 'large', 'giant')
        num_classes: Number of segmentation classes
        freeze: Whether to freeze backbone
        hybrid: Whether to use hybrid CNN+DINOv2 model
    """
    model_map = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14',
        'large': 'dinov2_vitl14',
        'giant': 'dinov2_vitg14'
    }
    
    model_name = model_map.get(variant, 'dinov2_vitb14')
    
    if hybrid:
        return HybridDINOv2UNet(dino_model=model_name, num_classes=num_classes,
                               freeze_dino=freeze)
    else:
        return DINOv2UNet(model_name=model_name, num_classes=num_classes,
                         freeze_backbone=freeze)


if __name__ == "__main__":
    print("Testing DINOv2 U-Net implementation...")
    
    # Test basic model
    model = DINOv2UNet(model_name='dinov2_vits14', num_classes=3, freeze_backbone=True)
    x = torch.randn(2, 1, 224, 224)  # Grayscale input
    
    print(f"Input shape: {x.shape}")
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nDINOv2 U-Net created successfully!")

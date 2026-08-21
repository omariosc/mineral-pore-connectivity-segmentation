"""
U-Net architecture for mineral pore segmentation.
Binary segmentation: inside yellow (class 0) vs outside yellow (class 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np
from pathlib import Path
import sys
import warnings

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import load_config


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for mineral pore segmentation.
    
    Args:
        n_channels: Number of input channels (1 for grayscale, 3 for RGB)
        n_classes: Number of output classes (2 for binary segmentation)
        bilinear: Use bilinear upsampling instead of transposed convolutions
        base_features: Number of features in first layer (doubled at each level)
    """
    
    def __init__(self, n_channels: int = 1, n_classes: int = 2, 
                 bilinear: bool = True, base_features: int = 64):
        super(UNet, self).__init__()
        self.architecture_name = "plain_unet"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_features = base_features
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)
        self.outc = OutConv(base_features, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class AttentionBlock(nn.Module):
    """Attention block for improved feature fusion"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # Upsample g to match x size if needed
        if g.size()[-2:] != x.size()[-2:]:
            g = F.interpolate(g, size=x.size()[-2:], mode='bilinear', align_corners=True)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(UNet):
    """U-Net with attention gates for mineral pore segmentation"""
    
    def __init__(self, n_channels: int = 1, n_classes: int = 2, 
                 bilinear: bool = True, base_features: int = 64):
        super().__init__(n_channels, n_classes, bilinear, base_features)
        self.architecture_name = "attention_unet"
        
        # Add attention blocks
        factor = 2 if bilinear else 1
        self.att1 = AttentionBlock(base_features * 16 // factor, base_features * 8, base_features * 4)
        self.att2 = AttentionBlock(base_features * 8 // factor, base_features * 4, base_features * 2)
        self.att3 = AttentionBlock(base_features * 4 // factor, base_features * 2, base_features)
        self.att4 = AttentionBlock(base_features * 2 // factor, base_features, base_features // 2)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention
        x4 = self.att1(g=x5, x=x4)
        x = self.up1(x5, x4)
        
        x3 = self.att2(g=x, x=x3)
        x = self.up2(x, x3)
        
        x2 = self.att3(g=x, x=x2)
        x = self.up3(x, x2)
        
        x1 = self.att4(g=x, x=x1)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class DeepSupervisionUNet(UNet):
    """U-Net with deep supervision for improved gradient flow"""
    
    def __init__(self, n_channels: int = 1, n_classes: int = 2, 
                 bilinear: bool = True, base_features: int = 64):
        super().__init__(n_channels, n_classes, bilinear, base_features)
        self.architecture_name = "deep_supervision_unet"
        
        # Deep supervision outputs
        factor = 2 if bilinear else 1
        self.ds1 = OutConv(base_features * 8 // factor, n_classes)
        self.ds2 = OutConv(base_features * 4 // factor, n_classes)
        self.ds3 = OutConv(base_features * 2 // factor, n_classes)
        self.ds4 = OutConv(base_features, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with deep supervision
        d4 = self.up1(x5, x4)
        ds1 = self.ds1(d4)
        
        d3 = self.up2(d4, x3)
        ds2 = self.ds2(d3)
        
        d2 = self.up3(d3, x2)
        ds3 = self.ds3(d2)
        
        d1 = self.up4(d2, x1)
        ds4 = self.ds4(d1)
        
        # Final output
        logits = self.outc(d1)
        
        # Return all outputs for deep supervision
        if self.training:
            return logits, [ds1, ds2, ds3, ds4]
        else:
            return logits


def create_model(
    model_type: str = 'unet',
    config_path: Optional[str] = None,
    num_classes: Optional[int] = None,
    in_channels: Optional[int] = None,
) -> nn.Module:
    """
    Create a U-Net model based on configuration.
    
    Args:
        model_type: Explicit model type. ``unet`` and ``plain_unet`` both mean
            the original U-Net without attention or deep supervision.
            ``legacy_configured_unet`` retains the old configuration-driven
            alias for loading historical workflows.
        config_path: Path to configuration file
        
    Returns:
        PyTorch model
    """
    config = load_config(config_path or "config/pipeline_config.yaml")
    model_config = config.get('model', {})
    
    # Get model parameters
    n_channels = (
        int(in_channels)
        if in_channels is not None
        else model_config.get('in_channels', 1)
    )
    n_classes = num_classes if num_classes is not None else model_config.get('out_channels', 3)  # Use provided num_classes or config
    base_features = model_config.get('base_features', 64)
    bilinear = model_config.get('bilinear', True)
    use_attention = model_config.get('use_attention', True)
    use_deep_supervision = model_config.get('use_deep_supervision', True)

    normalized_model_type = str(model_type).strip().lower().replace('-', '_')

    # Handle improved_unet model type
    if normalized_model_type == 'improved_unet':
        try:
            from .improved_unet import ImprovedUNet
            model = ImprovedUNet(
                encoder_name='efficientnet-b4',
                encoder_weights='imagenet',
                in_channels=n_channels,
                classes=n_classes,
                activation=None
            )
            print(f"Created ImprovedUNet with {n_classes} output classes")
        except ImportError:
            print("Warning: ImprovedUNet not available, falling back to AttentionUNet")
            model = AttentionUNet(
                n_channels,
                n_classes,
                bilinear=bilinear,
                base_features=base_features,
            )
    elif normalized_model_type in {'unet', 'plain_unet'}:
        # Model names used in controlled comparisons must resolve to one exact
        # implementation. Historically ``unet`` consulted use_attention and
        # could silently instantiate AttentionUNet, invalidating the label on
        # an ablation run.
        model = UNet(
            n_channels,
            n_classes,
            bilinear=bilinear,
            base_features=base_features,
        )
    elif normalized_model_type == 'attention_unet':
        model = AttentionUNet(
            n_channels,
            n_classes,
            bilinear=bilinear,
            base_features=base_features,
        )
    elif normalized_model_type == 'deep_supervision_unet':
        model = DeepSupervisionUNet(
            n_channels,
            n_classes,
            bilinear=bilinear,
            base_features=base_features,
        )
    elif normalized_model_type == 'legacy_configured_unet':
        warnings.warn(
            "legacy_configured_unet is configuration-dependent and is retained "
            "only for historical reproducibility; use plain_unet for a controlled "
            "plain U-Net comparator.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_attention:
            model = AttentionUNet(
                n_channels,
                n_classes,
                bilinear=bilinear,
                base_features=base_features,
            )
        elif use_deep_supervision:
            model = DeepSupervisionUNet(
                n_channels,
                n_classes,
                bilinear=bilinear,
                base_features=base_features,
            )
        else:
            model = UNet(
                n_channels,
                n_classes,
                bilinear=bilinear,
                base_features=base_features,
            )
    else:
        raise ValueError(
            f"Unsupported U-Net model type: {model_type!r}. Choose one of "
            "plain_unet, unet, attention_unet, deep_supervision_unet, "
            "legacy_configured_unet, or improved_unet."
        )

    # Preserve both requested and resolved identities in checkpoints/configs.
    # Third-party ImprovedUNet implementations may not expose a stable name.
    model.requested_model_type = normalized_model_type
    if not hasattr(model, 'architecture_name'):
        model.architecture_name = type(model).__name__

    return model


def test_model():
    """Test the U-Net model with sample input."""
    print("Testing U-Net models...")
    
    # Test parameters
    batch_size = 2
    height, width = 512, 512
    n_channels = 1  # Grayscale
    n_classes = 2   # Binary segmentation
    
    # Create sample input
    x = torch.randn(batch_size, n_channels, height, width)
    
    # Test basic U-Net
    print("\n1. Testing basic U-Net:")
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Attention U-Net
    print("\n2. Testing Attention U-Net:")
    model_att = AttentionUNet(n_channels=n_channels, n_classes=n_classes)
    output_att = model_att(x)
    print(f"   Output shape: {output_att.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model_att.parameters()):,}")
    
    # Test Deep Supervision U-Net
    print("\n3. Testing Deep Supervision U-Net:")
    model_ds = DeepSupervisionUNet(n_channels=n_channels, n_classes=n_classes)
    model_ds.train()
    output_ds, ds_outputs = model_ds(x)
    print(f"   Main output shape: {output_ds.shape}")
    print(f"   Deep supervision outputs: {[o.shape for o in ds_outputs]}")
    print(f"   Model parameters: {sum(p.numel() for p in model_ds.parameters()):,}")
    
    # Test model creation from config
    print("\n4. Testing model creation from config:")
    model_config = create_model('unet')
    print(f"   Model type: {type(model_config).__name__}")
    
    # Test different input sizes
    print("\n5. Testing different input sizes:")
    for size in [256, 512, 1024]:
        x_test = torch.randn(1, n_channels, size, size)
        out_test = model(x_test)
        print(f"   Input {size}x{size} -> Output {out_test.shape}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_model()

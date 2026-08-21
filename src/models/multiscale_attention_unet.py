"""
Multi-scale Attention UNet for high-resolution pore segmentation.
Specifically designed for detecting small pore regions in geological images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .pyramid_context import PyramidContextBlock


class SpatialAttentionGate(nn.Module):
    """
    Spatial attention gate for focusing on relevant regions.
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
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
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from coarser scale
            x: Skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Ensure same spatial dimensions
        if g1.shape[2:] != x1.shape[2:]:
            # Resize g1 to match x1's spatial dimensions
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear' if x1.dim() == 5 else 'bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extract features at multiple scales using dilated convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Different dilation rates for multi-scale features
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Combine features
        self.combine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate multi-scale features
        combined = torch.cat([b1, b2, b3, b4], dim=1)
        
        return self.combine(combined)


class BoundaryRefinementModule(nn.Module):
    """
    Refine boundaries for better pore-mineral separation.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Edge detection branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Refinement branch
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_features = self.edge_conv(x)
        enhanced = torch.cat([x, x * edge_features], dim=1)
        refined = self.refine_conv(enhanced)
        return refined


class DoubleConv(nn.Module):
    """Double convolution block with batch norm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, use_multiscale: bool = False):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(2)
        
        if use_multiscale:
            self.conv = nn.Sequential(
                DoubleConv(in_channels, out_channels),
                MultiScaleFeatureExtractor(out_channels, out_channels)
            )
        else:
            self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling with attention gate and double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True,
                 use_attention: bool = True, use_boundary: bool = False):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        # Attention gate
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttentionGate(
                F_g=in_channels // 2,
                F_l=in_channels // 2,
                F_int=in_channels // 4
            )
        
        # Boundary refinement
        self.use_boundary = use_boundary
        if use_boundary:
            self.boundary = BoundaryRefinementModule(out_channels)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Ensure x1 and x2 have the same spatial dimensions
        if x1.shape[2:] != x2.shape[2:]:
            # Resize x1 to match x2's dimensions
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        
        # Apply attention to skip connection
        if self.use_attention:
            x2 = self.attention(g=x1, x=x2)
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        # Apply boundary refinement
        if self.use_boundary:
            x = self.boundary(x)
        
        return x


class MultiScaleAttentionUNet(nn.Module):
    """
    Enhanced UNet with multi-scale features and attention mechanisms.
    
    Design elements (each based on established segmentation components):
    1. Multi-scale feature extraction using dilated convolutions
    2. Spatial attention gates for focusing on pore regions
    3. Boundary refinement modules for accurate pore-mineral separation
    4. Deep supervision for better gradient flow
    """
    
    def __init__(self, 
                 n_channels: int = 1,
                 n_classes: int = 3,
                 bilinear: bool = True,
                 base_features: int = 32,
                 deep_supervision: bool = True):
        super().__init__()
        self.architecture_name = "multiscale_attention_unet"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_features = base_features
        self.deep_supervision = deep_supervision
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, base_features)
        
        # Encoder path with multi-scale features
        self.down1 = Down(base_features, base_features * 2, use_multiscale=False)
        self.down2 = Down(base_features * 2, base_features * 4, use_multiscale=True)
        self.down3 = Down(base_features * 4, base_features * 8, use_multiscale=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor, use_multiscale=True)
        
        # Decoder path with attention and boundary refinement
        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear,
                      use_attention=True, use_boundary=False)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear,
                      use_attention=True, use_boundary=True)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear,
                      use_attention=True, use_boundary=True)
        self.up4 = Up(base_features * 2, base_features, bilinear,
                      use_attention=True, use_boundary=True)
        
        # Output layers
        self.outc = nn.Conv2d(base_features, n_classes, kernel_size=1)
        
        # Deep supervision outputs
        if deep_supervision:
            self.ds3 = nn.Conv2d(base_features * 4 // factor, n_classes, kernel_size=1)
            self.ds2 = nn.Conv2d(base_features * 2 // factor, n_classes, kernel_size=1)
            self.ds1 = nn.Conv2d(base_features, n_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        
        if self.deep_supervision and self.training:
            ds3 = self.ds3(x)
            
        x = self.up2(x, x3)
        
        if self.deep_supervision and self.training:
            ds2 = self.ds2(x)
            
        x = self.up3(x, x2)
        
        if self.deep_supervision and self.training:
            ds1 = self.ds1(x)
            
        x = self.up4(x, x1)
        
        # Final output
        logits = self.outc(x)
        
        if self.deep_supervision and self.training:
            # Return main output and deep supervision outputs
            return logits, [ds1, ds2, ds3]
        else:
            return logits


class MultiScaleAttentionUNetPyramid(MultiScaleAttentionUNet):
    """Conditional full-tile candidate with locked x5 pyramid context.

    The reference :class:`MultiScaleAttentionUNet` implementation is not
    modified.  This separately named architecture copies its forward sequence
    and inserts one prospectively fixed block immediately after ``down4``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        factor = 2 if self.bilinear else 1
        bottleneck_channels = self.base_features * 16 // factor
        self.pyramid_context = PyramidContextBlock(
            in_channels=bottleneck_channels,
            branch_channels=32,
            pool_grids=(1, 2, 4, 8),
            norm_groups=8,
            dropout=0.0,
        )
        self.architecture_name = "multiscale_attention_unet_pyramid"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.pyramid_context(self.down4(x4))

        x = self.up1(x5, x4)
        if self.deep_supervision and self.training:
            ds3 = self.ds3(x)
        x = self.up2(x, x3)
        if self.deep_supervision and self.training:
            ds2 = self.ds2(x)
        x = self.up3(x, x2)
        if self.deep_supervision and self.training:
            ds1 = self.ds1(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.deep_supervision and self.training:
            return logits, [ds1, ds2, ds3]
        return logits

    def resolved_pyramid_context_config(self):
        """Return the immutable candidate-specific architecture record."""
        return self.pyramid_context.resolved_config()


def create_multiscale_attention_unet(config: dict) -> nn.Module:
    """Create multi-scale attention UNet from config."""
    if isinstance(config, dict):
        model_config = config.get('model', {})
        n_channels = model_config.get('in_channels', 1)
        n_classes = model_config.get('num_classes', 3)
        bilinear = model_config.get('bilinear', True)
        base_features = model_config.get('base_features', 32)
        deep_supervision = model_config.get('deep_supervision', True)
    else:
        # ConfigLoader object with dotted-key access.
        n_channels = config.get('model.in_channels', 1)
        n_classes = config.get('model.num_classes', 3)
        bilinear = config.get('model.bilinear', True)
        base_features = config.get('model.base_features', 32)
        deep_supervision = config.get('model.deep_supervision', True)
    
    return MultiScaleAttentionUNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        base_features=base_features,
        deep_supervision=deep_supervision
    )


def create_multiscale_attention_unet_pyramid(config: dict) -> nn.Module:
    """Create the separately named, locked pyramid-context candidate."""
    if isinstance(config, dict):
        model_config = config.get('model', {})
        n_channels = model_config.get('in_channels', 1)
        n_classes = model_config.get('num_classes', 3)
        bilinear = model_config.get('bilinear', True)
        base_features = model_config.get('base_features', 32)
        deep_supervision = model_config.get('deep_supervision', True)
    else:
        n_channels = config.get('model.in_channels', 1)
        n_classes = config.get('model.num_classes', 3)
        bilinear = config.get('model.bilinear', True)
        base_features = config.get('model.base_features', 32)
        deep_supervision = config.get('model.deep_supervision', True)

    if int(base_features) != 32 or not bool(bilinear):
        raise ValueError(
            "multiscale_attention_unet_pyramid is locked to base_features=32 "
            "and bilinear=True"
        )
    return MultiScaleAttentionUNetPyramid(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        base_features=base_features,
        deep_supervision=deep_supervision,
    )

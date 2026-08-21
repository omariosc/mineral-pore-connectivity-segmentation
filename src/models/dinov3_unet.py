"""
DINOv3 UNet for segmentation.
Uses DINOv3 ViT-S/16 as encoder with UNet decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import warnings
from pathlib import Path

from src.training.checkpoint_io import load_weights_only_checkpoint

class DINOv3UNet(nn.Module):
    """
    UNet with DINOv3 ViT-S/16 encoder.
    DINOv3 provides better features than DINOv2, trained on larger dataset (LVD-1689M).
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = True,
        patch_size: int = 16,
        embed_dim: int = 384,  # ViT-S/16 has 384 dims
        depth: int = 12,  # ViT-S has 12 layers
        num_heads: int = 6  # ViT-S has 6 heads
    ):
        """
        Initialize DINOv3 UNet.
        
        Args:
            num_classes: Number of output classes
            pretrained_path: Optional path to a local DINOv3 checkpoint
            freeze_encoder: Whether to freeze encoder weights
            patch_size: Patch size (16 for ViT-S/16)
            embed_dim: Embedding dimension (384 for ViT-S)
            depth: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.freeze_encoder = freeze_encoder
        
        # Load DINOv3 backbone
        self.backbone = self._load_dinov3(
            pretrained_path, 
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature dimensions for ViT-S/16
        # We'll extract features from layers [3, 6, 9, 12]
        decoder_channels = [256, 128, 64, 32]
        
        # Projection layers to match decoder channels
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(embed_dim, decoder_channels[0], 1),
            nn.Conv2d(embed_dim, decoder_channels[1], 1),
            nn.Conv2d(embed_dim, decoder_channels[2], 1),
            nn.Conv2d(embed_dim, decoder_channels[3], 1)
        ])
        
        # UNet decoder blocks
        self.decoder4 = self._make_decoder_block(decoder_channels[0], decoder_channels[1])
        self.decoder3 = self._make_decoder_block(decoder_channels[1], decoder_channels[2])
        self.decoder2 = self._make_decoder_block(decoder_channels[2], decoder_channels[3])
        self.decoder1 = self._make_decoder_block(decoder_channels[3], 16)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )
    
    def _load_dinov3(self, checkpoint_path: str, embed_dim: int, depth: int, num_heads: int):
        """Load compatible weights into the bundled DINOv3-style encoder.

        Checkpoint-adjacent Python modules are deliberately never imported.
        Architecture code must come from this reviewed package, while the
        checkpoint contributes tensor weights and primitive metadata only.
        """
        if not checkpoint_path:
            warnings.warn("No DINOv3 checkpoint path provided. Using random initialization.")
            return self._create_simple_vit(embed_dim, depth, num_heads, None)

        checkpoint_path = str(Path(checkpoint_path).expanduser())
        checkpoint = load_weights_only_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        return self._create_simple_vit(embed_dim, depth, num_heads, checkpoint)
    
    def _create_simple_vit(self, embed_dim: int, depth: int, num_heads: int, checkpoint: Optional[dict]):
        """Create a simplified ViT compatible with DINOv3."""
        
        class SimpleViT(nn.Module):
            def __init__(self, patch_size, embed_dim, depth, num_heads):
                super().__init__()
                self.patch_size = patch_size
                self.embed_dim = embed_dim
                
                # Patch embedding
                self.patch_embed = nn.Conv2d(1, embed_dim, patch_size, stride=patch_size)
                
                # Position embedding (learnable)
                self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))  # Max 1024 patches
                
                # Transformer blocks
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True
                    ) for _ in range(depth)
                ])
                
                self.norm = nn.LayerNorm(embed_dim)
            
            def forward(self, x):
                # x: (B, C, H, W)
                B, C, H, W = x.shape
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
                x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
                
                # Add position embedding
                N = x.shape[1]
                x = x + self.pos_embed[:, :N, :]
                
                # Store intermediate features
                features = []
                for i, block in enumerate(self.blocks):
                    x = block(x)
                    if i in [2, 5, 8, 11]:  # Extract at layers 3, 6, 9, 12
                        features.append(x)
                
                return features
            
            def get_intermediate_layers(self, x, n=None):
                """Compatible interface with DINOv3."""
                return self.forward(x)
        
        model = SimpleViT(self.patch_size, embed_dim, depth, num_heads)
        
        # Load checkpoint weights if available
        if checkpoint is not None:
            try:
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    # Try to load compatible weights
                    state_dict = checkpoint
                    # Filter and rename keys for compatibility
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if 'patch_embed' in k or 'pos_embed' in k or 'blocks' in k:
                            new_state_dict[k] = v
                    if new_state_dict:
                        model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                warnings.warn(f"Failed to load checkpoint weights: {e}")
        
        return model
    
    def _make_decoder_block(self, in_channels: int, out_channels: int):
        """Create a decoder block with upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _extract_features(self, x):
        """Extract hierarchical features from DINOv3."""
        # Get intermediate features
        if hasattr(self.backbone, 'get_intermediate_layers'):
            features = self.backbone.get_intermediate_layers(x, n=[3, 6, 9, 12])
        else:
            features = self.backbone(x)
        
        # Reshape features from (B, N, C) to (B, C, H, W)
        B = x.shape[0]
        H_patches = W_patches = int((x.shape[-1] // self.patch_size))
        
        reshaped_features = []
        for feat in features:
            if len(feat.shape) == 3:  # (B, N, C)
                feat = feat.transpose(1, 2)  # (B, C, N)
                feat = feat.reshape(B, -1, H_patches, W_patches)  # (B, C, H, W)
            reshaped_features.append(feat)
        
        return reshaped_features
    
    def forward(self, x):
        """Forward pass."""
        # Ensure input is correct size (divisible by patch_size)
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            # Pad to nearest multiple of patch_size
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            padded = True
            original_size = (H, W)
        else:
            padded = False
        
        # Extract features
        features = self._extract_features(x)
        
        # Project features to decoder dimensions
        projected_features = []
        for feat, proj in zip(features, self.proj_layers):
            projected_features.append(proj(feat))
        
        # Decoder path with skip connections
        x = projected_features[-1]  # Start from deepest features
        
        x = self.decoder4(x)
        x = x + F.interpolate(projected_features[-2], size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.decoder3(x)
        x = x + F.interpolate(projected_features[-3], size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.decoder2(x)
        x = x + F.interpolate(projected_features[-4], size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.decoder1(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Resize to original input size
        if padded:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


def create_dinov3_unet(num_classes: int = 3, **kwargs) -> nn.Module:
    """
    Create DINOv3 UNet model.
    
    Args:
        num_classes: Number of segmentation classes
        **kwargs: Additional arguments for DINOv3UNet
    
    Returns:
        DINOv3UNet model
    """
    return DINOv3UNet(num_classes=num_classes, **kwargs)

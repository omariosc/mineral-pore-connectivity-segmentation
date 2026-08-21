"""
SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
Based on: Xie et al., NeurIPS 2021
Adapted for mineral pore segmentation with extreme class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig
import segmentation_models_pytorch as smp


class SegFormerForSegmentation(nn.Module):
    """
    SegFormer model for semantic segmentation.
    Uses pretrained SegFormer encoder with custom decoder for 3-class segmentation.
    
    Args:
        model_name (str): Pretrained model name from HuggingFace
        num_classes (int): Number of segmentation classes
        dropout (float): Dropout rate
    """
    
    def __init__(self, model_name="nvidia/segformer-b2-finetuned-ade-512-512", 
                 num_classes=3, dropout=0.1):
        super(SegFormerForSegmentation, self).__init__()
        
        # Load pretrained SegFormer encoder
        self.encoder = SegformerModel.from_pretrained(model_name)
        
        # Get encoder output dimensions
        config = self.encoder.config
        hidden_sizes = config.hidden_sizes  # [64, 128, 320, 512] for B2
        
        # Multi-level feature fusion decoder
        self.decoder = SegFormerDecoder(
            encoder_channels=hidden_sizes,
            decoder_channels=256,
            num_classes=num_classes,
            dropout=dropout
        )
        
    def forward(self, x):
        # SegFormer expects (B, C, H, W) format
        if x.shape[1] == 1:  # Grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        
        # Get multi-scale features from encoder
        outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        features = outputs.hidden_states[1:]  # Skip first, get 4 levels
        
        # Decode to segmentation map
        out = self.decoder(features)
        
        return out


class SegFormerDecoder(nn.Module):
    """
    All-MLP decoder for SegFormer.
    Fuses multi-scale features and produces segmentation map.
    """
    
    def __init__(self, encoder_channels, decoder_channels=256, 
                 num_classes=3, dropout=0.1):
        super(SegFormerDecoder, self).__init__()
        
        self.num_classes = num_classes
        
        # MLP layers to unify channel dimensions
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(in_ch, decoder_channels, 1)
            for in_ch in encoder_channels
        ])
        
        # Batch norm layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(decoder_channels)
            for _ in encoder_channels
        ])
        
        # Feature fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(encoder_channels), decoder_channels, 1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(decoder_channels // 2, num_classes, 1)
        )
        
    def forward(self, features):
        B = features[0].shape[0]
        
        # Get target size from highest resolution feature
        target_h, target_w = features[0].shape[2:]
        
        # Process each feature level
        outs = []
        for i, feature in enumerate(features):
            # Reshape transformer output if needed
            if len(feature.shape) == 3:  # (B, N, C)
                h = w = int(feature.shape[1] ** 0.5)
                feature = feature.transpose(1, 2).reshape(B, -1, h, w)
            
            # Apply MLP and batch norm
            out = self.linear_layers[i](feature)
            out = self.bn_layers[i](out)
            
            # Upsample to target size
            if out.shape[2:] != (target_h, target_w):
                out = F.interpolate(out, size=(target_h, target_w), 
                                  mode='bilinear', align_corners=False)
            outs.append(out)
        
        # Concatenate and fuse
        out = torch.cat(outs, dim=1)
        out = self.linear_fuse(out)
        
        # Generate segmentation map
        out = self.segmentation_head(out)
        
        # Note: Don't upsample here - let it match the input size naturally
        # The model will handle the proper output size based on input
        
        return out


class LightweightSegFormer(nn.Module):
    """
    Lightweight SegFormer variant using MiT-B0 backbone.
    More suitable for faster training while maintaining transformer benefits.
    """
    
    def __init__(self, num_classes=3, pretrained=True, dropout=0.1):
        super(LightweightSegFormer, self).__init__()
        
        # Use smaller B0 variant
        model_name = "nvidia/mit-b0" if pretrained else None
        
        # Encoder channels for B0: [32, 64, 160, 256]
        encoder_channels = [32, 64, 160, 256]
        
        if pretrained and model_name:
            try:
                from transformers import SegformerForSemanticSegmentation
                # Load pretrained model
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            except:
                # Fallback to custom implementation
                self.model = CustomMiTB0(num_classes=num_classes, dropout=dropout)
        else:
            self.model = CustomMiTB0(num_classes=num_classes, dropout=dropout)
    
    def forward(self, x):
        input_size = x.shape[2:]  # Save input size
        
        if x.shape[1] == 1:  # Convert grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        
        out = self.model(x).logits if hasattr(self.model, 'logits') else self.model(x)
        
        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out


class CustomMiTB0(nn.Module):
    """
    Custom MiT-B0 implementation for cases where HuggingFace model isn't available.
    Mix Transformer B0 - smallest variant.
    """
    
    def __init__(self, num_classes=3, dropout=0.1):
        super(CustomMiTB0, self).__init__()
        
        # Simplified transformer blocks
        self.patch_embed1 = PatchEmbed(3, 32, 7, 4, 3)
        self.patch_embed2 = PatchEmbed(32, 64, 3, 2, 1)
        self.patch_embed3 = PatchEmbed(64, 160, 3, 2, 1)
        self.patch_embed4 = PatchEmbed(160, 256, 3, 2, 1)
        
        # Transformer blocks (simplified)
        self.block1 = nn.Sequential(*[TransformerBlock(32, 1) for _ in range(2)])
        self.block2 = nn.Sequential(*[TransformerBlock(64, 2) for _ in range(2)])
        self.block3 = nn.Sequential(*[TransformerBlock(160, 5) for _ in range(2)])
        self.block4 = nn.Sequential(*[TransformerBlock(256, 8) for _ in range(2)])
        
        # Decoder
        self.decoder = SimpleDecoder([32, 64, 160, 256], num_classes, dropout)
        
    def forward(self, x):
        input_size = x.shape[2:]  # Save original input size
        
        # Stage 1
        x1, (H1, W1) = self.patch_embed1(x)
        x1 = self.block1(x1)
        x1 = x1.reshape(x1.shape[0], H1, W1, -1).permute(0, 3, 1, 2)
        
        # Stage 2
        x2, (H2, W2) = self.patch_embed2(x1)
        x2 = self.block2(x2)
        x2 = x2.reshape(x2.shape[0], H2, W2, -1).permute(0, 3, 1, 2)
        
        # Stage 3
        x3, (H3, W3) = self.patch_embed3(x2)
        x3 = self.block3(x3)
        x3 = x3.reshape(x3.shape[0], H3, W3, -1).permute(0, 3, 1, 2)
        
        # Stage 4
        x4, (H4, W4) = self.patch_embed4(x3)
        x4 = self.block4(x4)
        x4 = x4.reshape(x4.shape[0], H4, W4, -1).permute(0, 3, 1, 2)
        
        # Decode
        out = self.decoder([x1, x2, x3, x4])
        
        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out


class PatchEmbed(nn.Module):
    """Patch embedding with overlapping patches."""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class TransformerBlock(nn.Module):
    """Simplified transformer block."""
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleDecoder(nn.Module):
    """Simple all-MLP decoder."""
    
    def __init__(self, encoder_channels, num_classes, dropout=0.1):
        super().__init__()
        
        decoder_dim = 256
        
        # Fusion layers
        self.layers = nn.ModuleList([
            nn.Conv2d(ch, decoder_dim, 1) for ch in encoder_channels
        ])
        
        # Final segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(decoder_dim, num_classes, 1)
        )
        
    def forward(self, features):
        # Get the feature size from first (highest resolution) feature
        # Don't multiply by a fixed factor - let the features determine the size
        if len(features) > 0:
            # Use the largest feature map size
            target_size = max([f.shape[2:] for f in features], key=lambda x: x[0] * x[1])
        else:
            target_size = features[0].shape[2:]
        
        # Process each level
        outs = []
        for i, feat in enumerate(features):
            out = self.layers[i](feat)
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            outs.append(out)
        
        # Concatenate and process
        out = torch.cat(outs, dim=1)
        out = self.head(out)
        
        return out


# For compatibility with existing code
def get_segformer_model(variant="b0", num_classes=3, pretrained=True):
    """
    Factory function to get SegFormer model variants.
    
    Args:
        variant: Model variant ('b0', 'b1', 'b2', 'b3', 'b4', 'b5')
        num_classes: Number of segmentation classes
        pretrained: Whether to use pretrained weights
    """
    if variant == "b0":
        return LightweightSegFormer(num_classes=num_classes, pretrained=pretrained)
    else:
        model_map = {
            "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
            "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
            "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
            "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
            "b5": "nvidia/segformer-b5-finetuned-ade-640-640"
        }
        model_name = model_map.get(variant, model_map["b2"])
        return SegFormerForSegmentation(model_name=model_name, num_classes=num_classes)


if __name__ == "__main__":
    # Test implementation
    print("Testing SegFormer implementation...")
    
    # Test lightweight version
    model = LightweightSegFormer(num_classes=3, pretrained=False)
    x = torch.randn(2, 1, 256, 256)  # Grayscale input
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("SegFormer model created successfully!")
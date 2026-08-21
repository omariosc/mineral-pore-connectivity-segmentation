"""
Comprehensive augmentation pipeline for mineral/pore segmentation.
Includes domain-specific augmentations for geological microscopy images.

IMPORTANT: These augmentations are applied independently to each training
sample, whether that sample is a 683-pixel patch or a native 2048-pixel tile.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
import random
import albumentations as A
import os


class GeologicalAugmentations:
    """Domain-specific augmentations for geological microscopy images."""
    
    def __init__(self, 
                 patch_size: int = 683,
                 training: bool = True,
                 augmentation_strength: str = 'strong',  # 'light', 'medium', 'strong'
                 mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0,
                 use_advanced: bool = True,
                 seed: int = 42):
        """
        Initialize augmentation pipeline.
        
        Args:
            patch_size: Size of the patches
            training: Whether in training mode
            augmentation_strength: Level of augmentation
            mixup_alpha: Alpha parameter for mixup
            cutmix_alpha: Alpha parameter for cutmix
            use_advanced: Whether to use advanced augmentations (mixup, cutmix, etc.)
            seed: Run seed passed to the Albumentations v2 composition.
        """
        self.patch_size = patch_size
        self.training = training
        self.augmentation_strength = augmentation_strength
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.use_advanced = use_advanced
        self.seed = int(seed)
        self.transforms_enabled = False
        self.disable_reason = None
        self.effective_transforms = []
        
        # Define augmentation pipeline
        self.transform = self._build_augmentation_pipeline()
        
    def _build_augmentation_pipeline(self):
        """Build the augmentation pipeline based on strength."""
        
        # Check if transforms are disabled via environment variable
        disable_transforms = os.environ.get('DISABLE_TRANSFORMS', '0') == '1'
        
        if disable_transforms:
            self.disable_reason = 'DISABLE_TRANSFORMS=1'
            return A.Compose([], seed=self.seed)
        
        if not self.training:
            self.disable_reason = 'evaluation_mode'
            return A.Compose([], seed=self.seed)

        allowed_strengths = {'light', 'medium', 'geological', 'strong'}
        if self.augmentation_strength not in allowed_strengths:
            raise ValueError(
                f"Unsupported augmentation strength: {self.augmentation_strength!r}. "
                f"Choose one of {sorted(allowed_strengths)}."
            )
        
        # Connectivity-preserving discrete symmetries. This is the complete
        # light/conservative policy used by the confirmatory experiment.
        symmetry_augmentations = [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]

        if self.augmentation_strength == 'light':
            self.transforms_enabled = True
            self.effective_transforms = list(symmetry_augmentations)
            return A.Compose(symmetry_augmentations, seed=self.seed)
        
        # Medium augmentations
        medium_augmentations = [
            # Morphology-warping transforms are intentionally excluded from
            # light/conservative runs and available only at medium or strong.
            A.ElasticTransform(alpha=20, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),

            # Color/intensity augmentations relevant to microscopy
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),  # RGB only
            
            # Simulate different microscopy conditions
            # Albumentations v2 expresses noise standard deviation relative to
            # the dtype range. These bounds preserve the former variance range
            # of 10--50 for uint8 inputs.
            A.GaussNoise(
                std_range=(0.012401, 0.027730),
                mean_range=(0.0, 0.0),
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # Simulate focus variations
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=0.2),
            
            # Perspective changes that might occur in imaging
            A.Perspective(scale=(0.05, 0.1), p=0.2),
        ]

        if self.augmentation_strength == 'medium':
            augmentations = symmetry_augmentations + medium_augmentations
            self.transforms_enabled = True
            self.effective_transforms = list(augmentations)
            return A.Compose(augmentations, seed=self.seed)
        
        # Strong augmentations
        strong_augmentations = [
            # More aggressive color augmentations
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Equalize(p=0.2),
            
            # Simulate different lighting conditions
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=3,
                p=0.2,
            ),
            
            # Simulate artifacts
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2),
            
            # Optical distortions
            A.OpticalDistortion(distort_limit=(-0.5, 0.5), p=0.2),
            
            # Simulate different staining/imaging protocols
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),  # RGB only
            
            # Random crops and scales to increase variety
            A.RandomResizedCrop(size=(self.patch_size, self.patch_size), 
                               scale=(0.64, 1.0), ratio=(0.9, 1.1), p=0.3),  # scale: 0.64 = 0.8^2, 1.0 = full area
            
            # Coarse dropout to simulate occlusions/artifacts
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(10, 50),
                hole_width_range=(10, 50),
                fill=0,
                fill_mask=None,
                p=0.2,
            ),
            
            # === NEW DOMAIN-SPECIFIC GEOLOGICAL AUGMENTATIONS ===
            
            # Mineralogical variations
            # A.RandomToneCurve(scale=0.3, p=0.3),  # RGB only - Simulate different mineral compositions
            # A.Posterize(num_bits=4, p=0.2),  # May require RGB - Simulate crystallization boundaries
            # A.Solarize(threshold=128, p=0.1),  # May require RGB - Simulate polarized light effects
            
            # Thin section thickness variations
            # A.ChannelShuffle(p=0.2),  # RGB only - Simulate different light wavelengths through minerals
            # A.ChromaticAberration(primary_distortion_limit=0.02, secondary_distortion_limit=0.01, p=0.3),  # RGB only
            
            # Edge enhancement for better boundary detection
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.UnsharpMask(blur_limit=(3, 7), sigma_limit=(0.5, 1.5), p=0.3),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2),
            
            # Edge-aware noise reduction
            A.MedianBlur(blur_limit=5, p=0.1),  # Preserve edges while removing noise
            
            # Microscope-specific artifacts
            A.MotionBlur(blur_limit=7, p=0.2),  # Stage movement artifacts
            A.ZoomBlur(max_factor=1.1, p=0.2),  # Focus adjustment artifacts
            A.RingingOvershoot(blur_limit=(7, 15), cutoff=(0.7, 0.9), p=0.2),  # Optical artifacts
            
            # Illumination variations specific to microscopy
            # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
            #                num_flare_circles_lower=1, num_flare_circles_upper=2, p=0.1),  # May require RGB
            
            # Texture variations for geological patterns
            # A.Superpixels(p_replace=0.1, n_segments=100, p=0.2),  # May require RGB - Simulate grain boundaries
            
            # Advanced spatial augmentations for geological deformations
            A.PiecewiseAffine(scale=(0.03, 0.05), p=0.3),  # Simulate tectonic deformation
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), 
                    rotate=(-45, 45), shear=(-10, 10), p=0.3),
            A.SafeRotate(limit=180, border_mode=cv2.BORDER_REFLECT, p=0.3),  # Full rotation range
            
            # Image quality variations
            A.Downscale(scale_range=(0.5, 0.9), p=0.3),  # Simulate lower resolution imaging
            A.ImageCompression(quality_range=(70, 100), p=0.2),  # JPEG compression artifacts
            # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),  # RGB only - Color channel shifts
        ]

        if self.augmentation_strength == 'strong':
            augmentations = (
                symmetry_augmentations + medium_augmentations + strong_augmentations
            )
            self.transforms_enabled = True
            self.effective_transforms = list(augmentations)
            return A.Compose(augmentations, seed=self.seed)
        
        # Geological-specific augmentations (subset of strong for domain-specific training)
        geological_specific = [
            # Essential geological transforms (grayscale compatible)
            # A.RandomToneCurve(scale=0.3, p=0.4),  # RGB only
            # A.ChromaticAberration(primary_distortion_limit=0.02, secondary_distortion_limit=0.01, p=0.4),  # RGB only
            A.UnsharpMask(blur_limit=(3, 7), sigma_limit=(0.5, 1.5), p=0.4),
            A.PiecewiseAffine(scale=(0.03, 0.05), p=0.4),
            # A.Superpixels(p_replace=0.1, n_segments=100, p=0.3),  # May require RGB
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=0.3),
        ]
        
        # Only the geological policy reaches this point; all other validated
        # strengths return above so unused constructors cannot emit warnings or
        # consume startup time.
        augmentations = symmetry_augmentations + geological_specific

        self.transforms_enabled = True
        self.effective_transforms = list(augmentations)
        return A.Compose(augmentations, seed=self.seed)

    def resolved_config(self) -> Dict:
        """Return the exact per-patch policy used by this augmentor."""
        return {
            'enabled': self.transforms_enabled,
            'disable_reason': self.disable_reason,
            'training': self.training,
            'strength': self.augmentation_strength,
            'seed': self.seed,
            'albumentations_version': A.__version__,
            'transforms': [
                {
                    'name': type(transform).__name__,
                    'probability': float(transform.p),
                }
                for transform in self.effective_transforms
            ],
        }
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to image and mask."""
        # Convert grayscale to RGB for augmentations if needed
        is_grayscale = False
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            is_grayscale = True
            if image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze(-1)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply albumentation pipeline
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # Convert back to one grayscale channel before tensor conversion. This
        # keeps training inputs consistent with validation/test inputs.
        if is_grayscale:
            if isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[0] == 3:
                image = (
                    0.299 * image[0]
                    + 0.587 * image[1]
                    + 0.114 * image[2]
                )
            elif not isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            # Handle grayscale images
            if image.ndim == 2:
                image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
            elif image.ndim == 3 and image.shape[2] == 1:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        # All grayscale models in this repository are trained and evaluated on
        # the same [-1, 1] scale. Previously, augmented training patches used
        # ImageNet RGB normalisation while validation used [-1, 1].
        # Source microscopy tiles are uint8; Albumentations preserves that
        # scale for the transforms configured above.
        image = image.float() / 255.0
        image = (image - 0.5) / 0.5
        
        return image, mask
    
    def mixup(self, images: torch.Tensor, masks: torch.Tensor, 
              alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            masks: Batch of masks (B, H, W)
            alpha: Mixup alpha parameter
            
        Returns:
            Mixed images, mixed masks, mixing coefficients
        """
        if alpha is None:
            alpha = self.mixup_alpha
            
        batch_size = images.size(0)
        if batch_size == 1:
            return images, masks, torch.ones(1)
        
        # Sample mixing coefficients
        lam = torch.from_numpy(np.random.beta(alpha, alpha, size=batch_size)).float()
        lam = lam.to(images.device)
        
        # Random permutation for mixing
        indices = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam.view(-1, 1, 1, 1) * images + (1 - lam.view(-1, 1, 1, 1)) * images[indices]
        
        # For masks, we'll use the lambda to determine which mask dominates
        # This preserves the discrete nature of segmentation masks
        mask_selector = (lam.view(-1, 1, 1) > 0.5).float()
        mixed_masks = mask_selector * masks.float() + (1 - mask_selector) * masks[indices].float()
        mixed_masks = mixed_masks.long()
        
        return mixed_images, mixed_masks, lam
    
    def cutmix(self, images: torch.Tensor, masks: torch.Tensor,
               alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            masks: Batch of masks (B, H, W)
            alpha: CutMix alpha parameter
            
        Returns:
            Mixed images, mixed masks, mixing coefficients
        """
        if alpha is None:
            alpha = self.cutmix_alpha
            
        batch_size = images.size(0)
        if batch_size == 1:
            return images, masks, torch.ones(1)
        
        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        indices = torch.randperm(batch_size).to(images.device)
        
        # Get image dimensions
        _, _, H, W = images.shape
        
        # Sample box coordinates
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling of box center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Box boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        mixed_masks = masks.clone()
        mixed_masks[:, bby1:bby2, bbx1:bbx2] = masks[indices, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda for actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        lam = torch.tensor([lam] * batch_size).to(images.device)
        
        return mixed_images, mixed_masks, lam
    
    def mosaic(self, images: List[torch.Tensor], masks: List[torch.Tensor],
               size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a mosaic of 4 images (similar to YOLOv4).
        
        Args:
            images: List of 4 images
            masks: List of 4 masks
            size: Output size
            
        Returns:
            Mosaic image and mask
        """
        if size is None:
            size = self.patch_size
            
        if len(images) != 4:
            return images[0], masks[0]
        
        # Create mosaic
        mosaic_img = torch.zeros(3, size, size)
        mosaic_mask = torch.zeros(size, size).long()
        
        # Place each image in a quadrant
        h_split = size // 2
        w_split = size // 2
        
        for i, (img, mask) in enumerate(zip(images, masks)):
            h, w = img.shape[1:]
            
            if i == 0:  # Top-left
                mosaic_img[:, :h_split, :w_split] = F.interpolate(
                    img.unsqueeze(0), size=(h_split, w_split), mode='bilinear'
                ).squeeze(0)
                mosaic_mask[:h_split, :w_split] = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0), size=(h_split, w_split), mode='nearest'
                ).squeeze().long()
            elif i == 1:  # Top-right
                mosaic_img[:, :h_split, w_split:] = F.interpolate(
                    img.unsqueeze(0), size=(h_split, w_split), mode='bilinear'
                ).squeeze(0)
                mosaic_mask[:h_split, w_split:] = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0), size=(h_split, w_split), mode='nearest'
                ).squeeze().long()
            elif i == 2:  # Bottom-left
                mosaic_img[:, h_split:, :w_split] = F.interpolate(
                    img.unsqueeze(0), size=(h_split, w_split), mode='bilinear'
                ).squeeze(0)
                mosaic_mask[h_split:, :w_split] = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0), size=(h_split, w_split), mode='nearest'
                ).squeeze().long()
            else:  # Bottom-right
                mosaic_img[:, h_split:, w_split:] = F.interpolate(
                    img.unsqueeze(0), size=(h_split, w_split), mode='bilinear'
                ).squeeze(0)
                mosaic_mask[h_split:, w_split:] = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0), size=(h_split, w_split), mode='nearest'
                ).squeeze().long()
        
        return mosaic_img, mosaic_mask


def get_augmentation_pipeline(config: Dict, training: bool = True) -> GeologicalAugmentations:
    """
    Factory function to create augmentation pipeline.
    
    Args:
        config: Configuration dictionary
        training: Whether in training mode
        
    Returns:
        GeologicalAugmentations instance
    """
    aug_config = config.get('augmentation', {})
    
    return GeologicalAugmentations(
        patch_size=config.get('patch_size', 683),
        training=training,
        augmentation_strength=aug_config.get('strength', 'strong'),
        mixup_alpha=aug_config.get('mixup_alpha', 0.2),
        cutmix_alpha=aug_config.get('cutmix_alpha', 1.0),
        use_advanced=aug_config.get('use_advanced', True),
        seed=config.get('seed', 42),
    )

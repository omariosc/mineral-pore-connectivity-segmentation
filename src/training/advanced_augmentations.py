"""
Advanced augmentation techniques for extreme class imbalance.
Focus on minority class (disconnected pores) preservation and enhancement.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random
from typing import Tuple, List, Optional
import albumentations as A


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation specifically for minority classes.
    Copies instances of minority class (disconnected pores) and pastes them
    in other locations to increase their representation.
    """
    def __init__(self, 
                 minority_classes=[0],  # Disconnected pores
                 paste_probability=0.5,
                 max_paste_objects=3,
                 scale_range=(0.8, 1.2),
                 rotation_range=(-30, 30)):
        self.minority_classes = minority_classes
        self.paste_probability = paste_probability
        self.max_paste_objects = max_paste_objects
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        
    def extract_objects(self, image, mask, target_class):
        """
        Extract individual objects of target class from mask.
        """
        class_mask = (mask == target_class).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(class_mask)
        
        objects = []
        for label_id in range(1, num_labels):  # Skip background
            obj_mask = (labels == label_id).astype(np.uint8)
            
            # Get bounding box
            y_coords, x_coords = np.where(obj_mask > 0)
            if len(y_coords) == 0:
                continue
                
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            
            # Extract object
            obj_img = image[y_min:y_max+1, x_min:x_max+1].copy()
            obj_mask_crop = obj_mask[y_min:y_max+1, x_min:x_max+1]
            
            objects.append({
                'image': obj_img,
                'mask': obj_mask_crop,
                'class': target_class,
                'bbox': (x_min, y_min, x_max, y_max)
            })
            
        return objects
    
    def augment_object(self, obj):
        """
        Apply random transformations to object.
        """
        img = obj['image']
        mask = obj['mask']
        
        # Random scale
        scale = np.random.uniform(*self.scale_range)
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Random rotation
        angle = np.random.uniform(*self.rotation_range)
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        img = cv2.warpAffine(img, M, (new_w, new_h))
        mask = cv2.warpAffine(mask, M, (new_w, new_h))
        
        return img, mask
    
    def paste_object(self, image, mask, obj_img, obj_mask, obj_class):
        """
        Paste object at random valid location.
        """
        h, w = image.shape[:2]
        obj_h, obj_w = obj_img.shape[:2]
        
        if obj_h >= h or obj_w >= w:
            return image, mask
        
        # Find valid paste location (avoid overlapping with existing minority class)
        max_attempts = 50
        for _ in range(max_attempts):
            x = np.random.randint(0, w - obj_w)
            y = np.random.randint(0, h - obj_h)
            
            # Check if location is valid (not overlapping with same class)
            target_region = mask[y:y+obj_h, x:x+obj_w]
            if not np.any(target_region == obj_class):
                # Paste object
                paste_mask = obj_mask > 0
                image[y:y+obj_h, x:x+obj_w][paste_mask] = obj_img[paste_mask]
                mask[y:y+obj_h, x:x+obj_w][paste_mask] = obj_class
                break
                
        return image, mask
    
    def __call__(self, image, mask):
        """
        Apply Copy-Paste augmentation.
        """
        if np.random.random() > self.paste_probability:
            return image, mask
            
        # Extract minority class objects
        all_objects = []
        for target_class in self.minority_classes:
            objects = self.extract_objects(image, mask, target_class)
            all_objects.extend(objects)
            
        if not all_objects:
            return image, mask
            
        # Randomly select objects to paste
        num_paste = min(len(all_objects), 
                       np.random.randint(1, self.max_paste_objects + 1))
        paste_objects = random.sample(all_objects, num_paste)
        
        # Apply augmentation and paste
        augmented_image = image.copy()
        augmented_mask = mask.copy()
        
        for obj in paste_objects:
            obj_img, obj_mask = self.augment_object(obj)
            augmented_image, augmented_mask = self.paste_object(
                augmented_image, augmented_mask, 
                obj_img, obj_mask, obj['class']
            )
            
        return augmented_image, augmented_mask


class ClassAwareMixUp:
    """
    MixUp augmentation that preserves minority class regions.
    Mixes images while ensuring minority class pixels are not destroyed.
    """
    def __init__(self, 
                 minority_classes=[0],
                 alpha=0.2,
                 preserve_minority_weight=0.8):
        self.minority_classes = minority_classes
        self.alpha = alpha
        self.preserve_minority_weight = preserve_minority_weight
        
    def __call__(self, image1, mask1, image2, mask2):
        """
        Apply class-aware MixUp.
        """
        # Random mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Create minority class preservation mask
        minority_mask1 = np.zeros_like(mask1, dtype=np.float32)
        minority_mask2 = np.zeros_like(mask2, dtype=np.float32)
        
        for cls in self.minority_classes:
            minority_mask1[mask1 == cls] = 1.0
            minority_mask2[mask2 == cls] = 1.0
            
        # Compute mixing weights with minority preservation
        mix_weight1 = lam * np.ones_like(image1, dtype=np.float32)
        mix_weight2 = (1 - lam) * np.ones_like(image2, dtype=np.float32)
        
        # Increase weight for minority class regions
        if len(image1.shape) == 3:
            minority_mask1 = np.expand_dims(minority_mask1, axis=-1)
            minority_mask2 = np.expand_dims(minority_mask2, axis=-1)
            
        mix_weight1 = mix_weight1 * (1 - minority_mask1) + \
                     self.preserve_minority_weight * minority_mask1
        mix_weight2 = mix_weight2 * (1 - minority_mask2) + \
                     self.preserve_minority_weight * minority_mask2
                     
        # Normalize weights
        total_weight = mix_weight1 + mix_weight2
        mix_weight1 = mix_weight1 / (total_weight + 1e-8)
        mix_weight2 = mix_weight2 / (total_weight + 1e-8)
        
        # Mix images
        mixed_image = (image1 * mix_weight1 + image2 * mix_weight2).astype(image1.dtype)
        
        # Mix masks (use nearest neighbor for discrete labels)
        mixed_mask = mask1 if lam > 0.5 else mask2
        
        # Ensure minority class regions are preserved
        for cls in self.minority_classes:
            minority_regions = (mask1 == cls) | (mask2 == cls)
            if minority_regions.any():
                # Prefer mask with more minority pixels
                count1 = (mask1 == cls).sum()
                count2 = (mask2 == cls).sum()
                source_mask = mask1 if count1 >= count2 else mask2
                mixed_mask[minority_regions] = source_mask[minority_regions]
                
        return mixed_image, mixed_mask


class MosaicAugmentation:
    """
    Mosaic augmentation that combines 4 images into one.
    Ensures minority class representation in final mosaic.
    """
    def __init__(self, minority_classes=[0], target_size=683):
        self.minority_classes = minority_classes
        self.target_size = target_size
        
    def __call__(self, images, masks):
        """
        Create mosaic from 4 images.
        Args:
            images: List of 4 images
            masks: List of 4 masks
        """
        assert len(images) == 4 and len(masks) == 4
        
        # Create output arrays
        mosaic_img = np.zeros((self.target_size, self.target_size, images[0].shape[-1]), 
                             dtype=images[0].dtype)
        mosaic_mask = np.zeros((self.target_size, self.target_size), dtype=masks[0].dtype)
        
        # Calculate quadrant size
        h_split = self.target_size // 2
        w_split = self.target_size // 2
        
        # Place each image in a quadrant
        quadrants = [
            (0, 0, h_split, w_split),  # Top-left
            (0, w_split, h_split, self.target_size),  # Top-right
            (h_split, 0, self.target_size, w_split),  # Bottom-left
            (h_split, w_split, self.target_size, self.target_size)  # Bottom-right
        ]
        
        # Prioritize images with minority classes
        minority_counts = []
        for mask in masks:
            count = sum((mask == cls).sum() for cls in self.minority_classes)
            minority_counts.append(count)
            
        # Sort by minority class count (descending)
        sorted_indices = np.argsort(minority_counts)[::-1]
        
        for idx, (y1, x1, y2, x2) in enumerate(quadrants):
            img_idx = sorted_indices[idx]
            img = images[img_idx]
            mask = masks[img_idx]
            
            # Resize to fit quadrant
            quad_h, quad_w = y2 - y1, x2 - x1
            img_resized = cv2.resize(img, (quad_w, quad_h))
            mask_resized = cv2.resize(mask, (quad_w, quad_h), interpolation=cv2.INTER_NEAREST)
            
            # Place in mosaic
            mosaic_img[y1:y2, x1:x2] = img_resized
            mosaic_mask[y1:y2, x1:x2] = mask_resized
            
        return mosaic_img, mosaic_mask


class GridMaskAugmentation:
    """
    GridMask augmentation that drops out grid regions.
    Preserves minority class regions from dropout.
    """
    def __init__(self, 
                 minority_classes=[0],
                 num_grid=(4, 8),
                 rotate_angle=(-90, 90),
                 keep_probability=0.5):
        self.minority_classes = minority_classes
        self.num_grid = num_grid
        self.rotate_angle = rotate_angle
        self.keep_probability = keep_probability
        
    def __call__(self, image, mask):
        """
        Apply GridMask augmentation.
        """
        h, w = image.shape[:2]
        
        # Random grid parameters
        grid_h = np.random.randint(*self.num_grid)
        grid_w = np.random.randint(*self.num_grid)
        
        # Create grid mask
        grid_mask = np.ones((h, w), dtype=np.float32)
        
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        for i in range(grid_h):
            for j in range(grid_w):
                if np.random.random() > self.keep_probability:
                    y1, y2 = i * cell_h, min((i + 1) * cell_h, h)
                    x1, x2 = j * cell_w, min((j + 1) * cell_w, w)
                    
                    # Check if region contains minority class
                    region_mask = mask[y1:y2, x1:x2]
                    has_minority = any((region_mask == cls).any() 
                                     for cls in self.minority_classes)
                    
                    # Don't drop regions with minority class
                    if not has_minority:
                        grid_mask[y1:y2, x1:x2] = 0
                        
        # Random rotation
        if self.rotate_angle[0] != 0 or self.rotate_angle[1] != 0:
            angle = np.random.uniform(*self.rotate_angle)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            grid_mask = cv2.warpAffine(grid_mask, M, (w, h))
            
        # Apply grid mask
        if len(image.shape) == 3:
            grid_mask = np.expand_dims(grid_mask, axis=-1)
            
        augmented_image = (image * grid_mask).astype(image.dtype)
        
        return augmented_image, mask


class SmartCropAugmentation:
    """
    Intelligent cropping that ensures minority class representation.
    """
    def __init__(self, 
                 minority_classes=[0],
                 crop_size=(512, 512),
                 minority_threshold=0.01):
        self.minority_classes = minority_classes
        self.crop_size = crop_size
        self.minority_threshold = minority_threshold
        
    def find_minority_regions(self, mask):
        """
        Find regions containing minority classes.
        """
        h, w = mask.shape
        crop_h, crop_w = self.crop_size
        
        valid_regions = []
        
        for y in range(0, h - crop_h + 1, crop_h // 4):
            for x in range(0, w - crop_w + 1, crop_w // 4):
                crop_mask = mask[y:y+crop_h, x:x+crop_w]
                
                # Calculate minority class ratio
                minority_pixels = sum((crop_mask == cls).sum() 
                                    for cls in self.minority_classes)
                total_pixels = crop_h * crop_w
                minority_ratio = minority_pixels / total_pixels
                
                if minority_ratio >= self.minority_threshold:
                    valid_regions.append({
                        'bbox': (x, y, x+crop_w, y+crop_h),
                        'minority_ratio': minority_ratio
                    })
                    
        return valid_regions
    
    def __call__(self, image, mask):
        """
        Apply smart cropping.
        """
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size
        
        if h < crop_h or w < crop_w:
            return image, mask
            
        # Find regions with minority classes
        minority_regions = self.find_minority_regions(mask)
        
        if minority_regions:
            # Select region with highest minority ratio
            best_region = max(minority_regions, key=lambda r: r['minority_ratio'])
            x1, y1, x2, y2 = best_region['bbox']
        else:
            # Random crop if no minority regions found
            x1 = np.random.randint(0, w - crop_w + 1)
            y1 = np.random.randint(0, h - crop_h + 1)
            x2, y2 = x1 + crop_w, y1 + crop_h
            
        # Crop image and mask
        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]
        
        return cropped_image, cropped_mask


def create_advanced_augmentation_pipeline(config):
    """
    Create comprehensive augmentation pipeline for extreme class imbalance.
    """
    transforms = []
    
    # Copy-Paste for minority class enhancement
    if config.get('use_copy_paste', True):
        transforms.append(CopyPasteAugmentation(
            minority_classes=config.get('minority_classes', [0]),
            paste_probability=config.get('paste_probability', 0.5)
        ))
    
    # Smart cropping
    if config.get('use_smart_crop', True):
        transforms.append(SmartCropAugmentation(
            minority_classes=config.get('minority_classes', [0]),
            crop_size=config.get('crop_size', (512, 512))
        ))
    
    # GridMask with minority preservation
    if config.get('use_gridmask', True):
        transforms.append(GridMaskAugmentation(
            minority_classes=config.get('minority_classes', [0])
        ))
    
    # Standard augmentations from albumentations
    standard_transforms = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.0)
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.3),
    ])
    
    return transforms, standard_transforms
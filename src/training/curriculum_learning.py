"""
Curriculum learning strategy for pore segmentation.
Gradually increases difficulty from easy to hard samples.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class CurriculumScheduler:
    """
    Manages curriculum learning by controlling sample difficulty.
    
    Difficulty metrics for pore segmentation:
    1. Pore density - fewer pores are easier
    2. Pore size variance - uniform sizes are easier
    3. Boundary complexity - simpler boundaries are easier
    4. Class balance - balanced classes are easier
    """
    
    def __init__(self,
                 total_epochs: int,
                 warmup_epochs: int = 5,
                 strategy: str = 'progressive',
                 difficulty_percentiles: List[float] = None):
        """
        Args:
            total_epochs: Total number of training epochs
            warmup_epochs: Number of epochs for warmup phase
            strategy: 'progressive', 'staged', or 'adaptive'
            difficulty_percentiles: Percentiles for difficulty thresholds
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy
        
        if difficulty_percentiles is None:
            # Default percentiles for easy, medium, hard samples
            self.difficulty_percentiles = [0.3, 0.6, 0.9, 1.0]
        else:
            self.difficulty_percentiles = difficulty_percentiles
        
        self.current_epoch = 0
        self.sample_difficulties = {}
        self.difficulty_thresholds = None
        
    def compute_sample_difficulty(self, 
                                 image_path: str,
                                 mask: np.ndarray) -> float:
        """
        Compute difficulty score for a sample.
        
        Args:
            image_path: Path to the image
            mask: Ground truth mask (H, W) with values 0, 1, 2
            
        Returns:
            Difficulty score (0 = easy, 1 = hard)
        """
        # Calculate various difficulty metrics
        metrics = {}
        
        # 1. Pore density (ratio of pore pixels)
        pore_pixels = np.sum((mask == 0) | (mask == 1))
        total_pixels = mask.size
        metrics['pore_density'] = pore_pixels / total_pixels
        
        # 2. Class imbalance (deviation from equal distribution)
        unique, counts = np.unique(mask, return_counts=True)
        if len(unique) == 3:
            class_ratios = counts / total_pixels
            # Measure deviation from balanced (0.33, 0.33, 0.33)
            imbalance = np.std(class_ratios)
            metrics['class_imbalance'] = imbalance * 3  # Scale to [0, 1]
        else:
            metrics['class_imbalance'] = 1.0  # Max difficulty if missing classes
        
        # 3. Disconnected pore ratio (harder if more disconnected)
        if pore_pixels > 0:
            disconnected_ratio = np.sum(mask == 0) / pore_pixels
            metrics['disconnected_ratio'] = disconnected_ratio
        else:
            metrics['disconnected_ratio'] = 0.0
        
        # 4. Boundary complexity (using gradient magnitude)
        from scipy import ndimage
        grad_y, grad_x = np.gradient(mask.astype(float))
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        boundary_pixels = np.sum(grad_mag > 0)
        metrics['boundary_complexity'] = boundary_pixels / total_pixels
        
        # 5. Small component prevalence (harder if many small components)
        from skimage.measure import label
        # Check for small disconnected pore components
        disconnected_mask = (mask == 0).astype(np.uint8)
        if disconnected_mask.any():
            labeled, num_components = label(disconnected_mask, return_num=True)
            if num_components > 0:
                component_sizes = []
                for i in range(1, num_components + 1):
                    size = np.sum(labeled == i)
                    component_sizes.append(size)
                
                # Count components smaller than 100 pixels
                small_components = sum(1 for s in component_sizes if s < 100)
                metrics['small_component_ratio'] = small_components / max(num_components, 1)
            else:
                metrics['small_component_ratio'] = 0.0
        else:
            metrics['small_component_ratio'] = 0.0
        
        # Combine metrics with weights
        weights = {
            'pore_density': 0.15,
            'class_imbalance': 0.25,
            'disconnected_ratio': 0.20,
            'boundary_complexity': 0.20,
            'small_component_ratio': 0.20
        }
        
        difficulty = sum(metrics[k] * weights[k] for k in weights)
        
        # Clamp to [0, 1]
        difficulty = max(0.0, min(1.0, difficulty))
        
        return difficulty
    
    def precompute_difficulties(self, dataset_samples: List[Tuple[str, np.ndarray]]):
        """
        Precompute difficulty scores for all samples.
        
        Args:
            dataset_samples: List of (image_path, mask) tuples
        """
        print("Computing sample difficulties...")
        
        difficulties = []
        for image_path, mask in dataset_samples:
            difficulty = self.compute_sample_difficulty(image_path, mask)
            self.sample_difficulties[image_path] = difficulty
            difficulties.append(difficulty)
        
        # Compute thresholds based on percentiles
        difficulties = np.array(difficulties)
        self.difficulty_thresholds = [
            np.percentile(difficulties, p * 100)
            for p in self.difficulty_percentiles
        ]
        
        print(f"Difficulty statistics:")
        print(f"  Min: {difficulties.min():.3f}")
        print(f"  Max: {difficulties.max():.3f}")
        print(f"  Mean: {difficulties.mean():.3f}")
        print(f"  Std: {difficulties.std():.3f}")
        print(f"  Thresholds: {self.difficulty_thresholds}")
    
    def get_current_difficulty_threshold(self, epoch: int) -> float:
        """
        Get the maximum difficulty threshold for the current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Maximum difficulty score to include
        """
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Warmup phase: use easiest samples
            return self.difficulty_thresholds[0]
        
        if self.strategy == 'progressive':
            # Gradually increase difficulty
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            # Map progress to threshold indices
            threshold_idx = min(int(progress * len(self.difficulty_thresholds)),
                              len(self.difficulty_thresholds) - 1)
            return self.difficulty_thresholds[threshold_idx]
            
        elif self.strategy == 'staged':
            # Step-wise increases in difficulty
            stages = len(self.difficulty_thresholds)
            epochs_per_stage = (self.total_epochs - self.warmup_epochs) // stages
            stage = min((epoch - self.warmup_epochs) // epochs_per_stage,
                       stages - 1)
            return self.difficulty_thresholds[stage]
            
        elif self.strategy == 'adaptive':
            # Adaptive based on performance (requires validation metrics)
            # For now, use progressive as fallback
            return self.get_current_difficulty_threshold_progressive(epoch)
        
        else:
            # No curriculum: use all samples
            return 1.0
    
    def filter_samples(self, 
                      sample_paths: List[str],
                      epoch: int) -> List[str]:
        """
        Filter samples based on current curriculum stage.
        
        Args:
            sample_paths: List of sample paths
            epoch: Current epoch
            
        Returns:
            Filtered list of sample paths
        """
        threshold = self.get_current_difficulty_threshold(epoch)
        
        filtered = []
        for path in sample_paths:
            if path in self.sample_difficulties:
                if self.sample_difficulties[path] <= threshold:
                    filtered.append(path)
            else:
                # Include samples without precomputed difficulty
                filtered.append(path)
        
        print(f"Epoch {epoch}: Using {len(filtered)}/{len(sample_paths)} samples "
              f"(difficulty ≤ {threshold:.3f})")
        
        return filtered
    
    def get_sample_weights(self, 
                          sample_paths: List[str],
                          epoch: int) -> torch.Tensor:
        """
        Get importance weights for samples based on difficulty.
        
        Args:
            sample_paths: List of sample paths
            epoch: Current epoch
            
        Returns:
            Tensor of weights for each sample
        """
        weights = []
        
        for path in sample_paths:
            if path in self.sample_difficulties:
                difficulty = self.sample_difficulties[path]
                
                # Higher weight for harder samples in later epochs
                if epoch < self.warmup_epochs:
                    # Inverse weighting during warmup (prefer easy)
                    weight = 1.0 - difficulty * 0.5
                else:
                    # Progressive weighting (gradually increase hard sample importance)
                    progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                    weight = 1.0 + difficulty * progress
                
                weights.append(weight)
            else:
                weights.append(1.0)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def save_difficulties(self, save_path: Path):
        """Save computed difficulties to file."""
        save_data = {
            'difficulties': self.sample_difficulties,
            'thresholds': self.difficulty_thresholds,
            'percentiles': self.difficulty_percentiles
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load_difficulties(self, load_path: Path):
        """Load precomputed difficulties from file."""
        with open(load_path, 'r') as f:
            save_data = json.load(f)
        
        self.sample_difficulties = save_data['difficulties']
        self.difficulty_thresholds = save_data['thresholds']
        self.difficulty_percentiles = save_data['percentiles']


class HardNegativeMining:
    """
    Focus training on hard negative samples (false positives).
    """
    
    def __init__(self,
                 ratio: float = 0.3,
                 min_samples: int = 100):
        """
        Args:
            ratio: Ratio of hard negatives to keep
            min_samples: Minimum number of samples to keep
        """
        self.ratio = ratio
        self.min_samples = min_samples
    
    def select_hard_negatives(self,
                            predictions: torch.Tensor,
                            targets: torch.Tensor,
                            losses: torch.Tensor) -> torch.Tensor:
        """
        Select hard negative samples based on loss.
        
        Args:
            predictions: Model predictions (N, C, H, W)
            targets: Ground truth (N, H, W)
            losses: Per-sample losses (N,)
            
        Returns:
            Mask indicating which samples to keep
        """
        batch_size = predictions.shape[0]
        
        # Sort by loss (descending)
        sorted_indices = torch.argsort(losses, descending=True)
        
        # Keep top ratio of hardest samples
        num_keep = max(int(batch_size * self.ratio), 
                      min(self.min_samples, batch_size))
        
        # Create mask
        mask = torch.zeros(batch_size, dtype=torch.bool)
        mask[sorted_indices[:num_keep]] = True
        
        return mask
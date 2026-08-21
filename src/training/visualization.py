"""
Improved visualization functions for 3-class segmentation.
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, Optional


def create_color_map_3class() -> np.ndarray:
    """
    Create color map for 3-class segmentation.
    
    Returns:
        Color map array (3, 3) with RGB values for each class
    """
    color_map = np.array([
        [255, 0, 0],      # Class 0 (Disconnected) - Red
        [0, 255, 0],      # Class 1 (Connected) - Green  
        [0, 0, 255],      # Class 2 (Minerals) - Blue
    ], dtype=np.uint8)
    
    return color_map


def visualize_prediction_3class(prediction: np.ndarray, 
                               original_image: Optional[np.ndarray] = None,
                               alpha: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create visualization for 3-class prediction.
    
    Args:
        prediction: (H, W) array with class indices (0, 1, 2)
        original_image: Optional original image for overlay
        alpha: Transparency for overlay (0=transparent, 1=opaque)
        
    Returns:
        colored_prediction: RGB visualization of prediction
        overlay: Overlay on original image (if provided)
    """
    color_map = create_color_map_3class()
    
    # Create colored prediction
    h, w = prediction.shape
    colored_prediction = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(3):
        mask = prediction == class_idx
        colored_prediction[mask] = color_map[class_idx]
    
    # Create overlay if original image provided
    overlay = None
    if original_image is not None:
        # Convert grayscale to RGB if needed
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_image
            
        # Create overlay
        overlay = cv2.addWeighted(original_rgb, 1-alpha, colored_prediction, alpha, 0)
    
    return colored_prediction, overlay


class ImprovedPatchPredictor:
    """Improved predictor that handles 3-class segmentation properly."""
    
    def __init__(self, model, device, patch_size: int = 683, num_classes: int = 3):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.num_classes = num_classes
    
    def predict_full_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on full image by splitting into patches and stitching results.
        
        Args:
            image_path: Path to input image
            
        Returns:
            class_prediction: (H, W) array with predicted class indices
            class_probabilities: (H, W, C) array with class probabilities
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = image.shape
        
        # Initialize arrays for all class probabilities
        class_probabilities = np.zeros((h, w, self.num_classes), dtype=np.float32)
        
        # Process each patch in 3x3 grid
        step = self.patch_size
        
        for row in range(3):
            for col in range(3):
                # Calculate patch coordinates
                start_y = row * step
                start_x = col * step
                
                # Ensure we don't go beyond image boundaries
                end_y = min(start_y + self.patch_size, h)
                end_x = min(start_x + self.patch_size, w)
                
                # Adjust start if needed to maintain patch size
                if end_y - start_y < self.patch_size:
                    start_y = max(0, end_y - self.patch_size)
                if end_x - start_x < self.patch_size:
                    start_x = max(0, end_x - self.patch_size)
                
                # Extract patch
                patch = image[start_y:start_y + self.patch_size, 
                            start_x:start_x + self.patch_size]
                
                # Prepare for model
                patch_tensor = torch.from_numpy(patch).float() / 255.0
                # Normalize to [-1, 1]
                patch_tensor = (patch_tensor - 0.5) / 0.5
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    output = self.model(patch_tensor)
                    # Get probabilities for all classes
                    probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
                
                # Place back in full prediction
                actual_end_y = min(start_y + self.patch_size, h)
                actual_end_x = min(start_x + self.patch_size, w)
                
                # Store all class probabilities
                for c in range(self.num_classes):
                    class_probabilities[start_y:actual_end_y, start_x:actual_end_x, c] = probs[c][
                        :actual_end_y - start_y, :actual_end_x - start_x
                    ]
        
        # Get class predictions from probabilities
        class_prediction = np.argmax(class_probabilities, axis=2)
        
        return class_prediction, class_probabilities


def save_prediction_visualizations(image_path: Path, 
                                  class_prediction: np.ndarray,
                                  class_probabilities: np.ndarray,
                                  output_dir: Path):
    """
    Save comprehensive visualizations for predictions.
    
    Args:
        image_path: Path to original image
        class_prediction: (H, W) predicted class indices
        class_probabilities: (H, W, C) class probabilities
        output_dir: Directory to save outputs
    """
    # Load original image
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    # Create visualizations
    colored_pred, overlay = visualize_prediction_3class(class_prediction, original)
    
    # Save outputs
    stem = image_path.stem
    
    # 1. Raw prediction (class indices)
    cv2.imwrite(str(output_dir / f"{stem}_prediction.png"), class_prediction)
    
    # 2. Colored prediction
    cv2.imwrite(str(output_dir / f"{stem}_prediction_colored.png"), colored_pred)
    
    # 3. Overlay visualization
    if overlay is not None:
        cv2.imwrite(str(output_dir / f"{stem}_visualization.png"), overlay)
    
    # 4. Class probability maps
    for c in range(class_probabilities.shape[2]):
        class_name = ['disconnected', 'connected', 'minerals'][c]
        prob_map = (class_probabilities[:, :, c] * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{stem}_prob_{class_name}.png"), prob_map)
    
    # 5. Create summary image with all visualizations
    create_summary_visualization(
        original, class_prediction, colored_pred, overlay,
        output_dir / f"{stem}_summary.png"
    )


def create_summary_visualization(original: np.ndarray,
                               prediction: np.ndarray,
                               colored_pred: np.ndarray,
                               overlay: np.ndarray,
                               save_path: Path):
    """Create a summary image with multiple visualizations."""
    # Convert all to RGB
    if len(original.shape) == 2:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original
    
    # Create grid layout
    h, w = original.shape[:2]
    
    # Top row: original and prediction
    top_row = np.hstack([original_rgb, colored_pred])
    
    # Bottom row: overlay and class distribution
    if overlay is not None:
        # Create class distribution visualization
        class_dist = create_class_distribution_viz(prediction, (h, w))
        bottom_row = np.hstack([overlay, class_dist])
    else:
        bottom_row = top_row
    
    # Combine rows
    summary = np.vstack([top_row, bottom_row])
    
    # Add labels
    summary = add_labels_to_summary(summary, h, w)
    
    # Save
    cv2.imwrite(str(save_path), summary)


def create_class_distribution_viz(prediction: np.ndarray, 
                                 target_size: Tuple[int, int]) -> np.ndarray:
    """Create visualization showing class distribution."""
    h, w = target_size
    viz = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Calculate class percentages
    total_pixels = prediction.size
    class_counts = [
        np.sum(prediction == 0),  # Disconnected
        np.sum(prediction == 1),  # Connected
        np.sum(prediction == 2),  # Minerals
    ]
    
    # Create bar chart
    bar_height = h - 60
    bar_width = w // 4
    start_x = w // 8
    
    colors = create_color_map_3class()
    class_names = ['Disconn.', 'Connect.', 'Minerals']
    
    for i, (count, color, name) in enumerate(zip(class_counts, colors, class_names)):
        percentage = count / total_pixels * 100
        bar_h = int(bar_height * percentage / 100)
        
        x1 = start_x + i * bar_width
        x2 = x1 + bar_width - 10
        y1 = h - 30
        y2 = y1 - bar_h
        
        # Draw bar
        cv2.rectangle(viz, (x1, y1), (x2, y2), color.tolist(), -1)
        
        # Add percentage text
        text = f"{percentage:.1f}%"
        cv2.putText(viz, text, (x1 + 5, y2 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add class name
        cv2.putText(viz, name, (x1, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add title
    cv2.putText(viz, "Class Distribution", (w // 4, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return viz


def add_labels_to_summary(summary: np.ndarray, h: int, w: int) -> np.ndarray:
    """Add text labels to summary visualization."""
    # Add labels for each quadrant
    labels = [
        ("Original", (10, 20)),
        ("Prediction", (w + 10, 20)),
        ("Overlay", (10, h + 20)),
        ("Distribution", (w + 10, h + 20))
    ]
    
    for label, (x, y) in labels:
        cv2.putText(summary, label, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(summary, label, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    return summary
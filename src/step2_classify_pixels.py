#!/usr/bin/env python3
"""Historical pixel-classification recovery step.

This non-authoritative utility preserves the original yellow-ring and
black-pixel classification path. It does not reconstruct the canonical
annotation masks or the canonical confirmatory dataset. Follow
``docs/CONFIRMATORY_RERUN.md`` for the current locked rerun protocol.

Classes:
- 0: Disconnected pores (black pixels inside yellow rings) - Red
- 1: Connected pores (black pixels outside yellow rings) - Green
- 2: Minerals (non-black pixels) - Blue
"""

import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import json
from typing import Tuple, Dict


def find_pixels_inside_rings(yellow_mask: np.ndarray) -> np.ndarray:
    """
    Find pixels that are inside the yellow rings.
    
    Args:
        yellow_mask: Binary mask where yellow ring pixels are 255
        
    Returns:
        Binary mask where pixels inside rings are 255
    """
    # Find contours of the yellow rings
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for inside regions
    inside_mask = np.zeros(yellow_mask.shape, dtype=np.uint8)
    
    # Fill the inside of each contour
    for contour in contours:
        cv2.drawContours(inside_mask, [contour], -1, 255, -1)
    
    # Remove the yellow ring pixels themselves from the inside mask
    inside_mask = cv2.bitwise_and(inside_mask, cv2.bitwise_not(yellow_mask))
    
    return inside_mask


def classify_pixels(img_path: str, yellow_mask_path: str, black_threshold: int = 100) -> Tuple[np.ndarray, Dict]:
    """
    Classify all pixels into 3 classes: disconnected pores, connected pores, and minerals.
    
    Returns:
        classification: np.ndarray with values 0, 1, or 2
        stats: Dictionary with classification statistics
    """
    # Read the original image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Read the yellow mask
    yellow_mask = cv2.imread(yellow_mask_path, cv2.IMREAD_GRAYSCALE)
    if yellow_mask is None:
        raise ValueError(f"Could not read mask: {yellow_mask_path}")
    
    # Find black pixels (pores)
    black_mask = (img < black_threshold).astype(np.uint8) * 255
    
    # Find pixels inside rings
    inside_rings_mask = find_pixels_inside_rings(yellow_mask)
    
    # Create classification
    # Start with all pixels as minerals (class 2)
    classification = np.full(img.shape, 2, dtype=np.uint8)
    
    # Class 1: Black pixels outside rings (connected pores)
    outside_rings_mask = cv2.bitwise_not(inside_rings_mask)
    connected_pores = cv2.bitwise_and(black_mask, outside_rings_mask)
    classification[connected_pores > 0] = 1
    
    # Class 0: Black pixels inside rings (disconnected pores)
    disconnected_pores = cv2.bitwise_and(black_mask, inside_rings_mask)
    classification[disconnected_pores > 0] = 0
    
    # Calculate statistics for all classes
    unique, counts = np.unique(classification, return_counts=True)
    total_pixels = classification.size
    
    stats = {
        'total_pixels': total_pixels,
        'class_counts': dict(zip(unique.tolist(), counts.tolist())),
        'class_percentages': {},
        'black_pixels': int(np.sum(black_mask > 0)),
        'yellow_pixels': int(np.sum(yellow_mask > 0)),
        'inside_ring_pixels': int(np.sum(inside_rings_mask > 0))
    }
    
    # Calculate percentages for all classes
    for class_id in [0, 1, 2]:
        count = stats['class_counts'].get(class_id, 0)
        stats['class_percentages'][class_id] = (count / total_pixels) * 100
    
    return classification, stats


def create_visualization(classification: np.ndarray) -> np.ndarray:
    """
    Create a color visualization of the 3-class classification.
    Class 0 (disconnected pores) = Red
    Class 1 (connected pores) = Green
    Class 2 (minerals) = Blue
    """
    h, w = classification.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply colors
    color_img[classification == 0] = [255, 0, 0]    # Red for disconnected pores
    color_img[classification == 1] = [0, 255, 0]    # Green for connected pores
    color_img[classification == 2] = [0, 0, 255]    # Blue for minerals
    
    return color_img


def process_all_images(img_dir: str, mask_dir: str, output_dir: str, black_threshold: int = 100):
    """
    Process all images and create 3-class classifications.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    classification_dir = os.path.join(output_dir, 'pixel_classifications')
    visualization_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Find all images
    img_files = list(Path(img_dir).glob("*.png"))
    print(f"\nProcessing {len(img_files)} images for 3-class classification...")
    
    # Overall statistics
    overall_stats = {
        'total_images': len(img_files),
        'total_pixels': 0,
        'class_totals': {0: 0, 1: 0, 2: 0},
        'images_with_disconnected': 0,
        'images_with_connected': 0,
        'images_with_minerals': 0
    }
    
    for img_path in tqdm(img_files, desc="Classifying pixels"):
        # Get corresponding mask path
        mask_filename = img_path.stem + "_mask.png"
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {img_path.name}")
            continue
        
        # Classify the image
        classification, stats = classify_pixels(str(img_path), mask_path, black_threshold)
        
        # Update overall statistics
        overall_stats['total_pixels'] += stats['total_pixels']
        for class_id in [0, 1, 2]:
            if class_id in stats['class_counts']:
                overall_stats['class_totals'][class_id] += stats['class_counts'][class_id]
                if class_id == 0 and stats['class_counts'][class_id] > 0:
                    overall_stats['images_with_disconnected'] += 1
                elif class_id == 1 and stats['class_counts'][class_id] > 0:
                    overall_stats['images_with_connected'] += 1
                elif class_id == 2 and stats['class_counts'][class_id] > 0:
                    overall_stats['images_with_minerals'] += 1
        
        # Save classification
        classification_path = os.path.join(classification_dir, img_path.name)
        cv2.imwrite(classification_path, classification)
        
        # Create and save visualization
        visualization = create_visualization(classification)
        vis_path = os.path.join(visualization_dir, img_path.stem + "_vis.png")
        cv2.imwrite(vis_path, visualization)
    
    # Calculate overall percentages
    if overall_stats['total_pixels'] > 0:
        overall_stats['class_percentages'] = {
            class_id: (overall_stats['class_totals'][class_id] / overall_stats['total_pixels']) * 100
            for class_id in [0, 1, 2]
        }
    else:
        overall_stats['class_percentages'] = {0: 0.0, 1: 0.0, 2: 0.0}
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'pixel_classification_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("3-CLASS PIXEL CLASSIFICATION SUMMARY")
    print("="*50)
    print(f"Total images processed: {overall_stats['total_images']}")
    print(f"Total pixels: {overall_stats['total_pixels']:,}")
    print(f"\nClass distribution:")
    print(f"  Disconnected pores (0): {overall_stats['class_totals'][0]:,} ({overall_stats['class_percentages'][0]:.2f}%)")
    print(f"  Connected pores (1): {overall_stats['class_totals'][1]:,} ({overall_stats['class_percentages'][1]:.2f}%)")
    print(f"  Minerals (2): {overall_stats['class_totals'][2]:,} ({overall_stats['class_percentages'][2]:.2f}%)")
    print(f"\nImages with disconnected pores: {overall_stats['images_with_disconnected']}")
    print(f"Images with connected pores: {overall_stats['images_with_connected']}")
    print(f"Images with minerals: {overall_stats['images_with_minerals']}")
    print("="*50)
    
    return overall_stats


if __name__ == "__main__":
    # Use configuration from step 1
    project_root = Path(__file__).parent.parent
    img_dir = project_root / "original_images"
    mask_dir = project_root / "results" / "step1_yellow_masks"
    output_dir = project_root / "results" / "step2_pixel_classification"
    
    print("=== HISTORICAL 3-class pixel-classification recovery ===")
    print("Non-authoritative output; not the canonical confirmatory dataset.")
    print("Current protocol: docs/CONFIRMATORY_RERUN.md")

    # Process all images
    process_all_images(str(img_dir), str(mask_dir), str(output_dir))

"""Historical yellow-ring detection recovery step.

This non-authoritative utility preserves the original thin-line detector for
forensic recovery. It does not reconstruct the canonical annotation masks or
the canonical confirmatory dataset. Follow ``docs/CONFIRMATORY_RERUN.md`` for
the current locked rerun protocol.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy import ndimage
from skimage import morphology, measure


def detect_yellow_precise(img_path: str, visualize: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Precise detection of yellow pixels using multiple color spaces.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    h, w = img.shape[:2]
    
    # Initialize combined mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Method 1: HSV - very broad range for yellow
    # Yellow hue is around 30 degrees (15 in OpenCV scale)
    # But we'll use a broader range to catch variations
    lower_hsv = np.array([10, 50, 50])
    upper_hsv = np.array([50, 255, 255])
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    combined_mask = cv2.bitwise_or(combined_mask, mask_hsv)
    
    # Method 2: LAB color space
    # In LAB, yellow has positive a (red-green) and positive b (yellow-blue)
    mask_lab = ((img_lab[:,:,1] > 128) &  # a channel > 128 (towards red)
                (img_lab[:,:,2] > 150)).astype(np.uint8) * 255  # b channel > 150 (towards yellow)
    combined_mask = cv2.bitwise_or(combined_mask, mask_lab)
    
    # Method 3: RGB ratios - yellow has high R and G, low B
    # Avoid division by zero
    sum_rgb = np.sum(img_rgb, axis=2).astype(float) + 1e-6
    r_ratio = img_rgb[:,:,0].astype(float) / sum_rgb
    g_ratio = img_rgb[:,:,1].astype(float) / sum_rgb
    b_ratio = img_rgb[:,:,2].astype(float) / sum_rgb
    
    mask_ratio = ((r_ratio > 0.33) & 
                  (g_ratio > 0.33) & 
                  (b_ratio < 0.25) &
                  (sum_rgb > 100)).astype(np.uint8) * 255  # Not too dark
    combined_mask = cv2.bitwise_or(combined_mask, mask_ratio)
    
    # Method 4: Direct RGB thresholds - looking for bright yellow
    mask_rgb = ((img_rgb[:,:,0] > 150) &  # Red channel
                (img_rgb[:,:,1] > 150) &  # Green channel
                (img_rgb[:,:,2] < 100) &  # Blue channel
                (img_rgb[:,:,0] + img_rgb[:,:,1] > 350)).astype(np.uint8) * 255  # R+G high
    combined_mask = cv2.bitwise_or(combined_mask, mask_rgb)
    
    # Method 5: Yellow-specific detector
    # Yellow = high (R+G)/2 - B
    yellow_score = ((img_rgb[:,:,0].astype(float) + img_rgb[:,:,1].astype(float)) / 2.0) - img_rgb[:,:,2].astype(float)
    mask_yellow = (yellow_score > 80).astype(np.uint8) * 255
    combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)
    
    # Clean up the initial mask
    kernel_small = np.ones((2, 2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)
    
    # Since yellow rings are thin, we need to be careful with morphological operations
    # Use skeleton to preserve thin structures
    skeleton = morphology.skeletonize(combined_mask > 0).astype(np.uint8) * 255
    
    # Dilate skeleton slightly to make it more visible
    kernel_dilate = np.ones((3, 3), np.uint8)
    final_mask = cv2.dilate(skeleton, kernel_dilate, iterations=1)
    
    # Also keep pixels that are very clearly yellow (high confidence)
    high_conf_mask = np.zeros_like(combined_mask)
    # Very strict yellow criteria
    strict_yellow = ((img_rgb[:,:,0] > 200) & 
                     (img_rgb[:,:,1] > 200) & 
                     (img_rgb[:,:,2] < 50) &
                     (img_hsv[:,:,0] >= 15) & 
                     (img_hsv[:,:,0] <= 35)).astype(np.uint8) * 255
    high_conf_mask = cv2.bitwise_or(high_conf_mask, strict_yellow)
    
    # Combine skeleton and high confidence pixels
    final_mask = cv2.bitwise_or(final_mask, high_conf_mask)
    
    # Remove very small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    min_area = 5  # Very small threshold to keep thin lines
    
    cleaned_mask = np.zeros_like(final_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255
    
    # Statistics
    stats_dict = {
        'total_pixels': cleaned_mask.size,
        'yellow_pixels': int(np.sum(cleaned_mask > 0)),
        'methods': {
            'hsv': int(np.sum(mask_hsv > 0)),
            'lab': int(np.sum(mask_lab > 0)),
            'ratio': int(np.sum(mask_ratio > 0)),
            'rgb': int(np.sum(mask_rgb > 0)),
            'yellow_score': int(np.sum(mask_yellow > 0)),
            'strict': int(np.sum(strict_yellow > 0))
        }
    }
    
    if visualize:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(mask_hsv, cmap='gray')
        axes[1].set_title(f'HSV ({stats_dict["methods"]["hsv"]:,})')
        axes[1].axis('off')
        
        axes[2].imshow(mask_rgb, cmap='gray')
        axes[2].set_title(f'RGB ({stats_dict["methods"]["rgb"]:,})')
        axes[2].axis('off')
        
        axes[3].imshow(mask_yellow, cmap='gray')
        axes[3].set_title(f'Yellow Score ({stats_dict["methods"]["yellow_score"]:,})')
        axes[3].axis('off')
        
        axes[4].imshow(combined_mask, cmap='gray')
        axes[4].set_title(f'Combined ({np.sum(combined_mask > 0):,})')
        axes[4].axis('off')
        
        axes[5].imshow(skeleton, cmap='gray')
        axes[5].set_title('Skeleton')
        axes[5].axis('off')
        
        axes[6].imshow(cleaned_mask, cmap='gray')
        axes[6].set_title(f'Final ({stats_dict["yellow_pixels"]:,})')
        axes[6].axis('off')
        
        # Overlay
        overlay = img_rgb.copy()
        overlay[cleaned_mask > 0] = [255, 0, 255]
        axes[7].imshow(overlay)
        axes[7].set_title('Overlay')
        axes[7].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'yellow_detection_{Path(img_path).stem}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return cleaned_mask, stats_dict


def process_all_images(img_dir: str, output_dir: str):
    """
    Process all images with precise yellow detection.
    """
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    img_files = list(Path(img_dir).glob("*.png"))
    print(f"\nProcessing {len(img_files)} images with precise detection...")
    
    # First analyze a few samples in detail
    print("\nDetailed analysis of sample images:")
    sample_files = img_files[:5]
    for img_path in sample_files:
        print(f"\nAnalyzing {img_path.name}...")
        mask, stats = detect_yellow_precise(str(img_path), visualize=True)
        print(f"  Yellow pixels: {stats['yellow_pixels']:,}")
        for method, count in stats['methods'].items():
            print(f"  {method}: {count:,}")
    
    # Process all images
    total_stats = {
        'total_pixels': 0,
        'total_yellow': 0,
        'images_with_yellow': 0,
        'yellow_pixel_counts': []
    }
    
    print("\n\nProcessing all images...")
    for img_path in tqdm(img_files):
        try:
            # Detect yellow
            mask, stats = detect_yellow_precise(str(img_path), visualize=False)
            
            # Save mask
            mask_path = os.path.join(output_dir, f"{img_path.stem}_mask.png")
            cv2.imwrite(mask_path, mask)
            
            # Update statistics
            total_stats['total_pixels'] += stats['total_pixels']
            total_stats['total_yellow'] += stats['yellow_pixels']
            
            if stats['yellow_pixels'] > 0:
                total_stats['images_with_yellow'] += 1
                total_stats['yellow_pixel_counts'].append(stats['yellow_pixels'])
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
    
    # Print final statistics
    print(f"\n=== Precise Yellow Ring Detection Statistics ===")
    print(f"Total images: {len(img_files)}")
    print(f"Images with yellow: {total_stats['images_with_yellow']} ({total_stats['images_with_yellow']/len(img_files)*100:.1f}%)")
    print(f"Total yellow pixels: {total_stats['total_yellow']:,}")
    print(f"Yellow percentage: {(total_stats['total_yellow'] / total_stats['total_pixels']) * 100:.3f}%")
    
    if total_stats['yellow_pixel_counts']:
        counts = np.array(total_stats['yellow_pixel_counts'])
        print(f"\nYellow pixels per image (with yellow):")
        print(f"  Mean: {counts.mean():.0f}")
        print(f"  Median: {np.median(counts):.0f}")
        print(f"  Min: {counts.min()}")
        print(f"  Max: {counts.max()}")
    
    # Create final visualizations
    create_comprehensive_visualizations(img_dir, output_dir, total_stats)


def create_comprehensive_visualizations(img_dir: str, mask_dir: str, stats: Dict):
    """
    Create comprehensive visualizations of the detection results.
    """
    viz_dir = os.path.join(mask_dir, 'visualizations')
    
    # Find images with varying amounts of yellow
    img_files = list(Path(img_dir).glob("*.png"))
    images_with_masks = []
    
    for img_path in img_files:
        mask_path = os.path.join(mask_dir, f"{img_path.stem}_mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            yellow_count = np.sum(mask > 0)
            if yellow_count > 0:
                images_with_masks.append((img_path, yellow_count))
    
    # Sort by yellow count and select diverse samples
    images_with_masks.sort(key=lambda x: x[1], reverse=True)
    
    # Select up to 15 samples
    num_samples = min(15, len(images_with_masks))
    if num_samples > 0:
        # Take samples from different ranges
        indices = np.linspace(0, len(images_with_masks)-1, num_samples, dtype=int)
        selected_samples = [images_with_masks[i] for i in indices]
        
        print(f"\nCreating {num_samples} detailed visualizations...")
        
        for idx, (img_path, yellow_count) in enumerate(selected_samples):
            # Read original and mask
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask_path = os.path.join(mask_dir, f"{img_path.stem}_mask.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Create visualizations
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original
            axes[0].imshow(img_rgb)
            axes[0].set_title(f'Original: {img_path.stem}')
            axes[0].axis('off')
            
            # Mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title(f'Yellow Mask ({yellow_count:,} pixels)')
            axes[1].axis('off')
            
            # Overlay
            overlay = img_rgb.copy()
            overlay[mask > 0] = [255, 0, 0]  # Red for visibility
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay (Red)')
            axes[2].axis('off')
            
            # Zoomed region - find area with yellow
            if yellow_count > 0:
                y_coords, x_coords = np.where(mask > 0)
                y_center = int(np.mean(y_coords))
                x_center = int(np.mean(x_coords))
                
                # Create zoom window
                zoom_size = 200
                y_min = max(0, y_center - zoom_size//2)
                y_max = min(img.shape[0], y_center + zoom_size//2)
                x_min = max(0, x_center - zoom_size//2)
                x_max = min(img.shape[1], x_center + zoom_size//2)
                
                zoom_img = overlay[y_min:y_max, x_min:x_max]
                axes[3].imshow(zoom_img)
                axes[3].set_title(f'Zoomed Region ({x_min}:{x_max}, {y_min}:{y_max})')
            else:
                axes[3].text(0.5, 0.5, 'No yellow detected', ha='center', va='center')
                axes[3].set_title('Zoomed Region')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'sample_{idx+1:02d}_{yellow_count}px.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {viz_dir}")


def main():
    # Paths
    base_dir = Path(__file__).resolve().parents[1]
    img_dir = base_dir / "original_images"
    output_dir = base_dir / "results" / "step1_yellow_masks"
    
    print("=== HISTORICAL Precise Yellow Ring Detection ===")
    print("Non-authoritative recovery output; not a reconstruction of canonical masks.")
    print("Current protocol: docs/CONFIRMATORY_RERUN.md")
    print(f"Input directory: {img_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process all images
    process_all_images(str(img_dir), str(output_dir))
    
    print("\nPrecise yellow ring detection complete!")


if __name__ == "__main__":
    main()

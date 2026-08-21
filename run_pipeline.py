#!/usr/bin/env python3
"""Run the historical preprocessing recovery path.

This non-authoritative utility preserves the original three-step workflow for
forensic recovery. It does not reconstruct the canonical annotation masks or
the canonical confirmatory dataset. Follow ``docs/CONFIRMATORY_RERUN.md`` for
the current locked rerun protocol.
"""

import subprocess
import sys
from pathlib import Path
import json
import time


def run_command(cmd: str, description: str):
    """Run a command and check for errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error in {description}:")
        print(result.stderr)
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.1f} seconds")
    
    if result.stdout:
        print("\nOutput:")
        print(result.stdout)
    
    return result


def stat_value(mapping, key, default=0):
    """Read JSON stats that may have stringified numeric keys."""
    return mapping.get(key, mapping.get(str(key), default))


def main():
    """Run the historical annotation-to-COCO recovery pipeline."""
    print("="*80)
    print("HISTORICAL MINERAL PORE PREPROCESSING RECOVERY PATH")
    print("="*80)
    print("\nSTATUS: non-authoritative historical recovery utility.")
    print("It does not reconstruct the canonical annotation masks or canonical")
    print("confirmatory dataset. See docs/CONFIRMATORY_RERUN.md for the current protocol.")
    print("\nThis pipeline will:")
    print("1. Detect yellow rings in microscopy images")
    print("2. Classify pixels into disconnected pores, connected pores, and minerals")
    print("3. Generate a COCO-format dataset for training")
    print("\nClasses:")
    print("- Class 0: Disconnected pores (inside yellow rings)")
    print("- Class 1: Connected pores (outside yellow rings)")
    print("- Class 2: Mineral matrix")
    print("="*80)
    
    # Check if source images exist
    image_dir = Path("original_images")
    if not image_dir.exists():
        print(f"Error: {image_dir} directory not found!")
        print("Please ensure your annotated microscopy PNGs are in 'original_images'.")
        sys.exit(1)
    
    image_count = len(list(image_dir.glob("*.png")))
    print(f"\nFound {image_count} images in {image_dir}")
    
    if image_count == 0:
        print("Error: No PNG images found!")
        sys.exit(1)
    
    # Step 1: Detect yellow rings
    run_command(
        "python3 src/step1_detect_yellow_rings.py",
        "Step 1: Yellow Ring Detection"
    )
    
    # Step 2: Classify pixels
    run_command(
        "python3 src/step2_classify_pixels.py",
        "Step 2: Pixel Classification (3 classes)"
    )
    
    # Step 3: Generate COCO dataset
    run_command(
        "python3 src/step3_generate_coco_dataset.py",
        "Step 3: COCO Dataset Generation"
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    # Load and display statistics
    stats_file = Path("results/step2_pixel_classification/pixel_classification_stats.json")
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print("\nPixel Classification Summary:")
        print(f"  Total pixels: {stats['total_pixels']:,}")
        print(f"  Disconnected pores: {stat_value(stats['class_totals'], 0):,} ({stat_value(stats['class_percentages'], 0):.2f}%)")
        print(f"  Connected pores: {stat_value(stats['class_totals'], 1):,} ({stat_value(stats['class_percentages'], 1):.2f}%)")
        print(f"  Mineral matrix: {stat_value(stats['class_totals'], 2):,} ({stat_value(stats['class_percentages'], 2):.2f}%)")
    
    coco_file = Path("results/step3_coco_dataset/annotations.json")
    if coco_file.exists():
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        # Count annotations by category
        cat_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
        
        print(f"\nCOCO Dataset Summary:")
        print(f"  Total images: {len(coco_data['images'])}")
        print(f"  Total annotations: {len(coco_data['annotations'])}")
        print(f"  Disconnected pore annotations: {cat_counts.get(0, 0)}")
        print(f"  Connected pore annotations: {cat_counts.get(1, 0)}")
        print(f"  Mineral annotations: {cat_counts.get(2, 0)}")
    
    print("\nOutput directories:")
    print("  - results/step1_yellow_masks/       : Yellow ring detection masks")
    print("  - results/step2_pixel_classification/: Pixel classifications")
    print("  - results/step3_coco_dataset/       : COCO format dataset")
    print("\nHistorical recovery outputs only; do not use them as the canonical")
    print("confirmatory dataset. See docs/CONFIRMATORY_RERUN.md.")
    print("="*80)


if __name__ == "__main__":
    main()

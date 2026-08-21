#!/usr/bin/env python3
"""Historical COCO-generation recovery step.

This non-authoritative utility preserves the original conversion from
three-class pixel classifications. It does not reconstruct the canonical
annotation masks or the canonical confirmatory dataset. Follow
``docs/CONFIRMATORY_RERUN.md`` for the current locked rerun protocol.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import os
from datetime import datetime
from tqdm import tqdm
import shutil
from typing import List, Dict, Tuple
import hashlib


def extract_polygons(classification: np.ndarray, class_id: int, min_area: int = 20) -> List[List[List[float]]]:
    """
    Extract polygon coordinates for a specific class.
    
    Args:
        classification: Classification array with values 0, 1, or 2
        class_id: Class ID to extract (0, 1, or 2)
        min_area: Minimum area for polygons to keep
        
    Returns:
        List of polygons (each polygon is a list of [x, y] coordinates)
    """
    # Create binary mask for the specific class
    mask = (classification == class_id).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Skip small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplify contour to reduce points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to COCO format (flatten the array and convert to list)
        if len(approx) >= 3:  # Need at least 3 points for a valid polygon
            polygon = approx.reshape(-1).tolist()
            polygons.append([polygon])
    
    return polygons


def create_coco_dataset(classification_dir: str, img_dir: str, output_dir: str):
    """
    Create COCO format dataset from 3-class pixel classifications.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize COCO format
    coco_data = {
        "info": {
            "description": "Mineral Pore Segmentation Dataset (3-class)",
            "version": "3.0",
            "year": 2024,
            "contributor": "Automated Pipeline",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [{
            "id": 1,
            "name": "Research Use",
            "url": ""
        }],
        "categories": [
            {"id": 0, "name": "disconnected_pore", "supercategory": "pore"},
            {"id": 1, "name": "connected_pore", "supercategory": "pore"},
            {"id": 2, "name": "mineral", "supercategory": "material"}
        ],
        "images": [],
        "annotations": []
    }
    
    # Get all classification files
    classification_files = list(Path(classification_dir).glob("*.png"))
    print(f"\nProcessing {len(classification_files)} pixel classifications...")
    
    annotation_id = 1
    
    # Create images subdirectory and link to original images
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total_annotations': 0,
        'annotations_per_class': {0: 0, 1: 0, 2: 0},
        'images_with_annotations': 0
    }
    
    for idx, class_file in enumerate(tqdm(classification_files, desc="Generating COCO annotations")):
        # Read classification
        classification = cv2.imread(str(class_file), cv2.IMREAD_GRAYSCALE)
        if classification is None:
            print(f"Warning: Could not read {class_file}")
            continue
        
        # Get corresponding original image
        img_filename = class_file.name
        orig_img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(orig_img_path):
            print(f"Warning: Original image not found for {img_filename}")
            continue
        
        # Read original image to get dimensions
        orig_img = cv2.imread(orig_img_path)
        if orig_img is None:
            print(f"Warning: Could not read original image {orig_img_path}")
            continue
        
        height, width = orig_img.shape[:2]
        
        # Create symbolic link to original image
        link_path = os.path.join(images_dir, img_filename)
        if os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(os.path.abspath(orig_img_path), link_path)
        
        # Add image info
        image_info = {
            "id": idx + 1,
            "file_name": img_filename,
            "width": width,
            "height": height,
            "date_captured": "",
            "license": 1
        }
        coco_data["images"].append(image_info)
        
        # Extract annotations for each class
        image_has_annotations = False
        
        for class_id in [0, 1, 2]:  # All 3 classes
            polygons = extract_polygons(classification, class_id)
            
            for polygon_group in polygons:
                for polygon in polygon_group:
                    # Calculate area
                    points = np.array(polygon).reshape(-1, 2)
                    area = cv2.contourArea(points.astype(np.float32))
                    
                    # Calculate bounding box
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                    bbox = [float(x_min), float(y_min), 
                           float(x_max - x_min), float(y_max - y_min)]
                    
                    # Create annotation
                    annotation = {
                        "id": annotation_id,
                        "image_id": idx + 1,
                        "category_id": class_id,
                        "segmentation": [polygon],
                        "area": float(area),
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1
                    stats['annotations_per_class'][class_id] += 1
                    image_has_annotations = True
        
        if image_has_annotations:
            stats['images_with_annotations'] += 1
    
    stats['total_annotations'] = len(coco_data["annotations"])
    
    # Save COCO format JSON
    coco_path = os.path.join(output_dir, "annotations.json")
    with open(coco_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    # Create train/val/test splits
    create_data_splits(coco_data, output_dir)
    
    # Print statistics
    print("\n" + "="*50)
    print("COCO DATASET GENERATION COMPLETE (3-CLASS)")
    print("="*50)
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Images with annotations: {stats['images_with_annotations']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"\nAnnotations per class:")
    print(f"  Disconnected pores (0): {stats['annotations_per_class'][0]}")
    print(f"  Connected pores (1): {stats['annotations_per_class'][1]}")
    print(f"  Minerals (2): {stats['annotations_per_class'][2]}")
    print(f"\nOutput saved to: {output_dir}")
    print("="*50)
    
    return stats


def create_data_splits(coco_data: Dict, output_dir: str, 
                      train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Create train/val/test splits for the dataset.
    """
    np.random.seed(42)  # For reproducibility
    
    # Get all image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    np.random.shuffle(image_ids)
    
    # Calculate split sizes
    n_images = len(image_ids)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    
    # Split image IDs
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    
    # Create split dictionary
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    # Save splits
    splits_path = os.path.join(output_dir, "splits.json")
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nData splits created:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val: {len(val_ids)} images")
    print(f"  Test: {len(test_ids)} images")


if __name__ == "__main__":
    # Use configuration from previous steps
    project_root = Path(__file__).parent.parent
    classification_dir = project_root / "results" / "step2_pixel_classification" / "pixel_classifications"
    img_dir = project_root / "original_images"
    output_dir = project_root / "results" / "step3_coco_dataset"
    
    print("=== HISTORICAL COCO dataset recovery ===")
    print("Non-authoritative output; not the canonical confirmatory dataset.")
    print("Current protocol: docs/CONFIRMATORY_RERUN.md")

    # Generate COCO dataset
    create_coco_dataset(str(classification_dir), str(img_dir), str(output_dir))

"""Deterministic patch/full-tile datasets for segmentation training."""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union
from .augmentations import get_augmentation_pipeline
from .data_contract import (
    CANONICAL_CLASS_NAMES,
    INPUT_NORMALIZATION,
    SplitManifest,
    create_deterministic_image_splits,
    load_lossless_mask,
    resolve_split_manifest,
    validate_lossless_mask_directory,
)
import os


def seed_patch_dataloader_worker(worker_id: int) -> None:
    """Deterministically separate Albumentations RNG streams per worker.

    Albumentations 2.0.8 predates the upstream worker-RNG synchronization fix.
    This top-level function is picklable and explicitly reseeds the Compose
    object inherited by each training worker.
    """
    del worker_id  # The PyTorch-assigned initial seed already encodes worker ID.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    torch_worker_seed = int(torch.initial_seed() % (2**32))
    augmentor = getattr(dataset, "augmentor", None)
    augmentation_seed = int(
        getattr(augmentor, "seed", getattr(dataset, "augmentation_seed", 0))
    )
    effective_seed = int((augmentation_seed + torch_worker_seed) % (2**32))
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    transform = getattr(augmentor, "transform", None)
    if transform is not None:
        set_random_seed = getattr(transform, "set_random_seed", None)
        if set_random_seed is None:
            raise RuntimeError(
                "Albumentations Compose lacks set_random_seed; cannot guarantee "
                "independent deterministic training-worker streams"
            )
        set_random_seed(effective_seed)

def _canonical_category_name(name: str) -> str:
    """Normalise known COCO category aliases to the manuscript class names."""
    normalised = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "disconnected": "disconnected_pore",
        "isolated_pore": "disconnected_pore",
        "connected": "connected_pore",
        "background": "mineral",
        "matrix": "mineral",
    }
    return aliases.get(normalised, normalised)


def deterministic_axis_positions(
    length: int, patch_size: int, overlap: int = 0
) -> Sequence[int]:
    """Cover an axis deterministically, anchoring the final patch at the edge."""
    length = int(length)
    patch_size = int(patch_size)
    overlap = int(overlap)
    if length <= 0 or patch_size <= 0:
        raise ValueError("length and patch_size must be positive")
    if patch_size > length:
        raise ValueError(f"patch size {patch_size} exceeds axis length {length}")
    step = patch_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than patch_size")
    last_start = length - patch_size
    positions = list(range(0, last_start + 1, step))
    if positions[-1] != last_start:
        positions.append(last_start)
    return tuple(positions)


def build_category_id_map(coco_data: Mapping[str, Any], num_classes: int) -> Dict[int, int]:
    """
    Map source COCO category IDs to the canonical training IDs.

    Historical annotation files in this repository use two different ID
    orders. Mapping by category name prevents a mineral annotation from being
    silently interpreted as a disconnected pore.
    """
    if num_classes not in (2, 3):
        raise ValueError(f"PatchDataset supports 2 or 3 classes, got {num_classes}")

    target_by_name = {
        name: class_id
        for class_id, name in CANONICAL_CLASS_NAMES.items()
        if class_id < num_classes
    }
    category_id_map: Dict[int, int] = {}
    unknown_categories = []
    for category in coco_data.get("categories", []):
        source_id = int(category["id"])
        name = _canonical_category_name(str(category.get("name", "")))
        if name in target_by_name:
            category_id_map[source_id] = target_by_name[name]
        elif not (num_classes == 2 and name == "mineral"):
            unknown_categories.append((source_id, name))

    if unknown_categories:
        raise ValueError(
            "Unsupported COCO categories: "
            + ", ".join(f"{category_id}:{name}" for category_id, name in unknown_categories)
        )

    annotation_category_ids = {
        int(annotation["category_id"])
        for annotation in coco_data.get("annotations", [])
    }
    unmapped = sorted(annotation_category_ids - set(category_id_map))
    if unmapped:
        raise ValueError(
            "Annotations reference category IDs with no canonical mapping: "
            + ", ".join(map(str, unmapped))
        )
    return category_id_map


class PatchDataset(Dataset):
    """Dataset covering each source image with a deterministic patch grid."""
    
    def __init__(self, coco_data: dict, image_dir: str, patch_size: int = 683, overlap: int = 0,
                 augmentation_config: Optional[Dict] = None, training: bool = True, 
                 bootstrap_factor: int = 1, image_ids: Optional[Sequence[int]] = None,
                 num_classes: Optional[int] = None, mineral_class_id: int = 2,
                 ignore_index: int = -100, split_name: Optional[str] = None,
                 mask_dir: Optional[Union[str, Path]] = None,
                 validated_mask_paths: Optional[Mapping[int, Union[str, Path]]] = None,
                 target_provenance: Optional[Mapping[str, Any]] = None):
        """
        Initialize patch dataset.
        
        Args:
            coco_data: COCO format data
            image_dir: Directory containing images
            patch_size: Size of each patch (683 for 3x3 split of 2048x2048)
            overlap: Overlap between patches in pixels
            augmentation_config: Configuration for augmentations
            training: Whether this is for training (enables augmentations)
            bootstrap_factor: How many augmented versions per patch (training only)
            image_ids: Whole-image subset to include. No image is split at patch level.
            num_classes: Output classes. Three-class mode treats unannotated pixels as mineral.
            mineral_class_id: Canonical mineral class ID in three-class mode.
            ignore_index: Target value for unannotated pixels in two-class mode.
            split_name: Optional provenance label (train, val, or test).
            mask_dir: Optional directory of authoritative lossless masks whose
                file names match the COCO images. When omitted, masks are
                rasterized from legacy COCO polygons.
        """
        self.image_dir = Path(image_dir)  
        self.patch_size = patch_size
        self.overlap = overlap
        self.coco_data = coco_data
        self.training = training
        self.bootstrap_factor = bootstrap_factor if training else 1  # Only bootstrap training
        self.num_classes = num_classes or len(self.coco_data.get('categories', []))
        self.mineral_class_id = mineral_class_id
        self.ignore_index = ignore_index
        self.split_name = split_name
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.category_id_map = build_category_id_map(self.coco_data, self.num_classes)
        
        # Initialize augmentation pipeline
        if augmentation_config is None:
            augmentation_config = {
                'patch_size': patch_size,
                'seed': 42,
                'augmentation': {
                    'enabled': True,
                    'strength': 'strong',
                    'mixup_alpha': 0.2,
                    'cutmix_alpha': 1.0,
                    'use_advanced': True
                }
            }
        self.augmentations_enabled = bool(
            augmentation_config.get('augmentation', {}).get('enabled', True)
        )
        self.augmentor = (
            get_augmentation_pipeline(augmentation_config, training=True)
            if training and self.augmentations_enabled
            else None
        )
        self.augmentation_provenance = (
            self.augmentor.resolved_config()
            if self.augmentor is not None
            else {
                'enabled': False,
                'disable_reason': 'dataset_not_training_or_augmentations_disabled',
                'training': training,
                'strength': augmentation_config.get('augmentation', {}).get(
                    'strength'
                ),
                'seed': int(augmentation_config.get('seed', 42)),
                'albumentations_version': None,
                'transforms': [],
            }
        )
        self.augmentation_seed = int(augmentation_config.get('seed', 42))
        
        # Create mappings
        all_images = {int(img['id']): img for img in self.coco_data['images']}
        if image_ids is None:
            selected_ids = list(all_images)
        else:
            selected_ids = [int(image_id) for image_id in image_ids]
            unknown_ids = sorted(set(selected_ids) - set(all_images))
            if unknown_ids:
                raise ValueError(
                    "PatchDataset received image IDs absent from COCO data: "
                    + ", ".join(map(str, unknown_ids))
                )
        self.image_ids = tuple(selected_ids)
        self.images = {image_id: all_images[image_id] for image_id in selected_ids}
        if self.mask_dir is not None:
            if validated_mask_paths is None:
                self.mask_paths, validated_provenance = validate_lossless_mask_directory(
                    self.coco_data,
                    self.image_dir,
                    self.mask_dir,
                    image_ids=self.image_ids,
                )
                self.target_provenance = validated_provenance
            else:
                normalised_paths = {
                    int(image_id): Path(path)
                    for image_id, path in validated_mask_paths.items()
                }
                missing_ids = sorted(set(self.image_ids) - set(normalised_paths))
                if missing_ids:
                    raise ValueError(
                        "Validated mask mapping is missing selected image IDs: "
                        + ", ".join(map(str, missing_ids))
                    )
                self.mask_paths = {
                    image_id: normalised_paths[image_id]
                    for image_id in self.image_ids
                }
                self.target_provenance = dict(target_provenance or {})
            self.target_source = "lossless_png_masks"
        else:
            self.mask_paths = {}
            self.target_source = "coco_polygon_rasterization"
            self.target_provenance = {
                "target_source": self.target_source,
                "mask_directory": None,
                "mask_count": None,
                "mask_aggregate_sha256": None,
                "validated_source_values": None,
                "canonical_value_mapping": (
                    "COCO category names mapped to canonical class IDs; "
                    "unannotated pixels become mineral in three-class mode"
                ),
            }
        self.img_to_anns = {}
        
        for ann in self.coco_data['annotations']:
            img_id = int(ann['image_id'])
            if img_id not in self.images:
                continue
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Generate all patches
        self.patches = []
        self._generate_patches()
        
    def _generate_patches(self):
        """Generate a deterministic covering grid for every source image."""
        print(f"Generating patches for {len(self.images)} images...")
        
        for img_id, img_info in self.images.items():
            img_h = int(img_info.get('height', 2048))
            img_w = int(img_info.get('width', 2048))
            if self.patch_size > img_h or self.patch_size > img_w:
                raise ValueError(
                    f"Patch size {self.patch_size} exceeds image {img_id} dimensions "
                    f"{img_w}x{img_h}"
                )
            
            y_positions = deterministic_axis_positions(
                img_h, self.patch_size, self.overlap
            )
            x_positions = deterministic_axis_positions(
                img_w, self.patch_size, self.overlap
            )

            for row, start_y in enumerate(y_positions):
                for col, start_x in enumerate(x_positions):
                    patch_info = {
                        'img_id': img_id,
                        'img_info': img_info,
                        'patch_id': f"{img_id}_{row}_{col}",
                        'row': row,
                        'col': col,
                        'start_y': start_y,
                        'start_x': start_x,
                        'end_y': start_y + self.patch_size,
                        'end_x': start_x + self.patch_size
                    }
                    self.patches.append(patch_info)
        
        print(
            f"Generated {len(self.patches)} patches from {len(self.images)} images "
            f"at patch size {self.patch_size}"
        )

    def training_class_statistics(self) -> Dict[str, Any]:
        """Count canonical pixels from authoritative training masks only.

        Loss weighting and sampling candidates must never infer class ratios
        from validation or held-out targets.  This method therefore refuses to
        run for any dataset that is not explicitly the training partition, or
        when targets would be reconstructed from legacy COCO polygons.
        """
        cached = getattr(self, "_training_class_statistics", None)
        if cached is not None:
            return dict(cached)
        if not self.training or self.split_name != "train":
            raise RuntimeError(
                "class statistics may only be derived from the training dataset"
            )
        if self.num_classes not in (2, 3):
            raise RuntimeError(
                "pore class statistics require a two- or three-output dataset"
            )
        if self.mask_dir is None or self.target_source != "lossless_png_masks":
            raise RuntimeError(
                "class statistics require authoritative lossless training masks"
            )

        counts = np.zeros(3, dtype=np.int64)
        training_mask_digest = hashlib.sha256()
        training_mask_records = []
        for image_id in self.image_ids:
            image_info = self.images[image_id]
            expected_shape = (
                int(image_info.get("height", 2048)),
                int(image_info.get("width", 2048)),
            )
            mask = load_lossless_mask(
                self.mask_paths[image_id],
                expected_shape=expected_shape,
                # Always count canonical C0/C1/C2 source pixels, even for a
                # two-output conditional model whose dataset maps C2 to ignore.
                num_classes=3,
                mineral_class_id=self.mineral_class_id,
                ignore_index=self.ignore_index,
            )
            counts += np.bincount(mask.reshape(-1), minlength=3)[:3]
            mask_path = self.mask_paths[image_id]
            relative_name = mask_path.relative_to(self.mask_dir).as_posix()
            training_mask_records.append((relative_name, mask_path))

        for relative_name, mask_path in sorted(training_mask_records):
            training_mask_digest.update(relative_name.encode("utf-8"))
            training_mask_digest.update(b"\0")
            with mask_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    training_mask_digest.update(chunk)
            training_mask_digest.update(b"\0")

        total = int(counts.sum())
        if total <= 0 or np.any(counts <= 0):
            raise RuntimeError(
                "all three canonical classes must be present in the training masks"
            )
        image_ids_json = json.dumps(
            list(self.image_ids), separators=(",", ":")
        ).encode("utf-8")
        self._training_class_statistics = {
            "source": "authoritative_training_masks_only",
            "split_name": "train",
            "image_count": len(self.image_ids),
            "image_ids": list(self.image_ids),
            "image_ids_sha256": hashlib.sha256(image_ids_json).hexdigest(),
            "class_order": ["C0", "C1", "C2"],
            "counts": [int(value) for value in counts.tolist()],
            "frequencies": [float(value / total) for value in counts.tolist()],
            "total_pixels": total,
            "training_mask_count": len(training_mask_records),
            "training_mask_aggregate_sha256": training_mask_digest.hexdigest(),
            "training_mask_aggregate_sha256_algorithm": (
                "sha256 over lexicographically sorted training-only UTF-8 "
                "relative filename, NUL, raw file bytes, NUL"
            ),
        }
        return dict(self._training_class_statistics)
    
    def __len__(self):
        return len(self.patches) * self.bootstrap_factor
    
    def __getitem__(self, idx):
        # Handle bootstrap factor - map idx to actual patch
        actual_idx = idx % len(self.patches)
        patch_info = self.patches[actual_idx]
        
        # Load full image
        img_path = self.image_dir / patch_info['img_info']['file_name']
        full_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if full_image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Extract patch
        patch_image = full_image[
            patch_info['start_y']:patch_info['end_y'],
            patch_info['start_x']:patch_info['end_x']
        ]
        
        h, w = full_image.shape
        img_id = patch_info['img_id']
        if self.mask_dir is not None:
            mask_path = self.mask_paths[img_id]
            full_mask = load_lossless_mask(
                mask_path,
                expected_shape=full_image.shape,
                num_classes=self.num_classes,
                mineral_class_id=self.mineral_class_id,
                ignore_index=self.ignore_index,
            )
        else:
            # Legacy fallback: rasterize COCO polygons. In three-class mode the
            # unannotated material is explicitly mineral.
            if self.num_classes == 2:
                full_mask = np.full((h, w), 255, dtype=np.uint8)
            else:
                full_mask = np.full((h, w), self.mineral_class_id, dtype=np.uint8)

            if img_id in self.img_to_anns:
                for ann in self.img_to_anns[img_id]:
                    source_class_id = int(ann['category_id'])
                    class_id = self.category_id_map[source_class_id]

                    if 'segmentation' in ann and ann['segmentation']:
                        for seg in ann['segmentation']:
                            if len(seg) >= 6:
                                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(full_mask, [points], class_id)
        
        # Extract patch mask
        patch_mask = full_mask[
            patch_info['start_y']:patch_info['end_y'],
            patch_info['start_x']:patch_info['end_x']
        ]
        
        # Keep grayscale and add channel dimension for consistency
        if len(patch_image.shape) == 2:
            patch_image = np.expand_dims(patch_image, axis=-1)  # Add channel dimension
        
        # Handle mask based on number of classes
        if self.num_classes == 2:
            # For 2-class pore segmentation, convert 255 to -100 for ignore_index
            patch_mask = patch_mask.astype(np.int64)
            patch_mask[patch_mask == 255] = self.ignore_index
        # For 3-class, keep mask as is (0: disconnected, 1: connected, 2: minerals)
        
        # Apply augmentations if in training mode and not disabled
        disable_transforms = os.environ.get('DISABLE_TRANSFORMS', '0') == '1'
        if self.training and self.augmentations_enabled and not disable_transforms:
            # For 2-class with ignore pixels, temporarily convert -100 back to 255 for augmentations
            if self.num_classes == 2 and np.any(patch_mask == self.ignore_index):
                aug_mask = patch_mask.copy()
                aug_mask[aug_mask == self.ignore_index] = 255
                patch_image, aug_mask = self.augmentor(patch_image, aug_mask)
                # Check if augmentor returned tensors
                if isinstance(patch_image, torch.Tensor):
                    # If tensors returned, handle mask conversion here and return
                    if isinstance(aug_mask, torch.Tensor):
                        # Convert 255 to -100 in tensor
                        aug_mask[aug_mask == 255] = self.ignore_index
                        return patch_image, aug_mask, patch_info['patch_id']
                # Otherwise continue with numpy processing
                if isinstance(aug_mask, torch.Tensor):
                    aug_mask = aug_mask.numpy()
                # Convert back to -100
                aug_mask = aug_mask.astype(np.int64)
                aug_mask[aug_mask == 255] = self.ignore_index
                patch_mask = aug_mask
            else:
                patch_image, patch_mask = self.augmentor(patch_image, patch_mask)
            
            # If augmentor returns tensors, use them directly
            if isinstance(patch_image, torch.Tensor):
                return patch_image, patch_mask, patch_info['patch_id']
            else:
                # Convert to tensors - handle grayscale
                if patch_image.ndim == 3 and patch_image.shape[2] == 1:
                    patch_image = patch_image.squeeze(-1)  # Remove channel dim temporarily
                patch_image = torch.from_numpy(patch_image).float() / 255.0
                if patch_image.ndim == 2:
                    patch_image = patch_image.unsqueeze(0)  # Add channel dim for grayscale
                # Normalize to [-1, 1] for consistency
                patch_image = (patch_image - 0.5) / 0.5
                patch_mask = torch.from_numpy(patch_mask).long()
        else:
            # For validation, just normalize
            if patch_image.ndim == 3 and patch_image.shape[2] == 1:
                patch_image = patch_image.squeeze(-1)  # Remove channel dim
            patch_image = torch.from_numpy(patch_image).float() / 255.0
            if patch_image.ndim == 2:
                patch_image = patch_image.unsqueeze(0)  # Add channel dim
            # For grayscale, use simple normalization
            patch_image = (patch_image - 0.5) / 0.5  # Normalize to [-1, 1]
            patch_mask = torch.from_numpy(patch_mask).long()
        
        return patch_image, patch_mask, patch_info['patch_id']


class PatchPredictor:
    """Handles prediction on patches and stitching results back together."""
    
    def __init__(self, model, device, patch_size: int = 683):
        self.model = model
        self.device = device
        self.patch_size = patch_size
    
    def predict_full_image(self, image_path: str) -> np.ndarray:
        """
        Predict on full image by splitting into patches and stitching results.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Full prediction mask as numpy array
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = image.shape
        prediction = np.zeros((h, w), dtype=np.float32)
        
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
                patch_tensor = (patch_tensor - 0.5) / 0.5
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    output = self.model(patch_tensor)
                    prob = torch.softmax(output, dim=1)
                    patch_pred = prob[0, 1].cpu().numpy()  # Connected class probability
                
                # Place back in full prediction
                actual_end_y = min(start_y + self.patch_size, h)
                actual_end_x = min(start_x + self.patch_size, w)
                
                prediction[start_y:actual_end_y, start_x:actual_end_x] = patch_pred[
                    :actual_end_y - start_y, :actual_end_x - start_x
                ]
        
        return prediction
    
    def predict_batch_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Predict on a batch of patches."""
        with torch.no_grad():
            outputs = self.model(images)
            return torch.softmax(outputs, dim=1)


def create_patch_data_loaders(
    coco_data: dict,
    image_dir: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    test_split: float = 0.0,
    split_manifest: Optional[SplitManifest] = None,
    split_seed: int = 42,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augmentation_config: Optional[Dict] = None,
    bootstrap_factor: int = 1,
    patch_size: int = 683,
    evaluation_patch_size: Optional[int] = None,
    evaluation_batch_size: int = 1,
    num_classes: int = 3,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    return_test: bool = False,
    mask_dir: Optional[Union[str, Path]] = None,
):
    """
    Create patch loaders from disjoint sets of whole source images.
    
    Args:
        coco_data: COCO format data
        image_dir: Directory containing images
        batch_size: Batch size for patch training
        val_split: Fraction of images to use for validation
        test_split: Fraction of images to reserve for final testing when no manifest is supplied
        split_manifest: JSON path or mapping with train, val, and test image IDs/file names
        split_seed: Random seed used only when generating a split
        distributed: Whether to use distributed data loading
        rank: Process rank for distributed training
        world_size: Total number of processes
        augmentation_config: Configuration for augmentations
        bootstrap_factor: Number of augmented versions per training patch
        patch_size: Training patch side length
        evaluation_patch_size: Validation/test side length; defaults to the
            training size for backward compatibility
        evaluation_batch_size: Validation/test batch size
        num_classes: Number of target classes
        num_workers: DataLoader worker processes
        pin_memory: DataLoader pin-memory setting; defaults to CUDA availability
        return_test: Return a third, untouched test loader
        mask_dir: Optional authoritative lossless-mask directory. Omitting it
            preserves legacy COCO polygon rasterization.
        
    Returns:
        (train_loader, val_loader), or (train_loader, val_loader, test_loader)
        when return_test is true.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    if split_manifest is not None:
        split_ids = resolve_split_manifest(coco_data, split_manifest)
        split_source = str(split_manifest) if isinstance(split_manifest, (str, Path)) else "mapping"
    else:
        split_ids = create_deterministic_image_splits(
            coco_data,
            val_split=val_split,
            test_split=test_split,
            seed=split_seed,
        )
        split_source = f"generated(seed={split_seed})"

    if not split_ids["train"]:
        raise ValueError("Training split is empty")
    if not split_ids["val"]:
        raise ValueError("Validation split is empty")
    if return_test and not split_ids["test"]:
        raise ValueError("Test split is empty but return_test=True")

    validated_mask_paths = None
    target_provenance = None
    if mask_dir is not None:
        selected_split_names = (
            ("train", "val", "test") if return_test else ("train", "val")
        )
        selected_ids = [
            image_id
            for split_name in selected_split_names
            for image_id in split_ids[split_name]
        ]
        validated_mask_paths, target_provenance = validate_lossless_mask_directory(
            coco_data,
            image_dir,
            mask_dir,
            image_ids=selected_ids,
        )

    evaluation_patch_size = (
        patch_size if evaluation_patch_size is None else int(evaluation_patch_size)
    )
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    dataset_kwargs = {
        "coco_data": coco_data,
        "image_dir": image_dir,
        "augmentation_config": augmentation_config,
        "num_classes": num_classes,
        "mask_dir": mask_dir,
        "validated_mask_paths": validated_mask_paths,
        "target_provenance": target_provenance,
    }
    train_dataset = PatchDataset(
        **dataset_kwargs,
        patch_size=patch_size,
        training=True,
        bootstrap_factor=bootstrap_factor,
        image_ids=split_ids["train"],
        split_name="train",
    )
    val_dataset = PatchDataset(
        **dataset_kwargs,
        patch_size=evaluation_patch_size,
        training=False,
        image_ids=split_ids["val"],
        split_name="val",
    )
    test_dataset = PatchDataset(
        **dataset_kwargs,
        patch_size=evaluation_patch_size,
        training=False,
        image_ids=split_ids["test"],
        split_name="test",
    ) if return_test and split_ids["test"] else None

    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    train_generator = torch.Generator().manual_seed(split_seed)
    validation_generator = torch.Generator().manual_seed(split_seed + 1)
    test_generator = torch.Generator().manual_seed(split_seed + 2)

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=split_seed,
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = (
            DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if test_dataset is not None
            else None
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  worker_init_fn=seed_patch_dataloader_worker,
                                  generator=train_generator)
        val_loader = DataLoader(val_dataset, batch_size=evaluation_batch_size, sampler=val_sampler,
                                num_workers=num_workers, pin_memory=pin_memory,
                                worker_init_fn=seed_patch_dataloader_worker,
                                generator=validation_generator)
        test_loader = (
            DataLoader(test_dataset, batch_size=evaluation_batch_size, sampler=test_sampler,
                       num_workers=num_workers, pin_memory=pin_memory,
                       worker_init_fn=seed_patch_dataloader_worker,
                       generator=test_generator)
            if test_dataset is not None
            else None
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  worker_init_fn=seed_patch_dataloader_worker,
                                  generator=train_generator)
        val_loader = DataLoader(val_dataset, batch_size=evaluation_batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory,
                                worker_init_fn=seed_patch_dataloader_worker,
                                generator=validation_generator)
        test_loader = (
            DataLoader(test_dataset, batch_size=evaluation_batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory,
                       worker_init_fn=seed_patch_dataloader_worker,
                       generator=test_generator)
            if test_dataset is not None
            else None
        )

    print(f"Image split source: {split_source}")
    print(
        "Target source: "
        + (
            f"lossless PNG masks ({mask_dir})"
            if mask_dir is not None
            else "legacy COCO polygon rasterization"
        )
    )
    print(f"Train patches: {len(train_dataset)} ({len(split_ids['train'])} images)")
    print(f"Val patches: {len(val_dataset)} ({len(split_ids['val'])} images)")
    if test_dataset is not None:
        print(f"Test patches: {len(test_dataset)} ({len(split_ids['test'])} images)")

    if return_test:
        return train_loader, val_loader, test_loader
    return train_loader, val_loader

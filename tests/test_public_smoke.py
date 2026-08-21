import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_public_smoke import build_synthetic_smoke_dataset
from src.training.data_contract import (
    load_lossless_mask,
    resolve_split_manifest,
    validate_lossless_mask_directory,
)


class PublicSyntheticSmokeTests(unittest.TestCase):
    def test_generated_corpus_is_deterministic_and_maps_all_classes(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_synthetic_smoke_dataset(Path(first_dir))
            second = build_synthetic_smoke_dataset(Path(second_dir))

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "passed")
        self.assertEqual(first["mapped_target_values"], [0, 1, 2])
        self.assertEqual(first["resolved_splits"], {"train": [1], "val": [2], "test": [3]})
        self.assertEqual(sum(first["mapped_class_pixel_counts"].values()), 3 * 9 * 9)

    def test_split_contract_rejects_leakage(self):
        coco_data = {
            "images": [
                {"id": 1, "file_name": "a.png"},
                {"id": 2, "file_name": "b.png"},
                {"id": 3, "file_name": "c.png"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "image leakage"):
            resolve_split_manifest(
                coco_data,
                {"train": [1], "val": [2], "test": [1, 3]},
            )

        with self.assertRaisesRegex(ValueError, "must be a list"):
            resolve_split_manifest(
                coco_data,
                {"train": "1", "val": [2], "test": [3]},
            )

        duplicate_ids = {
            "images": [
                {"id": 1, "file_name": "a.png"},
                {"id": 1, "file_name": "b.png"},
                {"id": 3, "file_name": "c.png"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "image IDs must be unique"):
            resolve_split_manifest(
                duplicate_ids,
                {"train": [1], "val": [2], "test": [3]},
            )

    def test_lossless_loader_rejects_invalid_values_and_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mask.png"
            invalid = np.full((4, 4), 255, dtype=np.uint8)
            invalid[0, 0] = 2
            Image.fromarray(invalid).save(path)
            with self.assertRaisesRegex(ValueError, r"invalid values \[2\]"):
                load_lossless_mask(path, (4, 4))

            valid = np.full((4, 4), 255, dtype=np.uint8)
            Image.fromarray(valid).save(path)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_lossless_mask(path, (3, 4))

    def test_mask_directory_rejects_unsafe_indexed_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            for unsafe_name in ("../escape.png", "/tmp/escape.png", "nested\\escape.png", ""):
                coco_data = {
                    "images": [
                        {
                            "id": 1,
                            "file_name": unsafe_name,
                            "width": 4,
                            "height": 4,
                        }
                    ]
                }
                with self.assertRaisesRegex(ValueError, "Unsafe COCO image file name"):
                    validate_lossless_mask_directory(
                        coco_data, root / "images", root / "masks"
                    )

    def test_command_line_smoke_uses_temporary_data_by_default(self):
        completed = subprocess.run(
            [sys.executable, "scripts/run_public_smoke.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["data_classification"], "synthetic_smoke_only_not_scientific_evidence")


if __name__ == "__main__":
    unittest.main()

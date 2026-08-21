import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_dataset_split_table import (  # noqa: E402
    DatasetContract,
    GENERATOR_RELATIVE_PATH,
    GENERATOR_SCHEMA,
    GENERATOR_VERSION,
    MASK_AGGREGATE_ALGORITHM,
    OUTPUT_FILENAMES,
    build_summary,
    render_bundle,
    write_bundle,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class DatasetSplitTableTests(unittest.TestCase):
    def make_fixture(
        self,
        root: Path,
        *,
        filenames: tuple[str, str, str] = ("train.png", "val.png", "test.png"),
        masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> DatasetContract:
        split_path = root / "config" / "confirmatory_splits.json"
        annotation_path = (
            root / "results" / "step3_coco_dataset" / "pore_annotations.json"
        )
        mask_dir = (
            root
            / "results"
            / "step2_pore_classification"
            / "pore_classifications"
        )
        split_path.parent.mkdir(parents=True)
        annotation_path.parent.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        generator_path = root / GENERATOR_RELATIVE_PATH
        generator_path.parent.mkdir(parents=True)
        generator_path.write_bytes(b"fixture dataset-table generator\n")

        split_payload = (
            json.dumps(
                {"train": [1], "val": [2], "test": [3]},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        annotation_payload = (
            json.dumps(
                {
                    "images": [
                        {
                            "id": image_id,
                            "file_name": file_name,
                            "width": 2,
                            "height": 2,
                        }
                        for image_id, file_name in enumerate(filenames, 1)
                    ]
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        split_path.write_bytes(split_payload)
        annotation_path.write_bytes(annotation_payload)

        if masks is None:
            masks = (
                np.array([[0, 1], [255, 255]], dtype=np.uint8),
                np.array([[0, 0], [1, 255]], dtype=np.uint8),
                np.array([[1, 1], [1, 255]], dtype=np.uint8),
            )
        for file_name, mask in zip(filenames, masks):
            Image.fromarray(mask).save(mask_dir / file_name)

        digest = hashlib.sha256()
        for file_name in sorted(filenames):
            digest.update(file_name.encode())
            digest.update(b"\0")
            digest.update((mask_dir / file_name).read_bytes())
            digest.update(b"\0")
        return DatasetContract(
            split_manifest=Path("config/confirmatory_splits.json"),
            annotation_index=Path(
                "results/step3_coco_dataset/pore_annotations.json"
            ),
            mask_directory=Path(
                "results/step2_pore_classification/pore_classifications"
            ),
            split_manifest_sha256=sha256_bytes(split_payload),
            annotation_index_sha256=sha256_bytes(annotation_payload),
            mask_aggregate_sha256=digest.hexdigest(),
            image_count=3,
        )

    def test_exact_counts_percentages_and_latex_rows_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract = self.make_fixture(root)
            rows, provenance = build_summary(root, contract)
            self.assertEqual(
                rows,
                [
                    {
                        "split": "train",
                        "display_name": "Training",
                        "tile_count": 1,
                        "total_pixels": 4,
                        "c0_pixels": 1,
                        "c0_percent": "25.000000",
                        "c1_pixels": 1,
                        "c1_percent": "25.000000",
                        "c2_pixels": 2,
                        "c2_percent": "50.000000",
                    },
                    {
                        "split": "val",
                        "display_name": "Validation",
                        "tile_count": 1,
                        "total_pixels": 4,
                        "c0_pixels": 2,
                        "c0_percent": "50.000000",
                        "c1_pixels": 1,
                        "c1_percent": "25.000000",
                        "c2_pixels": 1,
                        "c2_percent": "25.000000",
                    },
                    {
                        "split": "test",
                        "display_name": "Locked retrospective",
                        "tile_count": 1,
                        "total_pixels": 4,
                        "c0_pixels": 0,
                        "c0_percent": "0.000000",
                        "c1_pixels": 3,
                        "c1_percent": "75.000000",
                        "c2_pixels": 1,
                        "c2_percent": "25.000000",
                    },
                    {
                        "split": "total",
                        "display_name": "Total",
                        "tile_count": 3,
                        "total_pixels": 12,
                        "c0_pixels": 3,
                        "c0_percent": "25.000000",
                        "c1_pixels": 5,
                        "c1_percent": "41.666667",
                        "c2_pixels": 4,
                        "c2_percent": "33.333333",
                    },
                ],
            )
            self.assertEqual(
                provenance["mask_aggregate_sha256"], contract.mask_aggregate_sha256
            )
            first = render_bundle(root, contract)
            second = render_bundle(root, contract)
            self.assertEqual(first, second)
            self.assertEqual(
                first[OUTPUT_FILENAMES[1]].decode(),
                "% Generated by scripts/generate_dataset_split_table.py; do not edit.\n"
                "Training & 1 & 25.000 & 25.000 & 50.000 \\\\\n"
                "Validation & 1 & 50.000 & 25.000 & 25.000 \\\\\n"
                "Locked retrospective & 1 & 0.000 & 75.000 & 25.000 \\\\\n"
                "\\midrule\n"
                "Total & 3 & 25.000 & 41.667 & 33.333 \\\\\n"
                "\\bottomrule\n",
            )
            manifest = json.loads(first[OUTPUT_FILENAMES[2]])
            self.assertEqual(
                manifest["generator"],
                {
                    "schema": GENERATOR_SCHEMA,
                    "version": GENERATOR_VERSION,
                    "script": {
                        "path": GENERATOR_RELATIVE_PATH.as_posix(),
                        "sha256": sha256_bytes(
                            (root / GENERATOR_RELATIVE_PATH).read_bytes()
                        ),
                    },
                },
            )
            self.assertFalse(manifest["prediction_or_metric_inputs_read"])
            self.assertEqual(
                manifest["inputs"]["mask_directory"][
                    "aggregate_sha256_algorithm"
                ],
                MASK_AGGREGATE_ALGORITHM,
            )
            self.assertEqual(
                manifest["outputs"][OUTPUT_FILENAMES[0]]["sha256"],
                sha256_bytes(first[OUTPUT_FILENAMES[0]]),
            )

    def test_refuses_to_overwrite_an_existing_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract = self.make_fixture(root)
            relative = Path("paper_assets/tables/fixture")
            output, hashes = write_bundle(
                relative, project_root=root, contract=contract
            )
            before = {
                name: (output / name).read_bytes() for name in OUTPUT_FILENAMES
            }
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_bundle(relative, project_root=root, contract=contract)
            self.assertEqual(
                before,
                {name: (output / name).read_bytes() for name in OUTPUT_FILENAMES},
            )
            self.assertEqual(
                hashes,
                {name: sha256_bytes(before[name]) for name in OUTPUT_FILENAMES},
            )

    def test_rejects_split_hash_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract = self.make_fixture(root)
            split_path = root / contract.split_manifest
            split_path.write_bytes(split_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Split manifest SHA-256 mismatch"):
                build_summary(root, contract)

    def test_rejects_mask_byte_drift_before_decoding_pixels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract = self.make_fixture(root)
            mask_path = root / contract.mask_directory / "test.png"
            mask_path.write_bytes(b"not a PNG")
            with self.assertRaisesRegex(ValueError, "Mask aggregate SHA-256 mismatch"):
                build_summary(root, contract)

    def test_rejects_authenticated_mask_with_invalid_source_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            invalid = np.array([[0, 1], [2, 255]], dtype=np.uint8)
            contract = self.make_fixture(
                root, masks=(invalid, invalid, invalid)
            )
            with self.assertRaisesRegex(ValueError, r"invalid values \[2\]"):
                build_summary(root, contract)

    def test_rejects_unsafe_authenticated_annotation_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract = self.make_fixture(
                root, filenames=("../escape.png", "val.png", "test.png")
            )
            with self.assertRaisesRegex(ValueError, "Unsafe annotation filename"):
                build_summary(root, contract)

    @unittest.skipUnless(hasattr(os, "symlink"), "symbolic links unavailable")
    def test_rejects_symlinked_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract = self.make_fixture(root)
            mask_dir = root / contract.mask_directory
            original = mask_dir / "train.png"
            replacement = root / "replacement.png"
            original.rename(replacement)
            original.symlink_to(replacement)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                build_summary(root, contract)


if __name__ == "__main__":
    unittest.main()

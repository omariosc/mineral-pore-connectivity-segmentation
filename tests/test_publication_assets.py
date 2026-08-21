import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.generate_publication_assets as publication_assets
from scripts.generate_publication_assets import (
    FONT_FAMILY,
    PUBLICATION_CLASS_COLORS,
    configure_style,
    confirmatory_split_summary_rows,
    dataset_summary_rows,
    plot_study_workflow,
)


SPLIT_MANIFEST = PROJECT_ROOT / "config" / "confirmatory_splits.json"


class ConfirmatorySplitSummaryTests(unittest.TestCase):
    def write_manifest(self, payload: object) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        manifest_path = Path(temp_dir.name) / "splits.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path

    def test_frozen_manifest_reports_exact_current_rerun_counts(self):
        self.assertEqual(
            confirmatory_split_summary_rows(SPLIT_MANIFEST),
            [
                {"metric": "declared_train_images", "value": "74"},
                {"metric": "declared_val_images", "value": "5"},
                {"metric": "declared_test_images", "value": "21"},
            ],
        )

    def test_malformed_manifest_is_rejected(self):
        malformed = self.write_manifest(
            {"train": [1, 2], "val": "3", "test": [4]}
        )
        with self.assertRaisesRegex(ValueError, "val.*list of integer IDs"):
            confirmatory_split_summary_rows(malformed)

    def test_overlapping_manifest_is_rejected(self):
        overlapping = self.write_manifest(
            {"train": [1, 2], "val": [3], "test": [2, 4]}
        )
        with self.assertRaisesRegex(ValueError, "train.*test.*overlap"):
            confirmatory_split_summary_rows(overlapping)

    def test_public_candidate_dataset_summary_has_one_denominator_family(self):
        with mock.patch.object(
            publication_assets,
            "measured_class_counts",
            side_effect=AssertionError(
                "dataset summary must not mix pixel-level denominators"
            ),
        ):
            rows = dataset_summary_rows()
        self.assertEqual(rows, confirmatory_split_summary_rows(SPLIT_MANIFEST))
        self.assertTrue(all(row["metric"].startswith("declared_") for row in rows))
        self.assertFalse(any("percent" in row["metric"] for row in rows))


class PublicationAssetCodeTests(unittest.TestCase):
    def test_publication_assets_use_locked_class_colours_and_sans_serif_fonts(self):
        self.assertEqual(
            PUBLICATION_CLASS_COLORS,
            {"0": "#B33A3A", "1": "#2E8B57", "2": "#4C78A8"},
        )
        self.assertEqual(FONT_FAMILY[:3], ["Arial", "Helvetica", "DejaVu Sans"])

    def test_generated_study_workflow_does_not_claim_future_test_opening(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            figures = Path(temp_dir) / "figures"
            figures.mkdir()
            with mock.patch.object(publication_assets, "FIGURES", figures):
                configure_style()
                plot_study_workflow()
                first = {
                    suffix: (figures / f"fig_10_study_workflow.{suffix}").read_bytes()
                    for suffix in ("pdf", "png", "svg")
                }
                plot_study_workflow()
                second = {
                    suffix: (figures / f"fig_10_study_workflow.{suffix}").read_bytes()
                    for suffix in ("pdf", "png", "svg")
                }
            svg = (figures / "fig_10_study_workflow.svg").read_text(
                encoding="utf-8"
            )
        self.assertEqual(first, second)
        self.assertNotIn("held-out test partition is opened", svg)
        self.assertIn("locked retrospective evaluation partition", svg)
        for word in ("Recovered", "label", "sources", "stored", "ring", "annotations"):
            self.assertIn(word, svg)
        self.assertIn("series-disjoint by", svg)
        self.assertIn("filename; 74 / 5 /", svg)
        self.assertIn("<!-- 21 -->", svg)
        self.assertNotIn("Label construction", svg)

    def test_class_balance_uses_the_authoritative_mask_alphabet_and_wording(self):
        with (
            mock.patch.object(
                publication_assets,
                "label_mask_paths",
                return_value=[Path("mask.png")],
            ),
            mock.patch.object(
                publication_assets.Image,
                "open",
                return_value=np.asarray([[0, 1, 255]], dtype=np.uint8),
            ),
        ):
            self.assertEqual(
                publication_assets.measured_class_counts(),
                {"0": 1, "1": 1, "2": 1},
            )

        with (
            mock.patch.object(
                publication_assets,
                "label_mask_paths",
                return_value=[Path("mask.png")],
            ),
            mock.patch.object(
                publication_assets.Image,
                "open",
                return_value=np.asarray([[0, 1, 2]], dtype=np.uint8),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "authoritative mask values 0/1/255"):
                publication_assets.measured_class_counts()

        with tempfile.TemporaryDirectory() as temp_dir:
            figures = Path(temp_dir) / "figures"
            figures.mkdir()
            with (
                mock.patch.object(publication_assets, "FIGURES", figures),
                mock.patch.object(
                    publication_assets,
                    "label_mask_paths",
                    return_value=[Path(f"mask_{index}.png") for index in range(100)],
                ),
                mock.patch.object(
                    publication_assets,
                    "measured_class_counts",
                    return_value={"0": 1, "1": 20, "2": 79},
                ),
            ):
                configure_style()
                publication_assets.plot_class_balance()
            svg = (figures / "fig_01_class_balance.svg").read_text(encoding="utf-8")
        self.assertIn("Direct counts from 100 lossless label masks", svg)
        self.assertIn("source values 0/1/255 map to C0/C1/C2", svg)
        self.assertNotIn("values 2 and 255", svg)

    def test_architecture_only_mode_never_loads_results_or_microscopy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            figures = Path(temp_dir) / "figures"
            tables = Path(temp_dir) / "tables"
            args = SimpleNamespace(
                architecture_only=True,
                class_balance_only=False,
                curated_only=False,
                study_workflow_only=False,
            )
            with (
                mock.patch.object(publication_assets, "ROOT", Path(temp_dir)),
                mock.patch.object(publication_assets, "FIGURES", figures),
                mock.patch.object(publication_assets, "TABLES", tables),
                mock.patch.object(publication_assets, "parse_args", return_value=args),
                mock.patch.object(publication_assets, "configure_style"),
                mock.patch.object(publication_assets, "plot_model_architecture") as plot_architecture,
                mock.patch.object(
                    publication_assets,
                    "load_run_summaries",
                    side_effect=AssertionError("architecture-only mode opened result summaries"),
                ),
            ):
                publication_assets.main()
            plot_architecture.assert_called_once_with()

    def test_class_balance_only_mode_never_loads_experiments_or_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = root / "paper_assets"
            figures = assets / "figures"
            tables = assets / "tables"
            args = SimpleNamespace(
                architecture_only=False,
                class_balance_only=True,
                curated_only=False,
                study_workflow_only=False,
            )
            with (
                mock.patch.object(publication_assets, "ROOT", root),
                mock.patch.object(publication_assets, "ASSETS", assets),
                mock.patch.object(publication_assets, "FIGURES", figures),
                mock.patch.object(publication_assets, "TABLES", tables),
                mock.patch.object(publication_assets, "parse_args", return_value=args),
                mock.patch.object(publication_assets, "configure_style"),
                mock.patch.object(publication_assets, "plot_class_balance") as plot_balance,
                mock.patch.object(publication_assets, "write_dataset_summary") as write_summary,
                mock.patch.object(
                    publication_assets,
                    "load_run_summaries",
                    side_effect=AssertionError(
                        "class-balance-only mode opened experiment summaries"
                    ),
                ),
                mock.patch.object(
                    publication_assets,
                    "plot_output_comparison_grid",
                    side_effect=AssertionError(
                        "class-balance-only mode opened predictions"
                    ),
                ),
                mock.patch.object(
                    publication_assets,
                    "plot_inference_times",
                    side_effect=AssertionError(
                        "class-balance-only mode opened evaluation metrics"
                    ),
                ),
            ):
                publication_assets.main()
            plot_balance.assert_called_once_with()
            write_summary.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.generate_publication_assets as publication_assets
from scripts.generate_publication_assets import (
    FONT_FAMILY,
    PUBLICATION_CLASS_COLORS,
    configure_style,
    confirmatory_split_summary_rows,
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
            svg = (figures / "fig_10_study_workflow.svg").read_text(
                encoding="utf-8"
            )
        self.assertNotIn("held-out test partition is opened", svg)
        self.assertIn("locked retrospective evaluation partition", svg)


if __name__ == "__main__":
    unittest.main()

# Reproducibility Guide

This guide separates public software verification from private historical
recovery. The current checkout does not contain a verified, versioned chain
that can rebuild the canonical lossless masks from raw inputs. Those stored
masks and their confirmed provenance are external local prerequisites for a
scientific rerun.

## Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate mineral-pore-seg
```

Alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Expected Local Inputs

```text
original_images/*.png
```

The private research checkout contains clean 2048 x 2048 microscopy tiles,
recovered ring-marked sources, and the stored canonical masks in separate
locations. These inputs and their redistribution rights are not supplied by the
public snapshot.

The committed repository intentionally does not include raw microscopy images, full result trees, checkpoints, logs, PDFs, or manuscript scratch files. Those remain local or should be released through a separate archive.

## Historical preprocessing recovery only

The following commands document recovered code paths. They are
non-authoritative, their exact input/version/review provenance is incomplete,
and they cannot recreate the canonical mask corpus from a clean public
checkout. Do not substitute their outputs for the verified stored masks used by
the confirmatory loader.

```bash
python3 run_pipeline.py
```

Equivalent step-by-step commands:

```bash
python3 src/step1_detect_yellow_rings.py
python3 src/step2_classify_pixels.py
python3 src/step3_generate_coco_dataset.py
```

Generated outputs:

```text
results/step1_yellow_masks/
results/step2_pixel_classification/
results/step3_coco_dataset/
```

The historical COCO split generation uses NumPy seed `42`; that split is not
the current confirmatory partition.

For a new publication-facing model run, use the leakage checks, explicit
three-way manifest, and metadata requirements in
[`CONFIRMATORY_RERUN.md`](CONFIRMATORY_RERUN.md). The historical split is by
COCO tile ID; it is not evidence of separation by upstream specimen or mosaic.
The current frozen manifest explicitly lists 74 training, 5 validation, and 21
locked retrospective-evaluation tiles. Those counts come from
`config/confirmatory_splits.json`, while their specimen/mosaic independence
remains a data-owner confirmation gate.

## Private paper-asset regeneration

```bash
python3 scripts/generate_publication_assets.py
```

Generated outputs:

```text
paper_assets/figures/
paper_assets/tables/
paper_assets/manifest.json
```

The paper asset script reads private microscopy images and saved local result
summaries; it does not rerun model training. It is not a clean-public
reproduction command and cannot be run from the exported public snapshot
alone.

The regenerated files remain private-checkout outputs. They are not part of
the canonical public snapshot, even when the same local result tree produces
byte-identical outputs:

```text
paper_assets/figures/fig_01_class_balance.{pdf,png,svg}
paper_assets/figures/fig_02_top_experiments.{pdf,png,svg}
paper_assets/figures/fig_03_architecture_comparison.{pdf,png,svg}
paper_assets/figures/fig_04_baseline_training_curves.{pdf,png,svg}
paper_assets/figures/fig_05_output_comparison_grid.{pdf,png,svg}
paper_assets/figures/fig_06_annotation_pipeline.{pdf,png,svg}
paper_assets/figures/fig_07_inference_time_distribution.{pdf,png,svg}
paper_assets/figures/fig_08_recovery_workflow.{pdf,png,svg}
paper_assets/figures/fig_09_model_architecture.{pdf,png,svg}
paper_assets/figures/fig_10_study_workflow.{pdf,png,svg}
paper_assets/tables/dataset_summary.csv
paper_assets/tables/experiment_summary.csv
paper_assets/tables/top_experiments.md
```

The canonical allowlist excludes all of these files. Figures 1, 9, and 10,
plus `tables/dataset_summary.csv`, are local candidates for a later, separately
approved asset release. Figure 6 remains local because it embeds
microscopy-derived raster panels; it requires written source, subject, and
data-redistribution rights and an explicit approval-manifest entry before it
can enter a release. Figures 2--5, 7, and 8 and the archived
experiment-ranking tables are retained for forensic traceability and must not
be cited as confirmatory performance.

## Public validation

```bash
python3 -m compileall src scripts config
python3 scripts/run_public_smoke.py
python3 -m unittest tests.test_public_smoke tests.test_public_release
python3 -m pytest -o addopts='' -q tests/test_publication_assets.py
python3 scripts/audit_public_snapshot.py --selection-only --list
python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('pyproject ok')"
```

The selection-only audit checks reviewed source paths but does not claim a
complete snapshot. Export to a fresh path outside the mixed checkout and verify
the exact tree and bytes:

```bash
python3 scripts/export_public_snapshot.py \
  --output-dir /fresh/path/outside/checkout
python3 scripts/audit_public_snapshot.py \
  --snapshot-root /fresh/path/outside/checkout
```

With no mode flag, the audit verifies that a Git tracked tree exactly equals
the allowlist and fails closed where Git metadata is absent. The exporter and
complete-tree audit each report the same deterministic SHA-256 over sorted
relative paths and file bytes.

## Private asset validation

Only in the private checkout with the complete local microscopy/result inputs:

```bash
python3 scripts/generate_publication_assets.py
python3 -m json.tool paper_assets/manifest.json
```

For Git hygiene:

```bash
git ls-files | rg '(^logs/|^papers/|^results/|^original_images/|^labelled images/|^Overleaf/|^figures/|^tables/|^docs/.*\.pdf$|^FINAL_MODEL\.pdf$|\.pptx$|\.zip$|\.pth$|\.pt$|__pycache__|\.DS_Store|^tasks/|^FINAL_MODEL|^PAPER\.txt|^ABLATION\.md|^MONITORING\.md|^CRON\.md|^SOTA_INNOVATIONS\.md|^TODO\.md|sample\.png|traditional_algorithms_comparison_table\.txt)'
```

The command should print no tracked paths.

## Current Limitations

- BTPN uncertainty and AI-ELT comparison files were not found in this checkout by filename.
- Final model metric JSON files in `results/final_model_evaluation/` contain zero/NaN placeholders and are not used for publication claims.
- Saved run-level summaries are retained only as an audit trail; a grouped, repeated-seed confirmatory evaluation is still required.

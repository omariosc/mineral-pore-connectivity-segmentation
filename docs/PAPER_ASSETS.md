# Paper Assets

Publication figures and local diagnostic graphics are generated into
`paper_assets/`. Generation does not imply that every output is suitable for a
public release or a manuscript claim.

The canonical code-and-documentation snapshot excludes the entire directory.
No current `paper_assets/` file has a completed approval-manifest entry.

## Regeneration

```bash
python3 scripts/generate_publication_assets.py
```

The generator requires local research inputs that are deliberately excluded
from Git, including microscopy-derived sources and saved result summaries. It
is a private-checkout regeneration command, not a clean-public reproduction
path.

The single exception is the code-only workflow schematic. Regenerate only
Figure 10 without opening microscopy, masks, predictions, or metrics with:

```bash
python3 scripts/generate_publication_assets.py --study-workflow-only
```

## Local Candidate Assets

| File stem | Purpose |
| --- | --- |
| `fig_01_class_balance` | Dataset class imbalance, including rare disconnected pores |
| `fig_09_model_architecture` | Source-derived multiscale attention U-Net schematic |
| `fig_10_study_workflow` | Locked retrospective group-disjoint evaluation workflow |
| `tables/dataset_summary.csv` | Split counts from the frozen 74/5/21 manifest |

Each figure is exported as PDF, PNG, and SVG. These files are candidates for a
future, separately reviewed asset release; none is selected by the canonical
public allowlist. Redistribution permission for any microscopy-derived
content must be confirmed before publication.

`fig_06_annotation_pipeline` is retained locally but is not a public candidate
because its SVG/PDF/PNG outputs embed microscopy-derived raster panels. It may
enter a later release only after written source, subject, and
data-redistribution rights are confirmed and its exact paths receive an
explicit approval-manifest entry based on
`config/public_asset_approvals.template.yml`.

## Local Diagnostic Outputs

Figures 2--5 and 7--8, `tables/experiment_summary.csv`,
`tables/top_experiments.md`, and the generator-wide `manifest.json` are local
forensic inventories. They contain or describe historical, estimated, or
non-independent model results and are excluded by both `.gitignore` and
`config/public_release_allowlist.txt`.

## Local Sources

- `results/**/training_summary.json`
- `results/**/training_metrics.csv`
- `results/step2_*/*classification_stats.json`
- `results/step3_coco_dataset/*.json`
- `results/final_evaluation/**`
- `original_images/*.png`

These sources and all generated assets remain local. The current
`config/public_release_allowlist.txt` names no path under `paper_assets/`.

Before adding any asset to the public boundary, verify its source evidence,
redistribution rights, labels, physical scale where relevant, and absence of
private metadata. For a separate asset release, copy
`config/public_asset_approvals.template.yml` to a private controlled location,
set `template_only` to `false`, and complete an `approved` entry for every
exact selected path. The manifest uses JSON syntax, which is valid YAML 1.2.
Create a separate exact custom allowlist, add only the approved asset paths,
and re-run the source-selection audit:

```bash
python3 scripts/audit_public_snapshot.py \
  --allowlist /controlled/path/asset-release-allowlist.txt \
  --approval-manifest /controlled/path/public_asset_approvals.yml \
  --selection-only --list
```

The exporter fails closed when any selected `paper_assets/` path lacks a
complete approval entry. Export the approved selection to a fresh path outside
the research checkout, then verify the complete exported tree:

```bash
python3 scripts/export_public_snapshot.py \
  --allowlist /controlled/path/asset-release-allowlist.txt \
  --approval-manifest /controlled/path/public_asset_approvals.yml \
  --output-dir /fresh/path/outside/checkout
python3 scripts/audit_public_snapshot.py \
  --allowlist /controlled/path/asset-release-allowlist.txt \
  --approval-manifest /controlled/path/public_asset_approvals.yml \
  --snapshot-root /fresh/path/outside/checkout
```

The completed approval manifest is an authorisation input and is not copied
into the public snapshot. The audit without a mode flag verifies the Git
tracked tree rather than the mixed working tree.

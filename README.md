<div align="center">

# Mineral Pore Connectivity Segmentation

### Reproducible three-class segmentation for rock microscopy

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![Public release checks](https://github.com/omariosc/mineral-pore-connectivity-segmentation/actions/workflows/public-smoke.yml/badge.svg)](https://github.com/omariosc/mineral-pore-connectivity-segmentation/actions/workflows/public-smoke.yml)

</div>

This repository contains the preprocessing, training, and locked evaluation
code for segmenting disconnected pores, connected pores, and mineral matrix in
rock microscopy tiles. It also includes a generated synthetic smoke corpus, so
the public data contract can be checked without access to the research images.

> **Evidence status, 21 August 2026:** archived experiment summaries are
> exploratory and are not independent test results. The historical pipeline
> allowed patches from the same source image to cross partitions, and several
> old result graphics used estimated or synthetic values. Those graphics and
> summaries are excluded from the public release. Publication metrics must come
> from the filename-series-disjoint confirmatory protocol and locked evaluator described
> below.

## Five-minute public smoke test

The smoke workflow generates three 9 × 9 grayscale images and matching masks
from fixed arrays in a temporary directory. It contains no microscopy data and
produces no scientific performance claim.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install "numpy>=1.24" "Pillow>=10"
python3 scripts/run_public_smoke.py
python3 tests/test_public_smoke.py
```

The workflow checks that:

- every generated image belongs to exactly one train, validation, or test set;
- each authoritative mask is single-channel, shape-matched, and restricted to
  source values `0`, `1`, and `255`;
- mask values map to canonical target IDs `0`, `1`, and `2`;
- the mask-corpus SHA-256 is deterministic.

Use `--output-dir <new-directory>` to retain the generated corpus and
`smoke_report.json`; the command refuses to overwrite an existing directory.

## Data contract

| Target ID | Class | Authoritative PNG value | Operational definition |
| ---: | --- | ---: | --- |
| 0 | Disconnected pore | 0 | Pore pixels enclosed by the recovered ring annotation |
| 1 | Connected pore | 1 | Pore pixels outside the recovered ring annotation |
| 2 | Mineral matrix | 255 | Non-pore mineral pixels |

The confirmatory loader uses the lossless PNG masks under
`results/step2_pore_classification/pore_classifications/`. COCO is used only as
an image-ID, file-name, and dimension index. Its polygon rasterizations are not
used as confirmatory targets because polygon filling can erase internal holes.

The provisional manifest at `config/confirmatory_splits.json` keeps every
visible acquisition series in one partition. Its explicit filename lists
contain 74 training, 5 validation, and 21 locked retrospective-evaluation
tiles. Those counts come from the frozen manifest rather than from an archived
experiment summary. Specimen-level independence must still be confirmed by the
data owner before results are treated as final.

## Method overview

| Stage | Entry point | Output | Role |
| --- | --- | --- | --- |
| Recovered ring evidence | Stored marked RGBA tiles and ring masks | `results/step1_yellow_masks/` | Historical label source; exact generator and review provenance pending |
| Pixel classification | Recovered threshold/ring rule | Lossless `0/1/255` masks | Authoritative stored targets; generator version pending confirmation |
| COCO indexing | `src/step3_generate_coco_dataset.py` | COCO JSON and image copies | Preserve IDs, names, dimensions, and legacy polygons |
| Patch training | `scripts/train_patches.py` | Validation-selected checkpoint and run metadata | Train without splitting source images across partitions |
| Locked evaluation | `scripts/evaluate_confirmatory_checkpoint.py` | JSON/CSV metrics and publication plots | Evaluate the selected checkpoint once on the native locked retrospective evaluation tiles |

`run_pipeline.py` and `src/step1_detect_yellow_rings.py` through
`src/step3_generate_coco_dataset.py` are historical recovery utilities, not the
authoritative source of the stored targets. They cannot rebuild the canonical
mask corpus from a clean checkout; the verified lossless masks are a separate
local prerequisite.

The repaired trainer selects checkpoints with the harmonic mean of C0 and C1
validation IoU, reloads the selected state, and records the checkpoint,
source-code, split, normalization, architecture, input, and target provenance.
It runs in validation-only mode and never constructs the locked retrospective
evaluation loader. The locked evaluator is the sole evaluation-partition
consumer and applies no learned or tuned
post-processing.

## Installation

For the complete training and evaluation stack:

```bash
conda env create -f environment.yml
conda activate mineral-pore-seg
```

or:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The code targets Python 3.11+. A CUDA GPU is useful for training, while the
synthetic smoke test is CPU-only.

## Confirmatory training

Scientific AIRE runs must use the mode-specific wrappers. Direct submission of
`aire_confirmatory.slurm` fails closed. The workflow first runs the three-path
GPU smoke, authenticates its three immutable outcomes, runs the 15-cell
validation-only method screen, and builds a deterministic selected-method lock.
The selected formulation is then retrained for the three fixed seeds with the
primary architecture and, separately, the controlled plain U-Net comparator.
Exact commands and artifact checks are documented in
`docs/CONFIRMATORY_RERUN.md`; do not add an array override at submission time.

```bash
export PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE=1
sbatch scripts/aire_validation_smoke.slurm
```

For non-scientific local development, the entry point still requires explicit
image, mask, and split paths and should remain validation-only:

```bash
python3 scripts/train_patches.py \
  --annotations-path results/step3_coco_dataset/pore_annotations.json \
  --image-dir results/step3_coco_dataset/images \
  --mask-dir results/step2_pore_classification/pore_classifications \
  --split-manifest config/confirmatory_splits.json \
  --patch-size 683 \
  --evaluation-patch-size 2048 \
  --evaluation-batch-size 1 \
  --model-type multiscale_attention_unet \
  --loss-type focal_dice \
  --class-weights 3 2 1 \
  --num-classes 3 \
  --seed 42 \
  --validation-only
```

Scientific primary/comparator retraining is lock-bound and must use
`aire_selected_retrain.slurm`; changing only a local model flag does not create
a confirmatory comparator result. After both three-seed neural retraining
campaigns complete, build and retain the content-addressed neural freeze before
any classical locked retrospective evaluation:

```bash
python3 scripts/build_neural_freeze_manifest.py \
  --primary-campaign-id <PRIMARY_RETRAIN_JOB_ID> \
  --plain-campaign-id <PLAIN_RETRAIN_JOB_ID>
```

## Locked evaluation

Evaluate only validation-selected checkpoints through the L40S scheduler
wrapper. The evaluator fails closed on incomplete or overlapping manifests,
unsafe paths, target-corpus hash mismatches, incompatible normalization, and
unexpected checkpoint structure. Submit the primary and plain-U-Net roles as
separate arrays, waiting for the primary array to finish before submitting the
comparator array. Each array is fixed to tasks `0-2%1`; do not supply an array
override.

```bash
export PORE_NEURAL_FREEZE_ID=<neural-freeze-id>
export PORE_SELECTED_ARCHITECTURE_ROLE=primary_multiscale
sbatch scripts/aire_locked_evaluation.slurm

# Only after the primary evaluation array reaches a terminal state:
export PORE_SELECTED_ARCHITECTURE_ROLE=plain_unet_comparator
sbatch scripts/aire_locked_evaluation.slurm
```

Outputs are written below
`results/confirmatory_evaluation/locked/<neural-freeze-id>/<architecture-role>/cell_XX/`
and remain ignored until deliberately curated. Metrics include confusion-based
class scores, tile-level records, whole-tile bootstrap intervals, and
histogram-based ROC/precision-recall summaries without persisting raw pixel
probabilities.

## Public release boundary

Research images, masks, generated result trees, checkpoints, logs, manuscript
sources, presentation files, publication assets, and local automation remain
on disk but are ignored. The canonical staging boundary is the exact,
wildcard-free list in `config/public_release_allowlist.txt`. It currently
contains code, configuration, documentation, tests, and CI metadata only; it
selects no file under `paper_assets/`.

```bash
python3 scripts/audit_public_snapshot.py --selection-only --list
```

This source-selection audit rejects private or bulky paths, diagnostic paper
assets, credential patterns, email addresses, institutional usernames, and
absolute home/HPC paths. It does not claim that the mixed research checkout is
a complete public snapshot. Materialize a fresh reviewed tree outside this
checkout, then verify its exact contents and bytes:

```bash
python3 scripts/export_public_snapshot.py \
  --output-dir /fresh/path/outside/checkout
python3 scripts/audit_public_snapshot.py \
  --snapshot-root /fresh/path/outside/checkout
```

Both commands report a deterministic SHA-256 over the sorted relative paths
and exact file bytes. The exported hash must equal the independently audited
hash.

With neither mode flag, `audit_public_snapshot.py` audits the Git tracked tree
and requires it to equal the allowlist exactly; it therefore fails closed in a
directory without Git metadata. Never replace the explicit allowlist with
`git add .`.

No publication figure or table is part of the canonical public snapshot. The
following local derivatives may be reconsidered for a separate asset release:

- `fig_01_class_balance` in PDF, PNG, and SVG;
- `fig_09_model_architecture` in PDF, PNG, and SVG;
- `fig_10_study_workflow` in PDF, PNG, and SVG;
- `dataset_summary.csv`.

Adding any `paper_assets/` path requires a separate exact custom allowlist and
a complete approval record for every selected file. Copy
`config/public_asset_approvals.template.yml` to a private controlled path and
pass it to both the exporter and auditor with `--approval-manifest`. The
template and pending entries cannot authorise an export, and the completed
manifest is not copied into the public snapshot. Historical rankings, training
curves, prediction grids, timing plots, recovery graphics, parsed experiment
tables, and microscopy-derived Figure 6 remain local.

Figure 10 alone can be regenerated without opening any research image, mask,
prediction, or metric file:

```bash
python3 scripts/generate_publication_assets.py --study-workflow-only
```

## Repository layout

```text
.github/workflows/       Public smoke and focused ML unit tests
config/                  Pipeline settings, split manifest, release allowlist
docs/                    Reproducibility and evidence-audit documentation
paper_assets/            Local derived assets; excluded pending separate approval
scripts/                 Public entry points plus ignored local research scripts
src/                     Preprocessing, model, loss, and training modules
tests/                   Loader, selection, evaluator, comparator, and smoke tests
original_images/         Local research images; ignored
results/                 Local masks, checkpoints, metrics, and outputs; ignored
Overleaf/                Local manuscript source; ignored
```

## Validation

The public workflow runs:

```bash
python3 -m compileall src scripts config tests
python3 scripts/run_public_smoke.py
python3 -m unittest tests.test_public_smoke tests.test_public_release
python3 scripts/audit_public_snapshot.py
python3 -m pytest -o addopts='' -q \
  tests/test_public_smoke.py \
  tests/test_public_release.py \
  tests/test_checkpoint_security.py \
  tests/test_publication_assets.py \
  tests/test_augmentations.py \
  tests/test_patch_dataset.py \
  tests/test_patch_trainer_selection.py \
  tests/test_conditional_pore_loss.py \
  tests/test_hierarchical_pore_loss.py \
  tests/test_pyramid_context.py \
  tests/test_screen_selection.py \
  tests/test_neural_freeze.py \
  tests/test_confirmatory_evaluator.py \
  tests/test_classical_comparators.py \
  tests/test_classical_evaluator.py \
  tests/test_dataset_split_table.py \
  tests/test_model_resolution.py \
  tests/test_aire_confirmatory_slurm.py \
  tests/test_validation_screen_reporter.py \
  tests/test_publication_results_manifest_builder.py \
  tests/test_publication_results_assembler.py
```

The focused ML tests require the complete ML environment. GitHub Actions uses
CPU PyTorch for these contract tests; it does not train a model or access the
private dataset. The default audit command above is appropriate only for a Git
tracked tree; use `--selection-only` during review of a mixed local checkout.

## Data, citation, and licence

The public repository does not grant rights to the research images, derived
masks, trained weights, manuscript, or locally generated publication assets.
See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the current data status.
`CITATION.cff` remains provisional until the software author list and release
identifier are confirmed. The reviewed code-only release is provided under the
MIT licence; that licence does not extend to the excluded research data,
manuscript, trained weights, or publication artwork.

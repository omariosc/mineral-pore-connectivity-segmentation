# Scripts

Public entry points for the synthetic data-contract smoke test, confirmatory
training, locked evaluation, prospective train-only classical comparator
fitting, paper-asset generation, and public-snapshot auditing. Historical
post-processing, monitoring, recovery, and one-off experiment scripts remain
local.

## Documented entry points

The recovered preprocessing entry point is historical and non-authoritative:

```bash
python3 run_pipeline.py
```

`run_pipeline.py` and `src/step1_detect_yellow_rings.py` through
`src/step3_generate_coco_dataset.py` are retained for recovery/audit work only.
They cannot rebuild the canonical stored masks from a clean public checkout.

Run the data-contract check without research data or PyTorch:

```bash
python3 scripts/run_public_smoke.py
```

Regenerate paper figures and tables only in the private checkout containing the
required microscopy and local result inputs:

```bash
python3 scripts/generate_publication_assets.py
```

The full generator is not a clean-public reproduction command and produces
local diagnostic outputs in addition to the explicitly allowlisted candidates.
Its `--study-workflow-only` mode is the narrow public-safe exception: it
regenerates Figure 10 without opening microscopy, masks, predictions, or
metrics.

Run a non-scientific local development fit with explicit lossless targets, a
complete split manifest, and no held-out loader:

```bash
python3 scripts/train_patches.py \
  --annotations-path results/step3_coco_dataset/pore_annotations.json \
  --image-dir results/step3_coco_dataset/images \
  --mask-dir results/step2_pore_classification/pore_classifications \
  --split-manifest config/confirmatory_splits.json \
  --num-classes 3 --epochs 1 --validation-only --no-checkpoints
```

Scientific AIRE training uses the smoke, screen, and selected-retraining
wrappers. Held-out evaluation uses the separate locked L40S wrapper only;
primary and plain-U-Net roles are submitted as separate serial arrays.

```bash
export PORE_NEURAL_FREEZE_ID=<neural-freeze-id>
export PORE_SELECTED_ARCHITECTURE_ROLE=primary_multiscale
sbatch scripts/aire_locked_evaluation.slurm
```

Fit and freeze the new task-matched classical comparators using canonical
training labels only. The campaign identifier is execution provenance; the
fitter derives the content-addressed output path
`results/classical_comparators/locked/classical-fit-<identity-prefix>`:

```bash
python3 scripts/fit_classical_comparators.py \
  --campaign-id <classical-campaign-id>
```

The source-controlled AIRE path is a single CPU job. If
`PORE_CLASSICAL_CAMPAIGN_ID` is unset, it derives the campaign as
`classical-fit-<SLURM_JOB_ID>`; otherwise the one supplied value must be a safe
identifier. The job prints the separate canonical `classical-fit-...` identity.
Data and output paths cannot be overridden:

```bash
sbatch scripts/aire_fit_classical_comparators.slurm
```

This command creates new B0/B1/B2 comparator definitions. The local, ignored
`traditional_algorithms_comparison.py` is historical recovery material with a
non-canonical label mapping; it is not a public entry point and its archived
output is not a comparable or citable result.

After the train-only lock has been reviewed and frozen, and only after the
neural selection and selected-winner retraining campaigns are frozen, its sole
supported held-out path is one separate non-array CPU job on AIRE. The
content-addressed fit and neural-freeze IDs derive every artifact/output path;
paths and predictor choices cannot be overridden:

```bash
python3 scripts/build_neural_freeze_manifest.py \
  --primary-campaign-id <PRIMARY_RETRAIN_JOB_ID> \
  --plain-campaign-id <PLAIN_RETRAIN_JOB_ID>
```

The printed neural-freeze identity is a mandatory precondition for the
classical held-out job. It authenticates the selected-method lock and all six
validation-only neural retraining outcomes without opening corpus bytes.

```bash
export PORE_CLASSICAL_FIT_ID=<classical-fit-identity>
export PORE_NEURAL_FREEZE_ID=<neural-freeze-identity>
sbatch scripts/aire_locked_classical_evaluation.slurm
```

This job reserves
`results/classical_evaluation/locked/<classical-fit-id>/<neural-freeze-id>/`
before reading any held-out image or target byte, then evaluates B0, B1, and B2
together once on all 21 native tiles. It is not a validation or model-selection
entry point.

## Important Scripts

| Script | Purpose |
| --- | --- |
| `run_public_smoke.py` | Generates and validates a tiny synthetic image/mask/split corpus |
| `train_patches.py` | Main patch-based training entry point |
| `aire_confirmatory.slurm` | Fail-closed shared implementation used only by the training wrappers |
| `aire_fit_classical_comparators.slurm` | Single `nodes`-partition train-only B0/B1/B2 freeze job |
| `aire_validation_smoke.slurm` | Immutable three-path L40S preflight array |
| `aire_validation_screen.slurm` | Immutable 15-cell validation-only method screen |
| `aire_selected_retrain.slurm` | Lock-bound three-seed selected-winner retraining array |
| `aire_locked_evaluation.slurm` | Serial L40S array for the sole held-out neural evaluation path |
| `evaluate_confirmatory_checkpoint.py` | Fail-closed full-tile evaluation of the validation-selected checkpoint |
| `build_neural_freeze_manifest.py` | Authenticates and freezes the selected-method lock plus six validation-only neural retraining cells before classical held-out evaluation |
| `fit_classical_comparators.py` | Fits and freezes B0/B1/B2 comparators from training labels only, outside the neural screen |
| `aire_locked_classical_evaluation.slurm` | Non-array `nodes`-partition wrapper for the sole classical held-out pass |
| `evaluate_locked_classical_comparators.py` | Authenticates and evaluates all three frozen classical comparators exactly once |
| `generate_publication_assets.py` | Rebuilds `paper_assets/` from private local inputs; not a clean-public reproduction command |
| `audit_public_snapshot.py` | Audits a source selection, Git tracked tree, or complete exported snapshot |
| `export_public_snapshot.py` | Materializes the reviewed allowlist at a fresh path outside the mixed checkout |

## Local/HPC Scripts

Some scripts were written for local AIRE/HPC monitoring or one-off ablation
recovery. They remain on disk but are ignored by Git. They are not part of the
public reproduction path. This local-only group includes
`traditional_algorithms_comparison.py`.

Review the public source selection with:

```bash
python3 scripts/audit_public_snapshot.py --selection-only --list
```

Then export to a new outside-checkout path and verify the exact tree:

```bash
python3 scripts/export_public_snapshot.py \
  --approval-manifest /controlled/path/public_asset_approvals.yml \
  --output-dir /fresh/path/outside/checkout
python3 scripts/audit_public_snapshot.py \
  --snapshot-root /fresh/path/outside/checkout
```

The audit without a mode flag verifies the Git tracked tree and fails closed if
Git metadata is absent.

When the allowlist contains any `paper_assets/` path, export also fails closed
unless every exact path has a complete `approved` record in the supplied
JSON-compatible YAML approval manifest. Code-only custom allowlists do not
require an asset manifest.

Scheduler logs, monitoring outputs, W&B folders, and checkpoint directories should remain untracked.

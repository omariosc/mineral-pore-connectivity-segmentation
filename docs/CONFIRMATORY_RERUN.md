# Confirmatory rerun workflow

This is the executable publication protocol. Historical jobs and the first
20 August protocol draft are audit records only. The scientific rules are
frozen in `CONFIRMATORY_PROTOCOL_ADDENDUM_BALANCED_PORES_2026-08-20.md`.

## Invalidated pre-repair campaigns

The validation screen `7433604` and its smoke campaign `7433018` are retained
for scheduler/audit history only. They were launched with a 27-file source map
that omitted repository-local Python modules executed by the trainer, including
the configuration loader. Do not use any cell from either campaign to build a
selected-method lock, retrain a winner, evaluate the locked partition, or
support a manuscript result. The repaired source map deliberately makes those
artifacts fail exact source-map verification; cells must not be mixed into a
new campaign.

The earlier `classical-fit-51ca0b9bcf413de5` lock is also audit-only after this
repair because it hashes the former `src/training/screen_selection.py`. Refit
the unchanged prespecified classical protocol through its canonical wrapper
after staging the repaired source tree. Numerically identical model bytes do
not make the old lock valid under a changed source snapshot.

## Non-negotiable boundaries

- Use the canonical lossless `0/1/255` target PNGs; COCO polygons are metadata
  only.
- Use `config/confirmatory_splits.json`. The filename-derived source grouping
  explicitly contains 74 training, 5 validation, and 21 locked
  retrospective-evaluation filenames. The grouping remains provisional until
  the data owner confirms the specimen/mosaic hierarchy.
- Training and method selection are validation-only. The trainer never creates
  a held-out loader and never writes test metrics.
- Do not submit `aire_confirmatory.slurm` directly and do not add an array
  override. Its three mode-specific wrappers fix every task mapping.
- Do not use the AIRE login-node GPU. Every neural smoke, training, or held-out
  inference must run in an allocated L40S Slurm job.
- The locked evaluator is the sole held-out consumer. Each selected checkpoint
  is evaluated exactly once through `aire_locked_evaluation.slurm`.
- A failed immutable cell is recorded, not selectively rerun. A new attempt
  requires a wholly new campaign.

## 1. CPU preflight

Run in the complete project environment before requesting a GPU:

```bash
python3 -m compileall src scripts config tests
python3 -m pytest -o addopts='' -q \
  tests/test_patch_dataset.py \
  tests/test_augmentations.py \
  tests/test_patch_trainer_selection.py \
  tests/test_conditional_pore_loss.py \
  tests/test_hierarchical_pore_loss.py \
  tests/test_pyramid_context.py \
  tests/test_model_resolution.py \
  tests/test_screen_selection.py \
  tests/test_execution_source_closure.py \
  tests/test_checkpoint_security.py \
  tests/test_aire_confirmatory_slurm.py \
  tests/test_confirmatory_evaluator.py \
  tests/test_neural_freeze.py \
  tests/test_classical_comparators.py \
  tests/test_classical_evaluator.py \
  tests/test_dataset_split_table.py \
  tests/test_publication_assets.py \
  tests/test_validation_screen_reporter.py \
  tests/test_publication_results_assembler.py \
  tests/test_publication_results_manifest_builder.py
python3 -m unittest tests.test_public_smoke tests.test_public_release
python3 scripts/audit_public_snapshot.py --selection-only --list
```

Retain the output and verify the canonical annotation, split, input-image, and
target-mask hashes reported by the tests. Do not sync a staged AIRE directory
while any job from that stage is pending or running.

## 2. Three-path GPU smoke

The smoke array is fixed to seed 42 and tasks R3, C2-F, and C2-FP. Each task
runs one epoch and at most two training/validation batches. It checks the flat
path plus both full-tile conditional paths without producing a scientific
result.

```bash
export PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE=1
sbatch scripts/aire_validation_smoke.slurm
```

After all three tasks terminate successfully, authenticate their canonical
outcomes. Replace `<SMOKE_JOB_ID>` only with the Slurm array job ID:

```bash
python3 scripts/build_smoke_preflight_manifest.py \
  --smoke-result results/patch_training/protocol_runs/validation_smoke_cell/<SMOKE_JOB_ID>/cell_00/metrics/model_selection.json \
  --smoke-result results/patch_training/protocol_runs/validation_smoke_cell/<SMOKE_JOB_ID>/cell_01/metrics/model_selection.json \
  --smoke-result results/patch_training/protocol_runs/validation_smoke_cell/<SMOKE_JOB_ID>/cell_02/metrics/model_selection.json \
  --output config/smoke_preflight_manifest.json
```

The builder verifies a common campaign, exact task/candidate mapping, source
and data hashes, successful checkpoints, AMP/full-tile evidence, and immutable
canonical paths. It refuses to overwrite an existing manifest.

## 3. Immutable 15-cell validation screen

The screen maps tasks `0..14` deterministically to five candidates crossed
with seeds 42, 123, and 2025. The wrapper already specifies `0-14%1`, so at most
one L40S is used at a time.

```bash
export PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE=1
export PORE_SMOKE_PREFLIGHT_MANIFEST=config/smoke_preflight_manifest.json
sbatch scripts/aire_validation_screen.slurm
```

When every task has a terminal outcome, build the selected-method lock by
passing the 15 canonical `metrics/model_selection.json` paths in task order:

```bash
python3 scripts/build_selected_method_lock.py \
  --screen-result results/patch_training/protocol_runs/validation_screen_cell/<SCREEN_JOB_ID>/cell_00/metrics/model_selection.json \
  ... \
  --screen-result results/patch_training/protocol_runs/validation_screen_cell/<SCREEN_JOB_ID>/cell_14/metrics/model_selection.json \
  --output config/selected_method_lock.json
```

The ellipsis is documentation shorthand, not a literal shell argument. Supply
all 15 repeated `--screen-result` arguments. The builder reconstructs the
balanced C0/C1 rule, authenticates every outcome/checkpoint, applies the frozen
failure policy, and records the deterministic winner. Never choose a method or
seed by inspecting held-out performance.

Generate the optional per-tile validation-screen report only through its
source-controlled non-array L40S wrapper:

```bash
sbatch scripts/aire_validation_report.slurm
```

It atomically writes an authenticated package below
`results/validation_screen_model_development/<selected-lock-sha256>/`: the
full JSON report, metric/margin/candidate-seed/tile CSV files, a publication
TeX fragment, CAS-width PDF, 600-dpi PNG, and a SHA-256 manifest covering every
payload. This is validation-only model-development evidence and cannot change
the frozen winner. The package is the prespecified screen publication output;
it is deliberately separate from, and does not depend on, the locked held-out
publication-results manifest and assembler.

## 4. Selected primary and plain-U-Net retraining

Read the selected candidate key from the verified lock. Submit the primary
three-seed array first:

```bash
export PORE_SELECTED_METHOD_KEY=<R3|H3|C2-P|C2-F|C2-FP>
export PORE_SELECTED_METHOD_LOCK=config/selected_method_lock.json
export PORE_SELECTED_ARCHITECTURE_ROLE=primary_multiscale
export PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE=1
sbatch scripts/aire_selected_retrain.slurm
```

After that array reaches a terminal successful state, submit a separate plain
U-Net array with the same method key and lock:

```bash
export PORE_SELECTED_METHOD_KEY=<R3|H3|C2-P|C2-F|C2-FP>
export PORE_SELECTED_METHOD_LOCK=config/selected_method_lock.json
export PORE_SELECTED_ARCHITECTURE_ROLE=plain_unet_comparator
export PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE=1
sbatch scripts/aire_selected_retrain.slurm
```

Do not allow the two arrays to run concurrently. Each wrapper is internally
serial (`0-2%1`), but two independent arrays would otherwise consume two GPUs
and alter fair-share behaviour. Both roles train on the same train/validation
partition and remain held-out blind.

After both three-seed retraining arrays finish successfully, freeze their
validation-only evidence together with the complete selected-method lock:

```bash
python3 scripts/build_neural_freeze_manifest.py \
  --primary-campaign-id <PRIMARY_RETRAIN_JOB_ID> \
  --plain-campaign-id <PLAIN_RETRAIN_JOB_ID>
```

This content-addressed manifest is the mandatory precondition for the later
classical held-out evaluator. It rebuilds the 15-cell selection lock and
authenticates all six canonical outcome/checkpoint pairs without reading any
corpus bytes. Record the printed `neural-freeze-...` ID; paths cannot be
overridden.

## 5. One locked held-out pass per authenticated freeze cell

Evaluate the three primary checkpoints through the scheduled wrapper using the
content-addressed ID printed by the neural-freeze builder:

```bash
export PORE_NEURAL_FREEZE_ID=<NEURAL_FREEZE_ID>
export PORE_SELECTED_ARCHITECTURE_ROLE=primary_multiscale
sbatch scripts/aire_locked_evaluation.slurm
```

After that array terminates, submit the plain comparator evaluation against
the same freeze:

```bash
export PORE_NEURAL_FREEZE_ID=<NEURAL_FREEZE_ID>
export PORE_SELECTED_ARCHITECTURE_ROLE=plain_unet_comparator
sbatch scripts/aire_locked_evaluation.slurm
```

The evaluation wrapper rejects arbitrary checkpoint, model, data, and device
overrides; the evaluator has no output-root override. Before any held-out byte
is opened, the evaluator rebuilds the full neural freeze and verifies the
selected-method lock, all six validation-only retraining checkpoints, semantic
model-state digests, role/task/seed mapping, split, development input/target
corpora, normalization, source hashes, and L40S execution. It derives the only
approved checkpoint from the freeze and atomically reserves
`results/confirmatory_evaluation/locked/<neural-freeze-id>/<architecture-role>/cell_XX`
before reading each held-out tile once. Campaign aliases and equivalent
checkpoint re-serializations resolve to the same scientific freeze identity;
alternate output roots, arbitrary or symlinked checkpoint paths, and repeated
attempts for the same freeze/role/cell are rejected.

## 6. External classical comparators after the neural freeze

Fit the prespecified B0, B1, and B2 comparators once on the 74 authenticated
training tiles through the CPU wrapper:

```bash
sbatch scripts/aire_fit_classical_comparators.slurm
```

The fit job prints a content-addressed `classical-fit-...` ID. The Slurm job ID
is execution provenance only and is not the later evaluator identity. After the
neural-freeze manifest in Section 4 exists, submit the sole classical held-out
pass with both canonical IDs:

```bash
export PORE_CLASSICAL_FIT_ID=<classical-fit-identity>
export PORE_NEURAL_FREEZE_ID=<neural-freeze-identity>
sbatch scripts/aire_locked_classical_evaluation.slurm
```

The evaluator accepts no campaign, lock, model, data, or output-path override.
It derives the exact canonical lock/model and output paths from the two IDs and
reserves
`results/classical_evaluation/locked/<classical-fit-id>/<neural-freeze-id>/`
before held-out discovery. A caller-campaign change, copied or symlinked
artifact, JSON reformat, timestamp edit, or identical model reserialization
cannot create a second pass for that pair.

## 7. Authenticate and assemble the publication results

After all six neural cells and the single classical evaluation report
`status: complete`, generate the assembler input from the two canonical IDs:

```bash
python3 scripts/build_publication_results_manifest.py \
  --neural-freeze-id <neural-freeze-identity> \
  --classical-fit-id <classical-fit-identity>
```

Do not hand-author this JSON. The builder accepts no result paths, method
names, checkpoint hashes, or scientific identities. It re-verifies the
content-addressed freeze and fit, derives the six canonical neural directories
and one canonical classical directory, computes every artifact SHA-256, and
cross-binds the neural evaluator/source/selected-lock attestations, the raw and
semantic identity of every neural checkpoint, and the classical
raw-lock/source/B2-model attestations before running the assembler's complete
evidence validation. It then creates the repository-root
`publication_results_inputs.json`. Existing manifests, symbolic paths, partial
curve pairs, undeclared qualitative figures, missing provenance, and identity
disagreement all fail closed. The generated manifest is local, ignored
evidence and must not be added to the public repository.

Then create a new publication-results directory:

```bash
python3 scripts/assemble_publication_results.py \
  --input-manifest publication_results_inputs.json \
  --output-dir results/publication_results
```

The assembler reauthenticates every bound byte, reconstructs metrics from
per-tile confusion counts, preserves the prespecified seed-42 qualitative
panel only when the evaluator declared its exact PDF/PNG pair, and refuses to
invent missing curves, panels, physical scale, timing, or metric values.
It intentionally does not ingest the separate validation-screen publication
package. Use that package directly for the screen result; do not manually
splice it into this locked held-out-evaluation assembly.

## Required evidence bundle

Retain, without manual editing:

- Slurm stdout/stderr and `sacct` records for every campaign;
- smoke-preflight and selected-method lock JSON files;
- all success/failure outcome records;
- validation-selected checkpoint files and SHA-256 values;
- resolved training configuration, environment, package versions, input and
  target provenance, source hashes, and worker/augmentation seeds;
- per-tile and aggregate JSON/CSV metrics, whole-tile intervals, curve tables,
  gate/reference diagnostics, and 2D structural diagnostics; and
- source-generated publication figures and tables.

No archived 2025 value, hand-entered metric, threshold sweep, morphology step,
or selectively rerun seed may enter the confirmatory manuscript.

## Remaining author gates

Even a technically complete campaign remains provisional until the data owner,
annotation lead, and co-authors confirm the independent specimen/mosaic
hierarchy, pore threshold, ring semantics and annotation procedure, SEM
calibration/scale, data rights, authorship, funding, and release licence.

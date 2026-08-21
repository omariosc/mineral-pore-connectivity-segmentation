# Prospective balanced-pore protocol addendum (20 August 2026)

Status: **pre-execution draft under code verification**  
Recorded: **20 August 2026, 22:25 BST**  
Scientific GPU runs started under this addendum: **none**

This document records a prospective change to
`docs/CONFIRMATORY_PROTOCOL_2026-08-20.md`. It does not rewrite the historical
protocol or retrospectively rename old experiments. It will be marked frozen,
assigned source hashes, and given an exact run matrix only after its focused
tests pass and before any candidate is trained for scientific comparison.

The already queued job `7430972_0` predates this addendum and was still pending
when the addendum was written. That two-epoch, two-batch job is a plumbing smoke
test only. Its augmentation and selection settings are not eligible to produce
scientific results.

## Reason for the amendment

The scientific objective is strong performance for **both** annotation-defined
pore classes:

- C0: isolated/disconnected pore candidates, shown in red; and
- C1: connected pore candidates, shown in green.

Overall pixel accuracy and frequency-weighted F1 are unsuitable selection
criteria because the mineral class occupies most pixels. A model that predicts
mineral well while collapsing C0 can score highly without addressing the study
question. This clarification was made before a confirmatory or method-screen
job began.

## Validation selection rule

Every candidate checkpoint is selected using the harmonic mean of the two
validation pore-class IoUs:

```text
S_pore = 2 * IoU_C0 * IoU_C1 / (IoU_C0 + IoU_C1 + 1e-8)
```

The harmonic mean is zero if either pore-class IoU is zero and strongly
penalises one-class collapse. Exact ties are resolved by higher validation
pore-union IoU and then lower validation loss. Mineral IoU, overall accuracy,
and frequency-weighted metrics are reported diagnostically but cannot select a
checkpoint or method.

Across candidates, an alternative is eligible to replace the reference only if
all of the following hold over the same prespecified screen seeds:

1. its mean `S_pore` is higher than the reference mean;
2. its mean C0 IoU is not lower than the reference mean; and
3. its mean C1 IoU is not lower than the reference mean.

The eligible candidate with the highest mean `S_pore` is frozen. A remaining
exact tie is resolved by mean pore-union IoU and then lower mean validation
loss. No candidate is selected from a single favourable seed.

Checkpoint selection is computed from one direct `2048 x 2048` forward pass per
validation tile with evaluation batch size one. The previous nine-patch
validation path is not used for selection because it introduces artificial
683-pixel borders and does not match the locked native-tile evaluator.

## Test embargo for method development

The method screen is train/validation only. Screen runs must:

- use `--validation-only`;
- construct only the training and validation datasets;
- never validate, load, or iterate the held-out test masks;
- write `held_out_dataset_constructed: false` and
  `held_out_test_evaluation_count: 0` as native JSON booleans/integers; and
- stop after restoring and recording the validation-selected checkpoint.

The held-out manifest entries may remain in the frozen split file so its hash is
stable. Earlier integrity work read held-out filenames, mask hashes, and class
counts to diagnose the recovered target-format error; no model predictions were
generated and no method may use those held-out counts. Loss balance and any
sampling rule are derived from training masks only. This procedural access is
retained in the audit trail rather than described as a perfectly unseen label
set.

A training-only patch audit found C0 pixels in 664 of the 666 deterministic
`683 x 683` training patches and C1 pixels in all 666. The locked screen
therefore uses the complete deterministic covering grid without target-aware
oversampling. This preserves each source tile's contribution and avoids adding
a sampler hyperparameter when 99.70% of patches already supervise both pore
classes.

## Candidate family under code verification

All candidates use the same source-grouped training/validation partitions,
lossless target masks, seeded symmetry augmentation, optimizer, schedule, epoch
budget, stopping rule, model family, native-tile validation, and screen seeds.
Only the explicitly named output formulation, objective, input channels, and
training context below may differ.

### R3: flat three-class reference

- three output logits for C0, C1, and C2;
- focal plus soft-Dice objective with the recorded `[3, 2, 1]` class weights;
- native three-class argmax at validation.

### H3: differentiable hierarchical three-class objective

- three output logits for C0, C1, and C2;
- equal normalized components for the three-class region objective, pore-union
  soft Dice, and conditional C0/C1 soft Dice;
- conditional C0/C1 balance derived once from authoritative training-mask
  counts using inverse square-root pore frequencies;
- no thresholding, connected-component labelling, NumPy conversion, or detached
  prediction term in the training objective; and
- native three-class argmax at validation.

### C2-P: patch-trained pore-gated conditional classifier

- two network outputs for C0 versus C1;
- two input channels: normalized grayscale and the fixed operational pore gate;
- mineral pixels mapped to `ignore_index=-100` in the learning objective;
- conditional focal plus equal-class macro soft Dice, with the focal balance
  derived once from authoritative training-mask pore frequencies;
- no heuristic prevalence regularizer or weight softening; and
- training on the deterministic 683-pixel covering grid, followed by native
  2048-pixel validation; and
- the recovered operational label rule `raw uint8 intensity < 100` defines the
  pore union at validation, the network distinguishes C0 from C1 only inside
  that union, and all other pixels are assigned C2.

C2-P must be scored from the resulting **composed three-class mask**, using the
same 3x3 confusion-matrix calculation as R3 and H3. Conditional metrics that
ignore mineral pixels may be reported as secondary diagnostics but cannot be
compared directly with the full-image reference score.

### C2-F: full-tile pore-gated conditional classifier

C2-F uses the same two-channel input, two-output objective, threshold gate, and
native validation as C2-P, but trains on one complete `2048 x 2048` tile at a
time with batch size one and AMP. It is eligible only if a bounded two-batch GPU
smoke establishes memory safety. The context change is prespecified because a
training-only audit found that 62.98% of C0 pixels lie in annotated ring regions
crossing a 683-pixel seam, 28.95% lie in ring bounding boxes larger than 683
pixels, and 83.93% of C1 pixels belong to pore components crossing those seams.

The intensity gate is scientifically interpretable because the recovered label
generator defines pore candidates by that rule and the authoritative masks
match it nearly, but not bit-for-bit. A training-only audit found
14,054 disagreements among 310,378,496 pixels (0.00453%): 12,980 authoritative
C2 pixels at recovered overlay traces satisfy the clean-image threshold, and
1,074 authoritative C1 pixels have clean-image intensity at or above 100. No
training C0 pixel violates the ring-interior factorisation. The evaluator must
report this gate-versus-reference disagreement as a rule diagnostic rather than
describing the gate as an exact reconstruction. Its C2/pore-union performance is therefore
definition-driven rather than learned generalisation. The manuscript must state
this explicitly, and the data owner and annotation lead must confirm the
authoritative threshold and label protocol before submission. The validation
screen acknowledges the recovered code and training-mask parity while
recording that confirmation as pending; it does not claim that confirmation
has already occurred.

### C2-FP: full-tile conditional classifier with pyramid context

C2-FP is the single prespecified context ablation. It is identical to C2-F
except that a residual pyramid-context block is applied at the 256-channel
U-Net bottleneck. The block uses adaptive-average-pooling grids `(1, 2, 4, 8)`,
32 channels per branch, GroupNorm with eight groups, ReLU, bilinear resizing,
concatenation, a dense `1 x 1` fusion, and residual addition. Dropout is fixed
at `0.0` for this candidate so the comparison isolates explicit context rather
than jointly changing regularisation. It adds 131,840 parameters to the
candidate implementation.

This is an established pyramid-pooling mechanism, not a claimed architectural
invention. It is included because the training-only context audit found many
annotation-defined structures larger than the effective local context and
because native full-tile input alone does not make a convolutional receptive
field global. The block may also overfit acquisition identity in a dataset with
few source groups. It therefore remains validation-only during screening and
is eligible only under the same C0 and C1 non-regression rule as every other
alternative. It receives no additional hyperparameter or seed if it fails.

Unconditional connected-component voting is excluded. A training-only audit
found that 45.584% of threshold-pore pixels lie in components containing both
C0 and C1 because tiny bridges join large networks. Even an oracle majority
label per component caps training C0 IoU at 0.9324, so this shortcut conflicts
with the stated requirement to protect the red class.

A component-area input channel was also considered before validation access
and deferred to a separately dated exploratory protocol. Although component
area is predictive on the training masks, 45.584% of gate pixels belong to
mixed-label components and 81.51% belong to components censored by a native
tile edge. Patch-local area is especially inconsistent with native inference.
The transform adds a brittle tile-window prior to the gate already supplied to
C2-F/C2-FP, requires a different input/evaluator contract, and is not added
post hoc to this bounded screen.

A dense ring-interior auxiliary target is also excluded from this locked screen.
It is task-aligned and gives substantially denser positive supervision than C0,
but the stored ring masks, recovered generator entry point, output locations,
and class-value conventions do not currently form a reproducible versioned
chain. The ring semantics and annotator provenance are also awaiting
confirmation from the data owner or annotation lead. It may be studied only under a
separately dated prospective protocol after those provenance gates are
resolved; it cannot be added after the present validation results are
inspected.

## Prespecified screen matrix

The method screen consists of exactly five candidates (`R3`, `H3`, `C2-P`,
`C2-F`, and `C2-FP`) crossed with seeds `42`, `123`, and `2025`, for 15 cells.
Each cell has at most 30 epochs and early-stopping patience 10. The conditional
focal exponent is fixed at gamma `2.0`; its focal and macro-Dice components have
equal normalized weight. Optimizer, schedule, augmentation, and all other
shared settings are those stated above and in the exact frozen commands.

The cells are submitted as an immutable Slurm array with at most one running
task at a time. This operational serialization does not change their scientific
independence. `C2-F` and `C2-FP` are admitted to the array only after bounded
two-batch AMP preflights establish memory safety at 2048 pixels. A failed
preflight excludes that candidate without substituting a smaller tile, altered
batching rule, or new hyperparameter.

If a cell produces a non-finite loss, invalid checkpoint, or fails its declared
runtime contract, that seed is recorded as a failed cell and is not rerun with
changed settings. A candidate with any failed prespecified seed is ineligible
to replace R3. If no alternative satisfies the C0 and C1 non-regression rule,
R3 remains the frozen method. No post-result candidate, threshold, loss weight,
or extra seed may be added to rescue a result.

## Evidence still required before freezing

The following fields must be resolved and committed to this document before the
current smoke is replaced:

- exact source-code hashes and resolved candidate commands;
- focused CPU-test result and AIRE environment versions; and
- successful bounded GPU smoke evidence for each distinct memory path.

No soft-clDice, Lovasz, component pooling, threshold sweep, or alternative
class weight may be added after validation results have been seen. Such a method
would require a separately dated prospective amendment and cannot silently join
this comparison.

## Confirmatory stage after the method freeze

The frozen winner is retrained from scratch with seeds 42, 123, and 2025 for the
primary multiscale attention U-Net. A controlled plain U-Net uses the same
formulation, objective, data, augmentation, optimizer, stopping rule, and seeds;
only the architecture changes. The locked evaluator is then run exactly once
per validation-selected checkpoint and reports every prespecified seed.

For a pore-gated winner, the evaluator must verify and record the two-output
checkpoint, uint8 threshold, normalization, label-mask checksum, split checksum,
and composed three-class inference rule before scoring native 2048x2048 tiles.
It must report C0 and C1 IoU, Dice, precision, and recall separately, their
harmonic IoU mean, pore-union agreement, C2 metrics, per-tile results, and
whole-tile bootstrap intervals. Overall accuracy remains descriptive only.

The locked evaluator also reports four secondary 2D structural diagnostics on
the native hard masks: C0 and C1 area-fraction error; the fraction of C1 pixels
in the largest 8-connected C1 component; 8-connected C0 component count per
megapixel; and pore-union boundary F1 at a fixed two-pixel tolerance. These
diagnostics never enter checkpoint or method selection, use no component-size
filter or tuned tolerance, and must be interpreted only as agreement with the
operational 2D labels, not as permeability or three-dimensional connectivity.

## AIRE execution boundary

Training and inference run through Slurm on an allocated L40S GPU. AIRE's login
GPU is for environment configuration and light setup, not for bypassing the
scheduler, so no training is run on the login node. A short current-code smoke
must complete before a validation screen or full repeated-seed job is submitted.

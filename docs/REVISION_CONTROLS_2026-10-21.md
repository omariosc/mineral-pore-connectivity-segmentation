# Post review computational controls

These analyses address gate composition, input information, architectural components,
loss weighting and reporting. They are a separately dated revision study. They do
not reopen the historical screen or replace its selected method. The 21 reserved
tiles remain a retrospective evaluation set with unresolved previous exposure and
physical independence; this protocol does not access their image or mask bytes.

## Existing checkpoints

Authenticate the original 15-cell screen and its selected checkpoints against the
existing selected-method lock. Preserve the source tree used for those runs. Run
new analysis scripts from a separate directory and record their checksums.

For Flat and Hierarchical, report native three-class argmax and a second prediction
that chooses only between C0/C1 logits below raw intensity 100 and assigns C2
everywhere else. The background logit does not participate inside the gate. Compare
the resulting complete-image confusion counts with conditional models using the
same tiles, seed pairing, hardware, precision and composition procedure.

Report pooled counts, individual tiles, every seed, C0 precision/recall, and
pore-union IoU to at least six decimals. Within-gate confusion has three reference
rows (C0/C1/C2) and two prediction columns, preserving reference-matrix pixels
that fall inside the gate. Its top two rows form the conditional pore-class
confusion; do not conceal gate/reference disagreements by dropping the third row
from the full-image metrics. Pore-class IoU changes are coupled.

Recomputed predictions can differ slightly from historical scores because inference
backends and numerical precision affect hard boundaries. Keep recorded scores,
new confusion counts, numerical backend flags and signed discrepancies separately.
No tolerance in the original reporter or selector is changed. An H200 migration is
an explicitly labelled hardware reanalysis, not an exact reproduction of L40S
predictions. The original-family diagnostic remains separate.

Threshold sensitivity uses 90, 95, 100, 105 and 110, with both the C2-P input channel
and final composition recomputed. This is an inference-only robustness analysis
without retraining, threshold choice or replacement of the fixed rule. It does
not establish why the original threshold was chosen or its historical timing.

C0 pore-body coverage recall uses 8-connected reference components and a fixed
50% reference-pixel coverage criterion. Include all sizes and report tile-edge
components separately. A merged prediction may cover more than one reference
component; this metric is not one-to-one instance matching or a demonstration of
3D connectivity. Preserve per-component areas and coverage for size-distribution
and missed-body analyses. Use all five validation tiles at seed 42 for panels,
rather than selecting favourable examples. Physical scale requires author evidence.

The frozen B0/B1/B2 classical fit can be evaluated on validation without refitting
or test access. B2 is a gate plus a conventional learned classifier. These are new
task-matched comparators, not faithful recreations of the unreleased companion-paper
methods. Source, model bytes, numeric model semantics and runtime versions must match
the original classical fit.

## Registered new training

`config/revision_protocol.json` contains the complete 8-arm by 3-seed matrix.
The four architecture arms share the same U-Net widths, normalization, convolution
bias convention, bilinear interpolation and He initialization. They cross ordinary
versus multiscale encoder blocks with absent versus present skip attention.
Boundary-refinement modules are absent in all four. A separate original-full arm
adds those boundary modules and exactly reconstructs the original architecture.
This avoids attributing their effect to attention or multiscale processing.
The plain arm is a standard U-Net topology using this shared implementation;
it is not claimed as a bit-for-bit reproduction of the original 2015 software.

The grayscale-only arm has the original-full architecture and two output logits,
but receives one input channel. It uses the same final gate composition, isolating
the effect of supplying the gate as an additional input. The two loss arms change
only normalized (focal,Dice) coefficients to (0.25,0.75) or (0.75,0.25); the matched
original-full arm uses (0.5,0.5). Equal coefficients do not imply equal loss scales
or gradient contributions.

Every arm uses the same 74 training and five validation tiles, canonical stored
masks, three fixed training seeds, 683-pixel patches, batch size four, historical
light symmetry augmentation and worker seeding, AdamW at 0.0005, weight decay
0.0001, 30-epoch cosine schedule, patience ten and validation-only selection by
harmonic C0/C1 IoU, then union IoU, then lower validation loss. Record actual
successful optimizer updates separately from AMP-skipped steps. The new campaign
uses a fixed initial AMP scale of 1024 for every arm, rather than the historical
PyTorch default of 65536. This is a recorded numerical-runtime departure, not a
change to the loss coefficients. Do not describe
an early-stop event or selected epoch as proof of convergence.

Qualification must first establish finite loss/gradients and native 2048-pixel
validation for every distinct arm with two finite-gradient training batches, two
successful optimizer updates and two validation batches. Inherited environment
settings cannot disable the registered augmentation.
All numerical work uses scheduled compute. AIRE L40S is the original-family
surface. Calder H200 may be used only as a separately recorded hardware migration
with an exact-entry smoke and matched hardware across all new arms. It must not
silently substitute into the old L40S campaign.

The finite envelope is 24 cells, at most 45 minutes per cell and one GPU at a time,
plus one bounded smoke and descriptive checkpoint analyses. Report every
registered cell, failure and null result. No additional seed, weight, threshold,
architecture or selective retry is added in response to validation outcomes.
If implementation repair is needed, preserve failed artifacts and use a new,
separately identified campaign rather than silently overwriting results.

## Interpretation and later evaluation

The multiscale/attention claim is unsupported if their controlled arms show no
consistent advantage. In that event, describe the system as rule-assisted pore
classification and revise the title accordingly. Gate-composed Flat approaching
the conditional result would support composition as an important explanation.
A grayscale-only arm matching the two-input arm would show that explicit gate
input provides little added benefit under this training scheme.

Before any reserved-set inference, freeze C2-P, Flat and the standard-U-Net
comparator checkpoints and write the exact evaluation protocol. Wurood's physical
mapping and previous-use history determine whether the set can support an
independent-test claim. More optimisation seeds do not replace independent mosaics
or specimens. No specimen-level uncertainty or significance is inferred from
millions of correlated pixels or five tiles from one visible filename series.

The submitted manuscript remains unchanged while laboratory and annotation details
are pending. Store proposed prose and numerical tables privately, with missing
author evidence explicitly marked for later completion.

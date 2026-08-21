# Superseded first confirmatory protocol (20 August 2026)

> **Audit record only. Do not execute the commands or selection rule below.**
> This first draft was replaced, before any new validation or held-out model
> result was generated, by
> `CONFIRMATORY_PROTOCOL_ADDENDUM_BALANCED_PORES_2026-08-20.md`. The addendum
> fixes the screen at 30 epochs, selects checkpoints with the harmonic mean of
> C0 and C1 IoU, requires immutable smoke/screen campaigns, keeps training
> validation-only, and makes the locked evaluator the sole held-out consumer.
> Only the mode-specific Slurm wrappers documented in
> `CONFIRMATORY_RERUN.md` are executable protocol entry points.

This file is retained to show how the protocol changed during the pre-result
integrity audit. It was never the basis of a valid confirmatory result and is
not an exact reproduction of the historical jobs.

## Why the historical final command is not rerun unchanged

The historical patch loader mixed patches from the same source tiles across
training and validation. The recovered topological loss also converts connected
components to NumPy arrays, so that term does not supply the claimed gradient,
and its implementation hard-codes class weights independently of the command
line. Repeating that command would reproduce neither a leakage-controlled study
nor the method described in the manuscript. The confirmatory run therefore uses
the established focal-plus-Dice implementation and makes this change explicit.

## Frozen data split

The split manifest is `config/confirmatory_splits.json` (SHA-256
`4f274f0eced3dfb096ff2e49fe2b9cac3901664fef2bf6da6943ccabad64c703`).
The COCO annotations used for the run have SHA-256
`fd79f8820b44ed1e8eed880be699f7d3428fcb18907ab8627d52ba4f0b1c471a`.

Training targets are loaded from the 100 lossless class-mask PNGs under
`results/step2_pore_classification/pore_classifications`. Their stored values
`0`, `1` and `255` are mapped to C0, C1 and C2. The COCO file supplies image IDs
and filenames but its polygon rasterisation is not used as the target: external
contours fill internal mineral holes and would change the connected-pore share
from 12.497% in the lossless masks to 25.045%. The mask corpus therefore remains
the authoritative target representation. The deterministic aggregate mask
SHA-256 is `36e6e8fde61fe8dc6dd75c74d8a098a4c215dbd84ddb1d132f9c2d4dce6dd830`
(lexicographically sorted relative filename, NUL, raw file bytes, NUL).

The 100 model-input PNGs under `results/step3_coco_dataset/images` are
byte-identical, filename for filename, to the clean greyscale files under
`original_images`; neither set contains the yellow annotation overlays. Their
aggregate SHA-256 under the same filename/NUL/raw-bytes/NUL algorithm is
`05ab6ab685a37015f9dd43294b10dc1c044fe3b104e08d3076f1ab333c265992`.

| Partition | Leading source identifiers | Tiles |
| --- | --- | ---: |
| Training | `pdo1`, `pdo4` | 74 |
| Validation | `pdo8` | 5 |
| Held-out test | `pdo2` | 21 |

No tile or visible acquisition-series prefix crosses a partition. The data
owner must still confirm that the four leading identifiers correspond to
independent source mosaics or specimens. Until then, results from this split
are described as provisional filename-series-disjoint evidence, without claiming that the
filename series are independent specimens, mosaics, or acquisitions.

## Frozen primary model configuration

- model: `multiscale_attention_unet`, one greyscale input channel and three
  output classes;
- base width: 32; bilinear upsampling; deep supervision disabled by the checked
  configuration;
- input: nine 683 x 683 patches per 2048 x 2048 tile, scaled from 8-bit values
  to `[-1,1]`;
- target: registered lossless PNG mask with `{0,1,255}` mapped to `{0,1,2}`;
- augmentation: Albumentations v2 `RandomRotate90`, `HorizontalFlip` and
  `VerticalFlip`, each with probability 0.5 and with the composition seeded by
  the run seed; no elastic/grid deformation, MixUp or CutMix;
- loss: focal plus soft Dice, equal component weights, focal gamma 2;
- class weights: `[3,2,1]` for C0, C1 and C2, fixed from the checked pipeline
  configuration rather than tuned on the held-out series;
- optimiser: AdamW, learning rate `5e-4`, weight decay `1e-4`;
- schedule: cosine annealing over at most 50 epochs;
- batch size: 4; bootstrap factor: 1; mixed precision enabled; gradient norm
  clipped to 1.0;
- seeds: 42, 123 and 2025;
- early stopping: patience 10 on the documented validation composite
  `0.7 * mean(C0 IoU, C1 IoU) + 0.3 * weighted F1`;
- checkpoint: the highest validation-composite state is reloaded before any
  held-out evaluation. Lowest validation loss and terminal training state are
  retained separately.

## Locked evaluation

The held-out partition is evaluated only after model selection. The primary
report contains per-class IoU and Dice, macro IoU, precision, recall and F1,
the aggregate confusion matrix, per-tile results, and tile-bootstrap 95%
confidence intervals. ROC and precision-recall curves must be generated only
from probabilities emitted by this selected checkpoint. No intensity threshold
or morphological post-processing is part of the primary confirmatory result.

The three seeds are a prespecified replicate set, not successive attempts chosen
after inspecting test performance. The controlled comparator is
`plain_unet`: the repository's original encoder-decoder U-Net with ordinary
skip concatenation, base width 32 and bilinear upsampling, without attention
gates, multiscale dilated blocks, boundary-refinement blocks or deep-supervision
heads. It uses the same split, loss, class weights, augmentation, optimiser,
schedule, stopping rule, checkpoint selection and seeds as the primary model;
the instantiated architecture is the only planned change. The former
configuration-driven `unet` behaviour is retained only as the explicit,
deprecated `legacy_configured_unet` option and is not used for this comparison.

## Retired execution recipe

The former direct-main-script workflow, architecture environment override,
50-epoch budget, and mineral-weighted selector are intentionally omitted. They
are rejected by the current launcher. Use `CONFIRMATORY_RERUN.md` and the
balanced-pore addendum instead.

# Training Utilities

This package contains patch-based training utilities for mineral pore segmentation.

## Main Entry Point

```bash
python3 scripts/train_patches.py
```

For confirmatory work, always pass `--mask-dir` and `--split-manifest`
explicitly. Omitting `--mask-dir` activates the legacy COCO-polygon fallback,
which is retained for compatibility and is not the publication target source.

The training stack supports:

- patch extraction from 2048 x 2048 microscopy tiles
- U-Net and related segmentation architectures
- focal, Dice, Tversky, Lovasz, boundary-aware, and combined losses
- complete, disjoint whole-image manifests
- authoritative lossless PNG targets through `data_contract.py`
- validation-composite checkpoint selection without constructing held-out data
- held-out evaluation only through `scripts/aire_locked_evaluation.slurm`
- experiment summaries in `training_summary.json`
- epoch metrics in `metrics/training_metrics.csv`
- checkpointing under generated `results/**/checkpoints/`

## Local Data

The preprocessing pipeline writes labels under:

```text
results/step2_pore_classification/pore_classifications/
results/step3_coco_dataset/
```

The mask loader accepts only source values `0`, `1`, and `255`; three-class
training maps them to disconnected pore, connected pore, and mineral IDs
`0`, `1`, and `2`. COCO supplies image metadata and legacy annotations, but its
polygons are not authoritative confirmatory targets.

Large checkpoints and run folders should remain local and ignored unless they are intentionally released through Git LFS or an external archive.

## Reproducibility

Use repository-relative paths and save the resolved architecture, source-code
hashes, split membership, normalization, and target-corpus provenance with
every run. The public synthetic smoke test checks the same dependency-light
data contract without research data:

```bash
python3 scripts/run_public_smoke.py
```

Generate manuscript assets separately with
`python3 scripts/generate_publication_assets.py`; that command requires the
ignored local research inputs and also creates diagnostic files that are not
part of the public release.

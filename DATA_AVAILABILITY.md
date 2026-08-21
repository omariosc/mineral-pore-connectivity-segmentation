# Data Availability

This repository is structured so code and documentation can be public while
research data and generated publication assets remain local or are released
through a separately approved archive.

## Local Data Expected By The Code

```text
original_images/*.png
results/step2_pore_classification/pore_classifications/*.png
results/step3_coco_dataset/images/*.png
results/step3_coco_dataset/pore_annotations.json
```

The first path contains the clean single-channel microscopy tiles and is
byte-identical to the confirmatory image copies under the COCO index. Recovered
ring-marked RGBA sources are stored separately under
`results/step1_yellow_masks/yellow_rings/` and are never model inputs. The
confirmatory target masks are single-channel PNGs with source values `0`, `1`,
and `255`, mapped to disconnected pore, connected pore, and mineral. The COCO
JSON supplies image IDs, file names, and dimensions; its polygons are not the
authoritative confirmatory targets. The exact historical ring-generation
version and annotation provenance remain data-owner confirmation gates.

The frozen filename lists in `config/confirmatory_splits.json` contain 74
training, 5 validation, and 21 locked retrospective-evaluation tiles. These
counts describe the manifest; independence at specimen or mosaic level remains
subject to data-owner confirmation.

## Synthetic Public Example

No private data is needed to verify the data contract:

```bash
python3 -m pip install "numpy>=1.24" "Pillow>=10"
python3 scripts/run_public_smoke.py
```

The script constructs three deterministic 9 × 9 images and masks in a
temporary directory, validates a complete disjoint split manifest, checks the
lossless mask values, maps them to target IDs `0/1/2`, and reports the mask
corpus SHA-256. This corpus is a software smoke fixture only. It is not a
geological dataset and cannot support model-performance claims.

## Recommended Public Release Pattern

- Publish only the code, documentation, configuration, tests, and CI metadata
  named in `config/public_release_allowlist.txt`.
- Release raw images, generated labels, and checkpoints only if licensing and
  participant/project permissions allow it.
- If the data is released separately, add the DOI or download URL here.
- If the research data cannot be public, retain the generated synthetic smoke
  fixture and document the controlled access route for scientific replication.

## Local Asset Candidates

The checkout includes derived figures and summary tables under `paper_assets/`.
The canonical public allowlist currently selects none of them. Figures 1, 9,
and 10 in PDF/PNG/SVG form, plus
`paper_assets/tables/dataset_summary.csv`, are local candidates for a later,
separately approved asset release. Historical experiment plots and parsed
rankings remain local and are not confirmatory results. None of these
derivatives substitutes for raw tile provenance, authoritative masks, or
complete run metadata.

Figure 6 is not a public candidate. It embeds microscopy-derived raster panels
and remains local until written source, subject, and data-redistribution rights
are confirmed and its exact files receive an explicit approval-manifest entry
based on `config/public_asset_approvals.template.yml`.

## Current Status

The development checkout contains local image and result files, but final public data-release terms are not yet specified. The Git repository ignores those local folders by default:

```text
original_images/
labelled images/
results/
papers/
Overleaf/
*.pptx
*.zip
*.pth
*.pt
*.ckpt
*.onnx
*.npy
*.npz
*.pdf
```

Before making final data-availability claims, confirm dataset ownership,
licensing, specimen/mosaic provenance, and whether microscopy-derived figure
panels may be redistributed. Then add either a DOI and licence or a precise,
durable controlled-access procedure.

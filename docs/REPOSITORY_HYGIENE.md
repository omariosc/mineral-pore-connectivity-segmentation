# Repository Hygiene

The repository should keep bulky or private research artifacts available locally while excluding them from public commits.

## Ignore Policy

Ignored by default:

- `logs/`
- `original_images/`
- `labelled images/`
- `results/`
- `papers/`
- `Overleaf/` and presentation/ZIP source files
- root-level historical `figures/`, `tables/`, and `manifest.json` copies
- private/editor PDFs in `docs/`
- checkpoints and model binaries (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`)
- Python caches and local environments
- generated datasets and experiment outputs
- experiment tracking folders such as `wandb/`
- older private root notes such as `FINAL_MODEL.md`, `PAPER.txt`, `ABLATION.md`, and `MONITORING.md`
- local planning files under `tasks/`
- local scheduler, email, monitoring, and exploratory scripts not named in the
  public script allowlist

Candidates for commit after the release audit:

- source code
- configuration
- documentation
- tests and CI metadata
- the synthetic smoke workflow and read-only CI checks

The canonical snapshot currently contains no `paper_assets/` file. Local
figures and tables require a separate exact allowlist and completed approval
manifest before they can enter any later asset release.

## Public Snapshot Allowlist

The canonical list is `config/public_release_allowlist.txt`. It deliberately
omits manuscript sources, raw and derived datasets, legacy automation, logs,
checkpoints, historical result graphics, parsed experiment rankings, and
one-off analysis scripts even when they remain present locally.

Inspect and scan the resolved list before staging:

```bash
python3 scripts/audit_public_snapshot.py --selection-only --list
```

This mode audits the selected source files but does not claim that the mixed
research checkout itself is a complete public tree. It fails on missing exact
entries, symlinks, forbidden private/bulky
paths, diagnostic paper assets, known credential formats, literal secret
assignments, email addresses, institutional usernames, and absolute home/HPC
paths.

Materialize the reviewed selection at a new path outside the checkout and audit
that complete snapshot exactly:

```bash
python3 scripts/export_public_snapshot.py \
  --output-dir /fresh/path/outside/checkout
python3 scripts/audit_public_snapshot.py \
  --snapshot-root /fresh/path/outside/checkout
```

With no mode option, `audit_public_snapshot.py` is a Git tracked-tree audit: it
requires the tracked tree to equal the allowlist and fails closed when Git
metadata is absent. Never use a blanket `git add .` for this research checkout.

## Clean one-commit publication sequence

Do not publish from the mixed research checkout. After the source-selection
audit passes, export to a new empty path and initialise Git only inside that
verified export. Because the exporter creates exactly the allowlisted tree,
`git add --all` is bounded there; it must not be run in the research checkout.

```bash
python3 scripts/audit_public_snapshot.py --selection-only --list
python3 scripts/export_public_snapshot.py \
  --output-dir /fresh/path/mineral-pore-segmentation-public
python3 scripts/audit_public_snapshot.py \
  --snapshot-root /fresh/path/mineral-pore-segmentation-public
cd /fresh/path/mineral-pore-segmentation-public
git init
git add --all
python3 scripts/audit_public_snapshot.py
git diff --cached --name-status
git commit -m "Initial public software release"
git status --short
```

The tracked-tree audit must pass after staging, the cached file list must match
the reviewed allowlist, and the final status must be empty. Configure a remote,
change repository visibility, or push only as separate authorised actions after
this local one-commit snapshot has been reviewed.

The canonical boundary excludes every file under `paper_assets/`. Figures 1,
9, and 10 in PDF/PNG/SVG form plus
`paper_assets/tables/dataset_summary.csv` remain local candidates for a later
asset-specific review. Figure 6, `paper_assets/manifest.json`, Figures 2--8,
the experiment summary, and the top-experiment table also remain local because
they contain or inventory restricted, historical, or non-evidential material.

If a future custom allowlist selects a `paper_assets/` path, the exporter and
auditor require every selected asset to have a complete `approved` entry in a
JSON-compatible YAML manifest. The default private path is
`config/public_asset_approvals.yml`, which is ignored; alternatively pass a
controlled external path with `--approval-manifest`. The public template is
not an approval and cannot authorise an export.

## If Files Are Already Tracked

Use `git rm --cached` to stop tracking files without deleting local copies.

Examples:

```bash
git rm --cached -r logs papers
git rm --cached -r original_images results
git rm --cached -r "labelled images" Overleaf figures tables
git rm --cached -r docs/*.pdf
git rm --cached "Fig1 & 2.pptx" manifest.json
git rm --cached FINAL_MODEL.pdf FINAL_MODEL.md FINAL_MODEL_COMPREHENSIVE_UPDATE.md
git rm --cached PAPER.txt MONITORING.md CRON.md CRON.log GITHUB_ACTIONS.log
git rm --cached ABLATION.md SOTA_INNOVATIONS.md TODO.md sample.png
git rm --cached -r tasks
git rm --cached -r '**/__pycache__'
```

After untracking, resolve the canonical allowlist, inspect every selected path,
and stage those paths explicitly. Staging, committing, pushing, and repository
visibility changes are separate actions and each should use the reviewed scope.

## Before Public Push

Run:

```bash
git status --short
git ls-files | rg '(^logs/|^papers/|^results/|^original_images/|^labelled images/|^Overleaf/|^figures/|^tables/|^docs/.*\\.pdf$|^FINAL_MODEL\\.pdf$|\\.pptx$|\\.zip$|\\.pth$|\\.pt$|__pycache__|\\.DS_Store)'
python3 -m compileall src scripts config
python3 scripts/run_public_smoke.py
python3 -m unittest tests.test_public_smoke tests.test_public_release
python3 -m pytest -o addopts='' -q tests/test_publication_assets.py
python3 scripts/audit_public_snapshot.py
```

If `git ls-files` still reports private or bulky files, untrack them with
`git rm --cached`. The unqualified audit above checks the Git tracked tree. For
a pre-staging source review use `--selection-only --list`, then export and audit
a fresh outside-checkout snapshot as shown above. Regenerate publication assets
only in the private checkout containing the required microscopy and local
result inputs; the full generator is not part of clean-public reproduction and
also creates deliberately excluded diagnostic files.

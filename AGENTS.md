# AGENTS.md

Repository guidance for coding agents.

## Objective

Make this repository reproducible, public-safe, and useful for paper preparation. Preserve local research artifacts while keeping the Git history focused on source code, documentation, configuration, and curated generated assets.

## Local Data And Git Hygiene

- Do not delete local files to make the repository smaller.
- Use `.gitignore` to keep bulky or private files untracked while leaving them available on disk.
- If a bulky/private file is already tracked in a real Git checkout, untrack it with `git rm --cached <path>` rather than removing it from the filesystem.
- Keep checkpoints, logs, scheduler outputs, raw experiment directories, private PDFs, and manuscript scratch material out of public commits.
- `paper_assets/` is the intended location for curated figures/tables, but only
  Figures 1, 9, and 10 plus `tables/dataset_summary.csv` are current public
  candidates. Figure 6 remains local until written source, subject, and data
  rights are confirmed and its exact files receive an explicit
  approval-manifest entry.

## Reproducibility

- Use `python3` in commands.
- Keep paths repository-relative.
- Default image input directory is `original_images/`.
- Default generated outputs are under `results/`.
- Treat `run_pipeline.py` and preprocessing steps 1--3 as historical recovery
  utilities only. They are non-authoritative and cannot reconstruct the
  canonical stored masks from a clean public checkout.
- Run `python3 scripts/generate_publication_assets.py` only in the private
  research checkout containing the required microscopy and result inputs. It
  is not a clean-public reproduction command.
- Do not claim metrics that are not backed by checked local source files.

## Validation Commands

```bash
python3 -m compileall src scripts config tests
python3 scripts/run_public_smoke.py
python3 -m unittest tests.test_public_smoke tests.test_public_release
python3 -m pytest -o addopts='' -q tests/test_publication_assets.py
python3 scripts/audit_public_snapshot.py --selection-only --list
```

The selection-only audit reviews allowlisted source files. Export a complete
candidate to a new path outside the mixed checkout with
`python3 scripts/export_public_snapshot.py --output-dir /fresh/path/outside/checkout`
and verify it with
`python3 scripts/audit_public_snapshot.py --snapshot-root /fresh/path/outside/checkout`.
The unqualified audit command checks the Git tracked tree and fails closed when
Git metadata is absent.

## Editing Rules

- Keep changes scoped and practical.
- Prefer documentation and ignore rules over destructive cleanup.
- Do not introduce new external services or network dependencies unless the user asks.
- Keep generated figures and tables reproducible from local files.

# Contributing

Contributions that improve reproducibility, scientific clarity, testing, and
public safety are welcome. Keep changes small enough to review and separate
scientific method changes from mechanical cleanup where practical.

## Before you contribute

- Use an issue to discuss material changes to the data contract, model,
  evaluation protocol, dependencies, or public release boundary.
- Never commit research images, authoritative masks, checkpoints, credentials,
  logs, unpublished metrics, manuscript scratch files, or presentation sources.
- Confirm that you have the right to share every submitted file. Do not submit
  third-party or collaboration data without documented redistribution rights.
- Report security concerns privately as described in `SECURITY.md`.

## Development and validation

Use Python 3.11 or later and repository-relative paths. A lightweight public
check requires only NumPy and Pillow; the focused model tests require the full
development environment.

```bash
python3 -m compileall src scripts config tests
python3 scripts/run_public_smoke.py
python3 -m unittest tests.test_public_smoke tests.test_public_release
python3 -m pytest -o addopts='' -q tests/test_publication_assets.py
python3 scripts/audit_public_snapshot.py --selection-only --list
```

Add or update tests for behavioural changes. Document any scientific assumption
that affects labels, splitting, training, checkpoint selection, or evaluation.
Report disconnected-pore (C0) and connected-pore (C1) behaviour separately;
aggregate accuracy alone is not sufficient. Do not add performance claims
unless checked source artifacts support them.

`run_pipeline.py` and preprocessing steps 1 through 3 are historical recovery
utilities. A clean public checkout cannot reconstruct the canonical stored
masks from them, so contributions must not present those steps as an
authoritative dataset rebuild.

## Public release boundary

The mixed research checkout is not itself a release artifact. Add a new public
file explicitly to `config/public_release_allowlist.txt`, keep private and bulky
paths covered by `.gitignore`, and export to a new directory outside the
checkout:

```bash
python3 scripts/export_public_snapshot.py --output-dir /fresh/path/outside/checkout
python3 scripts/audit_public_snapshot.py --snapshot-root /fresh/path/outside/checkout
```

Do not use blanket staging commands as a substitute for the allowlist. New
publication assets also require clear provenance, reproducible generation, and
confirmed redistribution rights.

## Pull requests

Describe what changed, why it is needed, the commands you ran, and any effect on
scientific interpretation or the public-data boundary. Note tests that could
not be run. Contributions are provided under the repository licence; academic
paper authorship is assessed separately using the applicable journal and
research-contribution criteria.

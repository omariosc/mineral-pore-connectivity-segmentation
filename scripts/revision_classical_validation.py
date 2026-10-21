#!/usr/bin/env python3
"""Score frozen train-only gate/classifier comparators on validation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from revision_metrics import (
    confusion,
    summarize_confusion,
    gate_confusion,
    component_recovery,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neural-root", type=Path, required=True)
    p.add_argument("--classical-root", type=Path, required=True)
    p.add_argument("--fit-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a scheduled CPU allocation")
    if a.output.exists():
        raise FileExistsError(a.output)
    sys.path.insert(0, str(a.neural_root.resolve()))
    from scripts import report_validation_screen_tiles as neural
    from scripts import evaluate_locked_classical_comparators as classical
    from scripts.fit_classical_comparators import _library_versions
    from src.classical import comparators

    lock, lock_hash, _, _ = neural.load_verified_screen_lock(
        Path("config/selected_method_lock.json"), a.neural_root
    )
    data = neural.independently_reattest_development_bytes(lock, a.neural_root)
    fit = classical.load_verified_classical_lock(
        a.fit_id, project_root=a.classical_root
    )
    for name, module in [
        ("scripts/evaluate_locked_classical_comparators.py", classical),
        ("src/classical/comparators.py", comparators),
    ]:
        if digest(module.__file__) != fit["source_code_sha256"][name]:
            raise ValueError("Actual imported classical source differs from fit")
    if _library_versions() != fit["environment_versions"]:
        raise ValueError("Classical runtime differs from frozen fit")
    estimator = classical.load_verified_b2_estimator(
        fit["b2_model_path"], fit["b2_model_sha256"], fit["b2_model_semantic_sha256"]
    )
    fit["b2_estimator"] = estimator
    tiles = neural.load_validation_tiles(
        neural.validation_tile_specs(lock), a.neural_root
    )
    records = []
    pooled = {}
    for tile in tiles:
        for name, prediction in classical.predict_all_locked_comparators(
            tile["image"], fit
        ).items():
            cm = confusion(tile["target"], prediction)
            pooled[name] = pooled.get(name, np.zeros((3, 3), dtype=np.int64)) + cm
            obj = component_recovery(tile["target"], prediction)
            records.append(
                {
                    "method": name,
                    "file_name": tile["file_name"],
                    "confusion": cm.tolist(),
                    "within_gate_confusion": gate_confusion(
                        tile["image"], tile["target"], prediction
                    ).tolist(),
                    **summarize_confusion(cm),
                    "object_recovery": obj,
                }
            )
            print(
                json.dumps(
                    {
                        "method": name,
                        "file_name": tile["file_name"],
                        **summarize_confusion(cm),
                    }
                ),
                flush=True,
            )
    a.output.mkdir(parents=True)
    report = {
        "status": "complete",
        "kind": "post_review_validation_only",
        "original_winner_reselected": False,
        "refitting_performed": False,
        "reserved_input_reads": 0,
        "reserved_target_reads": 0,
        "neural_lock_sha256": lock_hash,
        "classical_fit_id": a.fit_id,
        "classical_lock_sha256": fit["lock_file_sha256"],
        "model_sha256": fit["b2_model_sha256"],
        "model_semantic_sha256": fit["b2_model_semantic_sha256"],
        "development_bytes": data,
        "source_sha256": {
            Path(__file__).name: digest(__file__),
            "revision_metrics.py": digest(
                Path(__file__).with_name("revision_metrics.py")
            ),
        },
        "job_id": os.environ["SLURM_JOB_ID"],
        "library_versions": _library_versions(),
        "per_tile": records,
        "pooled": {
            name: {"confusion": cm.tolist(), **summarize_confusion(cm)}
            for name, cm in pooled.items()
        },
        "interpretation": "New task-matched baselines, not claimed reproductions of the companion paper; B2 supplies the threshold gate plus a conventional learned classifier.",
    }
    (a.output / "classical_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (a.output / "checksums.json").write_text(
        json.dumps(
            {
                "classical_validation.json": digest(
                    a.output / "classical_validation.json"
                )
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

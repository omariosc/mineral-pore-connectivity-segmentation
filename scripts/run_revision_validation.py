#!/usr/bin/env python3
"""Post-review validation analyses of the authenticated historical screen.

Never evaluates the reserved partition or changes the selected-method lock.
Historical source is imported from a read-only original root. This script and
revision_metrics.py are separately checksummed, rather than rewriting that root.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from contextlib import nullcontext
import numpy as np

try:
    from .revision_metrics import (
        compose_gate_logits,
        confusion,
        summarize_confusion,
        gate_confusion,
        component_recovery,
    )
except ImportError:
    from revision_metrics import (
        compose_gate_logits,
        confusion,
        summarize_confusion,
        gate_confusion,
        component_recovery,
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def runtime_guard(hardware):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a scheduled compute allocation")
    import torch

    if hardware == "cpu":
        return torch
    token = {"l40s": "L40S", "h200": "H200"}[hardware]
    if not torch.cuda.is_available() or token not in torch.cuda.get_device_name(0):
        raise RuntimeError(
            "The allocation does not match the explicitly requested GPU family"
        )
    return torch


def set_backend(torch, profile):
    torch.backends.cudnn.benchmark = profile == "training"
    torch.backends.cudnn.deterministic = profile == "deterministic"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def logits_for(
    torch, model, image, candidate, prepare, threshold=100, *, device="cuda"
):
    value = prepare(image, candidate)
    if value.shape[0] == 2:
        value[1] = (image < threshold).astype(np.float32)
    tensor = torch.from_numpy(value).unsqueeze(0).to(device)
    precision = (
        torch.autocast("cuda", dtype=torch.float16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), precision:
        logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
    result = logits[0].float().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite inference logits")
    return result


def selection_delta(values, recorded):
    keys = {
        "score": "harmonic_pore_iou",
        "c0_iou": "c0_iou",
        "c1_iou": "c1_iou",
        "pore_union_iou": "pore_union_iou",
    }
    return {k: values[v] - recorded[k] for k, v in keys.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--hardware", choices=["l40s", "h200", "cpu"], default="l40s")
    args = parser.parse_args()
    torch = runtime_guard(args.hardware)
    device = "cpu" if args.hardware == "cpu" else "cuda"
    root = args.source_root.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(root))
    from scripts import report_validation_screen_tiles as historical
    from scripts.evaluate_confirmatory_checkpoint import (
        create_model_from_state,
        prepare_locked_model_input,
    )

    lock, lock_hash, lock_id, cells = historical.load_verified_screen_lock(
        Path("config/selected_method_lock.json"), root
    )
    attestation = historical.independently_reattest_development_bytes(lock, root)
    tiles = historical.load_validation_tiles(
        historical.validation_tile_specs(lock), root
    )
    args.output.mkdir(parents=True)
    report = {
        "status": "running",
        "kind": "post_review_validation_only",
        "requested_hardware": args.hardware,
        "reserved_input_reads": 0,
        "reserved_target_reads": 0,
        "method_reselected": False,
        "source_root": str(root),
        "lock_sha256": lock_hash,
        "lock_identifier": lock_id,
        "selected_method_unchanged": lock["selected_method"],
        "development_bytes": attestation,
        "revision_sources": {
            Path(__file__).name: sha(__file__),
            "revision_metrics.py": sha(Path(__file__).with_name("revision_metrics.py")),
        },
        "runtime": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "hostname": platform.node(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "cpu_threads": torch.get_num_threads(),
            "historical_gpu_family_matched": args.hardware == "l40s",
        },
        "inference": {
            "amp": "float16" if device == "cuda" else "disabled_float32",
            "hard_prediction": "argmax over logits before probability conversion",
            "normalization": "historical prepare_locked_model_input",
            "gate": "raw uint8 intensity strictly less than threshold",
            "thresholds": [90, 95, 100, 105, 110],
            "sensitivity": "C2-P input gate and final composition both varied; no tuning or winner replacement",
        },
        "object_definition": "C0 8-connected reference components; detected when >=50% of pixels are predicted C0; coverage recall, not one-to-one instance matching; no size filter; edge strata reported",
        "cells": [],
    }
    write_json(args.output / "report.json", report)
    metrics = []
    objects = []
    curves = []
    budgets = []
    diagnostics = []
    saved = {}
    work = cells[:1] if args.diagnose_only else cells
    profiles = (
        ("cpu_float32",)
        if device == "cpu"
        else (("deterministic", "training") if args.diagnose_only else ("training",))
    )
    for cell in work:
        checkpoint, state, identity = historical.authenticate_screen_checkpoint(
            cell, lock, root
        )
        for profile in profiles:
            if device == "cuda":
                set_backend(torch, profile)
            model, model_config = create_model_from_state(state)
            model.to(device).eval()
            candidate = cell["candidate"]
            seed = int(cell["seed"])
            scope_rows = []
            totals = {}
            thresholds = (
                [90, 95, 100, 105, 110]
                if candidate == "C2-P" and not args.diagnose_only
                else [100]
            )
            for threshold in thresholds:
                for tile in tiles:
                    image = tile["image"]
                    target = tile["target"]
                    identifier = tile["file_name"]
                    logits = logits_for(
                        torch,
                        model,
                        image,
                        candidate,
                        prepare_locked_model_input,
                        threshold,
                        device=device,
                    )
                    native = logits.argmax(axis=0).astype(np.uint8)
                    predictions = (
                        {
                            "native": native,
                            "gate_composed": compose_gate_logits(
                                image, logits, threshold
                            ),
                        }
                        if candidate in ("R3", "H3")
                        else {
                            "gate_composed": compose_gate_logits(
                                image, logits, threshold
                            )
                        }
                    )
                    for composition, pred in predictions.items():
                        cm = confusion(target, pred)
                        key = (composition, threshold)
                        totals[key] = (
                            totals.get(key, np.zeros((3, 3), dtype=np.int64)) + cm
                        )
                        base = {
                            "candidate": candidate,
                            "seed": seed,
                            "backend": profile,
                            "composition": composition,
                            "threshold": threshold,
                            "file_name": identifier,
                        }
                        row = {
                            **base,
                            **summarize_confusion(cm),
                            "confusion_json": json.dumps(cm.tolist()),
                        }
                        if composition == "gate_composed":
                            within = gate_confusion(image, target, pred, threshold)
                            row[
                                "within_gate_reference_rows_C0_C1_C2_prediction_cols_C0_C1_json"
                            ] = json.dumps(within.tolist())
                        if threshold == 100 and not args.diagnose_only:
                            obj = component_recovery(target, pred)
                            row.update({k: v for k, v in obj.items() if k != "objects"})
                            objects.extend({**base, **o} for o in obj["objects"])
                        scope_rows.append(row)
                        metrics.append(row)
                        if (
                            seed == 42
                            and threshold == 100
                            and not args.diagnose_only
                            and candidate in ("R3", "C2-P")
                        ):
                            from PIL import Image

                            destination = args.output / "predictions"
                            destination.mkdir(exist_ok=True)
                            name = (
                                f"{candidate}_{composition}_{Path(identifier).stem}.png"
                            )
                            Image.fromarray(pred).save(destination / name)
                            saved[(identifier, candidate, composition)] = pred
                if threshold != 100 and device == "cuda":
                    torch.cuda.empty_cache()
            expected_composition = (
                "native" if candidate in ("R3", "H3") else "gate_composed"
            )
            native_metrics = summarize_confusion(totals[(expected_composition, 100)])
            deltas = selection_delta(native_metrics, cell["selection_metrics"])
            summary = {
                "candidate": candidate,
                "seed": seed,
                "backend": profile,
                "selected_epoch": checkpoint.get("epoch"),
                "checkpoint_identity": identity,
                "model": model_config,
                "recorded_selection": cell["selection_metrics"],
                "recomputed_selection_deltas": deltas,
                "exact_to_original_tolerance": all(
                    abs(x) <= 5e-7 for x in deltas.values()
                ),
                "pooled": [],
            }
            for (composition, threshold), cm in totals.items():
                item = {
                    "composition": composition,
                    "threshold": threshold,
                    "confusion": cm.tolist(),
                    **summarize_confusion(cm),
                }
                summary["pooled"].append(item)
                metrics.append(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "backend": profile,
                        "file_name": "POOLED",
                        **item,
                        "confusion_json": json.dumps(cm.tolist()),
                    }
                )
            report["cells"].append(summary)
            print(
                json.dumps(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "backend": profile,
                        "deltas": deltas,
                    }
                ),
                flush=True,
            )
            write_json(args.output / "report.json", report)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        if not args.diagnose_only:
            record_path = root / cell["result_artifact_repo_relative_identifier"]
            run = record_path.parent.parent
            history = run / "metrics/training_metrics.csv"
            config = run / "config/training_config.json"
            cfg = json.loads(config.read_text())
            recorded_outcome = json.loads(record_path.read_text())
            with history.open() as f:
                history_rows = list(csv.DictReader(f))
            curves.extend(
                {"candidate": cell["candidate"], "seed": int(cell["seed"]), **r}
                for r in history_rows
            )
            training = cfg["training_config"]
            contract = recorded_outcome["scientific_execution_contract"]
            patch_count = 666 if training["training_patch_size"] == 683 else 74
            nominal_updates = (patch_count + training["batch_size"] - 1) // training[
                "batch_size"
            ]
            budgets.append(
                {
                    "candidate": cell["candidate"],
                    "seed": int(cell["seed"]),
                    "completed_epochs": len(history_rows),
                    "selected_epoch_one_based": recorded_outcome[
                        "best_selection_epoch"
                    ],
                    "nominal_update_opportunities_per_epoch": nominal_updates,
                    "nominal_update_opportunities_total": nominal_updates
                    * len(history_rows),
                    "successful_updates_observed": None,
                    "amp_skipped_updates_observed": None,
                    "update_count_interpretation": "nominal opportunities derived from frozen patch count, batch size, no drop_last and no accumulation; successful AMP updates were not recorded",
                    "selected_checkpoint_epoch_raw": checkpoint.get("epoch"),
                    "epochs_planned": training["epochs"],
                    "batch_size": training["batch_size"],
                    "training_patch_size": training["training_patch_size"],
                    "early_stopping_patience": training["early_stopping_patience"],
                    "training_config_sha256": sha(config),
                    "history_sha256": sha(history),
                    "scientific_execution_contract_json": json.dumps(contract),
                    "original_system_info_json": json.dumps(cfg.get("system_info", {})),
                }
            )
        del checkpoint, state
    write_csv(args.output / "metrics.csv", metrics)
    write_csv(args.output / "objects.csv", objects)
    write_csv(args.output / "learning_curves.csv", curves)
    write_csv(args.output / "training_budgets.csv", budgets)
    if not args.diagnose_only:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        palette = ListedColormap(["#dc322f", "#2aa745", "#d2d4d6"])
        for tile in tiles:
            name = tile["file_name"]
            target = tile["target"]
            reference = [
                saved[(name, "R3", "native")],
                saved[(name, "R3", "gate_composed")],
                saved[(name, "C2-P", "gate_composed")],
            ]
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            axes[0, 0].imshow(tile["image"], cmap="gray", vmin=0, vmax=255)
            axes[0, 0].set_title("Raw SEM")
            axes[1, 0].imshow(target, cmap=palette, vmin=0, vmax=2)
            axes[1, 0].set_title("Reference C0 red C1 green")
            for j, (label, pred) in enumerate(
                zip(["Flat", "Flat + gate", "Patch conditional"], reference), start=1
            ):
                axes[0, j].imshow(pred, cmap=palette, vmin=0, vmax=2)
                axes[0, j].set_title(label)
                error = np.zeros_like(target)
                error[(pred != target) & (target < 2) & (pred < 2)] = 1
                error[(pred != target) & ((target == 2) | (pred == 2))] = 2
                axes[1, j].imshow(
                    error,
                    cmap=ListedColormap(["white", "#b83280", "#4169e1"]),
                    vmin=0,
                    vmax=2,
                )
                axes[1, j].set_title("Error pink C0/C1 blue pore/matrix")
            for ax in axes.ravel():
                ax.axis("off")
            fig.suptitle(
                f"{name} | seed 42 | full tile | physical calibration pending",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(args.output / (Path(name).stem + "_predictions.png"), dpi=200)
            plt.close(fig)
        fig, axes = plt.subplots(5, 3, figsize=(12, 15))
        for i, candidate in enumerate(["R3", "H3", "C2-P", "C2-F", "C2-FP"]):
            for seed in [42, 123, 2025]:
                rows = [
                    r
                    for r in curves
                    if r["candidate"] == candidate and r["seed"] == seed
                ]
                epochs = [int(r["epoch"]) for r in rows]
                for j, key in enumerate(
                    ["train_loss", "val_loss", "disconnected_pore_iou"]
                ):
                    axes[i, j].plot(
                        epochs, [float(r[key]) for r in rows], label=str(seed)
                    )
            for j, key in enumerate(
                ["training loss", "validation loss", "validation C0 IoU"]
            ):
                axes[i, j].set_title(f"{candidate} {key}")
                axes[i, j].set_xlabel("Epoch")
                axes[i, j].legend()
        fig.tight_layout()
        fig.savefig(args.output / "learning_curves.png", dpi=180)
        plt.close(fig)
    report["status"] = "complete"
    report["duration_seconds"] = time.monotonic() - START
    report["interpretation"] = (
        "Recomputed diagnostics are new validation-only observations. Original values retained with explicit deltas; no equality or independent-test claim is made where unsupported."
    )
    write_json(args.output / "report.json", report)
    manifest = {
        str(p.relative_to(args.output)): sha(p)
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    }
    write_json(args.output / "checksums.json", manifest)


if __name__ == "__main__":
    START = time.monotonic()
    main()

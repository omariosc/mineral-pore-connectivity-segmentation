#!/usr/bin/env python3
"""Train separately registered, validation-only revision controls on AIRE."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch


def _metrics(matrix):
    try:
        from scripts.revision_metrics import summarize_confusion
    except ModuleNotFoundError:
        from revision_metrics import summarize_confusion
    return summarize_confusion(matrix)


def build_inputs(images, channels, threshold):
    if channels == 1:
        return images
    if channels != 2:
        raise ValueError("Only one or two input channels are registered")
    gate = (images[:, 0:1] < (2.0 * threshold / 255.0 - 1.0)).to(images.dtype)
    return torch.cat((images, gate), dim=1)


def selection_key(matrix, loss):
    m = _metrics(matrix)
    if m["harmonic_pore_iou"] is None or not np.isfinite(loss):
        raise ValueError("Invalid selection metrics")
    return m["harmonic_pore_iou"], m["pore_union_iou"], -float(loss)


def project_development_metadata(partitions):
    images = []
    ids = set()
    names = set()
    for name in ["train", "val"]:
        part = partitions[name]
        if len(part["image_ids"]) != len(part["image_files"]):
            raise ValueError("Image metadata lengths differ")
        for identifier, filename in zip(part["image_ids"], part["image_files"]):
            if (
                identifier in ids
                or filename in names
                or Path(filename).name != filename
            ):
                raise ValueError("Duplicate or unsafe development identity")
            ids.add(identifier)
            names.add(filename)
            images.append(
                {"id": identifier, "file_name": filename, "height": 2048, "width": 2048}
            )
    return {
        "images": images,
        "annotations": [],
        "categories": [
            {"id": 0, "name": "disconnected_pore"},
            {"id": 1, "name": "connected_pore"},
            {"id": 2, "name": "mineral"},
        ],
    }


def validate_training_environment(environment):
    if environment.get("DISABLE_TRANSFORMS", "0") != "0":
        raise ValueError("Registered light augmentation cannot be disabled")


def qualify_smoke(gradient_norms, successful_updates):
    if len(gradient_norms) != 2 or not all(np.isfinite(x) for x in gradient_norms):
        raise FloatingPointError("Smoke requires two finite-gradient training batches")
    if successful_updates != 2:
        raise FloatingPointError("Smoke requires two successful optimizer updates")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--hardware", choices=["l40s", "h200"], default="l40s")
    args = p.parse_args()
    validate_training_environment(os.environ)
    token = {"l40s": "L40S", "h200": "H200"}[args.hardware]
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not torch.cuda.is_available()
        or token not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError(
            "Use a scheduled allocation matching the registered GPU family"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        sha(args.protocol) != args.protocol_sha256
        or sha(args.source_manifest) != args.source_manifest_sha256
    ):
        raise ValueError("Registration identity mismatch")
    registration = json.loads(args.protocol.read_text())
    if args.hardware not in registration["allowed_hardware"]:
        raise ValueError("Unregistered hardware")
    source_manifest = json.loads(args.source_manifest.read_text())
    code_root = Path(__file__).resolve().parents[1]
    for name, digest in source_manifest.items():
        if sha(code_root / name) != digest:
            raise ValueError(f"Revision source hash mismatch: {name}")
    if args.seed not in registration["seeds"]:
        raise ValueError("Unregistered seed")
    arm = next((x for x in registration["arms"] if x["name"] == args.arm), None)
    if arm is None:
        raise ValueError("Unregistered arm")
    root = args.source_root.resolve(strict=True)
    # Old source is preserved and checked using its own lock. Revision-only
    # model/analysis files remain separately bound by source_manifest.
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from revision_models import RevisionUNet
    from revision_metrics import confusion, summarize_confusion
    from scripts import report_validation_screen_tiles as original
    from src.training.patch_dataset import PatchDataset, seed_patch_dataloader_worker
    from src.models.conditional_pore_loss import ConditionalPoreFocalDiceLoss

    lock, lock_hash, _, _ = original.load_verified_screen_lock(
        Path("config/selected_method_lock.json"), root
    )
    if lock_hash != registration["original_screen_lock_sha256"]:
        raise ValueError("Different historical screen lock")
    data_attestation = original.independently_reattest_development_bytes(lock, root)
    partitions = lock["screen_selection_provenance"]["resolved_data_split"][
        "partitions"
    ]
    metadata = project_development_metadata(partitions)
    augmentation = {
        "patch_size": registration["patch_size"],
        "seed": args.seed,
        "augmentation": {"enabled": True, "strength": "light", "use_advanced": False},
    }
    image_dir = root / original.CANONICAL_IMAGE_ROOT
    mask_dir = root / original.CANONICAL_MASK_ROOT
    train = PatchDataset(
        metadata,
        str(image_dir),
        patch_size=registration["patch_size"],
        augmentation_config=augmentation,
        training=True,
        image_ids=partitions["train"]["image_ids"],
        num_classes=2,
        split_name="train",
        mask_dir=mask_dir,
    )
    val = PatchDataset(
        metadata,
        str(image_dir),
        patch_size=2048,
        augmentation_config=augmentation,
        training=False,
        image_ids=partitions["val"]["image_ids"],
        num_classes=2,
        split_name="val",
        mask_dir=mask_dir,
    )
    if len(train) != 666 or len(val) != 5:
        raise ValueError("Development patch/tile count differs")
    counts = train.training_class_statistics()["counts"]
    if counts != [1816734, 41571134, 266990628]:
        raise ValueError("Training-only class counts differ")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = RevisionUNet(
        arm["input_channels"],
        multiscale=arm["multiscale"],
        attention=arm["attention"],
        boundary=arm["boundary"],
        base_features=registration["base_features"],
    ).cuda()
    criterion = ConditionalPoreFocalDiceLoss(
        counts, focal_weight=arm["focal_weight"], dice_weight=1 - arm["focal_weight"]
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=registration["learning_rate"],
        weight_decay=registration["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=registration["epochs"]
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=registration["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=registration["workers"],
        worker_init_fn=seed_patch_dataloader_worker,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    valid = torch.utils.data.DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=registration["amp_initial_scale"])
    args.output.mkdir(parents=True)
    global RUN_OUTPUT
    RUN_OUTPUT = args.output
    start = time.monotonic()
    selected = None
    best_epoch = 0
    epochs = 1 if args.smoke else registration["epochs"]
    total_updates = 0
    report = {
        "status": "running",
        "scientific_result": not args.smoke,
        "protocol_id": registration["protocol_id"],
        "protocol_sha256": args.protocol_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_code_sha256": source_manifest,
        "historical_source_sha256": lock["screen_selection_provenance"][
            "source_code_sha256"
        ],
        "historical_lock_sha256": lock_hash,
        "arm": arm,
        "seed": args.seed,
        "job_id": os.environ["SLURM_JOB_ID"],
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "data_attestation": data_attestation,
        "reserved_input_reads": 0,
        "reserved_target_reads": 0,
        "annotation_index_reads": 0,
        "winner_reselected": False,
        "parameter_count": sum(x.numel() for x in model.parameters()),
        "train_patches": len(train),
        "validation_tiles": len(val),
        "updates_per_complete_epoch": len(loader),
        "augmentation": train.augmentation_provenance,
        "loss": criterion.resolved_config(),
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "epochs": [],
    }
    write_json(args.output / "run.json", report)
    best_tiles = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        epoch_updates = 0
        skipped_updates = 0
        gradient_norms = []
        for i, (images, targets, _) in enumerate(loader):
            if args.smoke and i >= 2:
                break
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            inputs = build_inputs(
                images, arm["input_channels"], registration["threshold"]
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=False
            )
            gradient_norms.append(float(gradient_norm))
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped_updates += 1
            else:
                epoch_updates += 1
                total_updates += 1
            train_losses.append(float(loss.detach()))
        if args.smoke:
            qualify_smoke(gradient_norms, epoch_updates)
        model.eval()
        validation_loss = []
        cm = np.zeros((3, 3), dtype=np.int64)
        tile_rows = []
        with torch.inference_mode():
            for i, (images, targets, patch_ids) in enumerate(valid):
                if args.smoke and i >= 2:
                    break
                images = images.cuda()
                targets = targets.cuda()
                inputs = build_inputs(
                    images, arm["input_channels"], registration["threshold"]
                )
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation output")
                pred = logits.argmax(1)
                pred = torch.where(
                    images[:, 0] < (2 * registration["threshold"] / 255 - 1), pred, 2
                )
                target = targets.clone()
                target[target == -100] = 2
                current = confusion(target[0].cpu().numpy(), pred[0].cpu().numpy())
                cm += current
                tile_rows.append(
                    {
                        "patch_id": str(patch_ids[0]),
                        "confusion": current.tolist(),
                        **summarize_confusion(current),
                    }
                )
                validation_loss.append(float(loss))
        if args.smoke and len(validation_loss) != 2:
            raise RuntimeError("Smoke must validate two native tiles")
        key = selection_key(cm, float(np.mean(validation_loss)))
        if selected is None or key > selected:
            selected = key
            best_epoch = epoch
            best_tiles = tile_rows
            # Save only selected model state; optimizer is not resumed or used
            # to manufacture additional independent runs.
            torch.save(
                {
                    "epoch_one_based": epoch,
                    "model_state_dict": model.state_dict(),
                    "arm": arm,
                    "seed": args.seed,
                    "registration_sha256": args.protocol_sha256,
                    "source_manifest_sha256": args.source_manifest_sha256,
                    "training_counts": counts,
                    "selection_key": key,
                    "validation_confusion": cm.tolist(),
                },
                args.output / "selected.pt",
            )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": float(np.mean(validation_loss)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_updates": epoch_updates,
            "amp_skipped_updates": skipped_updates,
            "cumulative_optimizer_updates": total_updates,
            "selected_epoch_so_far": best_epoch,
            **summarize_confusion(cm),
        }
        report["epochs"].append(row)
        report["selected_epoch_one_based"] = best_epoch
        report["selected_key"] = selected
        write_json(args.output / "run.json", report)
        print(json.dumps({"arm": args.arm, "seed": args.seed, **row}), flush=True)
        scheduler.step()
        if epoch - best_epoch >= registration["patience"]:
            break
    if total_updates == 0:
        raise FloatingPointError("No successful optimizer updates")
    if args.smoke:
        report["smoke_qualification"] = {
            "finite_gradient_norms": gradient_norms,
            "successful_optimizer_updates": total_updates,
            "native_validation_tiles": len(validation_loss),
        }
    report["status"] = "complete"
    report["selected_per_tile"] = best_tiles
    report["selected_checkpoint_sha256"] = sha(args.output / "selected.pt")
    report["actual_optimizer_updates"] = total_updates
    report["duration_seconds"] = time.monotonic() - start
    write_json(args.output / "run.json", report)
    with (args.output / "learning_curve.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report["epochs"][0]))
        w.writeheader()
        w.writerows(report["epochs"])
    write_json(
        args.output / "checksums.json",
        {
            n: sha(args.output / n)
            for n in ["run.json", "selected.pt", "learning_curve.csv"]
        },
    )


if __name__ == "__main__":
    RUN_OUTPUT = None
    try:
        main()
    except BaseException as error:
        if RUN_OUTPUT is not None and (RUN_OUTPUT / "run.json").exists():
            failure = json.loads((RUN_OUTPUT / "run.json").read_text())
            failure.update(
                status="failed",
                failure_type=type(error).__name__,
                failure_message=str(error),
            )
            write_json(RUN_OUTPUT / "run.json", failure)
        raise

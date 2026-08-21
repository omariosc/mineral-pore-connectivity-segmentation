#!/usr/bin/env python3
"""Generate paper-ready figures and tables from repository outputs.

The script is intentionally read-only with respect to experiment outputs. It
collects saved run summaries, training curves, dataset statistics, and selected
prediction images, then writes regenerated assets under ``paper_assets/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "paper_assets"
FIGURES = ASSETS / "figures"
TABLES = ASSETS / "tables"

FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono", "Menlo", "Consolas", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRAL = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

COLORS = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}

CLASS_NAMES = {
    "0": "Disconnected pores",
    "1": "Connected pores",
    "2": "Mineral matrix",
}

PUBLICATION_CLASS_COLORS = {
    "0": "#B33A3A",  # C0 disconnected pore: muted red
    "1": "#2E8B57",  # C1 connected pore: muted green
    "2": "#4C78A8",  # C2 mineral matrix: muted blue
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "text.color": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    if not title or not subtitle:
        raise ValueError("Every figure needs a title and subtitle.")
    ax.set_title("")
    left = ax.get_position().x0
    title_wrapped = textwrap.fill(title, width=82, break_long_words=False)
    subtitle_wrapped = textwrap.fill(subtitle, width=120, break_long_words=False)
    fig.text(left, 0.965, title_wrapped, ha="left", va="top", fontsize=11, weight="bold")
    fig.text(left, 0.915, subtitle_wrapped, ha="left", va="top", fontsize=8.5, color=TOKENS["muted"])
    fig.subplots_adjust(top=0.82)


def save_figure(fig: plt.Figure, name: str) -> None:
    png_path = FIGURES / f"{name}.png"
    svg_path = FIGURES / f"{name}.svg"
    pdf_path = FIGURES / f"{name}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(svg_path, bbox_inches="tight", facecolor=TOKENS["surface"], metadata={"Date": None})
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=TOKENS["surface"])
    normalize_text_file(svg_path)
    plt.close(fig)


def microscopy_image_dir() -> Path:
    """Return the best available local source for the 2048-pixel tiles.

    ``original_images`` is the public pipeline input, but this research checkout
    currently retains the corresponding grayscale tiles only in the generated
    COCO dataset.  Falling back keeps curated figures reproducible without
    pretending that the raw annotated source corpus is present.
    """
    candidates = [
        ROOT / "original_images",
        RESULTS / "step3_coco_dataset" / "images",
        RESULTS / "coco_dataset" / "images",
    ]
    for directory in candidates:
        if directory.exists() and any(directory.glob("*.png")):
            return directory
    return ROOT / "original_images"


def label_mask_paths() -> list[Path]:
    return sorted(
        (RESULTS / "step2_pore_classification" / "pore_classifications").glob("*.png")
    )


def measured_class_counts() -> dict[str, int]:
    """Measure the three saved label values directly from the mask files."""
    counts = {"0": 0, "1": 0, "2": 0}
    for path in label_mask_paths():
        values, frequencies = np.unique(np.asarray(Image.open(path)), return_counts=True)
        histogram = dict(zip(values.tolist(), frequencies.tolist(), strict=False))
        counts["0"] += int(histogram.get(0, 0))
        counts["1"] += int(histogram.get(1, 0))
        # Historical masks encode mineral as 255; newer masks may use 2.
        counts["2"] += int(histogram.get(2, 0)) + int(histogram.get(255, 0))
    return counts


def normalize_text_file(path: Path) -> None:
    """Keep generated text assets stable across platforms and reruns."""
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def parse_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def run_group(path: Path) -> str:
    parts = path.parts
    if "candidates" in parts:
        return "candidate"
    if "ablation" in parts:
        idx = parts.index("ablation")
        return parts[idx + 1] if idx + 1 < len(parts) else "ablation"
    if "patch_training" in parts:
        return "patch_training"
    return "other"


def clean_run_name(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("run_"):
        return parent.split("_", 3)[-1]
    return parent


def load_run_summaries() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(RESULTS.glob("**/training_summary.json")):
        try:
            data = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        metrics = data.get("final_metrics", {})
        config = data.get("training_config", {})
        row = {
            "run_name": clean_run_name(summary_path),
            "group": run_group(summary_path),
            "summary_path": str(summary_path.relative_to(ROOT)),
            "epochs_trained": parse_float(config.get("epochs_trained")),
            "early_stopped": str(config.get("early_stopped", "")).lower() == "true",
            "batch_size": parse_float(config.get("batch_size")),
            "learning_rate": parse_float(config.get("learning_rate")),
            "best_val_loss": parse_float(metrics.get("best_val_loss")),
            "best_pore_iou": parse_float(metrics.get("best_pore_iou")),
            "final_mean_iou": parse_float(metrics.get("final_mean_iou")),
            "final_pore_iou": parse_float(metrics.get("final_pore_iou")),
            "final_background_iou": parse_float(metrics.get("final_background_iou")),
            "final_val_acc": parse_float(metrics.get("final_val_acc")),
            "total_training_time_s": parse_float(metrics.get("total_training_time")),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No training_summary.json files found under results/.")
    return df.sort_values(["best_pore_iou", "final_mean_iou"], ascending=False, na_position="last")


def write_tables(runs: pd.DataFrame) -> None:
    keep = [
        "run_name",
        "group",
        "epochs_trained",
        "early_stopped",
        "best_pore_iou",
        "final_mean_iou",
        "final_pore_iou",
        "final_background_iou",
        "final_val_acc",
        "best_val_loss",
        "summary_path",
    ]
    rounded = runs[keep].copy()
    numeric_cols = rounded.select_dtypes(include="number").columns
    rounded[numeric_cols] = rounded[numeric_cols].round(4)
    rounded.to_csv(TABLES / "experiment_summary.csv", index=False, lineterminator="\n")
    write_markdown_table(rounded.head(20), TABLES / "top_experiments.md")

    dataset_rows = dataset_summary_rows()
    with (TABLES / "dataset_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(dataset_rows)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """Write a simple GitHub-flavored Markdown table without optional deps."""
    columns = list(df.columns)
    rows = [[format_cell(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]
    lines = []
    lines.append("| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |")
    lines.append("| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |")
    path.write_text("\n".join(lines) + "\n")


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def confirmatory_split_summary_rows(
    manifest_path: Path = ROOT / "config" / "confirmatory_splits.json",
) -> list[dict[str, str]]:
    """Return split counts from the frozen current-rerun manifest.

    The historical COCO ``splits.json`` records an obsolete 80/10/10 split
    and must not be used to describe the locked confirmatory rerun.
    """
    split_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_keys = ("train", "val", "test")
    split_ids: dict[str, set[int]] = {}
    for split_name in split_keys:
        values = split_data.get(split_name)
        if not isinstance(values, list) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in values
        ):
            raise ValueError(
                f"{manifest_path}: {split_name!r} must be a list of integer IDs"
            )
        if len(values) != len(set(values)):
            raise ValueError(
                f"{manifest_path}: {split_name!r} contains duplicate IDs"
            )
        split_ids[split_name] = set(values)

    for index, left in enumerate(split_keys):
        for right in split_keys[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise ValueError(
                    f"{manifest_path}: {left!r} and {right!r} overlap"
                )

    return [
        {"metric": "declared_train_images", "value": str(len(split_ids["train"]))},
        {"metric": "declared_val_images", "value": str(len(split_ids["val"]))},
        {"metric": "declared_test_images", "value": str(len(split_ids["test"]))},
    ]


def dataset_summary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_dir = microscopy_image_dir()
    image_count = len(list(source_dir.glob("*.png")))
    rows.append({"metric": "input_images", "value": str(image_count)})
    rows.append({"metric": "local_image_source", "value": str(source_dir.relative_to(ROOT))})

    measured = measured_class_counts()
    total_measured = sum(measured.values())
    if total_measured:
        for class_id, count in measured.items():
            class_name = CLASS_NAMES[class_id]
            rows.append({"metric": f"{class_name} pixels measured", "value": str(count)})
            rows.append(
                {
                    "metric": f"{class_name} percent measured",
                    "value": f"{100.0 * count / total_measured:.6f}",
                }
            )

    pixel_stats_path = RESULTS / "step2_pixel_classification" / "pixel_classification_stats.json"
    pore_stats_path = RESULTS / "step2_pore_classification" / "pore_classification_stats.json"
    stats_path = pixel_stats_path if pixel_stats_path.exists() else pore_stats_path
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for key, value in stats.items():
            if isinstance(value, (str, int, float)):
                rows.append({"metric": key, "value": str(value)})
        totals = stats.get("class_totals") or stats.get("pore_class_totals") or {}
        percentages = stats.get("class_percentages") or stats.get("pore_class_percentages") or {}
        for class_id, count in totals.items():
            class_name = CLASS_NAMES.get(str(class_id), f"class_{class_id}")
            rows.append({"metric": f"{class_name} pixels", "value": str(count)})
            pct = percentages.get(str(class_id), percentages.get(int(class_id), None) if str(class_id).isdigit() else None)
            if pct is not None:
                rows.append({"metric": f"{class_name} percent", "value": f"{float(pct):.4f}"})

    coco_path = RESULTS / "step3_coco_dataset" / "pore_annotations.json"
    if not coco_path.exists():
        coco_path = RESULTS / "step3_coco_dataset" / "annotations.json"
    if coco_path.exists():
        coco = json.loads(coco_path.read_text())
        rows.extend(
            [
                {"metric": "coco_images", "value": str(len(coco.get("images", [])))},
                {"metric": "coco_annotations", "value": str(len(coco.get("annotations", [])))},
            ]
        )

    rows.extend(confirmatory_split_summary_rows())
    return rows


def plot_class_balance() -> None:
    image_count = len(label_mask_paths())
    counts = measured_class_counts()
    disconnected = counts["0"]
    connected = counts["1"]
    mineral = counts["2"]
    total_pixels = disconnected + connected + mineral
    if total_pixels == 0:
        print("Skipping fig_01_class_balance: no saved label masks found.")
        return

    data = pd.DataFrame(
        [
            {"class": "C0  Disconnected pores", "pixels": disconnected, "colour": PUBLICATION_CLASS_COLORS["0"]},
            {"class": "C1  Connected pores", "pixels": connected, "colour": PUBLICATION_CLASS_COLORS["1"]},
            {"class": "C2  Mineral matrix", "pixels": mineral, "colour": PUBLICATION_CLASS_COLORS["2"]},
        ]
    )
    data["percent"] = data["pixels"] / max(total_pixels, 1) * 100.0
    data = data.sort_values("percent")

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    y_positions = np.arange(len(data))
    # A dot plot is used instead of bars because a logarithmic axis cannot
    # provide the zero baseline that bar-length comparisons require.
    ax.scatter(
        data["percent"],
        y_positions,
        s=115,
        c=data["colour"],
        edgecolors=NEUTRAL["dark"],
        linewidths=0.7,
        zorder=3,
    )
    ax.set_yticks(y_positions, labels=data["class"])
    ax.set_xlabel("Share of all pixels (%)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xscale("log")
    ax.set_xlim(0.35, 140)
    ax.grid(axis="x", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for y_pos, value in enumerate(data["percent"]):
        ax.text(value * 1.16, y_pos, f"{value:.3f}%", va="center", ha="left", fontsize=8.4, color=TOKENS["ink"])
    add_header(
        fig,
        ax,
        "Operational class distribution",
        f"Direct counts from {image_count} saved label masks; mineral values 2 and 255 are combined.",
    )
    save_figure(fig, "fig_01_class_balance")


def plot_top_experiments(runs: pd.DataFrame) -> None:
    top = runs.dropna(subset=["best_pore_iou"]).head(15).copy()
    top["label"] = top["run_name"].str.replace("_", " ", regex=False).str.slice(0, 42)
    top = top.sort_values("best_pore_iou")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.barplot(
        data=top,
        y="label",
        x="best_pore_iou",
        hue="group",
        dodge=False,
        ax=ax,
        palette={
            "candidate": COLORS["blue"]["base"],
            "patch_training": COLORS["gold"]["base"],
            "architecture": COLORS["olive"]["base"],
            "loss": COLORS["orange"]["base"],
            "augmentation": COLORS["pink"]["base"],
            "boundary": COLORS["blue"]["light"],
            "training": COLORS["olive"]["light"],
            "weight": COLORS["orange"]["light"],
            "hyperparameter": COLORS["pink"]["light"],
            "baseline": NEUTRAL["base"],
            "yolo": COLORS["gold"]["light"],
        },
        edgecolor=NEUTRAL["dark"],
        linewidth=0.5,
    )
    ax.set_xlabel("Archived composite selection score")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="x", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="Run group", frameon=True)
    add_header(
        fig,
        ax,
        "Historical runs ranked by the archived model-selection score",
        "The source field is named best_pore_iou but combines pore IoU and weighted F1; historical splits were not independent.",
    )
    save_figure(fig, "fig_02_top_experiments")


def plot_architecture_comparison(runs: pd.DataFrame) -> None:
    arch = runs[runs["group"].eq("architecture")].dropna(subset=["best_pore_iou", "final_mean_iou"]).copy()
    if arch.empty:
        return
    arch["label"] = arch["run_name"].str.replace("_", " ", regex=False)
    arch = arch.sort_values("best_pore_iou")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = range(len(arch))
    ax.hlines(y, arch["final_mean_iou"], arch["best_pore_iou"], color=NEUTRAL["base"], linewidth=1.4)
    ax.scatter(arch["final_mean_iou"], y, s=40, color=COLORS["blue"]["light"], edgecolor=COLORS["blue"]["dark"], label="Final mean IoU")
    ax.scatter(arch["best_pore_iou"], y, s=48, color=COLORS["orange"]["base"], edgecolor=COLORS["orange"]["dark"], label="Archived selection score")
    ax.set_yticks(list(y))
    ax.set_yticklabels(arch["label"])
    ax.set_xlabel("IoU")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True)
    add_header(
        fig,
        ax,
        "Archived architecture runs used a composite model-selection score",
        "Diagnostic only: historical splits were patch-randomized, and the score is not a pure pore IoU.",
    )
    save_figure(fig, "fig_03_architecture_comparison")


def plot_baseline_training_curve() -> None:
    csv_path = RESULTS / "patch_training" / "baseline" / "metrics" / "training_metrics.csv"
    if not csv_path.exists():
        return
    history = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], color=COLORS["blue"]["mid"], marker="o", markersize=3, label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], color=COLORS["orange"]["mid"], marker="o", markersize=3, linestyle="--", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(frameon=True)

    ax = axes[1]
    ax.plot(history["epoch"], history["mean_iou"], color=COLORS["blue"]["mid"], marker="o", markersize=3, label="Mean IoU")
    ax.plot(history["epoch"], history["pore_iou"], color=COLORS["olive"]["mid"], marker="o", markersize=3, label="Pore IoU")
    ax.plot(history["epoch"], history["background_iou"], color=COLORS["pink"]["mid"], marker="o", markersize=3, label="Background IoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(frameon=True)

    add_header(
        fig,
        axes[0],
        "Baseline training improves rapidly before plateauing",
        "Saved baseline run, 10 epochs; curves use validation metrics recorded in training_metrics.csv.",
    )
    save_figure(fig, "fig_04_baseline_training_curves")


def load_image(path: Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def preferred_candidate_plot_dir() -> Path | None:
    preferred_runs = [
        "C4_DINOv2_Foundation",
        "C3_Efficient_CNN",
        "C1_Conservative_Best",
        "C6_Nested_Architecture",
        "C5_YOLO_Detection",
        "C2_Aggressive_Minority",
    ]
    for run_name in preferred_runs:
        plot_dir = RESULTS / "candidates" / run_name / "plots"
        if plot_dir.exists():
            return plot_dir

    candidates_dir = RESULTS / "candidates"
    if not candidates_dir.exists():
        return None
    plot_dirs = sorted(path for path in candidates_dir.glob("*/plots") if path.is_dir())
    return plot_dirs[0] if plot_dirs else None


def available_samples(
    preferred: list[str],
    paths_for_sample: Callable[[str], list[Path]],
    fallback_glob: Path | None = None,
    fallback_suffix: str = "",
    limit: int = 3,
) -> list[str]:
    selected: list[str] = []
    for sample in preferred:
        if all(path.exists() for path in paths_for_sample(sample)):
            selected.append(sample)

    if fallback_glob is not None:
        for path in sorted(fallback_glob.glob(f"*{fallback_suffix}")):
            sample = path.name.removesuffix(fallback_suffix)
            if sample in selected:
                continue
            if all(candidate.exists() for candidate in paths_for_sample(sample)):
                selected.append(sample)
            if len(selected) >= limit:
                break

    return selected[:limit]


def draw_image_panel(ax: plt.Axes, path: Path) -> None:
    ax.imshow(load_image(path))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
        spine.set_linewidth(0.6)


def draw_ring_overlay_panel(ax: plt.Axes, image_path: Path, ring_mask_path: Path) -> None:
    """Render a recovered binary ring-mask boundary over its grayscale source tile."""
    source = load_image(image_path)
    grayscale = np.asarray(source.convert("L"), dtype=np.uint8)
    rgb = np.repeat(grayscale[..., None], 3, axis=2)

    ring_mask = Image.open(ring_mask_path).convert("L").resize(source.size, Image.Resampling.NEAREST)
    selected = np.asarray(ring_mask) > 0
    overlay_colour = np.asarray((213, 94, 0), dtype=np.float32)
    rgb_float = rgb.astype(np.float32)
    rgb_float[selected] = 0.15 * rgb_float[selected] + 0.85 * overlay_colour

    ax.imshow(rgb_float.astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
        spine.set_linewidth(0.6)


def draw_label_mask_panel(ax: plt.Axes, path: Path) -> None:
    raw = np.asarray(Image.open(path))
    rgb = np.full((*raw.shape, 3), (76, 120, 168), dtype=np.uint8)
    rgb[raw == 0] = (179, 58, 58)    # C0 disconnected pore: muted red
    rgb[raw == 1] = (46, 139, 87)    # C1 connected pore: muted green
    rgb[raw == 2] = (76, 120, 168)   # C2 mineral matrix: muted blue
    rgb[raw == 255] = (76, 120, 168)
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(TOKENS["axis"])
        spine.set_linewidth(0.6)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    detail: str,
    facecolor: str,
    edgecolor: str,
    linestyle: str = "-",
    title_fontsize: float = 7.8,
    detail_fontsize: float = 6.7,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=linestyle,
    )
    ax.add_patch(box)
    title_width = max(8, int(width * 100))
    detail_width = max(14, int(width * 150))
    title_text = "\n".join(
        textwrap.fill(line, width=title_width, break_long_words=False)
        for line in title.splitlines()
    )
    detail_text = "\n".join(
        textwrap.fill(line, width=detail_width, break_long_words=False)
        for line in detail.splitlines()
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.72,
        title_text,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        weight="bold",
        linespacing=1.05,
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.29,
        detail_text,
        ha="center",
        va="center",
        fontsize=detail_fontsize,
        color=TOKENS["muted"],
        linespacing=1.1,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=NEUTRAL["dark"],
            shrinkA=1,
            shrinkB=1,
        )
    )


def plot_recovery_workflow() -> None:
    """Draw the evidence-led workflow used by the submission preparation draft."""
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.95, "Recovered local artefacts", fontsize=8, weight="bold", color=COLORS["blue"]["dark"])
    ax.text(0.52, 0.95, "Confirmatory protocol before submission", fontsize=8, weight="bold", color=COLORS["orange"]["dark"])
    ax.plot([0.50, 0.50], [0.12, 0.91], color=NEUTRAL["light"], linewidth=1.0, linestyle=":")

    boxes = [
        ((0.02, 0.55), "BSE--SEM corpus", "4 mosaics claimed\nprovenance to verify", COLORS["blue"]["xlight"], COLORS["blue"]["mid"], "-"),
        ((0.18, 0.55), "Tile archive", "100 tiles\n2048 x 2048 pixels", COLORS["blue"]["xlight"], COLORS["blue"]["mid"], "-"),
        ((0.34, 0.55), "Operational labels", "C0 inside rings\nC1 outside; C2 matrix", COLORS["blue"]["xlight"], COLORS["blue"]["mid"], "-"),
        ((0.52, 0.55), "Grouped split", "No related tiles across\ntrain, validation, test", COLORS["orange"]["xlight"], COLORS["orange"]["mid"], "--"),
        ((0.68, 0.55), "Model selection", "Locked configuration\nvalidation data only", COLORS["orange"]["xlight"], COLORS["orange"]["mid"], "--"),
        ((0.84, 0.55), "Independent test", "Per-class metrics, CIs\nand error analysis", COLORS["orange"]["xlight"], COLORS["orange"]["mid"], "--"),
    ]
    width, height = 0.135, 0.27
    for xy, title, detail, face, edge, style in boxes:
        add_box(ax, xy, width, height, title, detail, face, edge, style)
    for left in [0.155, 0.315, 0.475, 0.655, 0.815]:
        add_arrow(ax, (left, 0.675), (left + 0.022, 0.675))

    ax.text(0.02, 0.31, "Evidence boundary", fontsize=8, weight="bold", color=TOKENS["ink"])
    ax.text(
        0.02,
        0.22,
        "Solid boxes are supported by files in this checkout. Dashed boxes are the locked rerun design; "
        "their results must replace every numeric placeholder in the manuscript.",
        fontsize=7.7,
        color=TOKENS["muted"],
        va="top",
        wrap=True,
    )
    handles = [
        Patch(facecolor=COLORS["blue"]["xlight"], edgecolor=COLORS["blue"]["mid"], label="Recovered evidence"),
        Patch(facecolor=COLORS["orange"]["xlight"], edgecolor=COLORS["orange"]["mid"], linestyle="--", label="Rerun required"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2, fontsize=7.5)
    save_figure(fig, "fig_08_recovery_workflow")


def plot_model_architecture() -> None:
    """Vector schematic derived from ``MultiScaleAttentionUNet`` source code."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    encoder = [
        (0.07, 0.74, "E1", "32 ch\nDoubleConv", False),
        (0.21, 0.62, "E2", "64 ch\nDoubleConv", False),
        (0.35, 0.50, "E3", "128 ch\nMS block", True),
        (0.49, 0.38, "E4", "256 ch\nMS block", True),
        (0.63, 0.26, "Bottleneck", "256 ch\nMS block", True),
    ]
    decoder = [
        (0.76, 0.38, "D4", "128 ch\nAG", False),
        (0.76, 0.54, "D3", "64 ch\nAG + BR", False),
        (0.76, 0.70, "D2", "32 ch\nAG + BR", False),
        (0.76, 0.86, "D1", "32 ch\nAG + BR", False),
    ]
    w, h = 0.105, 0.115
    for x, y, title, detail, multiscale in encoder:
        add_box(
            ax,
            (x, y),
            w,
            h,
            title,
            detail,
            COLORS["blue"]["xlight"] if not multiscale else COLORS["olive"]["xlight"],
            COLORS["blue"]["mid"] if not multiscale else COLORS["olive"]["mid"],
            title_fontsize=8.5,
            detail_fontsize=7.3,
        )
    for x, y, title, detail, _ in decoder:
        add_box(
            ax,
            (x, y),
            w,
            h,
            title,
            detail,
            COLORS["orange"]["xlight"],
            COLORS["orange"]["mid"],
            title_fontsize=8.5,
            detail_fontsize=7.3,
        )

    add_box(
        ax,
        (0.90, 0.86),
        0.085,
        h,
        "Output",
        "3 logits\nC0 / C1 / C2",
        COLORS["pink"]["xlight"],
        COLORS["pink"]["mid"],
        title_fontsize=8.5,
        detail_fontsize=7.3,
    )
    add_box(
        ax,
        (0.005, 0.86),
        0.055,
        h,
        "Input",
        "1 channel",
        NEUTRAL["xlight"],
        NEUTRAL["mid"],
        title_fontsize=8.5,
        detail_fontsize=7.3,
    )

    path = [(0.06, 0.91), (0.07, 0.79), (0.175, 0.67), (0.315, 0.55), (0.455, 0.43), (0.595, 0.31), (0.76, 0.43), (0.812, 0.54), (0.812, 0.70), (0.812, 0.86), (0.90, 0.91)]
    for start, end in zip(path[:-1], path[1:], strict=False):
        add_arrow(ax, start, end)

    # Attention-gated skip connections.
    skips = [
        ((0.122, 0.79), (0.76, 0.905)),
        ((0.262, 0.67), (0.76, 0.745)),
        ((0.402, 0.55), (0.76, 0.585)),
        ((0.542, 0.43), (0.76, 0.425)),
    ]
    for start, end in skips:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.10",
            mutation_scale=9,
            linewidth=0.9,
            linestyle=":",
            color=COLORS["orange"]["dark"],
        )
        ax.add_patch(arrow)

    ax.text(0.03, 0.13, "MS", fontsize=8.5, weight="bold", color=COLORS["olive"]["dark"], va="top")
    ax.text(0.075, 0.13, "parallel 1 x 1 and 3 x 3 branches\n(dilation 1, 2 and 4)", fontsize=7.3, color=TOKENS["muted"], va="top")
    ax.text(0.40, 0.13, "AG", fontsize=8.5, weight="bold", color=COLORS["orange"]["dark"], va="top")
    ax.text(0.445, 0.13, "spatial attention on\nskip connections", fontsize=7.3, color=TOKENS["muted"], va="top")
    ax.text(0.67, 0.13, "BR", fontsize=8.5, weight="bold", color=COLORS["orange"]["dark"], va="top")
    ax.text(0.715, 0.13, "boundary-refinement\nmodule", fontsize=7.3, color=TOKENS["muted"], va="top")
    ax.text(
        0.03,
        0.045,
        "Prespecified screen configuration: bilinear upsampling; base width 32; deep supervision off.",
        fontsize=7.2,
        color=TOKENS["muted"],
    )
    save_figure(fig, "fig_09_model_architecture")


def plot_study_workflow() -> None:
    """Publication-facing overview of the locked current-rerun workflow."""
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        ((0.02, 0.38), "Microscopy data", "100 retained tiles\n2048 x 2048 px", COLORS["blue"]["xlight"], COLORS["blue"]["mid"], "-"),
        ((0.185, 0.38), "Label construction", "pore mask +\nring mask", COLORS["blue"]["xlight"], COLORS["blue"]["mid"], "-"),
        ((0.35, 0.38), "Operational mask", "C0 / C1 / C2\nlabel mask", COLORS["olive"]["xlight"], COLORS["olive"]["mid"], "-"),
        ((0.515, 0.38), "Grouped partitions", "grouped split\nmanifest", COLORS["gold"]["xlight"], COLORS["gold"]["mid"], "--"),
        ((0.68, 0.38), "Model fitting", "training + validation\nonly", COLORS["orange"]["xlight"], COLORS["orange"]["mid"], "--"),
        ((0.845, 0.38), "Retrospective eval.", "one locked pass\n+ error audit", COLORS["pink"]["xlight"], COLORS["pink"]["mid"], "--"),
    ]
    width, height = 0.135, 0.31
    for xy, title, detail, face, edge, style in stages:
        add_box(ax, xy, width, height, title, detail, face, edge, style)
    for left in [0.155, 0.32, 0.485, 0.65, 0.815]:
        add_arrow(ax, (left, 0.535), (left + 0.025, 0.535))

    ax.text(0.02, 0.86, "Recovered inputs", fontsize=8.3, weight="bold", color=COLORS["blue"]["dark"])
    ax.text(0.515, 0.86, "Locked current-rerun workflow", fontsize=8.3, weight="bold", color=COLORS["orange"]["dark"])
    ax.plot([0.495, 0.495], [0.29, 0.91], color=NEUTRAL["mid"], linewidth=0.8, linestyle=":")

    ax.text(
        0.02,
        0.14,
        "For this rerun, fitting and selection exclude the locked retrospective evaluation partition.",
        fontsize=7.5,
        color=TOKENS["muted"],
    )
    save_figure(fig, "fig_10_study_workflow")


def plot_output_comparison_grid() -> None:
    plot_dir = preferred_candidate_plot_dir()
    if plot_dir is None:
        print("Skipping fig_05_output_comparison_grid: no candidate prediction plots found.")
        return

    label_dir = RESULTS / "step2_pore_classification" / "pore_visualizations"
    source_dir = microscopy_image_dir()
    preferred = ["pdo1_12_segment_10_5", "pdo1_12_segment_10_6", "pdo1_12_segment_10_10"]

    def paths_for_sample(sample: str) -> list[Path]:
        return [
            source_dir / f"{sample}.png",
            plot_dir / f"{sample}_prediction.png",
            plot_dir / f"{sample}_visualization.png",
            label_dir / f"{sample}_pore_vis.png",
        ]

    samples = available_samples(preferred, paths_for_sample, plot_dir, "_prediction.png")
    if not samples:
        print("Skipping fig_05_output_comparison_grid: no complete qualitative sample set found.")
        return

    columns = [
        ("Microscopy tile", lambda sample: source_dir / f"{sample}.png"),
        ("Model prediction", lambda sample: plot_dir / f"{sample}_prediction.png"),
        ("Model visualization", lambda sample: plot_dir / f"{sample}_visualization.png"),
        ("Annotation labels", lambda sample: label_dir / f"{sample}_pore_vis.png"),
    ]
    fig, axes = plt.subplots(len(samples), len(columns), figsize=(7.2, 5.6), squeeze=False)
    for row, sample in enumerate(samples):
        for col, (title, path_for_sample) in enumerate(columns):
            ax = axes[row][col]
            draw_image_panel(ax, path_for_sample(sample))
            if row == 0:
                ax.set_title(title, fontsize=8.5, pad=4)
            if col == 0:
                ax.set_ylabel(sample.replace("_", "\n"), fontsize=7.5)
    add_header(
        fig,
        axes[0][0],
        "Qualitative outputs link predictions to annotation-derived labels",
        "Saved qualitative records; split membership and checkpoint provenance require confirmation before publication.",
    )
    save_figure(fig, "fig_05_output_comparison_grid")


def plot_annotation_pipeline_grid() -> None:
    mask_dir = RESULTS / "step2_pore_classification" / "pore_classifications"
    ring_mask_dir = RESULTS / "step1_yellow_masks"
    source_dir = microscopy_image_dir()
    preferred = ["pdo1_12_segment_10_5", "pdo2_24_segment_5_10", "pdo8_21_segment_9_7"]

    def paths_for_sample(sample: str) -> list[Path]:
        return [
            source_dir / f"{sample}.png",
            ring_mask_dir / f"{sample}_mask.png",
            mask_dir / f"{sample}.png",
        ]

    samples = available_samples(preferred, paths_for_sample, mask_dir, ".png")
    if not samples:
        print("Skipping fig_06_annotation_pipeline: no complete annotation sample set found.")
        return

    columns = [
        ("(a) BSE--SEM tile", lambda sample: source_dir / f"{sample}.png"),
        ("(b) Recovered ring-mask boundary", lambda sample: ring_mask_dir / f"{sample}_mask.png"),
        ("(c) Operational labels", lambda sample: mask_dir / f"{sample}.png"),
    ]
    fig, axes = plt.subplots(len(samples), len(columns), figsize=(7.2, 5.35), squeeze=False)
    for row, sample in enumerate(samples):
        for col, (title, path_for_sample) in enumerate(columns):
            ax = axes[row][col]
            if col == 2:
                draw_label_mask_panel(ax, path_for_sample(sample))
            elif col == 1:
                draw_ring_overlay_panel(
                    ax,
                    source_dir / f"{sample}.png",
                    path_for_sample(sample),
                )
            else:
                draw_image_panel(ax, path_for_sample(sample))
            if row == 0:
                ax.set_title(title, fontsize=9.0, pad=5)
            if col == 0:
                specimen, _, tile = sample.partition("_segment_")
                row_label = f"{specimen.replace('_', '-')}\ntile {tile.replace('_', '-')}"
                ax.set_ylabel(
                    row_label,
                    fontsize=7.6,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#D55E00",
                linewidth=2.0,
                label="Derived boundary from recovered binary ring-mask pixels",
            ),
            Patch(facecolor=PUBLICATION_CLASS_COLORS["0"], label="C0: annotation-defined isolated pore"),
            Patch(facecolor=PUBLICATION_CLASS_COLORS["1"], label="C1: annotation-defined connected pore"),
            Patch(facecolor=PUBLICATION_CLASS_COLORS["2"], edgecolor=NEUTRAL["mid"], label="C2: mineral matrix"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        fontsize=7.5,
        frameon=True,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.93,
        bottom=0.18,
        wspace=0.10,
        hspace=0.14,
    )
    save_figure(fig, "fig_06_annotation_pipeline")


def plot_inference_times() -> None:
    metrics_path = RESULTS / "final_evaluation" / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text())
    per_image = metrics.get("per_image", [])
    data = pd.DataFrame(
        [
            {"image": item["name"], "inference_time_s": parse_float(item.get("inference_time"))}
            for item in per_image
            if parse_float(item.get("inference_time")) is not None
        ]
    )
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sns.histplot(data=data, x="inference_time_s", bins=8, ax=ax, color=COLORS["blue"]["base"], edgecolor=COLORS["blue"]["dark"])
    mean = data["inference_time_s"].mean()
    ax.axvline(mean, color=COLORS["orange"]["dark"], linestyle="--", linewidth=1.2, label=f"Mean {mean:.3f} s")
    ax.set_xlabel("Inference time per 2048 x 2048 tile (s)")
    ax.set_ylabel("Image count")
    ax.grid(axis="x", linestyle=":", linewidth=0.8)
    ax.legend(frameon=True)
    add_header(
        fig,
        ax,
        "Archived inference latency across 20 mixed-partition images",
        "Hardware diagnostic only: the images comprise 17 train, 2 validation and 1 test tile and use an older checkpoint.",
    )
    save_figure(fig, "fig_07_inference_time_distribution")


def write_manifest(runs: pd.DataFrame) -> None:
    manifest = {
        "generated_assets": {
            "figures": sorted(path.name for path in FIGURES.glob("*")),
            "tables": sorted(path.name for path in TABLES.glob("*")),
        },
        "source_counts": {
            "training_summaries": int(len(runs)),
            "input_images": len(list(microscopy_image_dir().glob("*.png"))),
            "label_masks": len(label_mask_paths()),
        },
        "notes": [
            "Figures are regenerated from local result summaries and images.",
            "If original_images is empty, microscopy panels use the preserved COCO tile copies.",
            "Historical validation metrics are not treated as independent test evidence.",
            "BTPN uncertainty assets were not present in this repository checkout by filename.",
        ],
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate reproducible publication figures and tables."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--curated-only",
        action="store_true",
        help=(
            "Regenerate only the evidence-backed manuscript figures 01, 06, "
            "09, and 10. This mode does not load experiment summaries, "
            "predictions, or evaluation metrics."
        ),
    )
    mode.add_argument(
        "--study-workflow-only",
        action="store_true",
        help=(
            "Regenerate only Figure 10. This mode does not load microscopy "
            "images, annotations, experiment summaries, predictions, or metrics."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    if args.study_workflow_only:
        plot_study_workflow()
        print(f"Wrote Figure 10 to {FIGURES.relative_to(ROOT)}")
        return

    if args.curated_only:
        plot_class_balance()
        plot_annotation_pipeline_grid()
        plot_model_architecture()
        plot_study_workflow()
        print(
            "Wrote curated publication figures 01, 06, 09, and 10 to "
            f"{FIGURES.relative_to(ROOT)}"
        )
        return

    runs = load_run_summaries()
    write_tables(runs)
    plot_class_balance()
    plot_top_experiments(runs)
    plot_architecture_comparison(runs)
    plot_baseline_training_curve()
    plot_output_comparison_grid()
    plot_annotation_pipeline_grid()
    plot_inference_times()
    plot_recovery_workflow()
    plot_model_architecture()
    plot_study_workflow()
    write_manifest(runs)
    print(f"Wrote publication assets to {ASSETS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

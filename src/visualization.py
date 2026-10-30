"""Figures for the degradation pipeline: severity grids and calibration curves.

Colours are the first four slots of the validated categorical palette (adjacent-pair CVD
separation >= 8, normal-vision >= 15). Two slots fall below 3:1 contrast on the light surface, so
every series also has its own marker, a direct label and a legend, and the numbers are available
as a table (results/metrics/severity_calibration.csv).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import calibration, config, degradations  # noqa: E402
from .data_loader import load_dataset  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = {"gaussian_noise": ("#2a78d6", "o"), "gaussian_blur": ("#eb6834", "s"),
          "contrast": ("#1baf7a", "^"), "brightness": ("#eda100", "D")}
FAMILY_LABELS = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur",
                 "contrast": "Contrast", "brightness": "Brightness"}
PARAM_SYMBOL = {"gaussian_noise": "σ", "gaussian_blur": "σ", "contrast": "α", "brightness": "β"}
DATASET_LABELS = {"fashion_mnist": "Fashion-MNIST", "cifair10": "ciFAIR-10 / CIFAR-10"}
GRID_CLASSES = 6  # one image of each of the first six classes


def plot_degradation_grid(dataset: str, family: str, path: Optional[Path] = None) -> Path:
    """Rows: one clean-training calibration image per class. Columns: severity 0 (clean) to 5."""
    ds = load_dataset(dataset, debug=False)
    clean, ids, labels = calibration.calibration_subset(ds)
    picks = [int(np.flatnonzero(labels == c)[0]) for c in range(GRID_CLASSES)]
    cmap = "gray" if clean.shape[-1] == 1 else None

    fig, axes = plt.subplots(GRID_CLASSES, 6, figsize=(8.4, 1.38 * GRID_CLASSES + 0.9),
                             facecolor=SURFACE)
    for col in range(6):
        shown = degradations.degrade(clean[picks], family, col, ids[picks], stream="train")
        value = degradations.severity_value(family, col)
        header = "clean" if value is None else (
            f"{col} {degradations.SEVERITY_NAMES[col].replace('_', ' ')}\n"
            f"{PARAM_SYMBOL[family]} = {value:g}")
        for row in range(GRID_CLASSES):
            ax = axes[row, col]
            ax.imshow(shown[row].squeeze(), cmap=cmap, vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(header, fontsize=8.5, color=INK_2)
            if col == 0:
                ax.set_ylabel(ds.class_names[row], rotation=0, ha="right", va="center",
                              fontsize=8.5, color=INK_2, labelpad=6)
    fig.suptitle(f"{DATASET_LABELS[dataset]}: {FAMILY_LABELS[family]}, severities 0-5",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.004, "Clean-training calibration images; uint8 after clipping and rounding",
             ha="center", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    path = path or config.PLOTS_DIR / f"degradation_grid_{dataset}_{family}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _spread(values: dict, min_gap: float) -> dict:
    """Nudge label positions apart (ascending order) so direct labels never overlap."""
    placed, last = {}, -np.inf
    for key in sorted(values, key=values.get):
        last = max(values[key], last + min_gap)
        placed[key] = last
    return placed


def plot_calibration_curves(audit: pd.DataFrame, dataset: str, path: Optional[Path] = None) -> Path:
    """Median PSNR and mean SSIM against severity for the four families (no dual axis)."""
    data = audit[audit["dataset"] == dataset]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), facecolor=SURFACE)
    for ax, column, title in ((axes[0], "psnr_median", "Median PSNR (dB)"),
                              (axes[1], "ssim_mean", "Mean SSIM")):
        ax.set_facecolor(SURFACE)
        ends = {}
        for family in degradations.FAMILIES:
            colour, marker = SERIES[family]
            rows = data[data["family"] == family].sort_values("severity")
            ax.plot(rows["severity"], rows[column], color=colour, marker=marker, markersize=6.5,
                    linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.2, label=FAMILY_LABELS[family])
            ends[family] = float(rows[column].iloc[-1])
        low, high = data[column].min(), data[column].max()
        for family, y in _spread(ends, 0.065 * (high - low)).items():
            ax.annotate(FAMILY_LABELS[family], (5, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)
        ax.set_xlim(0.8, 7.2)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("severity (1 = mild, 5 = extreme)", color=INK_2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{DATASET_LABELS[dataset]}: severity ladder audit (clean-training calibration subset)",
                 fontsize=11, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.915, "Equal PSNR/SSIM across families does not mean equal recognition difficulty.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    path = path or config.PLOTS_DIR / f"severity_calibration_{dataset}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    argparse.ArgumentParser(description="Regenerate all Phase 3 figures.").parse_args()
    audit = pd.read_csv(calibration.AUDIT_CSV)
    for dataset in calibration.CALIBRATION_DATASETS:
        for family in degradations.FAMILIES:
            print(plot_degradation_grid(dataset, family))
        print(plot_calibration_curves(audit, dataset))


if __name__ == "__main__":
    main()

"""Result figures for the comparison (reads the saved result files only; nothing is trained or scored here).

  results/plots/main_*.png                             the three-model comparison (LDA, SVM + HOG, CNN) for the paper:
                                                       main_clean_accuracy, main_robustness_curves, main_degraded_summary,
                                                       main_confusion_<dataset>
  results/plots/pipelines.png                          the implemented solutions as pipelines
  results/plots/clean_accuracy_comparison.png          clean test accuracy of every model, both datasets
  results/plots/robustness_curves_<dataset>.png        accuracy vs severity, one panel per degradation family
  results/plots/robustness_heatmap_<dataset>.png       accuracy of every model in every degraded condition
  results/plots/degraded_accuracy_summary.png          mean accuracy over the five severities per family
  results/plots/confusion_<dataset>.png                row-normalised confusion matrices, clean and degraded
  results/plots/feature_selection_effect.png           validation accuracy against the fraction of features kept
  results/plots/feature_selection_robustness_delta.png accuracy change caused by the selection stage, per severity
  results/plots/cnn_training_curves.png                CNN validation accuracy per epoch, three seeds
  results/plots/robustness_summary_ci.png              area under the severity curve with 95% bootstrap CIs (Phase 6)
  results/plots/retention_ci.png                       chance-adjusted retention with 95% bootstrap CIs (Phase 6)
  results/plots/significance_<dataset>.png             where paired McNemar tests (Holm) separate the models (Phase 6)

Colours follow the validated categorical palette (blue, orange, aqua for the three main models). A model with
a feature-selection stage keeps the hue of its base model and is drawn dashed / hatched; the LDA + HOG ablation
is grey. Noise accuracies are averaged over the 3 noise realisations; CNN curves are the mean of the 3 seeds
with the min-max range shaded.
"""
from __future__ import annotations

import json
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import config  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd5", "#fcfcfb"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
MODELS = ("lda", "lda_fs", "svm_hog", "svm_hog_fs", "cnn", "lda_hog")
LABEL = {"lda": "LDA", "lda_fs": "LDA + FS", "svm_hog": "SVM + HOG", "svm_hog_fs": "SVM + HOG + FS",
         "cnn": "CNN", "lda_hog": "LDA + HOG (ablation)"}
COLOR = {"lda": BLUE, "lda_fs": BLUE, "svm_hog": ORANGE, "svm_hog_fs": ORANGE, "cnn": AQUA, "lda_hog": GREY}
STYLE = {"lda": ("-", "o"), "lda_fs": ("--", "s"), "svm_hog": ("-", "o"), "svm_hog_fs": ("--", "s"),
         "cnn": ("-", "o"), "lda_hog": (":", "^")}
DATASETS = {"fashion_mnist": "Fashion-MNIST", "cifair10": "ciFAIR-10"}
FAMILIES = {"gaussian_noise": "Gaussian noise (σ)", "gaussian_blur": "Gaussian blur (σ)",
            "contrast": "Contrast reduction", "brightness": "Brightness increase"}
CHANCE = 0.10

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold", "legend.frameon": False})


def _save(fig, name: str) -> None:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _grid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load():
    clean = pd.read_csv(config.METRICS_DIR / "clean_test_metrics.csv")
    rob = pd.read_csv(config.METRICS_DIR / "robustness_results.csv")
    per_seed = rob.groupby(["dataset", "model", "seed", "family", "severity"]).accuracy.mean().reset_index()
    return clean, rob, per_seed


def curve(per_seed: pd.DataFrame, dataset: str, model: str, family: str):
    """(severity 0..5, mean accuracy, min, max over seeds) with the clean accuracy as severity 0."""
    q = per_seed[(per_seed.dataset == dataset) & (per_seed.model == model)]
    sev, mean, lo, hi = [], [], [], []
    for s in range(6):
        v = q[(q.family == "clean")] if s == 0 else q[(q.family == family) & (q.severity == s)]
        sev.append(s); mean.append(v.accuracy.mean()); lo.append(v.accuracy.min()); hi.append(v.accuracy.max())
    return np.array(sev), np.array(mean), np.array(lo), np.array(hi)


def present(df: pd.DataFrame) -> List[str]:
    return [m for m in MODELS if m in set(df.model)]


# --------------------------------------------------------------------------- #
def plot_pipelines() -> None:
    rows = [("LDA", ["image", "÷255, frozen\nscalar normalisation", "flatten\n(pixels)", "LDA\n(shrinkage)"], BLUE, False),
            ("LDA + FS", ["image", "÷255, frozen\nnormalisation", "flatten", "ANOVA F-test\nkeep top k", "LDA"], BLUE, True),
            ("SVM + HOG", ["image", "÷255", "HOG\n(9 orient., 4×4 cells)", "RBF-SVM\n(C, γ)"], ORANGE, False),
            ("SVM + HOG + FS", ["image", "÷255", "HOG", "ANOVA F-test\nkeep top k", "RBF-SVM"], ORANGE, True),
            ("CNN", ["image", "÷255, frozen\nnormalisation", "5 conv. layers\n(BN, ReLU, pool)", "global avg.\npool", "linear\n(10 classes)"], AQUA, False)]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 12.4); ax.set_ylim(-0.6, len(rows) - 0.4); ax.axis("off")
    for i, (name, steps, color, fs) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(0.0, y, name, ha="left", va="center", fontsize=10, fontweight="semibold", color=INK)
        previous = None
        for j, step in enumerate(steps):
            selection = "ANOVA" in step
            box = ax.text(3.2 + j * 2.2, y, step, ha="center", va="center", fontsize=8, color=INK,
                          bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=color, lw=2.0 if selection else 1.2,
                                    hatch="//" if selection else None))
            if previous is not None:
                ax.annotate("", xy=(0, 0), xycoords=box, xytext=(0, 0), textcoords=previous,
                            arrowprops=dict(arrowstyle="->", color=INK2, lw=1, patchA=previous.get_bbox_patch(),
                                            patchB=box.get_bbox_patch(), shrinkA=2, shrinkB=2))
            previous = box
    ax.set_title("The implemented solutions: every model owns its whole pipeline, applied to the (degraded) test image",
                 loc="left", fontsize=10)
    _save(fig, "pipelines.png")


def plot_clean(clean: pd.DataFrame, only=None, filename="clean_accuracy_comparison.png") -> None:
    models = [m for m in present(clean) if only is None or m in only]
    fig, ax = plt.subplots(figsize=(10 if only is None else 7, 4))
    width = 0.8 / len(models)
    for d, (dataset, title) in enumerate(DATASETS.items()):
        for i, m in enumerate(models):
            q = clean[(clean.dataset == dataset) & (clean.model == m)].accuracy
            x = d + (i - (len(models) - 1) / 2) * width
            ax.bar(x, q.mean(), width * 0.86, color=COLOR[m], alpha=0.55 if m.endswith("_fs") else 1,
                   hatch="///" if m.endswith("_fs") else None, edgecolor=SURFACE if not m.endswith("_fs") else COLOR[m], linewidth=0.8)
            if len(q) > 1:
                ax.scatter([x] * len(q), q, s=12, color=INK, zorder=3)
            ax.text(x, q.mean() + 0.012, f"{q.mean():.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(DATASETS)), list(DATASETS.values()))
    ax.set_ylim(0, 1.05); ax.set_ylabel("clean test accuracy"); _grid(ax)
    ax.legend(handles=[Patch(fc=COLOR[m], alpha=0.55 if m.endswith("_fs") else 1, hatch="///" if m.endswith("_fs") else None,
                             ec=COLOR[m] if m.endswith("_fs") else SURFACE, label=LABEL[m]) for m in models] +
                      [Line2D([], [], marker="o", ls="", color=INK, ms=4, label="CNN training seeds")],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Clean test accuracy (models trained and tuned on clean data only)", loc="left")
    _save(fig, filename)


def plot_curves(per_seed: pd.DataFrame, dataset: str) -> None:
    models = present(per_seed[per_seed.dataset == dataset])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, (family, title) in zip(axes.ravel(), FAMILIES.items()):
        for m in models:
            sev, mean, lo, hi = curve(per_seed, dataset, m, family)
            ls, mk = STYLE[m]
            ax.plot(sev, mean, ls=ls, marker=mk, ms=5, lw=2, color=COLOR[m], label=LABEL[m])
            if m == "cnn":
                ax.fill_between(sev, lo, hi, color=COLOR[m], alpha=0.18, lw=0)
        ax.axhline(CHANCE, color=INK2, lw=0.8, ls=(0, (2, 3)))
        ax.text(0.05, CHANCE - 0.015, "chance", fontsize=7, color=INK2, ha="left", va="top")
        ax.set_title(title, loc="left"); ax.set_xticks(range(6), ["clean", "1", "2", "3", "4", "5"])
        ax.set_ylim(0, 1); ax.set_xlabel("severity"); _grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{DATASETS[dataset]}: accuracy of the frozen models as degradation increases "
                 "(CNN: mean of 3 seeds, band = min–max)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, f"robustness_curves_{dataset}.png")


def plot_main_curves(per_seed: pd.DataFrame) -> None:
    models = [m for m in MODELS if m in ("lda", "svm_hog", "cnn")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharey=True)
    for r, (dataset, dtitle) in enumerate(DATASETS.items()):
        for c, (family, ftitle) in enumerate(FAMILIES.items()):
            ax = axes[r, c]
            for m in models:
                sev, mean, lo, hi = curve(per_seed, dataset, m, family)
                ax.plot(sev, mean, ls="-", marker="o", ms=5, lw=2, color=COLOR[m], label=LABEL[m])
                if m == "cnn":
                    ax.fill_between(sev, lo, hi, color=COLOR[m], alpha=0.18, lw=0)
            ax.axhline(CHANCE, color=INK2, lw=0.8, ls=(0, (2, 3)))
            ax.set_xticks(range(6), ["clean", "1", "2", "3", "4", "5"]); ax.set_ylim(0, 1); _grid(ax)
            if r == 0:
                ax.set_title(ftitle, loc="left")
            if c == 0:
                ax.set_ylabel(f"{dtitle}\ntest accuracy")
            if r == 1:
                ax.set_xlabel("severity")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h + [Line2D([], [], color=INK2, lw=0.8, ls=(0, (2, 3)))], l + ["chance (0.10)"], ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy of LDA, SVM + HOG and the CNN as test images are degraded (CNN: mean of 3 seeds, band = min–max)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, "main_robustness_curves.png")


def plot_heatmap(per_seed: pd.DataFrame, dataset: str) -> None:
    models = present(per_seed[per_seed.dataset == dataset])
    cols = [(f, s) for f in FAMILIES for s in range(1, 6)]
    data = np.array([[curve(per_seed, dataset, m, f)[1][s] for f, s in cols] for m in models])
    clean = np.array([curve(per_seed, dataset, m, "gaussian_noise")[1][0] for m in models])
    fig, ax = plt.subplots(figsize=(13, 0.6 * len(models) + 1.8))
    im = ax.imshow(np.column_stack([clean, data]), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(data.shape[1] + 1):
            v = np.column_stack([clean, data])[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.55 else INK)
    ax.set_yticks(range(len(models)), [LABEL[m] for m in models])
    ax.set_xticks(range(len(cols) + 1), ["clean"] + [str(s) for _, s in cols])
    for k, fam in enumerate(FAMILIES.values()):
        ax.text(1 + 5 * k + 2, -0.85, fam, ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="semibold")
        ax.axvline(0.5 + 5 * k, color=SURFACE, lw=3)
    ax.set_xlabel("severity within each degradation family")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="test accuracy")
    ax.set_title(f"{DATASETS[dataset]}: test accuracy in every condition", loc="left", pad=46)
    _save(fig, f"robustness_heatmap_{dataset}.png")


def plot_summary(per_seed: pd.DataFrame, only=None, filename="degraded_accuracy_summary.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        models = [m for m in present(per_seed[per_seed.dataset == dataset]) if only is None or m in only]
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            vals = [curve(per_seed, dataset, m, f)[1][1:].mean() for f in FAMILIES]
            x = np.arange(len(FAMILIES)) + (i - (len(models) - 1) / 2) * width
            ax.bar(x, vals, width * 0.86, color=COLOR[m], alpha=0.55 if m.endswith("_fs") else 1,
                   hatch="///" if m.endswith("_fs") else None,
                   edgecolor=COLOR[m] if m.endswith("_fs") else SURFACE, linewidth=0.8)
        ax.set_xticks(range(len(FAMILIES)), [f.split(" (")[0] for f in FAMILIES.values()])
        ax.set_title(title, loc="left"); ax.set_ylim(0, 1); _grid(ax)
        ax.axhline(CHANCE, color=INK2, lw=0.8, ls=(0, (2, 3)))
    axes[0].set_ylabel("mean test accuracy over severities 1–5")
    models = [m for m in present(per_seed) if only is None or m in only]
    fig.legend(handles=[Patch(fc=COLOR[m], alpha=0.55 if m.endswith("_fs") else 1, hatch="///" if m.endswith("_fs") else None,
                              ec=COLOR[m] if m.endswith("_fs") else SURFACE, label=LABEL[m]) for m in models],
               ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Accuracy averaged over the five severities of each degradation (dotted line = chance)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, filename)


ALL_CONFUSION = [("clean", "clean"), ("gaussian_noise_s3_r0", "Gaussian noise, severity 3"),
                 ("gaussian_blur_s3", "Gaussian blur, severity 3"), ("contrast_s4", "contrast reduction, severity 4")]


def plot_confusion(dataset: str, conditions=None, filename=None) -> None:
    conditions = conditions or ALL_CONFUSION
    models = [m for m in ("lda", "svm_hog", "cnn")
              if (config.PREDICTIONS_DIR / f"robustness_{dataset}_{m}_seed{config.SEED}.npz").exists()]
    names = pd.read_csv(config.CONFUSION_DIR / f"clean_{dataset}_lda_seed{config.SEED}.csv", index_col=0).columns
    fig, axes = plt.subplots(len(conditions), len(models), figsize=(4 * len(models), 3.9 * len(conditions)), squeeze=False)
    for j, m in enumerate(models):
        with np.load(config.PREDICTIONS_DIR / f"robustness_{dataset}_{m}_seed{config.SEED}.npz") as blob:
            runs = list(blob["runs"])
            for i, (run, text) in enumerate(conditions):
                cm = blob["confusion"][runs.index(run)].astype(float)
                norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
                ax = axes[i, j]
                ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
                ax.set_title(f"{LABEL[m]}: {text}\naccuracy {np.trace(cm) / cm.sum():.3f}", fontsize=8.5, loc="left")
                ax.set_xticks(range(10), names, rotation=90, fontsize=6.5)
                ax.set_yticks(range(10), names if j == 0 else [""] * 10, fontsize=6.5)
                for s in ax.spines.values():
                    s.set_visible(False)
    fig.suptitle(f"{DATASETS[dataset]}: confusion matrices, rows = true class (each row sums to 1), "
                 "columns = predicted class", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    _save(fig, filename or f"confusion_{dataset}.png")


def plot_fs_effect(per_seed: pd.DataFrame) -> None:
    pairs = (("lda", "lda_fs", BLUE), ("svm_hog", "svm_hog_fs", ORANGE))
    if not all(config.MODEL_META_DIR.joinpath(f"{d}__{p[1]}.json").exists() for d in DATASETS for p in pairs):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        for base, fs, color in pairs:
            table = json.loads((config.MODEL_META_DIR / f"{dataset}__{fs}.json").read_text())["selection"]["grid"]
            base_table = json.loads((config.MODEL_META_DIR / f"{dataset}__{base}.json").read_text())["selection"]["grid"]
            fracs = sorted({e["k_frac"] for e in table})
            best = [max(e["val_accuracy"] for e in table if e["k_frac"] == f) for f in fracs]
            ref = max(e["val_accuracy"] for e in base_table)
            ax.plot(fracs + [1.0], best + [ref], ls="--", marker="s", ms=5, lw=2, color=color, label=LABEL[fs])
            ax.scatter([1.0], [ref], s=60, facecolors=SURFACE, edgecolors=color, linewidths=2, zorder=3)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0], ["10%", "25%", "50%", "75%", "100%\n(no selection)"])
        ax.set_xlabel("fraction of features kept (ANOVA F-test)"); ax.set_title(title, loc="left"); _grid(ax)
    axes[0].set_ylabel("best clean validation accuracy")
    axes[0].legend(loc="lower right")
    fig.suptitle("Feature selection: clean validation accuracy against the number of features kept "
                 "(best of the other hyperparameters; SVM tuned on a 10,000-image subset)",
                 x=0.01, ha="left", fontsize=9.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "feature_selection_effect.png")

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey="row")
    for r, (dataset, title) in enumerate(DATASETS.items()):
        for c, (family, ftitle) in enumerate(FAMILIES.items()):
            ax = axes[r, c]
            for base, fs, color in pairs:
                sev, a, _, _ = curve(per_seed, dataset, base, family)
                _, b, _, _ = curve(per_seed, dataset, fs, family)
                ax.plot(sev, b - a, ls="--", marker="s", ms=5, lw=2, color=color, label=f"{LABEL[fs]} − {LABEL[base]}")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(6), ["clean", "1", "2", "3", "4", "5"]); _grid(ax)
            if r == 0:
                ax.set_title(ftitle, loc="left")
            if c == 0:
                ax.set_ylabel(f"{title}\nΔ test accuracy")
            if r == 1:
                ax.set_xlabel("severity")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Effect of the feature-selection stage on test accuracy (above 0: selection helps, below 0: it hurts)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, "feature_selection_robustness_delta.png")


def plot_cnn_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        meta = json.loads((config.MODEL_META_DIR / f"{dataset}__cnn.json").read_text())
        for entry, ls in zip(meta["files"], ("-", "--", ":")):
            ax.plot(range(1, len(entry["val_accuracy_by_epoch"]) + 1), entry["val_accuracy_by_epoch"], ls=ls,
                    marker="o", ms=4, lw=2, color=AQUA, label=f"seed {entry['seed']}")
        ax.set_title(title, loc="left"); ax.set_xlabel("epoch"); _grid(ax)
    axes[0].set_ylabel("clean validation accuracy"); axes[0].legend(loc="lower right")
    fig.suptitle("CNN training: validation accuracy of the selected configuration, three seeds",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "cnn_training_curves.png")


def _analysis(name: str):
    path = config.METRICS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def plot_summary_ci(metric: str, ylabel: str, title: str, filename: str, seed_range: bool) -> None:
    s = _analysis("analysis_robustness_summary.csv")
    if s is None:
        return
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey="row" if metric == "ausc" else False)
    for r, (dataset, dtitle) in enumerate(DATASETS.items()):
        models = present(s[s.dataset == dataset])
        for c, (family, ftitle) in enumerate(FAMILIES.items()):
            ax = axes[r, c]
            q = s[(s.dataset == dataset) & (s.family == family)].set_index("model")
            for x, m in enumerate(models):
                row = q.loc[m]
                if seed_range and row[f"{metric}_seed_max"] > row[f"{metric}_seed_min"]:
                    ax.plot([x, x], [row[f"{metric}_seed_min"], row[f"{metric}_seed_max"]], color=COLOR[m], lw=7, alpha=0.25,
                            solid_capstyle="butt")
                ax.errorbar(x, row[metric], yerr=[[row[metric] - row[f"{metric}_ci_low"]], [row[f"{metric}_ci_high"] - row[metric]]],
                            fmt=STYLE[m][1], color=COLOR[m], ms=6, lw=1.6, capsize=3,
                            mfc=SURFACE if m.endswith("_fs") else COLOR[m], mew=1.6)
            ax.set_xticks(range(len(models)), [LABEL[m].replace(" (ablation)", "").replace(" + ", "+\n", 1) for m in models],
                          fontsize=6.5, rotation=0)
            ax.set_xlim(-0.6, len(models) - 0.4); _grid(ax)
            if metric == "ausc":
                ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(ftitle, loc="left")
            if c == 0:
                ax.set_ylabel(f"{dtitle}\n{ylabel}")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, filename)


def plot_significance() -> None:
    p = _analysis("analysis_pairwise.csv")
    if p is None:
        return
    order = ["clean"] + [f"{f}_s{s}" for f in FAMILIES for s in range(1, 6)]
    colors = {"a": BLUE, "b": ORANGE, "ns": "#e6e5df", "seed": "#b9b8b0"}
    for dataset, dtitle in DATASETS.items():
        q = p[p.dataset == dataset]
        pairs = list(dict.fromkeys(zip(q.model_a, q.model_b)))
        fig, ax = plt.subplots(figsize=(13, 0.5 * len(pairs) + 2.2))
        for i, (a, b) in enumerate(pairs):
            for j, cond in enumerate(order):
                g = q[(q.model_a == a) & (q.model_b == b) & (q.condition == cond)]
                if "cnn" in (a, b):
                    if g.significant.all() and g.direction.nunique() == 1:
                        state = "a" if g.direction.iloc[0] > 0 else "b"
                    else:
                        state = "seed" if g.significant.any() else "ns"
                else:
                    state = ("a" if g.direction.iloc[0] > 0 else "b") if g.significant.iloc[0] else "ns"
                ax.add_patch(plt.Rectangle((j, len(pairs) - 1 - i), 0.92, 0.88, color=colors[state], lw=0))
        ax.set_xlim(0, len(order)); ax.set_ylim(0, len(pairs))
        ax.set_yticks([len(pairs) - 1 - i + 0.44 for i in range(len(pairs))], [f"{LABEL[a]}  vs  {LABEL[b]}" for a, b in pairs])
        ax.set_xticks([j + 0.46 for j in range(len(order))], ["clean"] + [str(s) for _ in FAMILIES for s in range(1, 6)], fontsize=7)
        for k, fam in enumerate(FAMILIES.values()):
            ax.text(1 + 5 * k + 2.5, len(pairs) + 0.15, fam, ha="center", va="bottom", fontsize=8.5, fontweight="semibold")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("severity within each degradation family")
        ax.legend(handles=[Patch(fc=colors["a"], label="first model more accurate"), Patch(fc=colors["b"], label="second model more accurate"),
                           Patch(fc=colors["ns"], label="no significant difference"),
                           Patch(fc=colors["seed"], label="CNN: significant for some training seeds only")],
                  ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
        ax.set_title(f"{dtitle}: paired exact McNemar tests, Holm-corrected over all tests (p < 0.05)", loc="left", pad=26)
        _save(fig, f"significance_{dataset}.png")


def main() -> None:
    clean, _, per_seed = load()
    plot_pipelines()
    plot_clean(clean)
    for dataset in DATASETS:
        plot_curves(per_seed, dataset)
        plot_heatmap(per_seed, dataset)
        plot_confusion(dataset)
    plot_summary(per_seed)
    main3 = ("lda", "svm_hog", "cnn")
    plot_clean(clean, main3, "main_clean_accuracy.png")
    plot_summary(per_seed, main3, "main_degraded_summary.png")
    plot_main_curves(per_seed)
    for dataset in DATASETS:
        plot_confusion(dataset, [ALL_CONFUSION[0], ALL_CONFUSION[3]], f"main_confusion_{dataset}.png")
    plot_fs_effect(per_seed)
    plot_cnn_curves()
    plot_summary_ci("ausc", "area under the severity curve\n(mean accuracy, severity 0–5)",
                    "Robustness summary: mean accuracy over the severity curve with 95% bootstrap CI over test images\n"
                    "(CIs are narrower than the markers; shaded bar = CNN seed range; open marker = feature-selection model)",
                    "robustness_summary_ci.png", True)
    plot_summary_ci("mean_retention", "mean chance-adjusted retention",
                    "Chance-adjusted retention (acc − 0.1)/(clean acc − 0.1), averaged over severities 1–5, 95% bootstrap CI",
                    "retention_ci.png", False)
    plot_significance()


if __name__ == "__main__":
    main()

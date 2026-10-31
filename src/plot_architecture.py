"""Architecture diagrams (drawn from the code's own definitions; nothing is trained or scored here).

  results/plots/architecture_system.(png|pdf)   the experiment: data -> clean training -> freeze -> degraded test -> evaluation
  results/plots/architecture_cnn.(png|pdf)      the CNN, with feature-map shapes taken from build_cnn itself
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from . import config  # noqa: E402
from .plot_results import AQUA, BLUE, GRID, INK, INK2, SURFACE  # noqa: E402  (shared palette and rcParams)

LANE_TRAIN, LANE_TEST = "#e9edf2", "#f3ece4"


def _save(fig, stem: str) -> None:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = config.PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def _box(ax, x, y, text, *, edge=INK2, lw=1.3, fs=8.5, weight="normal", fc="#ffffff", pad=0.5):
    return ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, fontweight=weight, zorder=3,
                   bbox=dict(boxstyle=f"round,pad={pad}", fc=fc, ec=edge, lw=lw))


def _extent(ax, text):
    """(x0, y0, x1, y1) of a text box in data coordinates (needs a drawn canvas)."""
    window = text.get_bbox_patch().get_window_extent(ax.figure.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(window.x0, window.y0), (window.x1, window.y1)])
    return x0, y0, x1, y1


def _arrow(ax, a, b, *, dashed=False):
    """Arrow from the edge of box a to the edge of box b along the line between their centres."""
    ax0, ay0, ax1, ay1 = _extent(ax, a)
    bx0, by0, bx1, by1 = _extent(ax, b)
    ca, cb = ((ax0 + ax1) / 2, (ay0 + ay1) / 2), ((bx0 + bx1) / 2, (by0 + by1) / 2)
    dx, dy = cb[0] - ca[0], cb[1] - ca[1]

    def exit_t(half_w, half_h):  # how far along (dx, dy) a ray from the centre leaves a box
        return min(half_w / abs(dx) if dx else float("inf"), half_h / abs(dy) if dy else float("inf"))

    ta, tb = exit_t((ax1 - ax0) / 2, (ay1 - ay0) / 2), exit_t((bx1 - bx0) / 2, (by1 - by0) / 2)
    start, end = (ca[0] + dx * ta, ca[1] + dy * ta), (cb[0] - dx * tb, cb[1] - dy * tb)
    ax.annotate("", xy=end, xytext=start, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3, ls=(0, (4, 3)) if dashed else "-",
                                shrinkA=2, shrinkB=2, mutation_scale=13))


def _lane(ax, x0, y0, x1, y1, color, label, bottom=False):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.25", fc=color, ec="none", zorder=0))
    ax.text(x0 + 0.2, (y0 + 0.2) if bottom else (y1 - 0.2), label, ha="left", va="bottom" if bottom else "top", fontsize=8.5,
            color=INK2, fontweight="semibold", zorder=1)


def system() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 7.2); ax.axis("off")
    _lane(ax, 4.75, 3.95, 12.75, 6.95, LANE_TRAIN, "TRAINING SIDE — clean images only; no degraded image is ever used for training, tuning or model selection")
    _lane(ax, 4.75, 0.55, 12.75, 3.0, LANE_TEST, "TEST SIDE — degradation is applied to the test images only", bottom=True)

    data = _box(ax, 1.3, 3.5, "Dataset\nFashion-MNIST (primary)\nciFAIR-10 (replication)", weight="semibold")
    split = _box(ax, 3.75, 3.5, "Stratified split +\nnear-duplicate filter\n(RMS pixel distance)")
    train = _box(ax, 6.3, 5.9, "Train split")
    val = _box(ax, 6.3, 4.65, "Validation split")
    test = _box(ax, 6.3, 1.75, "Test split\n(near-duplicates removed)")
    tune = _box(ax, 9.0, 5.25, "Train and tune\nLDA · SVM + HOG · CNN\n+ ANOVA feature selection\n(LDA, SVM + HOG)\nhyperparameters chosen on\nclean validation accuracy", edge=BLUE, lw=1.8)
    freeze = _box(ax, 11.8, 5.25, "Freeze\nmodel files SHA-256 hashed;\ntraining-data digests,\nHOG settings and selected\nfeatures recorded", edge=BLUE, lw=2.4, weight="semibold")
    ladder = _box(ax, 9.0, 1.75, "Degradation ladder (frozen)\nGaussian noise · Gaussian blur\ncontrast reduction · brightness\n5 severities each, audited with\nPSNR / SSIM before any evaluation", edge="#eb6834", lw=1.8)
    conditions = _box(ax, 11.8, 1.75, "Test conditions\nclean + 4 × 5 degraded\n(3 noise realisations)\nidentical images for every\nmodel (SHA-256 checked)", edge="#eb6834", lw=2.4, weight="semibold")
    evaluate = _box(ax, 14.25, 3.5, "Evaluate\nfrozen models on\nevery condition\n(clean run must\nreproduce clean test)", weight="semibold", edge=AQUA, lw=2.2)
    outputs = _box(ax, 14.25, 1.65, "Predictions per image\n→ metrics: accuracy, macro-F1,\nconfusion matrices", edge=AQUA, lw=1.6, fs=8)
    stats = _box(ax, 14.25, 0.45, "Statistics and figures\nbootstrap CIs · McNemar + Holm\nAUSC · retention · plots", edge=AQUA, lw=1.6, fs=8)

    fig.canvas.draw()
    for a, b in ((data, split), (split, train), (split, val), (split, test), (train, tune), (val, tune), (tune, freeze),
                 (test, ladder), (ladder, conditions), (freeze, evaluate), (conditions, evaluate), (evaluate, outputs), (outputs, stats)):
        _arrow(ax, a, b)
    ax.text(0.2, 0.1, "Seed 42. Every step reads and writes files in results/, so each stage can be re-run and checked separately.",
            ha="left", va="bottom", fontsize=7.5, color=INK2)
    ax.set_title("Experiment architecture: train once on clean data, freeze, then score the frozen models on degraded test images",
                 loc="left", fontsize=10.5, pad=8)
    _save(fig, "architecture_system")


def cnn() -> None:
    import torch

    from .models.cnn import build_cnn

    def stages(channels: int, size: int):
        net = build_cnn(channels, 10)
        x = torch.zeros(1, channels, size, size)
        shapes = {}
        for i, layer in enumerate(net):
            x = layer(x)
            shapes[i] = tuple(x.shape[1:])
        return net, shapes

    net28, s28 = stages(1, 28)
    net32, s32 = stages(3, 32)
    params = lambda n: sum(p.numel() for p in n.parameters())  # noqa: E731
    names = [type(layer).__name__ for layer in net28]
    pools = [i for i, n in enumerate(names) if n == "MaxPool2d"]
    convs = [i for i, n in enumerate(names) if n == "Conv2d"]
    # feature-map stages: (label, shape key index)
    blocks = [("Input", None, (1, 28, 28)), ("Conv 3×3 ×2", convs[1], None), ("MaxPool 2\n+ Dropout 0.25", pools[0] + 1, None),
              ("Conv 3×3 ×2", convs[3], None), ("MaxPool 2\n+ Dropout 0.25", pools[1] + 1, None), ("Conv 3×3", convs[4], None),
              ("Global average\npool", names.index("AdaptiveAvgPool2d"), None), ("Linear", names.index("Linear"), None)]

    fig, ax = plt.subplots(figsize=(15, 4.9))
    ax.set_xlim(0, 15.2); ax.set_ylim(-1.7, 3.6); ax.axis("off")
    x = 0.4
    prev = None
    for label, idx, fixed in blocks:
        shape = fixed if idx is None else s28[idx]
        c, h = shape[0], (shape[1] if len(shape) == 3 else 1)
        height = 0.35 + 2.2 * h / 28 if len(shape) == 3 else 0.45
        width = 0.35 + 0.95 * (c / 128) if len(shape) == 3 else 0.35
        if label == "Linear":
            height, width = 0.45, 0.5
        y0 = 1.1 - height / 2
        color = AQUA if label.startswith("Conv") else (BLUE if label == "Input" else "#9a9993")
        if label == "Linear":
            color = INK2
        rect = Rectangle((x, y0), width, height, fc=color, ec=INK, lw=1, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x + width / 2, y0 + height + 0.18, label, ha="center", va="bottom", fontsize=8.3, color=INK, fontweight="semibold")
        dims = "×".join(str(d) for d in shape[::-1]) if len(shape) == 3 else (f"{shape[0]}" if label.startswith("Global") else f"{shape[0]} classes")
        ax.text(x + width / 2, y0 - 0.15, dims if len(shape) != 3 else f"{shape[1]}×{shape[2]}×{shape[0]}", ha="center", va="top", fontsize=8, color=INK2)
        if prev is not None:
            ax.annotate("", xy=(x - 0.05, 1.1), xytext=(prev + 0.05, 1.1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1), zorder=1)
        prev = x + width
        x += width + 1.1
    for xx, text in ((0.0, "Every convolution: 3×3 kernel, padding 1, no bias, then BatchNorm and ReLU. Channels 32, 32, 64, 64, 128."),):
        ax.text(0.1, -0.95, text, fontsize=8.5, color=INK, ha="left")
    ax.text(0.1, -1.3, f"Fashion-MNIST input 28×28×1: {params(net28):,} parameters.   ciFAIR-10 input 32×32×3: {params(net32):,} parameters "
            f"(feature maps 32×32 → 16×16 → 8×8).   Trained with Adam, one-cycle schedule, batch 128, 10 epochs, no augmentation.",
            fontsize=8.5, color=INK2, ha="left")
    ax.set_title("CNN architecture (five convolutions, two poolings, global average pooling, linear classifier)", loc="left", fontsize=10.5, pad=6)
    _save(fig, "architecture_cnn")


def main() -> None:
    system()
    cnn()


if __name__ == "__main__":
    main()

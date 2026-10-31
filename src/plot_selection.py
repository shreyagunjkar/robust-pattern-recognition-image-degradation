"""Figure of what the ANOVA F-test feature selection kept (reads the frozen masks, fits nothing).

Left of each pair: the fraction of colour channels kept at every pixel by `lda_fs`; right: how many of the
36 HOG features of every 2x2-cell block `svm_hog_fs` kept. Output: results/plots/feature_selection_maps.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

DATASETS = {"fashion_mnist": ("Fashion-MNIST", 28, 1), "cifair10": ("ciFAIR-10", 32, 3)}
BLOCK_FEATURES = 36  # 2 x 2 cells x 9 orientations


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    for row, (dataset, (title, size, channels)) in zip(axes, DATASETS.items()):
        panels = (("lda_fs", "pixels kept (LDA + FS)"), ("svm_hog_fs", "HOG features kept per block (SVM + HOG + FS)"))
        for ax, (model, label) in zip(row, panels):
            mask = np.load(config.MODEL_META_DIR / f"{dataset}__{model}__selected_mask.npy")
            meta = json.loads((config.MODEL_META_DIR / f"{dataset}__{model}.json").read_text())
            if model == "lda_fs":
                shown = mask.reshape(size, size, channels).mean(axis=2)
                vmax = 1.0
            else:
                blocks = size // 4 - 1
                shown = mask.reshape(blocks, blocks, BLOCK_FEATURES).sum(axis=2)
                vmax = BLOCK_FEATURES
            im = ax.imshow(shown, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"{title}: {label}\nk = {meta['feature_selection']['k']} of "
                         f"{meta['feature_selection']['n_features_in']}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PLOTS_DIR / "feature_selection_maps.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Train/validation/test leakage audit: exact and near-duplicate images.

Exact duplicates are found by hashing the raw pixel bytes. Near-duplicates are
found with a nearest-neighbour search in pixel space; the distance is the RMS
pixel difference on the [0, 1] scale (0 = identical). Thresholds on that
distance are heuristics for flagging candidates, not a verdict - inspect the
closest pairs visually (see `save_closest_pairs`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .data_loader import ImageDataset

DEFAULT_THRESHOLDS = (0.0, 0.01, 0.02, 0.05)


def hash_images(x: np.ndarray) -> List[str]:
    return [hashlib.md5(img.tobytes()).hexdigest() for img in x]


def exact_overlap(x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """Indices of images in `x_b` whose pixels are byte-identical to some image in `x_a`."""
    seen = set(hash_images(x_a))
    return np.array([i for i, h in enumerate(hash_images(x_b)) if h in seen], dtype=np.int64)


def nearest_neighbour_distance(query: np.ndarray, reference: np.ndarray,
                               chunk: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """RMS distance (on [0, 1] pixels) from each query image to its nearest reference image.

    Returns (distance, index into `reference`). Uses the expansion
    ||q - r||^2 = ||q||^2 + ||r||^2 - 2 q.r in float32; the numerical floor is ~1e-3
    RMS, far below the thresholds of interest.
    """
    q = query.reshape(len(query), -1).astype(np.float32) / 255.0
    r = reference.reshape(len(reference), -1).astype(np.float32) / 255.0
    r_sq = (r ** 2).sum(axis=1)
    dist = np.empty(len(q), dtype=np.float32)
    idx = np.empty(len(q), dtype=np.int64)
    for start in range(0, len(q), chunk):
        block = q[start:start + chunk]
        # numpy 2.0 + Apple Accelerate raises spurious FP warnings on float32 matmul; the
        # result is verified against float64 in tests, and the isfinite check below
        # still catches any genuine overflow.
        with np.errstate(all="ignore"):
            sq = (block ** 2).sum(axis=1, keepdims=True) + r_sq[None, :] - 2.0 * block @ r.T
        if not np.isfinite(sq).all():
            raise FloatingPointError("non-finite distances in nearest_neighbour_distance")
        nearest = sq.argmin(axis=1)
        idx[start:start + chunk] = nearest
        dist[start:start + chunk] = np.sqrt(
            np.maximum(sq[np.arange(len(block)), nearest], 0.0) / q.shape[1]
        )
    return dist, idx


def leakage_report(ds: "ImageDataset", thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                   near_duplicates: bool = True) -> Dict[str, object]:
    """Audit overlap between train, validation and test images."""
    report: Dict[str, object] = {
        "dataset": ds.name,
        "exact_duplicates": {
            "train_vs_test": int(len(exact_overlap(ds.x_train, ds.x_test))),
            "val_vs_test": int(len(exact_overlap(ds.x_val, ds.x_test))),
            "train_vs_val": int(len(exact_overlap(ds.x_train, ds.x_val))),
            "within_train": int(len(ds.x_train) - len(set(hash_images(ds.x_train)))),
        },
        "index_overlap_train_val": int(len(np.intersect1d(ds.train_idx, ds.val_idx))),
    }
    if near_duplicates:
        # Test images against everything the model could have seen or been tuned on.
        reference = np.concatenate([ds.x_train, ds.x_val])
        dist, idx = nearest_neighbour_distance(ds.x_test, reference)
        report["near_duplicates_test_vs_train_val"] = {
            "n_test": int(len(dist)),
            "nn_distance_quantiles": {
                q: round(float(np.quantile(dist, q)), 4) for q in (0.0, 0.001, 0.01, 0.1, 0.5)
            },
            "count_at_or_below": {str(t): int((dist <= t + 1e-3).sum()) if t == 0.0
                                  else int((dist <= t).sum()) for t in thresholds},
        }
        report["_nn"] = (dist, idx, reference)
    return report


def save_closest_pairs(ds: "ImageDataset", dist: np.ndarray, idx: np.ndarray,
                       reference: np.ndarray, path: Path, k: int = 12) -> Path:
    """Figure of the k test images closest to the training pool, with their nearest match."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(dist)[:k]
    fig, axes = plt.subplots(2, k, figsize=(1.4 * k, 3.4))
    for col, test_i in enumerate(order):
        for row, (img, title) in enumerate((
            (ds.x_test[test_i], f"test\nd={dist[test_i]:.3f}"),
            (reference[idx[test_i]], "nearest\ntrain/val"),
        )):
            ax = axes[row, col]
            ax.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None, vmin=0, vmax=255)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

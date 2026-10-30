"""Preprocessing shared by every model.

Order of operations at evaluation time:
    uint8 image -> (degradation, Phase 3) -> uint8 -> /255 -> global normalisation
The normalisation constants are scalars fitted on the CLEAN training images only
and then frozen, so degraded images are never re-normalised with their own
statistics (which would silently hide brightness/contrast changes).

HOG features (Phase 4) must be computed from `to_unit_float` output, NOT from `normalize`
output: normalised pixels are negative, which breaks HOG's optional square-root
compression (it produced non-finite features in a check).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormStats:
    mean: float
    std: float


def to_unit_float(x: np.ndarray) -> np.ndarray:
    """uint8 pixels in [0, 255] -> float32 in [0, 1]."""
    if x.dtype != np.uint8:
        raise TypeError(f"expected uint8 images, got {x.dtype}")
    return x.astype(np.float32) / 255.0


def fit_norm_stats(x_train: np.ndarray) -> NormStats:
    """Global scalar mean/std over all pixels (and channels) of the clean training set."""
    unit = to_unit_float(x_train)
    return NormStats(mean=float(unit.mean()), std=float(unit.std()))


def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """uint8 images -> float32 images with the frozen training-set normalisation."""
    return ((to_unit_float(x) - stats.mean) / stats.std).astype(np.float32)


def flatten(x: np.ndarray) -> np.ndarray:
    """(N, H, W, C) -> (N, H*W*C), the input layout of the non-convolutional models."""
    return x.reshape(len(x), -1)

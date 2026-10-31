"""LDA, SVM + HOG, LDA + HOG and their feature-selection variants (HOG policy and the selection stage:
see the package docstring in src/models/__init__.py)."""
from __future__ import annotations

import hashlib
import time
import warnings
from typing import Dict, Optional

import joblib
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

from ..preprocessing import NormStats, fit_norm_stats, flatten, normalize, to_unit_float

HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (4, 4), "cells_per_block": (2, 2),
              "block_norm": "L2-Hys", "transform_sqrt": False}


# --------------------------------------------------------------------------- #
# HOG
# --------------------------------------------------------------------------- #
def normalized_hog_params(params: Dict[str, object]) -> Dict[str, object]:
    """JSON-comparable form of a HOG parameter set (tuples become lists), as stored in the metadata."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}


def _hog_one(image: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    if image.shape[-1] == 1:
        return hog(image[..., 0], **params)
    return hog(image, channel_axis=-1, **params)


def _hog_batch(images: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    return np.stack([_hog_one(image, params) for image in images])


def hog_features(x: np.ndarray, n_jobs: int = -1, params: Optional[Dict[str, object]] = None) -> np.ndarray:
    """uint8 (N, H, W, C) -> float32 HOG features (N, D), from the [0, 1] image."""
    params = dict(HOG_PARAMS if params is None else params)
    unit = to_unit_float(x)
    parts = Parallel(n_jobs=n_jobs)(delayed(_hog_batch)(unit[i:i + 500], params) for i in range(0, len(unit), 500))
    features = np.concatenate(parts).astype(np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("non-finite HOG features")
    return features


# --------------------------------------------------------------------------- #
# LDA, SVM + HOG, LDA + HOG
# --------------------------------------------------------------------------- #
SKLEARN_MODELS = ("lda", "svm_hog", "lda_hog", "lda_fs", "svm_hog_fs")


class SklearnModel:
    """LDA on normalised pixels, SVM on HOG, or LDA on HOG; `hparams` fix the classifier.

    The `_fs` variants insert an ANOVA F-test feature-selection stage between the features (pixels or HOG)
    and the classifier: the `k_frac` hyperparameter keeps the top `k_frac` fraction of features, ranked by the
    F statistic of each feature on the clean training split. The selected set is frozen with the model.
    """

    def __init__(self, name: str, hparams: Dict[str, object]):
        if name not in SKLEARN_MODELS:
            raise ValueError(f"unknown sklearn model {name!r}")
        self.name, self.hparams = name, dict(hparams)
        self.use_hog = self.base != "lda"
        self.selector: Optional[SelectKBest] = None
        # snapshot of the HOG settings used at fit time: prediction never reads the mutable global
        self.hog_params: Optional[Dict[str, object]] = dict(HOG_PARAMS) if self.use_hog else None
        self.stats: Optional[NormStats] = None
        self.estimator = None
        self.gamma: Optional[float] = None
        self.fit_seconds: Optional[float] = None

    @property
    def base(self) -> str:
        """The model without its selection stage ('svm_hog_fs' -> 'svm_hog')."""
        return self.name[:-3] if self.name.endswith("_fs") else self.name

    @property
    def use_fs(self) -> bool:
        return self.name.endswith("_fs")

    def feature_selection_record(self) -> Optional[Dict[str, object]]:
        """What the frozen selection stage did (None for models without one); stored in the metadata."""
        selector = getattr(self, "selector", None)
        if selector is None:
            return None
        mask = selector.get_support()
        return {"method": "ANOVA F-test (sklearn f_classif) with SelectKBest, fitted on the clean training split",
                "n_features_in": int(mask.size), "k": int(mask.sum()), "k_frac": float(self.hparams["k_frac"]),
                "selected_mask_sha256": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()}

    def frozen_hog_params(self) -> Optional[Dict[str, object]]:
        """HOG settings this model was fitted with (models saved before the snapshot existed used HOG_PARAMS)."""
        if not self.use_hog:
            return None
        return getattr(self, "hog_params", None) or dict(HOG_PARAMS)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.use_hog:
            return hog_features(x, params=self.frozen_hog_params())
        if self.stats is None:
            raise RuntimeError("model has no frozen normalisation statistics: fit it first")
        return flatten(normalize(x, self.stats))

    def fit_features(self, features: np.ndarray, y: np.ndarray) -> "SklearnModel":
        t0 = time.time()
        if self.use_fs:
            k = max(1, int(round(float(self.hparams["k_frac"]) * features.shape[1])))
            with warnings.catch_warnings():  # constant features (e.g. always-black pixels) have an undefined F score
                warnings.simplefilter("ignore")
                self.selector = SelectKBest(f_classif, k=k).fit(features, y)  # ranked on clean training data only
            features = self.selector.transform(features)
        if self.base == "svm_hog":
            # gamma = multiplier / (D * var(X)): the "scale" heuristic times a tuned factor (D = features kept)
            self.gamma = float(self.hparams["gamma_mult"]) / (features.shape[1] * float(features.var()))
            self.estimator = SVC(kernel="rbf", C=float(self.hparams["C"]), gamma=self.gamma,
                                 cache_size=2000)
        else:
            self.estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.hparams["shrinkage"])
        self.estimator.fit(features, y)
        self.fit_seconds = time.time() - t0
        return self

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "SklearnModel":
        if not self.use_hog:
            self.stats = fit_norm_stats(x_train)  # clean training images only, then frozen
        return self.fit_features(self.transform(x_train), y_train)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        if getattr(self, "selector", None) is not None:
            features = self.selector.transform(features)
        if self.base != "svm_hog":
            return self.estimator.predict(features).astype(np.int64)
        # Kernel prediction is the slow step; each image is independent, so chunks run in threads
        # (libsvm releases the GIL) and the labels are identical to a single call.
        parts = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self.estimator.predict)(features[i:i + 500]) for i in range(0, len(features), 500))
        return np.concatenate(parts).astype(np.int64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_features(self.transform(x))

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "SklearnModel":
        return joblib.load(path)

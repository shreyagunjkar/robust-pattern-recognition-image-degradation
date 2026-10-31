"""The compared models. Every model maps uint8 images (N, H, W, C) to integer labels.

Each model owns its whole input pipeline, so the same (possibly degraded) uint8 array can be fed to
all of them and nothing is computed from test-time statistics:

  lda       uint8 -> /255 -> (x - mean) / std with scalars frozen from the clean training set ->
            flatten -> Linear Discriminant Analysis
  svm_hog   uint8 -> /255 -> HOG -> RBF-kernel SVM          (HOG is computed AFTER degradation)
  cnn       uint8 -> /255 -> frozen normalisation -> small convolutional network
  lda_hog   uint8 -> /255 -> HOG -> LDA                     (appendix ablation, not a main model)

HOG policy (fixed, not tuned): input on the [0, 1] scale, 9 orientations, 4x4-pixel cells, 2x2-cell
blocks, L2-Hys block normalisation, no square-root compression. Colour images use skimage's
multichannel rule (gradients per channel, the strongest channel's gradient per pixel). 28x28 gives
1,296 features, 32x32 gives 1,764. HOG has no frozen statistics, so nothing is normalised.
"""
from __future__ import annotations

from .classical import HOG_PARAMS, SklearnModel, hog_features
from .cnn import CNNModel, build_cnn

MODEL_NAMES = ("lda", "svm_hog", "cnn", "lda_hog")
MAIN_MODELS = ("lda", "svm_hog", "cnn")
MODEL_TITLES = {"lda": "Linear Discriminant Analysis (LDA)",
                "svm_hog": "Support Vector Machine with Histogram of Oriented Gradients features (SVM + HOG)",
                "cnn": "Convolutional Neural Network (CNN)",
                "lda_hog": "LDA on HOG features (ablation)"}


def load_model(name: str, path):
    return CNNModel.load(path) if name == "cnn" else SklearnModel.load(path)

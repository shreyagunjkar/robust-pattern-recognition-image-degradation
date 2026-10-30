"""Central configuration: paths, seeds, dataset specifications and split sizes.

Everything that defines the experiment (and therefore must be reported in the
paper) lives here or in the degradation/model modules added in later phases.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
CACHE_DIR = DATA_DIR / "cache"  # degraded test sets (Phase 3+)
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

SEED = 42

# Debug mode: small stratified subsets so the whole pipeline runs in minutes.
# Enable with `PRJ_DEBUG=1` or by passing debug=True to the loader.
DEBUG_MODE = os.environ.get("PRJ_DEBUG", "0") == "1"
DEBUG_PER_CLASS = {"train": 200, "val": 50, "test": 100}

# Test images whose nearest train/val image is closer than this RMS distance
# (on the [0, 1] pixel scale) are treated as the same source image and removed
# from the evaluation set. See src/leakage.py; thresholds were chosen by visual
# inspection of ranked nearest-neighbour pairs.
DEDUP_TEST = True
# The same rule applied between the validation and the training images, so that
# clean-validation hyperparameter selection cannot reward memorised near-copies.
DEDUP_VAL = True


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    md5: str  # published by the dataset authors / widely documented
    sha256: str  # pinned from the verified download


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_shape: Tuple[int, int, int]  # (H, W, C); grayscale uses C = 1
    class_names: Tuple[str, ...]
    n_val: int  # size of the stratified validation set carved from the official train set
    files: Tuple[RemoteFile, ...]
    dedup_rms: Optional[float] = None  # near-duplicate threshold, see DEDUP_TEST


FASHION_MNIST = DatasetSpec(
    name="fashion_mnist",
    image_shape=(28, 28, 1),
    class_names=(
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ),
    n_val=10_000,  # 50k train / 10k val / 10k test
    dedup_rms=0.04,
    files=(
        RemoteFile(
            "train-images-idx3-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
            "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        ),
        RemoteFile(
            "train-labels-idx1-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "25c81989df183df01b3e8a0aad5dffbe",
            "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        ),
        RemoteFile(
            "t10k-images-idx3-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554ea6380940ec79",
            "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        ),
        RemoteFile(
            "t10k-labels-idx1-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a12a480ee83cd310",
            "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
        ),
    ),
)

# Original CIFAR-10. Its test set contains near-duplicates of training images (Barz &
# Denzler, "Do We Train on Test Data? Purging CIFAR of Near-Duplicates"), which a pixel
# distance threshold cannot separate from look-alikes. Use it only for the sensitivity
# analysis; the primary CIFAR experiment uses ciFAIR-10 below.
CIFAR10 = DatasetSpec(
    name="cifar10",
    image_shape=(32, 32, 3),
    class_names=(
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ),
    n_val=5_000,  # 45k train / 5k val / 10k test
    files=(
        RemoteFile(
            "cifar-10-binary.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
            "c32a1d4ab5d03f1284b67883e8d87530",
            "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
        ),
    ),
)

# ciFAIR-10 (Barz & Denzler): identical training set, 308 test-set duplicates replaced by
# new images (verified against CIFAR-10). Pairs that look identical at 32x32 still remain
# up to an RMS distance of ~0.08, so those test images are removed as well. The authors
# publish an archive MD5 and a test_batch MD5 in their official loader
# (github.com/cvjena/cifair, loaders/pytorch/cifair.py); the SHA-256 below was computed by us.
CIFAIR10 = DatasetSpec(
    name="cifair10",
    image_shape=(32, 32, 3),
    class_names=CIFAR10.class_names,
    n_val=5_000,  # 45k train / 5k val / 10k test
    dedup_rms=0.08,
    files=(
        RemoteFile(
            "ciFAIR-10.zip",
            "https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip",
            "ca08fd390f0839693d3fc45c4e49585f",
            "24b6bec47fb01c56fb150b501d053ea1d1f983cdb8410d4ce5bba1d25b5cff0e",
        ),
    ),
)

# MD5 of the ciFAIR-10 test_batch member, as published by the authors (same loader).
CIFAIR10_TEST_BATCH_MD5 = "01290e6b622a1977a000eff13650aca2"

DATASETS = {spec.name: spec for spec in (FASHION_MNIST, CIFAR10, CIFAIR10)}

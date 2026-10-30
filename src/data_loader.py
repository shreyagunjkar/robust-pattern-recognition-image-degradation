"""Download, verify, parse and split the image-classification datasets.

Images are returned as uint8 arrays of shape (N, H, W, C); grayscale datasets
use C = 1. Labels are int64 in [0, 9]. Every split is deterministic given the
seed, and the split indices are saved under data/splits/ so another student can
reproduce (or inspect) exactly which images were used.
"""
from __future__ import annotations

import argparse
import codecs
import gzip
import hashlib
import io
import pickle
import shutil
import struct
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from . import config
from .config import DatasetSpec, RemoteFile
from .leakage import nearest_neighbour_distance


class ChecksumError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Download and verification
# --------------------------------------------------------------------------- #
def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, remote: RemoteFile) -> None:
    """Raise ChecksumError unless both the MD5 and the pinned SHA-256 match."""
    for algorithm, expected in (("md5", remote.md5), ("sha256", remote.sha256)):
        actual = _file_digest(path, algorithm)
        if actual != expected:
            raise ChecksumError(
                f"{path.name}: {algorithm} mismatch (expected {expected}, got {actual}). "
                f"Delete the file and re-run to download it again."
            )


def download_file(remote: RemoteFile, dest_dir: Path) -> Path:
    """Download `remote` into `dest_dir` (resuming a partial download) and verify it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / remote.name
    if dest.exists():
        verify_file(dest, remote)
        return dest

    part = dest.with_name(dest.name + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    request = urllib.request.Request(remote.url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        append = resume_from > 0 and response.status == 206
        with open(part, "ab" if append else "wb") as f:
            shutil.copyfileobj(response, f, 1 << 20)

    try:
        verify_file(part, remote)
    except ChecksumError:
        part.unlink()
        raise
    part.replace(dest)
    return dest


# --------------------------------------------------------------------------- #
# Parsers (raw bytes only; nothing is unpickled)
# --------------------------------------------------------------------------- #
def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path.name}: bad IDX image magic number {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n * h * w:
        raise ValueError(f"{path.name}: expected {n * h * w} bytes, got {data.size}")
    return data.reshape(n, h, w, 1).copy()


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path.name}: bad IDX label magic number {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n:
        raise ValueError(f"{path.name}: expected {n} labels, got {data.size}")
    return data.astype(np.int64)


def _load_fashion_mnist(raw_dir: Path) -> Tuple[np.ndarray, ...]:
    spec = config.FASHION_MNIST
    paths = {f.name: download_file(f, raw_dir) for f in spec.files}
    return (
        _read_idx_images(paths["train-images-idx3-ubyte.gz"]),
        _read_idx_labels(paths["train-labels-idx1-ubyte.gz"]),
        _read_idx_images(paths["t10k-images-idx3-ubyte.gz"]),
        _read_idx_labels(paths["t10k-labels-idx1-ubyte.gz"]),
    )


def _load_cifar10(raw_dir: Path) -> Tuple[np.ndarray, ...]:
    spec = config.CIFAR10
    archive = download_file(spec.files[0], raw_dir)
    wanted = [f"data_batch_{i}.bin" for i in range(1, 6)] + ["test_batch.bin"]
    blobs: Dict[str, bytes] = {}
    meta = ""
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name in wanted:
                blobs[name] = tar.extractfile(member).read()
            elif name == "batches.meta.txt":
                meta = tar.extractfile(member).read().decode()
    missing = [n for n in wanted if n not in blobs]
    if missing:
        raise ValueError(f"{archive.name} is missing {missing}")
    if tuple(meta.split()) != spec.class_names:
        raise ValueError(f"CIFAR-10 class names in archive differ from config: {meta.split()}")

    def parse(names):
        records = np.concatenate(
            [np.frombuffer(blobs[n], dtype=np.uint8).reshape(-1, 3073) for n in names]
        )
        labels = records[:, 0].astype(np.int64)
        # Stored as 3 planes of 32x32 (R, G, B); convert to (N, H, W, C).
        images = records[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return np.ascontiguousarray(images), labels

    x_train, y_train = parse(wanted[:5])
    x_test, y_test = parse(wanted[5:])
    return x_train, y_train, x_test, y_test


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only builds numpy arrays; anything else is rejected.

    The ciFAIR-10 files are pickles. Scanning their opcodes (without executing them)
    shows they reference exactly these globals, so nothing else is allowed to load.
    """

    def find_class(self, module: str, name: str):
        try:
            from numpy._core.multiarray import _reconstruct
        except ImportError:  # numpy < 2
            from numpy.core.multiarray import _reconstruct
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("_codecs", "encode"): codecs.encode,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"blocked pickle global {module}.{name}")
        return allowed[(module, name)]


def _load_cifair10(raw_dir: Path) -> Tuple[np.ndarray, ...]:
    spec = config.CIFAIR10
    archive = download_file(spec.files[0], raw_dir)

    def parse(zf: zipfile.ZipFile, member: str):
        batch = _RestrictedUnpickler(io.BytesIO(zf.read(member)), encoding="latin1").load()
        data = batch["data"] if "data" in batch else batch[b"data"]
        labels = batch["labels"] if "labels" in batch else batch[b"labels"]
        images = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return np.ascontiguousarray(images), np.asarray(labels, dtype=np.int64)

    with zipfile.ZipFile(archive) as zf:
        train = [parse(zf, f"ciFAIR-10/data_batch_{i}") for i in range(1, 6)]
        x_test, y_test = parse(zf, "ciFAIR-10/test_batch")
    x_train = np.concatenate([x for x, _ in train])
    y_train = np.concatenate([y for _, y in train])
    return x_train, y_train, x_test, y_test


_LOADERS = {"fashion_mnist": _load_fashion_mnist, "cifar10": _load_cifar10,
            "cifair10": _load_cifair10}
_OFFICIAL_SIZES = {"fashion_mnist": (60_000, 10_000), "cifar10": (50_000, 10_000),
                   "cifair10": (50_000, 10_000)}


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_split(y_train: np.ndarray, n_val: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/validation split of the official training set (index arrays)."""
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=n_val, stratify=y_train, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def load_or_create_split(name: str, y_train: np.ndarray, n_val: int, seed: int,
                         split_dir: Path = config.SPLIT_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Reuse the saved split if present so results never depend on library versions."""
    path = split_dir / f"{name}_seed{seed}.npz"
    if path.exists():
        saved = np.load(path)
        return saved["train_idx"], saved["val_idx"]
    train_idx, val_idx = make_split(y_train, n_val, seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


def stratified_subsample(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Sorted indices of `per_class` randomly chosen samples from every class."""
    rng = np.random.default_rng(seed)
    chosen = [
        rng.permutation(np.flatnonzero(y == c))[:per_class] for c in np.unique(y)
    ]
    return np.sort(np.concatenate(chosen))


def test_keep_mask(name: str, x_train_full: np.ndarray, x_test: np.ndarray, threshold: float,
                   split_dir: Path = config.SPLIT_DIR) -> np.ndarray:
    """Boolean mask over the official test set: False = near-duplicate of a training image.

    Compared against the FULL official training set (train + val, whatever the seed or
    debug setting), so the mask depends only on the dataset and the threshold; it is cached.
    """
    path = split_dir / f"{name}_test_keep_rms{threshold:g}.npy"
    if path.exists():
        return np.load(path)
    distance, _ = nearest_neighbour_distance(x_test, x_train_full)
    keep = distance > threshold
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, keep)
    return keep


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass
class ImageDataset:
    name: str
    class_names: Tuple[str, ...]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray  # positions of x_train / x_val in the official training set
    val_idx: np.ndarray
    test_idx: np.ndarray  # positions of x_test in the official test set
    n_test_removed: int  # official test images dropped as near-duplicates of training images
    seed: int
    debug: bool

    def summary(self) -> Dict[str, object]:
        n_classes = len(self.class_names)
        return {
            "dataset": self.name,
            "seed": self.seed,
            "debug": self.debug,
            "test_near_duplicates_removed": self.n_test_removed,
            "image_shape": tuple(self.x_train.shape[1:]),
            "dtype": str(self.x_train.dtype),
            "sizes": {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)},
            "class_counts": {
                split: np.bincount(y, minlength=n_classes).tolist()
                for split, y in (("train", self.y_train), ("val", self.y_val), ("test", self.y_test))
            },
            "pixel_range": (int(self.x_train.min()), int(self.x_train.max())),
        }


def load_dataset(name: str, seed: int = config.SEED, debug: Optional[bool] = None,
                 dedup_test: Optional[bool] = None, raw_dir: Path = config.RAW_DIR,
                 split_dir: Path = config.SPLIT_DIR) -> ImageDataset:
    """Load `name` with a stratified train/val split of the official train set.

    With `dedup_test` (default: config.DEDUP_TEST) official test images that are
    near-duplicates of a training/validation image are removed, for datasets that define a
    `dedup_rms` threshold (datasets without one, such as ciFAIR-10, rely on their
    construction); otherwise the official test set is used as is. In debug mode every split is replaced by a small stratified
    subset of itself, so debug train/val/test remain disjoint.
    """
    if name not in config.DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(config.DATASETS)}")
    spec: DatasetSpec = config.DATASETS[name]
    debug = config.DEBUG_MODE if debug is None else debug
    dedup_test = config.DEDUP_TEST if dedup_test is None else dedup_test

    x_full, y_full, x_test, y_test = _LOADERS[name](raw_dir / name)
    expected_train, expected_test = _OFFICIAL_SIZES[name]
    if (len(y_full), len(y_test)) != (expected_train, expected_test):
        raise ValueError(f"{name}: unexpected official split sizes {len(y_full)}/{len(y_test)}")
    if x_full.shape[1:] != spec.image_shape or x_test.shape[1:] != spec.image_shape:
        raise ValueError(f"{name}: unexpected image shape {x_full.shape[1:]}")

    train_idx, val_idx = load_or_create_split(name, y_full, spec.n_val, seed, split_dir)
    test_idx = np.arange(len(y_test))
    n_test_removed = 0
    if dedup_test and spec.dedup_rms is not None:
        keep = test_keep_mask(name, x_full, x_test, spec.dedup_rms, split_dir)
        n_test_removed = int((~keep).sum())
        test_idx = test_idx[keep]

    if debug:
        per_class = config.DEBUG_PER_CLASS
        train_idx = train_idx[stratified_subsample(y_full[train_idx], per_class["train"], seed)]
        val_idx = val_idx[stratified_subsample(y_full[val_idx], per_class["val"], seed)]
        test_idx = test_idx[stratified_subsample(y_test[test_idx], per_class["test"], seed)]

    return ImageDataset(
        name=name,
        class_names=spec.class_names,
        x_train=x_full[train_idx], y_train=y_full[train_idx],
        x_val=x_full[val_idx], y_val=y_full[val_idx],
        x_test=x_test[test_idx], y_test=y_test[test_idx],
        train_idx=train_idx, val_idx=val_idx,
        test_idx=test_idx, n_test_removed=n_test_removed,
        seed=seed, debug=debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, verify and summarise a dataset.")
    parser.add_argument("--dataset", choices=sorted(config.DATASETS), default="fashion_mnist")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    ds = load_dataset(args.dataset, debug=True if args.debug else None)  # None -> PRJ_DEBUG
    for key, value in ds.summary().items():
        print(f"{key:>13}: {value}")


if __name__ == "__main__":
    main()

"""Download, verify, parse and split the image-classification datasets.

Images are returned as uint8 arrays of shape (N, H, W, C); grayscale datasets
use C = 1. Labels are int64 in [0, 9]. Every split is deterministic given the
seed. The split indices and the near-duplicate masks are cached under data/splits/;
each cache file is named after a fingerprint of the data and of every parameter that
produced it, so a stale cache can never be silently reused for different inputs.
"""
from __future__ import annotations

import argparse
import codecs
import gzip
import hashlib
import io
import json
import pickle
import pickletools
import re
import shutil
import struct
import tarfile
import urllib.error
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

# Bump when the splitting / filtering algorithm changes; it is part of every cache fingerprint.
SPLIT_VERSION = 2


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


def _content_range_start(response) -> Optional[int]:
    match = re.match(r"bytes (\d+)-", response.headers.get("Content-Range", ""))
    return int(match.group(1)) if match else None


def download_file(remote: RemoteFile, dest_dir: Path, _retry: bool = True) -> Path:
    """Download `remote` into `dest_dir` (resuming a partial download) and verify it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / remote.name
    if dest.exists():
        verify_file(dest, remote)
        return dest

    part = dest.with_name(dest.name + ".part")
    resume_from = part.stat().st_size if part.exists() else 0

    def restart() -> Path:
        part.unlink(missing_ok=True)
        if not _retry:
            raise RuntimeError(f"{remote.name}: could not resume or restart the download")
        return download_file(remote, dest_dir, _retry=False)

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(remote.url, headers=headers), timeout=60
        )
    except urllib.error.HTTPError as error:
        if error.code != 416 or not resume_from:
            raise
        # "Range not satisfiable": the partial file may already be the complete file.
        try:
            verify_file(part, remote)
        except ChecksumError:
            return restart()
        part.replace(dest)
        return dest

    with response:
        append = resume_from > 0 and response.status == 206
        if append and _content_range_start(response) != resume_from:
            return restart()  # the server answered a different range than we asked for
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
# Parsers
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


# Opcodes that bypass find_class (extension registry, persistent ids) or build arbitrary
# instances. The ciFAIR-10 files use none of them.
_FORBIDDEN_PICKLE_OPCODES = frozenset(
    {"EXT1", "EXT2", "EXT4", "INST", "OBJ", "PERSID", "BINPERSID"}
)
_MAX_PICKLE_BYTES = 64 * 1024 * 1024  # ciFAIR-10's largest member is ~44 MB


def _check_pickle_opcodes(raw: bytes) -> None:
    for opcode, _, _ in pickletools.genops(raw):
        if opcode.name in _FORBIDDEN_PICKLE_OPCODES:
            raise pickle.UnpicklingError(f"blocked pickle opcode {opcode.name}")


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only resolves the globals a numpy array pickle needs.

    Scanning the ciFAIR-10 files' opcodes (without executing them) shows they reference
    exactly these globals. This is NOT a general security boundary for untrusted pickles: it
    is one layer for loading a trusted, checksum-pinned artifact, together with the opcode
    filter, the size limit and the schema checks in `_parse_cifair_batch`.
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


def _parse_cifair_batch(zf: zipfile.ZipFile, member: str, expected_md5: Optional[str] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Read one ciFAIR-10 pickle batch from the zip and validate it before use."""
    if zf.getinfo(member).file_size > _MAX_PICKLE_BYTES:
        raise ValueError(f"{member}: larger than the {_MAX_PICKLE_BYTES}-byte limit")
    raw = zf.read(member)
    if expected_md5 is not None and hashlib.md5(raw).hexdigest() != expected_md5:
        raise ChecksumError(f"{member}: md5 differs from the value published by the authors")
    _check_pickle_opcodes(raw)
    batch = _RestrictedUnpickler(io.BytesIO(raw), encoding="latin1").load()
    if not isinstance(batch, dict):
        raise ValueError(f"{member}: expected a dict, got {type(batch).__name__}")
    data = batch["data"] if "data" in batch else batch.get(b"data")
    labels = batch["labels"] if "labels" in batch else batch.get(b"labels")
    if not (isinstance(data, np.ndarray) and data.dtype == np.uint8
            and data.ndim == 2 and data.shape[1] == 3072):
        raise ValueError(f"{member}: 'data' is not a (N, 3072) uint8 array")
    if not isinstance(labels, (list, np.ndarray)):
        raise ValueError(f"{member}: 'labels' must be a list or array")
    labels = np.asarray(labels)  # ragged input raises ValueError here
    if (labels.ndim != 1 or labels.size == 0 or len(labels) != len(data)
            or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError(f"{member}: 'labels' must be a one-dimensional integer vector "
                         f"matching 'data'")
    labels = labels.astype(np.int64)
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError(f"{member}: labels outside [0, 9]")
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(images), labels


def _load_cifair10(raw_dir: Path) -> Tuple[np.ndarray, ...]:
    archive = download_file(config.CIFAIR10.files[0], raw_dir)
    with zipfile.ZipFile(archive) as zf:
        train = [_parse_cifair_batch(zf, f"ciFAIR-10/data_batch_{i}") for i in range(1, 6)]
        x_test, y_test = _parse_cifair_batch(zf, "ciFAIR-10/test_batch",
                                             config.CIFAIR10_TEST_BATCH_MD5)
    x_train = np.concatenate([x for x, _ in train])
    y_train = np.concatenate([y for _, y in train])
    return x_train, y_train, x_test, y_test


_LOADERS = {"fashion_mnist": _load_fashion_mnist, "cifar10": _load_cifar10,
            "cifair10": _load_cifair10}
_OFFICIAL_SIZES = {"fashion_mnist": (60_000, 10_000), "cifar10": (50_000, 10_000),
                   "cifair10": (50_000, 10_000)}


# --------------------------------------------------------------------------- #
# Splits and near-duplicate masks (fingerprinted caches)
# --------------------------------------------------------------------------- #
def array_digest(*arrays: np.ndarray) -> str:
    """SHA-256 over dtype, shape and bytes, so the digest changes with content AND order."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(array)
    return digest.hexdigest()


def _fingerprint(**params: object) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def _threshold_repr(threshold: Optional[float]) -> Optional[str]:
    # repr() round-trips a float exactly, unlike "{:g}", which collapses nearby values.
    return None if threshold is None else repr(float(threshold))


def make_split(y_train: np.ndarray, n_val: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/validation split of the official training set (index arrays)."""
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=n_val, stratify=y_train, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def _validate_split(train_idx: np.ndarray, val_idx: np.ndarray, removed_idx: np.ndarray,
                    n_total: int, n_val: int) -> None:
    parts = {"train_idx": train_idx, "val_idx": val_idx, "val_removed_idx": removed_idx}
    for name, idx in parts.items():
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"split cache: {name} must be a 1-D integer array")
        if len(idx) and (idx.min() < 0 or idx.max() >= n_total):
            raise ValueError(f"split cache: {name} has indices outside [0, {n_total})")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"split cache: {name} contains repeated indices")
    everything = np.concatenate(list(parts.values()))
    if len(np.unique(everything)) != len(everything):
        raise ValueError("split cache: train / val / removed indices overlap")
    if len(everything) != n_total:
        raise ValueError("split cache: indices do not cover the official training set")
    if len(val_idx) + len(removed_idx) != n_val:
        raise ValueError(f"split cache: validation size is not the requested n_val={n_val}")


def load_or_create_split(name: str, x_full: np.ndarray, y_full: np.ndarray, n_val: int, seed: int,
                         val_dedup_rms: Optional[float], data_digest: str,
                         split_dir: Path = config.SPLIT_DIR
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split of the official train set, with val near-duplicates dropped.

    Validation images whose nearest TRAINING image is within `val_dedup_rms` (RMS on the
    [0, 1] scale) are removed from the validation set, so clean-validation hyperparameter
    selection cannot reward memorising near-copies. Returns (train_idx, val_idx, removed_idx).
    The result is cached in a file named after a fingerprint of the data and all parameters.
    """
    fingerprint = _fingerprint(version=SPLIT_VERSION, name=name, seed=seed, n_val=n_val,
                               val_dedup_rms=_threshold_repr(val_dedup_rms), data=data_digest)
    path = split_dir / f"{name}_split_seed{seed}_{fingerprint[:12]}.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        if str(saved["fingerprint"]) != fingerprint:
            raise ValueError(f"{path.name}: fingerprint inside the file does not match its name")
        train_idx, val_idx, removed_idx = (saved[k] for k in ("train_idx", "val_idx", "val_removed_idx"))
    else:
        train_idx, base_val = make_split(y_full, n_val, seed)
        val_idx, removed_idx = base_val, np.empty(0, dtype=base_val.dtype)
        if val_dedup_rms is not None:
            distance, _ = nearest_neighbour_distance(x_full[base_val], x_full[train_idx])
            keep = distance > val_dedup_rms
            val_idx, removed_idx = base_val[keep], base_val[~keep]
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, train_idx=train_idx, val_idx=val_idx, val_removed_idx=removed_idx,
                 fingerprint=np.array(fingerprint))
    _validate_split(train_idx, val_idx, removed_idx, len(y_full), n_val)
    return train_idx, val_idx, removed_idx


def stratified_subsample(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Sorted indices of `per_class` randomly chosen samples from every class."""
    rng = np.random.default_rng(seed)
    chosen = [
        rng.permutation(np.flatnonzero(y == c))[:per_class] for c in np.unique(y)
    ]
    return np.sort(np.concatenate(chosen))


def load_or_create_test_keep_mask(name: str, x_train_full: np.ndarray, x_test: np.ndarray,
                                  threshold: float, data_digest: str,
                                  split_dir: Path = config.SPLIT_DIR) -> np.ndarray:
    """Boolean mask over the official test set: False = near-duplicate of a training image.

    Compared against the FULL official training set (train + val, whatever the seed or
    debug setting), so the mask depends only on the data and the threshold. Cached in a file
    named after a fingerprint of both.
    """
    fingerprint = _fingerprint(version=SPLIT_VERSION, name=name, kind="test_keep",
                               threshold=_threshold_repr(threshold), data=data_digest)
    path = split_dir / f"{name}_test_keep_{fingerprint[:12]}.npy"
    if path.exists():
        keep = np.load(path, allow_pickle=False)
    else:
        distance, _ = nearest_neighbour_distance(x_test, x_train_full)
        keep = distance > threshold
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, keep)
    if keep.dtype != np.bool_ or keep.shape != (len(x_test),):
        raise ValueError(f"{path.name}: expected a boolean mask of length {len(x_test)}")
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
    n_val_removed: int  # validation images dropped as near-duplicates of training images
    n_test_removed: int  # official test images dropped as near-duplicates of training images
    seed: int
    debug: bool

    def summary(self) -> Dict[str, object]:
        n_classes = len(self.class_names)
        return {
            "dataset": self.name,
            "seed": self.seed,
            "debug": self.debug,
            "val_near_duplicates_removed": self.n_val_removed,
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
                 dedup_test: Optional[bool] = None, dedup_val: Optional[bool] = None,
                 raw_dir: Path = config.RAW_DIR, split_dir: Path = config.SPLIT_DIR) -> ImageDataset:
    """Load `name` with a stratified train/val split of the official train set.

    For datasets that define a `dedup_rms` threshold, `dedup_test` (default
    config.DEDUP_TEST) removes official test images that are near-duplicates of a
    train/val image, and `dedup_val` (default config.DEDUP_VAL) removes validation images
    that are near-duplicates of a training image. Datasets without a threshold (the
    original CIFAR-10) are used as is. In debug mode every split is replaced by a small
    stratified subset of itself, so debug train/val/test remain disjoint.
    """
    if name not in config.DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(config.DATASETS)}")
    spec: DatasetSpec = config.DATASETS[name]
    debug = config.DEBUG_MODE if debug is None else debug
    dedup_test = config.DEDUP_TEST if dedup_test is None else dedup_test
    dedup_val = config.DEDUP_VAL if dedup_val is None else dedup_val

    x_full, y_full, x_test, y_test = _LOADERS[name](raw_dir / name)
    expected_train, expected_test = _OFFICIAL_SIZES[name]
    if (len(y_full), len(y_test)) != (expected_train, expected_test):
        raise ValueError(f"{name}: unexpected official split sizes {len(y_full)}/{len(y_test)}")
    if x_full.shape[1:] != spec.image_shape or x_test.shape[1:] != spec.image_shape:
        raise ValueError(f"{name}: unexpected image shape {x_full.shape[1:]}")
    data_digest = array_digest(x_full, y_full, x_test, y_test)

    train_idx, val_idx, val_removed = load_or_create_split(
        name, x_full, y_full, spec.n_val, seed,
        spec.dedup_rms if dedup_val else None, data_digest, split_dir)
    test_idx = np.arange(len(y_test))
    n_test_removed = 0
    if dedup_test and spec.dedup_rms is not None:
        keep = load_or_create_test_keep_mask(name, x_full, x_test, spec.dedup_rms,
                                             data_digest, split_dir)
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
        test_idx=test_idx, n_val_removed=len(val_removed), n_test_removed=n_test_removed,
        seed=seed, debug=debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, verify and summarise a dataset.")
    parser.add_argument("--dataset", choices=sorted(config.DATASETS), default="fashion_mnist")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    ds = load_dataset(args.dataset, debug=True if args.debug else None)  # None -> PRJ_DEBUG
    for key, value in ds.summary().items():
        print(f"{key:>28}: {value}")


if __name__ == "__main__":
    main()

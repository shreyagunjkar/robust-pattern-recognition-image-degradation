"""Controlled image degradations: Gaussian noise, Gaussian blur, contrast, brightness.

Definitions (also documented in docs/degradations.md):

Arithmetic   Images are uint8 (N, H, W, C). They are converted to float32 on [0, 1] (v / 255),
             degraded in float32, CLIPPED to [0, 1], multiplied by 255 and ROUNDED to the nearest
             integer (ties to even, numpy.rint), then stored as uint8. Severity 0 is the identity:
             the input array is returned unchanged (a copy), with no float round trip.
Noise        x' = x + sigma * z with z ~ N(0, 1) drawn independently for every pixel AND channel.
             z depends only on (seed, realisation, split stream, original image id), never on
             batching or order, and is shared across severities (sigma scales one fixed field).
Blur         Separable Gaussian over the two SPATIAL axes (H, W) only, each channel independently
             (never across channels or images). Kernel radius r = ceil(3 * sigma) (truncated at
             3 sigma, size 2r + 1), weights normalised to sum 1, borders mirrored without repeating
             the edge pixel (reflect-101, numpy "reflect"), so flat regions keep their value.
Contrast     x' = m + alpha * (x - m), m = ONE scalar per image: the mean of the clean float image
             over all pixels and all channels (H, W, C). For RGB the colour differences between
             channels are therefore scaled by alpha as well (alpha -> 0 gives a uniform grey).
             For 0 <= alpha <= 1 the result stays inside [0, 1], so clipping never acts.
Brightness   x' = x + beta (beta > 0 only: brightening, with saturation at 1 by clipping).

What the freeze guard does and does not detect
  The frozen manifest (results/metrics/severity_ladder_frozen.json) stores a fingerprint of the
  DECLARED specification: DEGRADATION_VERSION, severity names, the four ladders, the arithmetic
  description, STREAM_IDS and RNG_SPEC. degrade() and get_degraded() refuse to run if the manifest
  is missing or its fingerprint differs from the current specification.
  The fingerprint does NOT cover the code. Editing an implementation so that pixels change, without
  touching one of the declared items above, goes undetected: whoever changes the semantics must bump
  DEGRADATION_VERSION by hand, re-run the audit and re-freeze. No automatic implementation check
  exists in the repository (the local, git-ignored test suite has pixel-level regression tests that
  catch some such edits, but a clone does not carry them).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from . import config
from .data_loader import array_digest
from .preprocessing import to_unit_float

DEGRADATION_VERSION = 1

FAMILIES: Tuple[str, ...] = ("gaussian_noise", "gaussian_blur", "contrast", "brightness")
SEVERITY_NAMES: Tuple[str, ...] = ("clean", "mild", "moderate", "high", "very_high", "extreme")

# Severity ladders, levels 1..5. Provisional values from the Phase 1 design, audited with
# PSNR/SSIM on a clean-training calibration subset (src/calibration.py) and then frozen
# (results/metrics/severity_ladder_frozen.json) before any model is trained or evaluated.
LADDERS: Dict[str, Tuple[float, ...]] = {
    "gaussian_noise": (0.05, 0.10, 0.20, 0.35, 0.50),  # sigma, on the [0, 1] intensity scale
    "gaussian_blur": (0.5, 1.0, 1.5, 2.0, 3.0),        # sigma, in pixels
    "contrast": (0.75, 0.50, 0.30, 0.15, 0.05),        # alpha (1 = unchanged)
    "brightness": (0.10, 0.20, 0.30, 0.40, 0.50),      # beta, added on the [0, 1] intensity scale
}

# Separate random streams so that train / val / test images with equal ids get unrelated noise.
STREAM_IDS = {"test": 0, "train": 1, "val": 2}

# The concrete noise generator. Part of the frozen identity and of every cache identity, because a
# different bit generator, entropy layout or draw method gives different pixels.
RNG_SPEC = {
    "bit_generator": "numpy.random.default_rng (PCG64, seeded through numpy SeedSequence)",
    "entropy": ["seed", "realisation", "stream_id (STREAM_IDS)", "original image id"],
    "draw": "Generator.standard_normal(shape=(H, W, C), dtype=float32), C order, one generator per image",
    "use": "x + sigma * z; one fixed field per (seed, realisation, stream, image) shared by all severities",
}

FROZEN_LADDER_PATH = config.METRICS_DIR / "severity_ladder_frozen.json"

_ARITHMETIC_SPEC = {
    "float": "float32 on [0, 1]",
    "clip_then_round": "clip to [0, 1], x255, numpy.rint (ties to even), uint8",
    "noise": "sigma * N(0,1) float32 per pixel and channel; seed=[seed, realisation, stream, image_id]",
    "blur": "separable, axes (H, W) per channel, radius ceil(3*sigma), normalised, mirror borders",
    "contrast": "m + alpha*(x - m), m = scalar mean over (H, W, C) of the clean image",
    "brightness": "x + beta, clipped",
}


class FreezeGuardError(RuntimeError):
    """Base class: the degradations may not be used because the frozen record does not vouch for them."""


class LadderNotFrozenError(FreezeGuardError):
    """The frozen manifest is missing or unreadable."""


class LadderChangedError(FreezeGuardError):
    """The current specification differs from the frozen manifest."""


# --------------------------------------------------------------------------- #
# Ladder bookkeeping and the freeze guard
# --------------------------------------------------------------------------- #
def severity_value(family: str, severity: int) -> Optional[float]:
    """Parameter of `family` at `severity` (1..5); None for severity 0 (clean)."""
    if severity not in range(0, 6):
        raise ValueError(f"severity must be 0..5, got {severity!r}")
    if severity == 0:  # identity for "clean" or any valid family
        if family != "clean" and family not in LADDERS:
            raise KeyError(f"unknown degradation family {family!r}; choose from {FAMILIES}")
        return None
    if family not in LADDERS:
        raise KeyError(f"unknown degradation family {family!r}; choose from {FAMILIES}")
    return LADDERS[family][severity - 1]


def conditions() -> List[Tuple[str, int]]:
    """The 21 evaluation conditions: clean, then 4 families x severities 1..5."""
    return [("clean", 0)] + [(family, level) for family in FAMILIES for level in range(1, 6)]


def condition_name(family: str, severity: int) -> str:
    return "clean" if severity == 0 else f"{family}_s{severity}"


def ladder_spec() -> Dict[str, object]:
    """The declared specification that the frozen fingerprint covers (JSON-serialisable)."""
    return {"version": DEGRADATION_VERSION, "names": list(SEVERITY_NAMES),
            "ladders": {f: list(v) for f, v in LADDERS.items()}, "arithmetic": _ARITHMETIC_SPEC,
            "stream_ids": STREAM_IDS, "rng": RNG_SPEC}


def _spec_fingerprint(spec: Dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def ladder_fingerprint() -> str:
    return _spec_fingerprint(ladder_spec())


def verify_frozen_ladder(path: Optional[Path] = None, *, allow_missing: bool = False) -> None:
    """Raise unless the frozen manifest exists and matches the current specification.

    `allow_missing=True` is the explicit unfrozen path used while the ladder is being audited: a
    missing manifest is then accepted, but an existing one must still match.
    """
    path = FROZEN_LADDER_PATH if path is None else path
    if not path.exists():
        if allow_missing:
            return
        raise LadderNotFrozenError(
            f"{path.name} not found: the severity ladder has not been frozen. Run the audit and "
            f"`python -m src.calibration --freeze` first (calibration code may call the unguarded "
            f"degrade_float, or degrade(..., allow_unfrozen=True)).")
    try:
        manifest = json.loads(path.read_text())
        frozen = manifest["ladder_fingerprint"]
    except (ValueError, KeyError, TypeError) as exc:
        raise LadderNotFrozenError(f"{path.name} is unreadable or has no ladder_fingerprint: {exc!r}")
    recorded = manifest.get("specification")
    if recorded is not None and _spec_fingerprint(recorded) != frozen:
        raise LadderChangedError(f"{path.name} is internally inconsistent: its specification does "
                                 f"not hash to its ladder_fingerprint (edited by hand?)")
    if frozen != ladder_fingerprint():
        current = json.loads(json.dumps(ladder_spec()))
        differing = sorted(k for k in set(current) | set(recorded)
                           if recorded.get(k) != current.get(k)) if recorded else []
        raise LadderChangedError(
            f"severity ladder / degradation definitions differ from {path.name}"
            f"{' in: ' + ', '.join(differing) if differing else ''}. Results from different "
            f"definitions are not comparable; revert the change or re-run the audit and re-freeze "
            f"before evaluating any model.")


# --------------------------------------------------------------------------- #
# Float <-> uint8 conversion
# --------------------------------------------------------------------------- #
def to_uint8(x: np.ndarray) -> np.ndarray:
    """float [0, 1] -> uint8: clip to [0, 1], scale by 255, round half to even."""
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# The four degradations (float32 in, float32 out, NOT clipped)
# --------------------------------------------------------------------------- #
def noise_field(image_ids: Sequence[int], shape: Tuple[int, ...], seed: int = config.SEED,
                realisation: int = 0, stream: str = "test") -> np.ndarray:
    """Standard-normal float32 field, one independent generator per original image id."""
    field = np.empty((len(image_ids),) + tuple(shape), dtype=np.float32)
    for row, image_id in enumerate(image_ids):
        rng = np.random.default_rng([seed, realisation, STREAM_IDS[stream], int(image_id)])
        field[row] = rng.standard_normal(shape, dtype=np.float32)
    return field


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Normalised 1-D Gaussian, radius ceil(3 sigma), size 2 * radius + 1 (float64)."""
    radius = int(np.ceil(3.0 * sigma))
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (taps / sigma) ** 2)
    return weights / weights.sum()


def gaussian_blur_float(x: np.ndarray, sigma: float) -> np.ndarray:
    """Blur (N, H, W, C) over axes 1 and 2 only; channels and images are never mixed."""
    kernel = gaussian_kernel_1d(sigma)
    out = ndimage.correlate1d(x, kernel, axis=1, mode="mirror", output=np.float32)
    return ndimage.correlate1d(out, kernel, axis=2, mode="mirror", output=np.float32)


def contrast_float(x: np.ndarray, alpha: float) -> np.ndarray:
    """m + alpha (x - m) with ONE scalar mean per image over all of (H, W, C)."""
    mean = x.mean(axis=(1, 2, 3), dtype=np.float64, keepdims=True).astype(np.float32)
    return mean + np.float32(alpha) * (x - mean)


def _degrade_float(x_unit: np.ndarray, family: str, value: float, image_ids: Sequence[int],
                   seed: int, realisation: int, stream: str) -> np.ndarray:
    if family == "gaussian_noise":
        z = noise_field(image_ids, x_unit.shape[1:], seed, realisation, stream)
        return x_unit + np.float32(value) * z
    if family == "gaussian_blur":
        return gaussian_blur_float(x_unit, value)
    if family == "contrast":
        return contrast_float(x_unit, value)
    if family == "brightness":
        return x_unit + np.float32(value)
    raise KeyError(f"unknown degradation family {family!r}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _check(x: np.ndarray, image_ids: Sequence[int]) -> np.ndarray:
    if x.dtype != np.uint8 or x.ndim != 4:
        raise TypeError(f"expected a uint8 array of shape (N, H, W, C), got {x.dtype} {x.shape}")
    ids = np.asarray(image_ids)
    if ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer) or ids.min(initial=0) < 0:
        raise ValueError("image_ids must be one non-negative integer (original image id) per image")
    return ids.astype(np.int64)


def _check_chunk(chunk: object) -> int:
    """A chunk size must be a positive integer (not a bool); checked before any allocation."""
    if isinstance(chunk, (bool, np.bool_)) or not isinstance(chunk, (int, np.integer)):
        raise TypeError(f"chunk must be a positive integer, got {chunk!r} ({type(chunk).__name__})")
    if chunk <= 0:
        raise ValueError(f"chunk must be a positive integer, got {chunk}")
    return int(chunk)


def degrade_float(x: np.ndarray, family: str, severity: int, image_ids: Sequence[int], *,
                  seed: int = config.SEED, realisation: int = 0, stream: str = "test",
                  chunk: int = 1000) -> np.ndarray:
    """Un-clipped float32 degraded images on the [0, 1] scale (used for clipping statistics).

    This is the calibration path: it does NOT consult the freeze guard, so the ladder can be
    audited before it is frozen. Model evaluation must use `get_degraded` / `degrade` instead.
    """
    chunk = _check_chunk(chunk)
    ids = _check(x, image_ids)
    value = severity_value(family, severity)
    if value is None:
        return to_unit_float(x)
    out = np.empty(x.shape, dtype=np.float32)
    for start in range(0, len(x), chunk):
        stop = start + chunk
        out[start:stop] = _degrade_float(to_unit_float(x[start:stop]), family, value,
                                         ids[start:stop], seed, realisation, stream)
    return out


def degrade(x: np.ndarray, family: str, severity: int, image_ids: Sequence[int], *,
            seed: int = config.SEED, realisation: int = 0, stream: str = "test",
            chunk: int = 1000, allow_unfrozen: bool = False) -> np.ndarray:
    """Degrade uint8 images (N, H, W, C); returns uint8 of the same shape.

    `image_ids` are the ORIGINAL ids of the images (their positions in the official train or
    test set), so a given image receives the same noise whatever the batch, order or subset.
    Severity 0 returns an exact copy of `x`.

    Refuses to run unless the frozen manifest exists and matches the current specification.
    `allow_unfrozen=True` (illustrations, calibration) tolerates a MISSING manifest only.
    """
    _check_chunk(chunk)
    _check(x, image_ids)
    verify_frozen_ladder(allow_missing=allow_unfrozen)
    if severity_value(family, severity) is None:
        return x.copy()
    return to_uint8(degrade_float(x, family, severity, image_ids, seed=seed,
                                  realisation=realisation, stream=stream, chunk=chunk))


# --------------------------------------------------------------------------- #
# Fingerprinted cache
# --------------------------------------------------------------------------- #
def degradation_fingerprint(*, source_digest: str, ids_digest: str, family: str, severity: int,
                            seed: int, realisation: int, stream: str) -> str:
    params: Dict[str, object] = {
        "version": DEGRADATION_VERSION, "family": family, "severity": severity,
        "value": repr(severity_value(family, severity)), "source": source_digest, "ids": ids_digest,
        "specification": ladder_fingerprint(),  # ladders, arithmetic, STREAM_IDS and RNG_SPEC
    }
    if family == "gaussian_noise":  # the other families are deterministic: seed is irrelevant
        # numpy does not promise identical Generator streams across versions, so the numpy
        # version is part of the noise identity (an upgrade regenerates instead of reusing).
        params.update(seed=seed, realisation=realisation, stream=stream,
                      stream_id=STREAM_IDS[stream], numpy=np.__version__)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def get_degraded(x: np.ndarray, image_ids: Sequence[int], family: str, severity: int, *,
                 dataset: str, split: str, seed: int = config.SEED, realisation: int = 0,
                 cache_dir: Path = config.CACHE_DIR,
                 source_digest: Optional[str] = None) -> np.ndarray:
    """Cached `degrade`. Every model must obtain its degraded images through this function.

    The cache file name embeds a fingerprint of the clean source array, the image ids, the
    degradation parameters, the declared specification (ladders, arithmetic, STREAM_IDS, RNG_SPEC,
    DEGRADATION_VERSION) and, for noise, seed / realisation / stream id / numpy version. It does
    not cover the implementation code (see the module docstring). Requires the frozen manifest.
    """
    ids = _check(x, image_ids)
    verify_frozen_ladder()
    if severity_value(family, severity) is None:
        return x.copy()
    fingerprint = degradation_fingerprint(
        source_digest=source_digest or array_digest(x), ids_digest=array_digest(ids),
        family=family, severity=severity, seed=seed, realisation=realisation, stream=split)
    path = cache_dir / f"{dataset}_{split}_{condition_name(family, severity)}_{fingerprint[:12]}.npy"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if cached.dtype != np.uint8 or cached.shape != x.shape:
            raise ValueError(f"{path.name}: cached array has dtype {cached.dtype}, shape "
                             f"{cached.shape}; expected uint8 {x.shape}")
        return cached
    degraded = degrade(x, family, severity, ids, seed=seed, realisation=realisation, stream=split)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, degraded)
    return degraded


def iter_degraded(x: np.ndarray, image_ids: Sequence[int], *, dataset: str, split: str,
                  seed: int = config.SEED, realisation: int = 0,
                  cache_dir: Path = config.CACHE_DIR
                  ) -> Iterator[Tuple[str, int, np.ndarray]]:
    """Yield (family, severity, uint8 images) for all 21 conditions, clean first."""
    digest = array_digest(x)
    for family, severity in conditions():
        yield family, severity, get_degraded(
            x, image_ids, family, severity, dataset=dataset, split=split, seed=seed,
            realisation=realisation, cache_dir=cache_dir, source_digest=digest)

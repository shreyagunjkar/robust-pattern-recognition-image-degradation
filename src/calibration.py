"""Model-independent audit of the severity ladders (no model is trained or evaluated here).

Calibration subset  A fixed stratified subset of CLEAN TRAINING images (100 per class, seed =
                    config.SEED), identified by their official training-set ids. Validation and
                    test images are never used.
Metrics (settings are fixed and recorded in the output)
  PSNR   per image, 10 log10(255^2 / MSE) on the uint8 images, MSE averaged over all pixels and
         channels; infinite when the image is unchanged (counted, excluded from percentiles).
  SSIM   skimage.metrics.structural_similarity on uint8 images with data_range=255, the canonical
         Wang et al. setting (Gaussian window sigma=1.5, win_size=11, no sample covariance),
         borders cropped by skimage; RGB is averaged over channels (channel_axis=-1).
  Also   clip fraction (pixel values outside [0, 1] before clipping), mean intensity shift,
         mean per-image intensity range, fraction of changed / constant images and, for blur,
         the -3 dB cutoff of the discrete kernel.
Freeze criteria (declared BEFORE the audit was run; failing any of them blocks the freeze)
  C1  For every dataset and family, median PSNR and mean SSIM strictly decrease from severity 1 to 5.
  C2  At severity 1, at least 99% of calibration images differ from the clean image.
  C3  At severity 5, at most 1% of calibration images are constant (a single grey value).
Action rule: if all criteria hold for every dataset, the provisional ladder is frozen unchanged. If
one fails, only the failing family may be adjusted, the audit is re-run once, and both versions are
recorded.
Matching PSNR/SSIM across families does NOT mean matching recognition difficulty; the metrics only
check that each ladder is monotone and non-degenerate.
"""
from __future__ import annotations

import argparse
import datetime
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from . import config, degradations
from .data_loader import ImageDataset, load_dataset, stratified_subsample

CALIBRATION_PER_CLASS = 100
SSIM_SETTINGS = {"data_range": 255, "gaussian_weights": True, "sigma": 1.5, "win_size": 11,
                 "use_sample_covariance": False}
CRITERIA = {
    "C1": "median PSNR and mean SSIM strictly decrease from severity 1 to 5 (every dataset, family)",
    "C2": "at severity 1, >= 99% of calibration images differ from the clean image",
    "C3": "at severity 5, <= 1% of calibration images are constant",
}
CALIBRATION_DATASETS = ("fashion_mnist", "cifair10")
AUDIT_CSV = config.METRICS_DIR / "severity_calibration.csv"


def calibration_subset(ds: ImageDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(images, official training-set ids, labels) of the fixed clean-training calibration subset."""
    if ds.debug:
        raise ValueError("the calibration subset must be built from the full (non-debug) dataset")
    chosen = stratified_subsample(ds.y_train, CALIBRATION_PER_CLASS, config.SEED)
    return ds.x_train[chosen], ds.train_idx[chosen], ds.y_train[chosen]


def psnr_per_image(clean: np.ndarray, degraded: np.ndarray) -> np.ndarray:
    mse = ((clean.astype(np.float64) - degraded.astype(np.float64)) ** 2).mean(axis=(1, 2, 3))
    with np.errstate(divide="ignore"):
        return np.where(mse > 0, 10.0 * np.log10(255.0 ** 2 / np.where(mse > 0, mse, 1.0)), np.inf)


def ssim_per_image(clean: np.ndarray, degraded: np.ndarray) -> np.ndarray:
    channel_axis = -1 if clean.shape[-1] > 1 else None
    out = np.empty(len(clean), dtype=np.float64)
    for i, (a, b) in enumerate(zip(clean, degraded)):
        if channel_axis is None:
            a, b = a[..., 0], b[..., 0]
        out[i] = structural_similarity(a, b, channel_axis=channel_axis, **SSIM_SETTINGS)
    return out


def blur_cutoff_3db(sigma: float, n_fft: int = 8192) -> float:
    """Frequency (cycles/pixel) where the discrete kernel's amplitude response falls to 1/sqrt(2)."""
    kernel = degradations.gaussian_kernel_1d(sigma)
    response = np.abs(np.fft.rfft(kernel, n_fft))
    freqs = np.fft.rfftfreq(n_fft)
    below = np.flatnonzero(response <= 1.0 / np.sqrt(2.0))[0]
    f0, f1, r0, r1 = freqs[below - 1], freqs[below], response[below - 1], response[below]
    return float(f0 + (1.0 / np.sqrt(2.0) - r0) * (f1 - f0) / (r1 - r0))


def condition_metrics(clean: np.ndarray, ids: np.ndarray, family: str, severity: int,
                      stream: str = "train") -> Dict[str, float]:
    value = degradations.severity_value(family, severity)
    unclipped = degradations.degrade_float(clean, family, severity, ids, stream=stream)
    degraded = degradations.to_uint8(unclipped)
    psnr, ssim = psnr_per_image(clean, degraded), ssim_per_image(clean, degraded)
    finite = psnr[np.isfinite(psnr)]
    per_image_range = degraded.reshape(len(degraded), -1).max(axis=1).astype(int) \
        - degraded.reshape(len(degraded), -1).min(axis=1).astype(int)
    return {
        "param": value,
        "n_images": len(clean),
        "psnr_median": float(np.median(finite)) if len(finite) else np.inf,
        "psnr_p10": float(np.percentile(finite, 10)) if len(finite) else np.inf,
        "psnr_p90": float(np.percentile(finite, 90)) if len(finite) else np.inf,
        "n_unchanged": int((~np.isfinite(psnr)).sum()),
        "ssim_mean": float(ssim.mean()),
        "ssim_median": float(np.median(ssim)),
        "ssim_p10": float(np.percentile(ssim, 10)),
        "ssim_p90": float(np.percentile(ssim, 90)),
        "clip_fraction": float(((unclipped < 0.0) | (unclipped > 1.0)).mean()),
        "mean_shift_levels": float(degraded.mean() - clean.mean()),
        "mean_range_levels": float(per_image_range.mean()),
        "frac_changed": float((degraded != clean).reshape(len(clean), -1).any(axis=1).mean()),
        "frac_constant": float((per_image_range == 0).mean()),
        "cutoff_3db_cycles_per_px": blur_cutoff_3db(value) if family == "gaussian_blur" else np.nan,
    }


def run_audit(datasets=CALIBRATION_DATASETS) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for name in datasets:
        clean, ids, _ = calibration_subset(load_dataset(name, debug=False))
        for family in degradations.FAMILIES:
            for severity in range(1, 6):
                rows.append({"dataset": name, "family": family, "severity": severity,
                             "severity_name": degradations.SEVERITY_NAMES[severity],
                             **condition_metrics(clean, ids, family, severity)})
    return pd.DataFrame(rows)


def evaluate_criteria(audit: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Check C1-C3 on an audit table; returns {criterion: {"passed": bool, "failures": [...]}}."""
    failures: Dict[str, List[str]] = {key: [] for key in CRITERIA}
    for (dataset, family), group in audit.groupby(["dataset", "family"]):
        group = group.sort_values("severity")
        if not (np.all(np.diff(group["psnr_median"]) < 0) and np.all(np.diff(group["ssim_mean"]) < 0)):
            failures["C1"].append(f"{dataset}/{family}")
        if group.iloc[0]["frac_changed"] < 0.99:
            failures["C2"].append(f"{dataset}/{family}")
        if group.iloc[-1]["frac_constant"] > 0.01:
            failures["C3"].append(f"{dataset}/{family}")
    return {key: {"description": CRITERIA[key], "passed": not failures[key], "failures": failures[key]}
            for key in CRITERIA}


def freeze_ladder(audit: pd.DataFrame, criteria: Dict[str, Dict[str, object]]) -> None:
    """Write the frozen-ladder record. Refuses to freeze if any criterion failed."""
    failed = [key for key, result in criteria.items() if not result["passed"]]
    if failed:
        raise RuntimeError(f"cannot freeze: criteria {failed} failed; see evaluate_criteria()")
    import skimage

    record = {
        "ladder_fingerprint": degradations.ladder_fingerprint(),
        "degradation_version": degradations.DEGRADATION_VERSION,
        "frozen_on": datetime.date.today().isoformat(),
        "statement": "Frozen after a model-independent PSNR/SSIM audit on a clean-training subset. "
                     "No model had been trained or evaluated at this point.",
        "ladders": {family: list(values) for family, values in degradations.LADDERS.items()},
        "severity_names": list(degradations.SEVERITY_NAMES),
        "calibration": {"datasets": list(CALIBRATION_DATASETS), "per_class": CALIBRATION_PER_CLASS,
                        "seed": config.SEED, "source": "clean training images",
                        "ssim_settings": SSIM_SETTINGS, "skimage_version": skimage.__version__,
                        "audit_csv": AUDIT_CSV.name},
        "criteria": criteria,
    }
    degradations.FROZEN_LADDER_PATH.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit (and optionally freeze) the severity ladder.")
    parser.add_argument("--freeze", action="store_true", help="write severity_ladder_frozen.json if all criteria pass")
    args = parser.parse_args()
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    audit = run_audit()
    audit.to_csv(AUDIT_CSV, index=False, float_format="%.5g")
    criteria = evaluate_criteria(audit)
    for key, result in criteria.items():
        print(f"{key}: {'PASS' if result['passed'] else 'FAIL'}  {result['description']}  {result['failures'] or ''}")
    if args.freeze:
        freeze_ladder(audit, criteria)
        print(f"frozen -> {degradations.FROZEN_LADDER_PATH}")


if __name__ == "__main__":
    main()

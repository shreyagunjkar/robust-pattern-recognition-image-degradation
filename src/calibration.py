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
Enforcement  freeze_ladder() takes ONLY the audit table. It validates the table (exactly the 2
             datasets x 4 families x 5 severities, no missing / duplicate / unexpected rows, expected
             sample counts, finite and in-range metrics, parameters equal to the current ladder) and
             recomputes C1-C3 itself; no caller-supplied verdict is accepted.
Matching PSNR/SSIM across families does NOT mean matching recognition difficulty; the metrics only
check that each ladder is monotone and non-degenerate.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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
REQUIRED_CRITERIA = ("C1", "C2", "C3")
CALIBRATION_DATASETS = ("fashion_mnist", "cifair10")
N_CLASSES = 10  # both calibration datasets have ten classes
CALIBRATION_N_IMAGES = CALIBRATION_PER_CLASS * N_CLASSES
AUDIT_CSV = config.METRICS_DIR / "severity_calibration.csv"
AUDIT_CSV_FLOAT_FORMAT = "%.5g"


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


class AuditError(ValueError):
    """The audit table is not the complete, well-formed audit of the current ladder."""


class FreezeRefused(RuntimeError):
    """A freeze criterion failed, so the ladder may not be frozen."""


_KEY_COLUMNS = ("dataset", "family", "severity")
_METRIC_COLUMNS = ("param", "n_images", "psnr_median", "psnr_p10", "psnr_p90", "n_unchanged",
                   "ssim_mean", "ssim_median", "ssim_p10", "ssim_p90", "clip_fraction",
                   "mean_shift_levels", "mean_range_levels", "frac_changed", "frac_constant",
                   "cutoff_3db_cycles_per_px")
AUDIT_COLUMNS = _KEY_COLUMNS + ("severity_name",) + _METRIC_COLUMNS


def expected_audit_keys(datasets=CALIBRATION_DATASETS) -> List[Tuple[str, str, int]]:
    return [(d, f, s) for d in datasets for f in degradations.FAMILIES for s in range(1, 6)]


def validate_audit(audit: pd.DataFrame, datasets=CALIBRATION_DATASETS) -> pd.DataFrame:
    """Return the audit sorted canonically, or raise AuditError listing every problem found.

    Requires exactly one row per (dataset, family, severity) of the expected grid (2 x 4 x 5), the
    expected number of calibration images per row, finite in-range metrics, and parameters and
    severity names equal to the CURRENT ladder (so an audit of a different ladder cannot freeze it).
    """
    if not isinstance(audit, pd.DataFrame):
        raise AuditError(f"audit must be a DataFrame, got {type(audit).__name__}")
    missing_cols = [c for c in AUDIT_COLUMNS if c not in audit.columns]
    if missing_cols:
        raise AuditError(f"audit is missing columns: {missing_cols}")
    problems: List[str] = []

    keys = list(zip(audit["dataset"], audit["family"], audit["severity"]))
    expected = expected_audit_keys(datasets)
    seen: Dict[tuple, int] = {}
    for key in keys:
        seen[key] = seen.get(key, 0) + 1
    duplicated = [k for k, n in seen.items() if n > 1]
    unexpected = [k for k in seen if k not in set(expected)]
    absent = [k for k in expected if k not in seen]
    if duplicated:
        problems.append(f"duplicate rows: {duplicated}")
    if unexpected:
        problems.append(f"unexpected rows: {unexpected}")
    if absent:
        problems.append(f"missing rows: {absent}")
    if len(audit) != len(expected):
        problems.append(f"expected {len(expected)} rows, found {len(audit)}")
    if problems:  # the per-row checks below assume the grid is right
        raise AuditError("; ".join(problems))

    for column in _METRIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(audit[column]) or pd.api.types.is_bool_dtype(audit[column]):
            problems.append(f"column {column!r} is not numeric (dtype {audit[column].dtype})")
    if not pd.api.types.is_integer_dtype(audit["severity"]):
        problems.append("column 'severity' is not an integer column")
    if problems:
        raise AuditError("; ".join(problems))

    def bad(row, why: str) -> None:
        problems.append(f"{row.dataset}/{row.family}/s{row.severity}: {why}")

    for row in audit.itertuples(index=False):
        if row.severity_name != degradations.SEVERITY_NAMES[row.severity]:
            bad(row, f"severity_name {row.severity_name!r} != {degradations.SEVERITY_NAMES[row.severity]!r}")
        if not np.isclose(row.param, degradations.severity_value(row.family, row.severity), rtol=1e-6, atol=0):
            bad(row, f"param {row.param!r} differs from the current ladder "
                     f"{degradations.severity_value(row.family, row.severity)!r}")
        if row.n_images != CALIBRATION_N_IMAGES:
            bad(row, f"n_images {row.n_images!r} != {CALIBRATION_N_IMAGES}")
        finite = {c: getattr(row, c) for c in _METRIC_COLUMNS if c != "cutoff_3db_cycles_per_px"}
        for column, value in finite.items():
            if not np.isfinite(value):
                bad(row, f"{column} is not finite ({value!r})")
        if row.family == "gaussian_blur":
            if not (np.isfinite(row.cutoff_3db_cycles_per_px) and 0 < row.cutoff_3db_cycles_per_px <= 0.5):
                bad(row, f"cutoff_3db_cycles_per_px {row.cutoff_3db_cycles_per_px!r} not in (0, 0.5]")
        elif not np.isnan(row.cutoff_3db_cycles_per_px):
            bad(row, "cutoff_3db_cycles_per_px must be empty for non-blur families")
        if all(np.isfinite(v) for v in finite.values()):
            for column in ("clip_fraction", "frac_changed", "frac_constant"):
                if not 0.0 <= getattr(row, column) <= 1.0:
                    bad(row, f"{column} {getattr(row, column)!r} outside [0, 1]")
            for column in ("ssim_mean", "ssim_median", "ssim_p10", "ssim_p90"):
                if not -1.0 <= getattr(row, column) <= 1.0:
                    bad(row, f"{column} {getattr(row, column)!r} outside [-1, 1]")
            if not 0 <= row.n_unchanged <= row.n_images or row.n_unchanged != int(row.n_unchanged):
                bad(row, f"n_unchanged {row.n_unchanged!r} not an integer in [0, n_images]")
            if not 0.0 <= row.mean_range_levels <= 255.0 or abs(row.mean_shift_levels) > 255.0:
                bad(row, "mean_range_levels / mean_shift_levels outside the 8-bit range")
            if not row.psnr_p10 <= row.psnr_median <= row.psnr_p90:
                bad(row, "PSNR p10 <= median <= p90 violated")
            if not row.ssim_p10 <= row.ssim_median <= row.ssim_p90:
                bad(row, "SSIM p10 <= median <= p90 violated")
            if abs(row.frac_changed - (1.0 - row.n_unchanged / row.n_images)) > 1e-6:
                bad(row, "frac_changed is inconsistent with n_unchanged")
    if problems:
        raise AuditError("; ".join(problems[:20]) + (f"; ... (+{len(problems) - 20} more)" if len(problems) > 20 else ""))
    order = {k: i for i, k in enumerate(expected)}
    return audit.assign(_o=[order[k] for k in keys]).sort_values("_o").drop(columns="_o").reset_index(drop=True)


def evaluate_criteria(audit: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Check C1-C3 on a VALIDATED audit table; returns {criterion: {"passed": bool, "failures": [...]}}."""
    audit = validate_audit(audit)
    failures: Dict[str, List[str]] = {key: [] for key in CRITERIA}
    for (dataset, family), group in audit.groupby(["dataset", "family"], sort=False):
        group = group.sort_values("severity")
        if not (np.all(np.diff(group["psnr_median"]) < 0) and np.all(np.diff(group["ssim_mean"]) < 0)):
            failures["C1"].append(f"{dataset}/{family}")
        if group.iloc[0]["frac_changed"] < 0.99:
            failures["C2"].append(f"{dataset}/{family}")
        if group.iloc[-1]["frac_constant"] > 0.01:
            failures["C3"].append(f"{dataset}/{family}")
    return {key: {"description": CRITERIA[key], "passed": not failures[key], "failures": failures[key]}
            for key in CRITERIA}


def serialise_audit(audit: pd.DataFrame) -> str:
    """The exact CSV text written to AUDIT_CSV (its SHA-256 is recorded in the frozen manifest)."""
    return audit.to_csv(index=False, float_format=AUDIT_CSV_FLOAT_FORMAT, lineterminator="\n")


def freeze_ladder(audit: pd.DataFrame) -> None:
    """Validate the audit, recompute C1-C3 from it, and write the frozen-ladder record.

    Takes only the audit table: the criteria are never accepted from the caller. Raises AuditError
    if the table is not the complete audit of the current ladder and FreezeRefused if any
    criterion fails; in both cases nothing is written.
    """
    audit = validate_audit(audit)
    criteria = evaluate_criteria(audit)
    if set(criteria) != set(REQUIRED_CRITERIA):  # an empty or partial evaluation must never pass
        raise FreezeRefused(f"cannot freeze: criteria {sorted(criteria)} evaluated, "
                            f"{list(REQUIRED_CRITERIA)} required")
    failed = {key: result["failures"] for key, result in criteria.items() if not result["passed"]}
    if failed:
        raise FreezeRefused(f"cannot freeze: criteria failed: {failed}")
    import skimage

    previous = None
    if degradations.FROZEN_LADDER_PATH.exists():
        try:
            previous = json.loads(degradations.FROZEN_LADDER_PATH.read_text()).get("ladder_fingerprint")
        except ValueError:
            previous = None
    record = {
        "ladder_fingerprint": degradations.ladder_fingerprint(),
        "supersedes_fingerprint": previous if previous != degradations.ladder_fingerprint() else None,
        "degradation_version": degradations.DEGRADATION_VERSION,
        "frozen_on": datetime.date.today().isoformat(),
        "statement": "Frozen after a model-independent PSNR/SSIM audit on a clean-training subset. "
                     "No model had been trained or evaluated at this point.",
        "ladders": {family: list(values) for family, values in degradations.LADDERS.items()},
        "severity_names": list(degradations.SEVERITY_NAMES),
        "specification": json.loads(json.dumps(degradations.ladder_spec())),
        "calibration": {"datasets": list(CALIBRATION_DATASETS), "per_class": CALIBRATION_PER_CLASS,
                        "n_images_per_row": CALIBRATION_N_IMAGES,
                        "seed": config.SEED, "source": "clean training images",
                        "ssim_settings": SSIM_SETTINGS, "skimage_version": skimage.__version__,
                        "audit_csv": AUDIT_CSV.name,
                        "audit_csv_sha256": hashlib.sha256(serialise_audit(audit).encode()).hexdigest()},
        "criteria": criteria,
    }
    degradations.FROZEN_LADDER_PATH.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit (and optionally freeze) the severity ladder.")
    parser.add_argument("--freeze", action="store_true", help="write severity_ladder_frozen.json if all criteria pass")
    args = parser.parse_args()
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    audit = validate_audit(run_audit())
    with open(AUDIT_CSV, "w", newline="") as handle:
        handle.write(serialise_audit(audit))
    criteria = evaluate_criteria(audit)
    for key, result in criteria.items():
        print(f"{key}: {'PASS' if result['passed'] else 'FAIL'}  {result['description']}  {result['failures'] or ''}")
    if args.freeze:
        freeze_ladder(audit)
        print(f"frozen -> {degradations.FROZEN_LADDER_PATH}")


if __name__ == "__main__":
    main()

# Degradation pipeline (Phase 3)

Status: **implemented and frozen** (`DEGRADATION_VERSION = 1`, frozen 2026-10-30 in
`results/metrics/severity_ladder_frozen.json`). The freeze happened before any model was trained
or evaluated. Code: `src/degradations.py` (transformations, cache, freeze guard),
`src/calibration.py` (audit), `src/visualization.py` (figures).

## Definitions

All arithmetic is done in **float32 on the [0, 1] scale** (uint8 value / 255).

| Step | Definition |
|---|---|
| Clip | Every degraded float image is clipped to [0, 1]. |
| Round | Multiply by 255 and round to the nearest integer, **ties to even** (`numpy.rint`); store as `uint8`. |
| Clean (severity 0) | The identity: an exact copy of the input, with no float round trip. (The round trip is also exact for all 256 values, which is tested.) |
| Gaussian noise | `x' = x + σ·z`, `z ~ N(0, 1)` drawn independently for every pixel **and channel**. σ is on the [0, 1] intensity scale. |
| Gaussian blur | Separable Gaussian over the two **spatial axes (H, W) only**, applied to each channel independently; it never mixes channels or images. Kernel radius `r = ceil(3σ)` (truncated at 3σ, size `2r + 1` = 5, 7, 11, 13, 19), weights normalised to sum 1. Borders are **mirrored without repeating the edge pixel** (reflect-101, the same as OpenCV `BORDER_REFLECT_101`), so flat regions keep their value and borders do not darken. The implementation matches OpenCV's `GaussianBlur` to ~1e-7 (tested). |
| Contrast | `x' = m + α·(x − m)` with **one scalar mean `m` per image**: the mean of the clean float image over all pixels and all channels `(H, W, C)`. For RGB the differences between channels are scaled by α too, so α → 0 gives a uniform grey, not the average colour. (A per-channel mean is a different, also common, convention; it was not used, and we have not verified which convention other benchmarks use.) For 0 ≤ α ≤ 1 the result stays inside [0, 1], so clipping never acts. |
| Brightness | `x' = x + β`, **β > 0 only** (brightening), saturating at 1 through clipping. Darkening is not tested. |

## Reproducibility and identical inputs

- **Noise is keyed by seed and original image id.** Each image gets its own generator,
  `default_rng([seed, realisation, stream, image_id])`, where `image_id` is the image's position in the
  official train or test set and `stream` ∈ {test, train, val}. The noise therefore does not depend
  on batch size, order or subset (tested), and is the same in debug mode, in the primary dataset and in
  every test variant for images that are unchanged.
- **One fixed noise field per image**, scaled by σ, is shared by all five severities (so a severity-4
  image is a stronger version of the same noise pattern, before clipping). `realisation` (default 0)
  selects an independent field; Phase 5 uses it for repeated noise realisations.
- **Every model must obtain degraded images through `get_degraded` / `iter_degraded`.** The cache file
  name embeds a SHA-256 fingerprint of the clean source array, the image ids, the severity parameter,
  and (for noise only) seed, realisation and stream, plus `DEGRADATION_VERSION`. A different source,
  parameter, seed or version gives a different file, so a stale cache is not reused; the loaded array's
  dtype and shape are validated. As in the data caches, file contents are not hashed, so a modified
  but well-formed cache file would not be detected. Phase 5 should record `array_digest` of every
  degraded array next to its results.
- **Freeze guard.** `degrade()` and `get_degraded()` raise `LadderChangedError` if the ladders or
  definitions no longer match the frozen record. Changing anything requires bumping
  `DEGRADATION_VERSION`, re-running the audit and re-freezing, which invalidates comparisons with earlier results.

## Pipeline order

`clean uint8 test image -> degrade (this module) -> uint8 -> [HOG branch: /255 only] or [other models: /255 then (x - mean) / std]`

Degradation always comes **before** HOG and before normalisation. The normalisation constants stay the
frozen clean-training scalars from `fit_norm_stats`; degraded images are never re-normalised with their
own statistics (otherwise a brightness shift would be normalised away; tested). HOG must be computed
from `to_unit_float` output, not from normalised pixels (negative values give non-finite features with
square-root compression).

## Severity ladder and audit

Severity 0 is clean; 1 mild, 2 moderate, 3 high, 4 very high, 5 extreme. The values below are the
Phase 1 provisional values, kept unchanged because the audit passed all criteria.

**Audit design (settings fixed in `src/calibration.py`).** A fixed stratified subset of 1,000
**clean training** images (100 per class, seed 42) per dataset; validation and test images are not
used. PSNR: per image on the uint8 images, `10·log10(255² / MSE)`, MSE over all pixels and channels,
median over images. SSIM: `skimage.metrics.structural_similarity`, `data_range=255`, Gaussian window
σ = 1.5, `win_size=11`, `use_sample_covariance=False`, channel-averaged for RGB, mean over images.
Noise uses realisation 0 only. "Clipped values" is the fraction of pixel values outside [0, 1] before
clipping.

**Criteria, declared before the audit was run:** (C1) median PSNR and mean SSIM strictly decrease from
severity 1 to 5 for every dataset and family; (C2) at severity 1, at least 99% of images differ from
clean; (C3) at severity 5, at most 1% of images are constant. **Result: all three pass on both datasets.**

| Family | Sev | Parameter | PSNR dB (F-MNIST / CIFAR) | SSIM (F-MNIST / CIFAR) | Clipped values (F / C) | Mean shift, grey levels (F / C) |
|---|---|---|---|---|---|---|
| Gaussian noise | 1 | σ = 0.05 | 27.5 / 26.1 | 0.85 / 0.85 | 27.6% / 2.8% | +2.6 / -0.1 |
| Gaussian noise | 2 | σ = 0.1 | 21.6 / 20.2 | 0.73 / 0.66 | 29.4% / 5.4% | +5.2 / -0.0 |
| Gaussian noise | 3 | σ = 0.2 | 16.0 / 14.7 | 0.54 / 0.40 | 33.5% / 12.3% | +9.8 / +0.3 |
| Gaussian noise | 4 | σ = 0.35 | 11.7 / 10.9 | 0.37 / 0.22 | 39.4% / 25.2% | +16.3 / +1.4 |
| Gaussian noise | 5 | σ = 0.5 | 9.3 / 9.1 | 0.26 / 0.15 | 45.8% / 37.4% | +22.3 / +2.5 |
| Gaussian blur | 1 | σ (px) = 0.5 | 27.6 / 34.5 | 0.96 / 0.98 | 0.0% / 0.0% | +0.3 / -0.0 |
| Gaussian blur | 2 | σ (px) = 1 | 19.1 / 25.1 | 0.72 / 0.84 | 0.0% / 0.0% | +0.8 / -0.1 |
| Gaussian blur | 3 | σ (px) = 1.5 | 17.1 / 22.3 | 0.55 / 0.69 | 0.0% / 0.0% | +1.1 / -0.1 |
| Gaussian blur | 4 | σ (px) = 2 | 15.9 / 20.7 | 0.43 / 0.57 | 0.0% / 0.0% | +1.4 / -0.2 |
| Gaussian blur | 5 | σ (px) = 3 | 14.3 / 18.9 | 0.30 / 0.41 | 0.0% / 0.0% | +2.0 / -0.2 |
| Contrast | 1 | α = 0.75 | 21.7 / 25.7 | 0.88 / 0.95 | 0.0% / 0.0% | +0.0 / +0.0 |
| Contrast | 2 | α = 0.5 | 15.7 / 19.7 | 0.71 / 0.80 | 0.0% / 0.0% | -0.0 / +0.0 |
| Contrast | 3 | α = 0.3 | 12.8 / 16.7 | 0.49 / 0.58 | 0.0% / 0.0% | -0.0 / +0.0 |
| Contrast | 4 | α = 0.15 | 11.1 / 15.0 | 0.28 / 0.37 | 0.0% / 0.0% | +0.0 / -0.0 |
| Contrast | 5 | α = 0.05 | 10.1 / 14.1 | 0.14 / 0.21 | 0.0% / 0.0% | +0.0 / -0.0 |
| Brightness | 1 | β = 0.1 | 19.9 / 20.0 | 0.86 / 0.96 | 5.0% / 6.0% | +25.3 / +24.8 |
| Brightness | 2 | β = 0.2 | 14.1 / 14.1 | 0.78 / 0.89 | 13.0% / 12.0% | +48.1 / +47.8 |
| Brightness | 3 | β = 0.3 | 10.9 / 10.8 | 0.69 / 0.79 | 20.9% / 20.6% | +69.1 / +69.4 |
| Brightness | 4 | β = 0.4 | 8.7 / 8.6 | 0.60 / 0.68 | 26.4% / 31.0% | +88.7 / +88.1 |
| Brightness | 5 | β = 0.5 | 7.0 / 7.1 | 0.51 / 0.55 | 31.0% / 44.1% | +107.1 / +104.2 |

Blur −3 dB cutoff of the discrete kernel (cycles/pixel; Nyquist = 0.5), severities 1–5:
0.311, 0.133, 0.088, 0.066, 0.044 (for σ ≥ 1 this is close to the continuous-Gaussian value 0.1325/σ).
Full table: `results/metrics/severity_calibration.csv`. Figures: `results/plots/severity_calibration_*.png`
(curves) and `results/plots/degradation_grid_*.png` (every severity of every family, one image per class).

### What the audit shows

- **Each ladder is monotone and non-degenerate**, which is all the audit is meant to check.
- **PSNR and SSIM disagree across families.** At severity 5, brightness has the lowest PSNR
  (7.0 / 7.1 dB) but the highest SSIM (0.51 / 0.55); contrast has the lowest SSIM on Fashion-MNIST. **Matching or
  ordering by these metrics must not be read as matching recognition difficulty**, and the severity
  scales of different families are not commensurable. Compare models *within* a family; use PSNR/SSIM
  only as a descriptive x-axis.
- **The same parameter does not mean the same quality loss on both datasets.** Blur σ = 1 costs 19.1 dB
  on Fashion-MNIST but 25.1 dB on CIFAR; noise σ = 0.2 gives SSIM 0.54 vs 0.40. Cross-dataset comparisons
  at equal parameters are comparisons of parameters, not of image quality.
- **Ladders are front-loaded in PSNR.** Blur loses 8.5 / 9.3 dB between severity 1 and 2 but only
  1.2–2.8 dB per step afterwards (SSIM still falls steadily, 0.55 → 0.30 on Fashion-MNIST); contrast steps shrink
  from 6.0 dB to 1.0 dB. Whether the upper levels are separable for the models is an empirical question for Phase 5.
- **Noise is not zero-mean after clipping.** On Fashion-MNIST ~50% of pixels are exactly black, so
  27.6% of values are already clipped at σ = 0.05, the mean intensity rises by 2.6 grey levels
  (22.3 at σ = 0.5) and the PSNR (27.5 dB) is above the nominal −20·log10 σ = 26.0 dB. CIFAR is affected much less (+2.5 levels at σ = 0.5).
- **Brightness saturates.** At β = 0.5, 31% (Fashion-MNIST) and 44% (CIFAR) of values are clipped, and the mean rises by ~105 grey levels.
- **Contrast α = 0.05 leaves about 12–13 grey levels** (mean per-image range 12.7 / 11.5 of 255). The
  information is not gone, but uint8 quantisation makes the remaining signal coarse.

## Evaluation protocol records

- **Models are trained and tuned once, on the primary data, and then frozen.** The primary CIFAR
  dataset is ciFAIR-10 (45,000 train / 4,928 validation after near-duplicate filtering). **Every later
  sensitivity analysis must reuse exactly the same tuned, frozen models across all test variants:**
  the original CIFAR-10 test set, ciFAIR-10 without the pixel filter, and the unfiltered Fashion-MNIST
  test set. Do not retrain or retune per variant. The original `cifar10` is a **test-set variant
  only**: its 5,000-image validation set contains the 72 near-duplicates removed from the primary
  validation set and must not be used for tuning.
- Variants differ only in which test images are scored. Because noise is keyed by official test id, an
  image present in several variants receives identical degraded pixels in all of them.
- The test set is evaluated through the frozen ladder only; no severity value may be changed after
  looking at model results.

## Limitations

- Synthetic corruptions on 28×28 grayscale and 32×32 colour images are not real-world degradation.
- Brightness is positive only; contrast uses a single RGB-wide mean; blur is isotropic and ignores
  motion/defocus shapes. Gaussian noise is i.i.d. and independent across channels, unlike sensor noise.
- uint8 storage means every degraded image is quantised; the most extreme contrast levels are coarsely quantised.
- The audit uses 1,000 images per dataset, one noise realisation and no confidence intervals.
- Severity ladders are not equal in "difficulty" across families, and the ladder values were not
  re-tuned: they are the provisional Phase 1 values that passed the audit.
- Caches are fingerprinted but not content-hashed (see above).

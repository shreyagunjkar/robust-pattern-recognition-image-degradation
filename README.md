# Robust Pattern Recognition Under Noise and Image Degradation

A controlled, reproducible comparison of three pattern-recognition approaches
(**LDA**, **SVM + HOG**, **CNN**) trained on clean images and evaluated on
progressively degraded test images (Gaussian noise, Gaussian blur, brightness /
contrast). Datasets: Fashion-MNIST (primary) and CIFAR-10 (replication).

This is a comparative study, not a new algorithm. **Status: Phase 2 of 7 (project
setup and data loading) is complete.** No degradations, models or results exist yet.

## Setup

Developed and tested on Python 3.12 (macOS, CPU). A GPU is optional: the CNN uses CUDA
if PyTorch can see one and falls back to CPU otherwise.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -r requirements-lock.txt (exact versions)
python -m src.data_loader --dataset fashion_mnist [--debug]   # first run downloads and checksum-verifies the data
python -m src.data_loader --dataset cifair10                  # also: cifar10 (original, sensitivity analysis)
python -m src.utils                      # writes results/metrics/environment.json
```

`PRJ_DEBUG=1` (or `debug=True`) switches every loader to small stratified subsets so the
whole pipeline can be tested in minutes before the full run.

## Data

| Key | Source | Train | Val | Test | Role |
|---|---|---|---|---|---|
| `fashion_mnist` | Zalando Research, 28×28 gray | 50,000 | 9,986 | 9,982 | primary |
| `cifair10` | ciFAIR-10 (CIFAR-10, duplicate test images replaced), 32×32 RGB | 45,000 | 4,928 | 9,901 | replication |
| `cifar10` | original CIFAR-10 | 45,000 | 5,000 | 10,000 | sensitivity analysis only |

- Images are `uint8` arrays of shape `(N, H, W, C)` (`C = 1` for Fashion-MNIST), labels `int64`.
- Validation is a stratified split of the official training set (seed 42), after which
  validation images that are near-duplicates of a training image are removed (see below).
  The validation set is used only for hyperparameter selection, on clean images.
- Split indices and near-duplicate masks are cached in `data/splits/`. Each file is named
  after a SHA-256 fingerprint of the image/label arrays and of every parameter that produced
  it (seed, `n_val`, thresholds, algorithm version) and is validated on load (index types,
  bounds, uniqueness, disjointness, coverage; mask dtype and length), so a stale or corrupted
  cache cannot be reused silently. Saved indices fix the split itself across library versions;
  they do not make feature computation or training version-independent.
- Downloads are verified against pinned MD5 and SHA-256 checksums (`src/config.py`).
  The Fashion-MNIST and CIFAR-10 MD5s match the values published by their authors; the
  ciFAIR-10 archive MD5 and `test_batch` MD5 match the values in the authors' official loader.
  All SHA-256 values were computed by us from the verified downloads.
- ciFAIR-10 only ships as pickles, so loading it is handled as reading a trusted,
  checksum-pinned artifact, not as a safe parser for untrusted pickles: checksum first, then
  an opcode filter (rejects extension-registry, persistent-id and instance opcodes, which
  would bypass the global allowlist), a restricted unpickler allowing only the five
  numpy/codec globals the files reference, a size limit, and schema/dtype validation.
  CIFAR-10 and Fashion-MNIST are read from raw bytes, nothing is unpickled.

### Leakage control (train images must never appear in the test set)

No dataset contains byte-identical images across splits (hash check). Near-duplicates do
exist, so the loader removes images whose nearest training image is closer than an RMS
pixel distance (on the [0, 1] scale). The same rule is applied to every model:

| Dataset | Threshold (RMS) | Test images removed | Val images removed |
|---|---|---|---|
| Fashion-MNIST | 0.04 | 18 | 14 |
| ciFAIR-10 | 0.08 (on top of the authors' 308 replacements) | 99 | 72 |
| CIFAR-10 (original) | none | 0 | 0 |

Test sets are therefore **distance-filtered benchmarks**. A pixel threshold flags candidates;
it does not establish that a removed image is the same source image, nor that every
surviving image is source-independent.

**Test-set curation policy (disclosure).** The thresholds were chosen on 2026-10-30 by looking at
ranked test/train image pairs (images only; no model existed and no labels or predictions were
used) and are frozen in `src/config.py` before any model is trained. Fashion-MNIST: pairs up to
~0.04 looked like the same garment, from ~0.045 like different items of the same type. CIFAR-10:
pairs that look identical at 32×32 keep appearing up to ~0.08, interleaved with look-alikes.
Because the test set was inspected to build the benchmark, it is not strictly untouched.

Caveats to report in the paper:
- On CIFAR a pixel threshold cannot separate duplicates from look-alikes, and 246 of the 308
  ciFAIR replacements lie above RMS 0.08, so a pixel filter alone would miss most true
  duplicates. Shifted or rescaled duplicates can remain.
- Many pairs that look identical at 32×32 survive ciFAIR's replacement. We did not check
  whether ciFAIR classifies them as "very similar" rather than same-source duplicates.
- Filtering conditions on similarity to the training set, which changes the evaluated
  population; whether that makes the benchmark easier or harder has not been measured.
- The filtered sets are not exactly class-balanced (Fashion-MNIST test 994–1000 per class,
  ciFAIR-10 test 971–1000), so macro recall no longer equals accuracy exactly.
- The ciFAIR page states 3.25% of CIFAR-10 test images are duplicates; we measured 308
  changed images (3.08%). We did not reconcile the difference.
- Near-duplicates inside the training set are left in place (they do not leak into held-out data).

Planned sensitivity analyses (Phase 5/6), all supported by flags: Fashion-MNIST unfiltered
(`dedup_test=False`), original CIFAR-10, ciFAIR-10 unfiltered, and accuracy / performance drop
stratified by nearest-training-image distance. Conclusions should hold across variants.

Inspection figures: `results/plots/leakage_*_closest_pairs.png`.

## Preprocessing (shared by all models)

`uint8 image -> degradation (Phase 3) -> uint8 -> /255 -> (x - mean) / std`. The scalars are
fitted on the clean training images only and frozen, so degraded images are never
re-normalised with their own statistics.

## Project layout

```
src/            config, data_loader, preprocessing, leakage, utils   (degradations, models,
                evaluation, experiments, visualization are added in later phases)
data/raw|splits|cache   downloaded data (git-ignored), saved split indices, degraded-image cache
results/        metrics/, plots/, confusion_matrices/
notebooks/      exploration only; the pipeline lives in src/
```

## Reproducibility record

Seed 42 (`src/config.py`); PyTorch deterministic kernels are requested but results can still
differ across hardware, drivers and library versions. Software versions:
`results/metrics/environment.json` and `requirements-lock.txt`. Degradation parameters, model hyperparameters and the final
experimental matrix will be added here as each phase is completed.

## Roadmap

1. Experimental design (done)  2. Project setup and data loading (done)
3. Degradation pipeline  4. Models  5. Robustness experiment  6. Analysis  7. Paper support

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
| `fashion_mnist` | Zalando Research, 28×28 gray | 50,000 | 10,000 | 9,982 | primary |
| `cifair10` | ciFAIR-10 (CIFAR-10, duplicate test images replaced), 32×32 RGB | 45,000 | 5,000 | 9,901 | replication |
| `cifar10` | original CIFAR-10 | 45,000 | 5,000 | 10,000 | sensitivity analysis only |

- Images are `uint8` arrays of shape `(N, H, W, C)` (`C = 1` for Fashion-MNIST), labels `int64`.
- Validation is a stratified split of the official training set (seed 42); the indices are
  saved in `data/splits/` and reused, so results never depend on library versions.
  The validation set is used only for hyperparameter selection, on clean images.
- Downloads are verified against pinned MD5 and SHA-256 checksums
  (`src/config.py`). Fashion-MNIST and CIFAR-10 MD5s match the values published by their
  authors. ciFAIR-10 publishes no checksums, so both of its hashes were computed by us.
- Nothing is unpickled with the standard unpickler. CIFAR-10 uses the binary release;
  ciFAIR-10 only ships as pickles, so it is read with a restricted unpickler that allows
  just the five numpy/codec globals the files reference (found by scanning the opcodes
  without executing them) and rejects everything else.

### Leakage control (train images must never appear in the test set)

No dataset contains byte-identical images across splits (hash check). Near-duplicates do
exist, so the loader removes test images whose nearest train/val image is closer than an
RMS pixel distance (on the [0, 1] scale), applied identically for every model:

| Dataset | Rule | Test images removed |
|---|---|---|
| Fashion-MNIST | RMS ≤ 0.04 (visual inspection: pairs up to ~0.04 are the same item, from ~0.045 they are different items of the same type) | 18 |
| ciFAIR-10 | the authors' 308 replacements, plus RMS ≤ 0.08 | 99 further |

Caveats to report in the paper:
- The thresholds are judgement calls. On CIFAR a pixel threshold cannot separate duplicates
  from look-alikes, and 246 of the 308 ciFAIR replacements lie above RMS 0.08, so a
  pixel-distance filter alone would miss most true duplicates. This is a mitigation, not a
  guarantee: shifted or rescaled duplicates can remain.
- Many pairs that look identical at 32×32 survive ciFAIR's replacement. We did not check
  whether ciFAIR classifies them as "very similar" rather than duplicates.
- The filtered test sets are not exactly class-balanced (Fashion-MNIST 994–1000 per class,
  ciFAIR-10 971–1000), so macro recall is no longer exactly equal to accuracy.
- Sensitivity analysis (Phase 5/6): repeat the evaluation on the original CIFAR-10 test set
  and on ciFAIR without the pixel filter (`dedup_test=False`). Conclusions should hold on all variants.
- The project page states 3.25% of CIFAR-10 test images are duplicates; we measured 308
  changed images (3.08%). We did not investigate the difference.

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

Seed 42 (`src/config.py`). Software versions: `results/metrics/environment.json` and
`requirements-lock.txt`. Degradation parameters, model hyperparameters and the final
experimental matrix will be added here as each phase is completed.

## Roadmap

1. Experimental design (done)  2. Project setup and data loading (done)
3. Degradation pipeline  4. Models  5. Robustness experiment  6. Analysis  7. Paper support

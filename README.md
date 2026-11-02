# Robust Pattern Recognition Under Noise and Image Degradation

A controlled, reproducible comparison of three pattern-recognition approaches
(**LDA**, **SVM + HOG**, **CNN**) trained on clean images and evaluated on progressively degraded test
images (Gaussian noise, Gaussian blur, contrast reduction, brightness increase; five severities each).
Datasets: Fashion-MNIST (primary) and ciFAIR-10 (replication). LDA and SVM + HOG are also trained with an
ANOVA F-test feature-selection stage (`lda_fs`, `svm_hog_fs`), and LDA + HOG is an ablation.

This is a comparative study, not a new algorithm. **Status: all seven phases are implemented and run**
(design, data, degradation pipeline, models, robustness experiment, statistical analysis, paper). Every model is
trained and tuned on clean data only, frozen (SHA-256 hashed), and scored on degraded test images that are
bit-identical for all models.

## Results at a glance

Clean test accuracy (CNN: mean of 3 training seeds):

| Model | Fashion-MNIST | ciFAIR-10 |
|---|---|---|
| LDA | 0.817 | 0.405 |
| SVM + HOG | 0.911 | 0.625 |
| CNN | 0.928 | 0.802 |

- The clean ranking does **not** persist under degradation: on Fashion-MNIST the CNN is significantly *less* accurate than
  SVM + HOG in 16 of 21 conditions (paired exact McNemar tests, Holm-corrected).
- HOG-based models are almost unaffected by contrast reduction and brightness increase (Fashion-MNIST) but sensitive to
  Gaussian noise; LDA is the most tolerant of noise and blur; failing models collapse onto one or two predicted classes.
- CNN accuracy under a single degradation varies by up to 0.12 across training seeds although clean accuracy varies by ~0.001.
- Feature selection gave no consistent significant gain on clean data; its effect on robustness was large for LDA on
  Fashion-MNIST, mixed on ciFAIR-10 and small for SVM + HOG.

These are descriptive findings for two small datasets, four synthetic degradations and one test set per dataset;
see the limitations in [`docs/project_documentation.md`](docs/project_documentation.md) and the paper.

| Where | What |
|---|---|
| [`paper/main.pdf`](paper/main.pdf) | IEEE two-column paper (LaTeX source `paper/main.tex`) |
| [`docs/project_documentation.md`](docs/project_documentation.md) | full technical documentation (methods, results, discussion) |
| [`docs/literature_survey.md`](docs/literature_survey.md) | literature survey and verification notes |
| [`docs/models.md`](docs/models.md), [`docs/degradations.md`](docs/degradations.md) | model and degradation definitions |
| [`dataset/`](dataset/) | the Fashion-MNIST files used by the case study (ciFAIR-10 is a download link, see `dataset/README.md`) |
| `results/plots/` | figures (`main_*.png` = three-model comparison; `architecture_*.png` = diagrams) |
| `results/metrics/` | clean and degraded test scores, statistics (`analysis_*.csv`), frozen records |

## Setup

Developed and tested on Python 3.12 (macOS, CPU). A GPU is optional: the CNN uses CUDA
if PyTorch can see one and falls back to CPU otherwise.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -r requirements-lock.txt (exact versions)
python -m src.data_loader --dataset fashion_mnist [--debug]   # first run downloads and checksum-verifies the data
python -m src.data_loader --dataset cifair10                  # also: cifar10 (original, sensitivity analysis)
python -m src.utils                      # writes results/metrics/environment.json
python -m src.calibration                # audit the severity ladder (add --freeze to write the record)
python -m src.visualization              # regenerate the degradation grids and calibration curves
```

Training, evaluation and analysis (CPU is enough; the full run takes several hours, dominated by the SVM and CNN):

```bash
python -m src.train --dataset fashion_mnist cifair10 --model lda svm_hog cnn lda_hog lda_fs svm_hog_fs
python -m src.evaluate_clean                 # clean test scores (verifies model hashes and training data)
python -m src.run_robustness                 # all degraded conditions, 31 runs per model and dataset
python -m src.analysis                       # bootstrap CIs, McNemar + Holm, AUSC, retention
python -m src.plot_results && python -m src.plot_selection && python -m src.plot_architecture
python -m src.make_ieee_tables               # LaTeX tables for the paper; build the paper with: cd paper && tectonic -X compile main.tex
```

Trained model weights (`models/`), logs and the local test suite are **not** in the repository; each model's SHA-256 and training-data
digests are recorded in `results/metrics/models/*.json`, so a retrained model can be checked against the record.

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
  bounds, uniqueness, disjointness, coverage; mask dtype and length). This catches caches built
  for different data or parameters and structurally invalid files; it does not detect every
  corruption (a modified but still valid-looking index set or mask passes, because file contents
  are not hashed against the fingerprint). Saved indices fix the split itself across library
  versions; they do not make feature computation or training version-independent.
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

`uint8 image -> degradation -> uint8 -> /255 -> (x - mean) / std`. The scalars are fitted on the
clean training images only and frozen, so degraded images are never re-normalised with their own
statistics. HOG features are computed from the degraded image at unit scale (`/255` only), not from
normalised pixels.

## Degradations (Phase 3)

Gaussian noise, Gaussian blur, contrast scaled around the per-image mean, and additive brightness,
five severities each plus clean (21 conditions). Full definitions, the audit and its findings are in
[`docs/degradations.md`](docs/degradations.md); the essentials:

- Float32 arithmetic on [0, 1]; clip to [0, 1], multiply by 255, round half to even, store as `uint8`.
  Severity 0 returns an exact copy of the input.
- Blur is separable over the spatial axes only (never across channels), radius `ceil(3σ)`, mirrored borders.
  Contrast uses one scalar mean per image over all pixels and channels. Brightness is positive only.
- Noise is keyed by `(seed, realisation, split, original image id)`, so it is independent of
  batching and order, and every model receives identical degraded arrays through `get_degraded`.
  Caches are named after a fingerprint of the source data, parameters, declared specification (including the
  noise generator and stream ids), seed and `DEGRADATION_VERSION`. The fingerprint covers the declared specification, not the
  code: changing what a transformation does requires a manual version bump and re-freeze.
- Severities were audited with PSNR/SSIM on a fixed subset of clean **training** images against criteria
  declared in advance, then **frozen** (`results/metrics/severity_ladder_frozen.json`) before any model
  existed. `degrade()` / `get_degraded()` refuse to run if the frozen manifest is missing or the declared definitions differ from it
  (`freeze_ladder` recomputes the criteria from a validated audit; nothing is accepted from the caller). PSNR/SSIM comparisons across
  families do not indicate equal recognition difficulty (brightness is worst by PSNR but best by SSIM at severity 5).

**Evaluation rule for later phases:** models are trained and tuned once on the primary data and then
frozen. Every sensitivity analysis (original CIFAR-10 test set, ciFAIR-10 without the pixel filter,
unfiltered Fashion-MNIST) must reuse those same models; nothing is retrained or retuned per test variant.

## Project layout

```
src/            config, data_loader, preprocessing, leakage, degradations, calibration, models/ (LDA, SVM + HOG, CNN),
                train, evaluate_clean, run_robustness, metrics, analysis, plot_results, plot_selection,
                plot_architecture, make_paper_tables, make_ieee_tables, visualization, utils
docs/           project_documentation.md, models.md, degradations.md, literature_survey.md, paper_draft.md
paper/          main.tex, refs.bib, tables/, figs/, main.pdf (IEEE conference format)
data/splits     saved split indices (raw data and caches are git-ignored)
results/        metrics/ (scores, statistics, frozen records), predictions/, confusion_matrices/, plots/, paper_tables/
```

## Reproducibility record

Seed 42 (`src/config.py`); PyTorch deterministic kernels are requested but results can still
differ across hardware, drivers and library versions. Software versions:
`results/metrics/environment.json` and `requirements-lock.txt`. Degradation parameters are in `src/degradations.py` and the frozen ladder record; model
hyperparameters are recorded per model in `results/metrics/models/*.json`.

## Roadmap

All seven phases are done: 1 experimental design, 2 setup and data, 3 degradation pipeline, 4 models, 5 robustness experiment,
6 analysis, 7 paper support. Possible extensions: more CNN seeds, other feature-selection methods, the original-CIFAR-10 and
unfiltered-test sensitivity runs (supported by the loader, not yet run), augmentation or noise-aware training, real degraded data.

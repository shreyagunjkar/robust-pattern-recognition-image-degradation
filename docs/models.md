# Models (Phase 4)

Three compared models (SVM + HOG is **one hybrid model**: HOG feature extraction and the SVM form a single
pipeline, and HOG is never evaluated on its own) plus one optional appendix ablation, all trained on **clean** data, tuned on the clean
**validation** split, frozen, and scored once on the clean test set. No degraded image is used for
training, tuning or model selection. Code: `src/models/` (models), `src/train.py` (tuning and
training), `src/evaluate_clean.py` (clean test scoring), `src/metrics.py`.

| Name | Pipeline (uint8 image in, label out) |
|---|---|
| **LDA** — Linear Discriminant Analysis | `/255` → `(x − μ) / σ` with scalar μ, σ frozen from the clean training set → flatten → LDA (`solver="lsqr"`, shrinkage tuned) |
| **SVM + HOG** — Support Vector Machine with Histogram of Oriented Gradients features | `/255` → HOG → RBF-kernel SVM (C and γ tuned) |
| **CNN** — Convolutional Neural Network | `/255` → frozen `(x − μ) / σ` → 5 conv layers → global average pooling → linear |
| *LDA + HOG* (ablation, appendix) | `/255` → HOG → LDA |
| **LDA + FS** — LDA with feature selection | `/255` → `(x − μ) / σ` → flatten → ANOVA F-test, keep the top `k_frac` of the pixels → LDA |
| **SVM + HOG + FS** — SVM + HOG with feature selection | `/255` → HOG → ANOVA F-test, keep the top `k_frac` of the HOG features → RBF-SVM |

Each model owns its whole input pipeline, so the same degraded uint8 array (from `get_degraded`) is
passed to every model. HOG is computed **after** degradation, as part of the model. Normalisation
constants are fitted on the clean training images only and never re-estimated on degraded data.

## HOG policy (fixed, not tuned)

Input on the [0, 1] scale (never the normalised, negative-valued image); 9 orientations; 4×4-pixel
cells; 2×2-cell blocks; L2-Hys block normalisation; no square-root compression. Colour images use
scikit-image's multichannel rule (gradients per channel, the strongest channel's gradient per pixel).
This gives 1,296 features for 28×28 and 1,764 for 32×32 images. The geometry was fixed before seeing
any result and is not part of the tuning grids (limitation: it may be suboptimal for these image sizes).
Each fitted HOG model stores a snapshot of these settings (`SklearnModel.hog_params`) and predicts with that snapshot, never with the
module-level constant; evaluation refuses a model whose settings differ from the ones recorded in its metadata.

## CNN

Conv(32)–Conv(32)–MaxPool–Dropout(0.25)–Conv(64)–Conv(64)–MaxPool–Dropout(0.25)–Conv(128)–global
average pool–Linear(10); 3×3 kernels, BatchNorm + ReLU, no bias in conv layers (about 140k parameters).
Adam with a one-cycle learning-rate schedule, batch size 128, 10 epochs, cross-entropy, **no data
augmentation** (augmentation would confound a robustness comparison). The epoch with the best clean
**validation** accuracy is kept. Evaluation runs in eval mode, so BatchNorm uses its stored running
statistics and never statistics of a degraded batch. Device: CUDA if present, otherwise CPU. Seeds
42, 43 and 44 are trained with the selected configuration to expose training-seed variance; LDA and
SVM are deterministic.

## Feature selection (ANOVA F-test)

The course requires a feature-selection algorithm. `lda_fs` and `svm_hog_fs` insert scikit-learn
`SelectKBest(f_classif)` between the features and the classifier. The F scores are fitted on the **clean training
split only**; `k = round(k_frac · D)` with `k_frac` ∈ {0.10, 0.25, 0.50, 0.75} chosen on clean validation accuracy
together with the classifier's hyperparameters (100% is deliberately not in the grid, so a `_fs` model always
selects; the model without selection is the 100% reference). The selection is frozen with the model: its size and a
hash of the selected mask are stored in the metadata (`feature_selection`), the mask is saved as
`results/metrics/models/<dataset>__<model>__selected_mask.npy`, and evaluation refuses a model whose selection differs
from the record. For the SVM, γ = multiplier / (D_kept · var(X_kept)). The CNN learns its features and has no
separate selection stage. The un-selected models are kept as the baselines.

## Tuning budgets

Selection metric: clean validation accuracy; ties go to the earlier grid entry. The budgets are small
and stated so that no model is silently favoured:

| Model | Grid | Configurations | Notes |
|---|---|---|---|
| LDA, LDA + HOG | shrinkage ∈ {0.001, 0.01, 0.05, 0.1, 0.3, auto} | 6 | full training set |
| LDA + FS | shrinkage (6 values) × `k_frac` (4 values) | 24 | full training set |
| SVM + HOG | C ∈ {1, 10} × γ-multiplier ∈ {0.5, 1, 2}, γ = multiplier / (D · var(X)) | 6 | tuned on a stratified 10,000-image training subset (kernel SVM cost), then **refitted on the full training set** |
| SVM + HOG + FS | C × γ-multiplier (6) × `k_frac` (4 values) | 24 | same 10,000-image tuning subset, then refit on the full training set |
| CNN | learning rate ∈ {1e-3, 3e-3} × weight decay ∈ {0, 5e-4} | 4 | full training set, 10 epochs each |

`selected_at_grid_edge` in each metadata record flags hyperparameters that landed on the boundary of
their grid (the optimum may lie outside it).

## Test-set discipline

* `train.py` receives only training and validation arrays into its tuning functions; a local test
  runs the whole training with the test arrays replaced by `None`.
* Each frozen model file is SHA-256 hashed in `results/metrics/models/<dataset>__<model>.json`, together
  with the digests of the training and validation arrays it was trained on and the full tuning table.
* `evaluate_clean.py` refuses to score a model whose file hash or training data differ from that
  record, then writes `results/metrics/clean_test_metrics.csv`, confusion matrices and per-image
  predictions (for the paired tests of Phase 6).
* `run_robustness.py` scores every frozen model on all degraded test conditions (clean, 4 families x 5 severities,
  3 noise realisations). Each saved results file carries the digests of the test pixels, ids and labels, the ladder
  fingerprint, the noise-realisation count and the model hash, and is reused on resume (or aggregated) only if all of
  them match; the clean condition must reproduce the clean-test predictions exactly. A partial rerun replaces only
  the rows it recomputed.
* The hash and the record are stored next to the test result, so retraining a model after looking at
  its test score would be visible in the git history; it is not technically prevented.
* Model files are loaded with `joblib` (pickle) after the hash check. They are your own files; do not
  load model files from untrusted sources.

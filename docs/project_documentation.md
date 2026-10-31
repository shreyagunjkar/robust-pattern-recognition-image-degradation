# Robust Pattern Recognition Under Noise and Image Degradation — Technical Documentation

This document describes what is implemented in this repository and reports the results that exist in it.
Every parameter, architecture detail and number below was taken from the code (`src/`), the frozen
records in `results/metrics/`, or the figures in `results/plots/`. Where a statement is a theoretical
expectation rather than an observation, it is labelled as such.

---

## 1. Project Overview

### 1.1 Background and motivation

Pattern-recognition systems are usually trained and evaluated on clean images, but images met in use are
often degraded: sensor noise, optical blur, poor illumination or low contrast. A classifier that scores
well on clean test data can lose accuracy when the input quality changes, and different families of
classifiers (statistical, kernel-based on hand-crafted features, deep learning) can fail in different ways.
Knowing *how* each family degrades, and under *which* degradation, is useful when choosing a method for a
setting where input quality is not controlled.

### 1.2 Problem statement

Given classifiers trained only on clean images, how does their classification performance change when the
**test** images are degraded by controlled amounts of Gaussian noise, Gaussian blur, contrast reduction
and brightness increase, and does this change differ between a linear statistical classifier, a
hand-crafted-feature kernel classifier and a convolutional neural network?

### 1.3 Main research question

*How robust are LDA, SVM + HOG and a CNN, trained on clean images, to progressively stronger test-time
degradations of four types, on two image datasets?*

### 1.4 Objectives

1. Build a reproducible data pipeline with leakage control for Fashion-MNIST and ciFAIR-10.
2. Implement four image degradations with five calibrated severity levels each, and freeze them before any model is evaluated.
3. Train three pattern-recognition approaches (LDA, SVM + HOG, CNN) on clean data only, with hyperparameters chosen on clean validation data.
4. Evaluate every frozen model on the clean test set and on all degraded test conditions using identical degraded images for every model.
5. Report accuracy, macro-averaged precision/recall/F1, error rate and confusion matrices per condition.

### 1.5 Research significance

The study is a **controlled comparison**, not a new algorithm. Its value lies in the experimental
control: the same degraded images reach every model, models are never retrained or re-tuned on degraded
data, the severity ladder is frozen and audited before any model result is seen, and the test sets are
filtered for near-duplicates of training images. The design makes differences between models
attributable to the models rather than to the evaluation setup.

---

## 2. Dataset

Two datasets are used for the main experiment. A third (original CIFAR-10) is supported by the loader for
sensitivity analysis only and is not part of the main experiment.

### 2.1 Summary

| Property | Fashion-MNIST (`fashion_mnist`) | ciFAIR-10 (`cifair10`) |
|---|---|---|
| Role | Primary dataset | Replication dataset |
| Source | Zalando Research (`fashion-mnist.s3-website.eu-central-1.amazonaws.com`) | Barz & Denzler, CIFAR-10 with replaced duplicate test images (`github.com/cvjena/cifair`, release v1.0) |
| Classes | 10 | 10 |
| Class names | T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Image size | 28 × 28, 1 channel (grayscale) | 32 × 32, 3 channels (RGB) |
| Pixel type | `uint8`, range 0–255 | `uint8`, range 0–255 |
| Official split | 60,000 train / 10,000 test | 50,000 train / 10,000 test |
| Split used in this project (train / validation / test) | **50,000 / 9,986 / 9,982** | **45,000 / 4,928 / 9,901** |
| Validation images removed as near-duplicates | 14 | 72 |
| Test images removed as near-duplicates | 18 | 99 |
| Train class counts | 5,000 per class | 4,500 per class |
| Test class counts | 994–1,000 per class | 971–1,000 per class |

ciFAIR-10 has the same training set as CIFAR-10; the authors replaced 308 test images that were
near-duplicates of training images. The 308 figure is what this project measured by comparing against
CIFAR-10 (3.08% of the test set).

### 2.2 How the split is built (`src/data_loader.py`)

1. The official training set is split into training and validation parts with a **stratified** split
   (`sklearn.model_selection.train_test_split`, `random_state = 42`): validation size 10,000 for
   Fashion-MNIST and 5,000 for ciFAIR-10.
2. **Near-duplicate filtering** (`src/leakage.py`). For each validation image, the nearest retained
   training image is found by RMS pixel distance on the [0, 1] scale; for each test image the nearest
   image in the *full* official training set is found. Images closer than a threshold are removed:
   **0.04** for Fashion-MNIST and **0.08** for ciFAIR-10 (`DatasetSpec.dedup_rms` in `src/config.py`).
   Exact (byte-identical) duplicates were checked by hashing and none were found across splits.
3. Split indices and removal masks are stored in `data/splits/`, named after a fingerprint of the data and parameters.

The thresholds were chosen by visually inspecting ranked test/train image pairs (before any model
existed); this is disclosed in the README and means the test set was inspected while building the
benchmark. Figures `results/plots/leakage_*_closest_pairs.png` show the closest pairs. Pixel distance
cannot separate duplicates from look-alikes on CIFAR-style images, so the filtered sets are called
*distance-filtered benchmarks*.

### 2.3 Preprocessing (`src/preprocessing.py`)

| Step | Definition |
|---|---|
| Unit scaling | `x / 255`, giving float32 in [0, 1] |
| Global normalisation (LDA and CNN) | `(x − μ) / σ` with **one scalar** mean and standard deviation over all pixels (and channels) of the **clean training images**, then frozen: Fashion-MNIST μ = 0.2863, σ = 0.3532; ciFAIR-10 μ = 0.4732, σ = 0.2515 |
| HOG input | the unit-scale image (`x / 255`) is used directly; no normalisation (see §4.2) |

Degraded images are never re-normalised with their own statistics, so a brightness or contrast change
remains visible to the model.

### 2.4 Why these datasets

* **Fashion-MNIST** (primary): small grayscale images of 10 balanced classes, so all three model families
  can be trained on CPU in reasonable time and the images are structured (clothing silhouettes and textures), which makes blur, contrast and noise visually interpretable.
* **ciFAIR-10** (replication): colour natural-image classes at 32 × 32, a harder recognition problem with
  different image statistics, used to check whether findings from the primary dataset hold on a different
  kind of data. ciFAIR-10 is used instead of CIFAR-10 because the original CIFAR-10 test set contains
  near-duplicates of training images.

---

## 3. Complete Methodology

### 3.1 Pipeline

```
Dataset ─► split + leakage filter ─► preprocessing ─► train (clean) ─► tune on clean validation ─► freeze
                                                                                         │
 clean test images ──────────────────────────────► clean evaluation ◄───────────────────┤
        │                                                                                │
        └─► degradation (4 types × 5 severities) ─► the same degraded images to every model ─► degraded evaluation
                                                                                         │
                                          metrics + confusion matrices ─► tables / figures
```

### 3.2 Stages and the code that implements them

| # | Stage | What happens | Code |
|---|---|---|---|
| 1 | Download and verification | Files are downloaded (resumable) and checked against pinned MD5/SHA-256 checksums | `src/data_loader.py`, `src/config.py` |
| 2 | Split and leakage control | Stratified train/validation split; near-duplicate removal (§2.2) | `src/data_loader.py`, `src/leakage.py` |
| 3 | Preprocessing | Unit scaling; scalar normalisation constants fitted on clean training images | `src/preprocessing.py` |
| 4 | Severity audit and freeze | Each ladder is audited with PSNR/SSIM on 1,000 clean **training** images per dataset (100 per class) against three criteria declared in advance; the ladder is then frozen in `results/metrics/severity_ladder_frozen.json` | `src/calibration.py`, `src/degradations.py` |
| 5 | Model training and tuning | Each model is trained on clean training data; hyperparameters (including the fraction of features kept by the feature-selection stage, where there is one) are chosen by clean **validation** accuracy; tuning functions receive no test data | `src/train.py`, `src/models/` |
| 6 | Freezing | Each model file is SHA-256 hashed in `results/metrics/models/<dataset>__<model>.json` together with the digests of the training/validation arrays it was trained on | `src/train.py` |
| 7 | Clean evaluation | Verifies the model hash and the training-data digests, then scores the clean test set; writes metrics, confusion matrices and per-image predictions | `src/evaluate_clean.py` |
| 8 | Degradation | The clean test images are degraded once through `get_degraded` (cache keyed by a fingerprint of data, parameters, specification and seed); the freeze guard refuses to run if the frozen manifest is missing or differs | `src/degradations.py` |
| 9 | Degraded evaluation | Every frozen model predicts every degraded test array; the SHA-256 digest of each degraded array is stored and must be identical for all models; the clean run must reproduce the clean-evaluation predictions exactly | `src/run_robustness.py` |
| 10 | Metrics | Accuracy, error rate, macro precision/recall/F1, confusion matrices (§7) | `src/metrics.py` |
| 11 | Statistical analysis | Paired bootstrap intervals, exact McNemar tests with Holm correction, area under the severity curve, chance-adjusted retention, prediction collapse (§7.1) | `src/analysis.py` |
| 12 | Figures | Degradation grids, severity-audit curves, leakage pairs, and the result figures of §8 | `src/visualization.py`, `src/leakage.py`, `src/plot_results.py`, `src/plot_selection.py` |

### 3.3 Design rules that make the comparison controlled

* Training and tuning use **clean data only**; no degraded image is ever used for training, tuning, epoch selection or model selection.
* The same degraded uint8 array is given to every model; each model applies its own input pipeline to it (LDA: normalisation + flatten; SVM + HOG: HOG; CNN: normalisation; the `+ FS` models additionally apply their frozen feature selection).
* Feature selection is fitted on the clean training split only and frozen with the model; a degraded test image is always reduced to the same features, and the selected set is recorded (count and hash) in the model's metadata, which is checked before evaluation.
* Noise is generated per image from `(seed, realisation, split, original image id)`, so it does not depend on batch size or order.
* The severity ladder is frozen before evaluation; `degrade()` and `get_degraded()` raise an error if the manifest is missing or the declared definitions differ from it.
* Seed 42 is used throughout (`src/config.py`); the environment is recorded in `results/metrics/environment.json`.
* Models run on CUDA if available and otherwise on CPU; all recorded runs used CPU.

---

## 4. Pattern Recognition Solutions

Three main models are compared. **LDA + HOG** is trained as an appendix ablation (it separates the effect of the HOG features from the effect of the SVM classifier), and **LDA + FS** and **SVM + HOG + FS** are the same two classical models with an ANOVA F-test feature-selection stage (§4.4), as the Pattern Recognition course requires a feature-selection algorithm. The CNN learns its features end to end and has no separate selection stage.

| Name | Input pipeline | Classifier |
|---|---|---|
| **LDA** — Linear Discriminant Analysis | `/255` → frozen scalar normalisation → flatten (784 or 3,072 raw-pixel features) | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage=…)` |
| **SVM + HOG** — Support Vector Machine with Histogram of Oriented Gradients features | `/255` → HOG (1,296 or 1,764 features) | `SVC(kernel="rbf", C=…, gamma=…)` |
| **CNN** — Convolutional Neural Network | `/255` → frozen scalar normalisation | 5-convolution network (§4.3) |
| *LDA + HOG* (ablation) | `/255` → HOG | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage=…)` |
| **LDA + FS** | `/255` → frozen normalisation → flatten → ANOVA F-test, keep top *k* pixels | LDA as above |
| **SVM + HOG + FS** | `/255` → HOG → ANOVA F-test, keep top *k* HOG features | `SVC(kernel="rbf", …)`, γ computed from the kept features |

**SVM + HOG is one hybrid model:** HOG feature extraction and the SVM are a single pipeline. HOG is
computed from the (possibly degraded) image at prediction time, inside the model, and is never evaluated on its own.

### 4.1 LDA — Linear Discriminant Analysis

**What it is.** LDA is a supervised, linear, generative classifier. It models each class as a Gaussian
with its own mean but a **shared covariance matrix** and assigns a sample to the class with the highest
posterior probability. The resulting decision boundaries between classes are linear.

**Mathematical intuition.** For classes $k = 1,\dots,K$ with means $\mu_k$, priors $\pi_k$ and shared
covariance $\Sigma$, the linear discriminant function is

$$\delta_k(x) = x^{\top}\Sigma^{-1}\mu_k - \tfrac12 \mu_k^{\top}\Sigma^{-1}\mu_k + \log \pi_k,\qquad \hat y = \arg\max_k \delta_k(x).$$

**Scatter matrices (Fisher's view).** With $N_k$ samples in class $k$, class mean $\mu_k$ and overall mean $\mu$:

$$S_W = \sum_{k}\sum_{x_i \in k}(x_i-\mu_k)(x_i-\mu_k)^{\top},\qquad S_B = \sum_{k} N_k(\mu_k-\mu)(\mu_k-\mu)^{\top}.$$

$S_W$ (within-class scatter) measures how spread out samples are around their own class mean; $S_B$
(between-class scatter) measures how far the class means are from the overall mean. Fisher's criterion
finds the projection $w$ that maximises

$$J(w)=\frac{w^{\top}S_B\,w}{w^{\top}S_W\,w},$$

i.e. directions where classes are far apart relative to their internal spread. The pooled within-class
covariance used by the discriminant function is $\Sigma \approx S_W/(N-K)$. With $K=10$ classes there are
at most $K-1 = 9$ discriminant directions; in this project LDA is used as a classifier (the `lsqr`
solver does not project to a lower-dimensional space), so all input dimensions enter the decision function.

**Implementation in this project** (`src/models/classical.py`, tuning in `src/train.py`):

| Item | Value |
|---|---|
| Input | Frozen-normalised raw pixels, flattened: 784 features (Fashion-MNIST), 3,072 (ciFAIR-10) |
| Estimator | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)`; class priors are the scikit-learn default (empirical class frequencies) |
| Hyperparameter tuned | `shrinkage` ∈ {0.001, 0.01, 0.05, 0.1, 0.3, `"auto"`}, selected by clean validation accuracy (6 configurations, full training set) |
| Selected | Fashion-MNIST: 0.05; ciFAIR-10: 0.1 |
| Why shrinkage | Image pixels are highly correlated and some are constant, so the covariance estimate of a 784- or 3,072-dimensional input is poorly conditioned; shrinkage regularises it. `"auto"` is the Ledoit–Wolf estimate. |

**Advantages.** Closed-form, deterministic and fast; few hyperparameters; a transparent linear baseline.

**Limitations.** Assumes Gaussian classes with equal covariance and linearly separable class structure;
works on raw pixels, so it has no built-in invariance to translation, blur or noise; very high-dimensional
inputs need regularisation.

**Relevance to this experiment.** LDA is the linear statistical baseline that works directly on pixel
intensities. Because it uses frozen normalisation constants, a brightness or contrast change moves its
input directly in feature space, which makes it a clear reference point for how much of a model's
robustness comes from its feature representation rather than the classifier.

### 4.2 SVM + HOG

**What HOG is.** The Histogram of Oriented Gradients is a hand-crafted descriptor of local shape and edge
structure. It is computed as follows (scikit-image `skimage.feature.hog`):

1. **Gradient.** Horizontal and vertical intensity gradients $g_x, g_y$ are computed at every pixel (centred differences).
2. **Magnitude and orientation.** $m=\sqrt{g_x^2+g_y^2}$ and $\theta=\arctan(g_y/g_x)$, with orientation taken as *unsigned* (0°–180°).
3. **Cells.** The image is divided into small non-overlapping cells. Inside each cell the gradient magnitudes vote (weighted by $m$) into orientation bins, giving one histogram per cell.
4. **Blocks and normalisation.** Cell histograms are grouped into overlapping blocks of cells (stride of one cell) and each block vector is normalised. The block normalisation used is **L2-Hys**: L2-normalise, clip values at 0.2, renormalise. Normalising over blocks reduces sensitivity to local changes in illumination and contrast.
5. **Feature vector.** All normalised block vectors are concatenated.

**HOG configuration used (fixed in advance, not tuned)** — `HOG_PARAMS` in `src/models/classical.py`:

| Parameter | Value |
|---|---|
| Orientation bins | 9 |
| Cell size | 4 × 4 pixels |
| Block size | 2 × 2 cells (8 × 8 pixels), stride one cell |
| Block normalisation | L2-Hys |
| Square-root compression | off |
| Input | unit-scale image (`/255`), never the negative-valued normalised image |
| Colour images | `channel_axis=-1`: gradients computed per channel, the channel with the strongest gradient at each pixel is used |
| Feature length | Fashion-MNIST: 7 × 7 cells → 6 × 6 blocks × 36 = **1,296**; ciFAIR-10: 8 × 8 cells → 7 × 7 blocks × 36 = **1,764** |

**What an SVM is.** A Support Vector Machine finds the separating surface with the largest margin between
classes; the training samples that define the margin are the support vectors. The soft-margin formulation
allows some violations, controlled by $C$:

$$\min_{w,b,\xi}\ \tfrac12\|w\|^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(w^{\top}\phi(x_i)+b)\ge 1-\xi_i,\ \xi_i\ge 0.$$

With the **RBF kernel** $K(x,x')=\exp(-\gamma\|x-x'\|^2)$ the boundary is non-linear. Multi-class
classification in `sklearn.svm.SVC` (libsvm) is one-vs-one: 45 binary classifiers for 10 classes, combined by voting.

**Implementation in this project:**

| Item | Value |
|---|---|
| Estimator | `sklearn.svm.SVC(kernel="rbf", C=C, gamma=γ, cache_size=2000)` |
| γ | `γ = m / (D · Var(X))`, where $D$ is the number of HOG features, $\mathrm{Var}(X)$ the variance of all feature values in the training set, and $m$ a tuned multiplier. $m=1$ is the scikit-learn `"scale"` heuristic. |
| Grid | $C \in \{1, 10\}$ × $m \in \{0.5, 1, 2\}$ (6 configurations), scored on the full clean validation set |
| Tuning data | a stratified 10,000-image training subset (1,000 per class), because kernel-SVM cost grows quickly with training size; the selected configuration is **refitted on the full training set** |
| Selected (Fashion-MNIST) | $C=10$, $m=2$ ($\gamma = 0.08021$ on the full training set; 23,480 support vectors per the training log) |

**Complete pipeline.** `uint8 image → /255 → HOG (computed on the possibly degraded image) → RBF-SVM → class`.

**Advantages.** The gradient-based descriptor captures edge structure and is insensitive to a constant
brightness offset; block normalisation reduces dependence on local contrast; the kernel SVM gives
non-linear boundaries with a strong regularisation parameter.

**Limitations.** HOG parameters are fixed, so they may not be optimal for 28 × 28 or 32 × 32 images; HOG
is computed from gradients, which are strongly affected by noise; prediction cost grows with the number of
support vectors; kernel SVM training is expensive on large sets (hence the 10,000-image tuning subset).

**Expected sensitivity (theoretical expectation, not an observed result).**

| Degradation | Expectation from how HOG and the RBF kernel work |
|---|---|
| Gaussian noise | Noise adds spurious gradients in every cell and flattens the orientation histograms; gradient-based features are expected to be noise-sensitive. |
| Gaussian blur | Blur removes high-frequency gradients, reducing gradient magnitudes and fine edge detail; sensitivity is expected to grow with σ. |
| Contrast | Scaling $x-m$ by α scales gradient magnitudes by α; L2-Hys block normalisation should offset much of this, until quantisation and noise-like detail dominate at very low α. |
| Brightness | A constant additive offset leaves gradients unchanged, so HOG should be largely insensitive until clipping at 1.0 destroys structure (31% of values are clipped at β = 0.5 on Fashion-MNIST, 44% on ciFAIR-10). |

### 4.3 CNN

**Why a CNN suits image pattern recognition.** Convolutional layers share small filters across the image,
so they learn local patterns (edges, textures, parts) with far fewer parameters than a fully connected
network, and stacked layers with pooling build features of increasing size and some tolerance to small
translations. The features are learned from data instead of being hand-designed.

**Architecture (`build_cnn` in `src/models/cnn.py`).** Every convolution is 3 × 3, stride 1, padding 1, no
bias, followed by BatchNorm and ReLU.

| # | Layer | Output channels | Notes |
|---|---|---|---|
| 1 | Conv 3×3 + BatchNorm + ReLU | 32 | input channels: 1 (Fashion-MNIST) or 3 (ciFAIR-10) |
| 2 | Conv 3×3 + BatchNorm + ReLU | 32 | |
| 3 | MaxPool 2×2 | 32 | halves height and width |
| 4 | Dropout | 32 | p = 0.25 |
| 5 | Conv 3×3 + BatchNorm + ReLU | 64 | |
| 6 | Conv 3×3 + BatchNorm + ReLU | 64 | |
| 7 | MaxPool 2×2 | 64 | halves height and width |
| 8 | Dropout | 64 | p = 0.25 |
| 9 | Conv 3×3 + BatchNorm + ReLU | 128 | |
| 10 | Global average pooling | 128 | `AdaptiveAvgPool2d(1)` |
| 11 | Flatten → Fully connected (Linear) | 10 | one linear layer: 128 → 10 class scores; no hidden fully connected layer |

Parameters: 140,458 (1-channel input) and 141,034 (3-channel input).

**Training configuration:**

| Item | Value |
|---|---|
| Loss | Cross-entropy |
| Optimiser | Adam with `weight_decay` from the grid |
| Learning-rate schedule | `OneCycleLR`, `max_lr` = the learning rate from the grid (PyTorch default shape) |
| Batch size | 128 |
| Epochs | 10 |
| Data augmentation | none (it would confound a robustness comparison) |
| Input | `/255` then frozen scalar normalisation |
| Validation use | after every epoch the clean validation accuracy is computed and the weights of the best-validation epoch are kept |
| Grid | learning rate ∈ {0.001, 0.003} × weight decay ∈ {0, 0.0005} (4 configurations, full training set, 10 epochs each) |
| Selected (Fashion-MNIST) | learning rate 0.003, weight decay 0 |
| Seeds | the selected configuration is trained with seeds 42, 43 and 44 (the seed-42 model is the tuning winner; 43 and 44 are retrained with the same hyperparameters) |
| Evaluation mode | eval mode: BatchNorm uses its stored running statistics, never statistics of a degraded batch |
| Device | CUDA if available, otherwise CPU (all recorded runs: CPU) |

**Advantages.** Learned hierarchical features; typically strong on image data. **Limitations.** Needs
more compute; behaviour depends on training seed (hence three seeds); no augmentation or noise-aware
training is used, so any robustness comes only from the clean-trained network.

### 4.4 Feature selection — ANOVA F-test (`SelectKBest`)

**What it is.** A filter method: each feature is scored on its own by how well it separates the classes, and
only the top-scoring features are kept. The score is the one-way ANOVA F statistic of feature $j$ against the class
label, the ratio of between-class to within-class variance of that feature:

$$F_j=\frac{\sum_{k}N_k(\bar x_{kj}-\bar x_j)^2/(K-1)}{\sum_{k}\sum_{i\in k}(x_{ij}-\bar x_{kj})^2/(N-K)}.$$

A large $F_j$ means the class means of feature $j$ are far apart relative to its spread inside classes. The
selector keeps the $k$ features with the largest $F_j$ (scikit-learn `SelectKBest(f_classif, k)`), and the original
feature order is preserved.

**Where it sits.** `lda_fs`: normalised pixels (784 or 3,072) → selection → LDA. `svm_hog_fs`: HOG features (1,296 or
1,764) → selection → RBF-SVM. It is applied after degradation, as part of the model, to the features of whatever image is
presented. The CNN has no separate selection stage (its convolutional layers learn the features).

**Controlled-experiment rules.** The F scores are computed on the clean **training** split only; the number kept is
$k=\text{round}(k_{\text{frac}}\cdot D)$ with $k_{\text{frac}}\in\{0.10, 0.25, 0.50, 0.75\}$ chosen by clean validation
accuracy together with the classifier's own hyperparameters; the selected set is then frozen (its count and a hash of the
mask are stored in the model's metadata and re-checked before evaluation, and the masks are saved as
`results/metrics/models/<dataset>__<model>__selected_mask.npy`). The selection is never refitted on degraded data. For the SVM the RBF
parameter is $\gamma=\text{multiplier}/(D_{\text{kept}}\cdot\text{var}(X_{\text{kept}}))$, so it adapts to the number of kept features.
A fraction of 100% is deliberately not in the grid, so a `+ FS` model always selects; the model without selection is the 100% reference.

**Implementation.** `src/models/classical.py` (`SklearnModel`, names `lda_fs`, `svm_hog_fs`; `feature_selection_record()`),
tuning grids in `src/train.py` (LDA: 6 shrinkage values × 4 fractions; SVM: 6 configurations × 4 fractions, same 10,000-image tuning subset protocol),
masks drawn by `src/plot_selection.py`.

**Advantages.** Cheap, deterministic, model-independent and easy to interpret (which pixels or HOG blocks survive).

**Limitations.** Each feature is scored independently, so redundant features are all kept and features that are
informative only in combination can be dropped; the F test assumes roughly Gaussian class-conditional feature values; the score is computed
on clean data and says nothing about how stable a feature is under degradation.

**Relevance to this experiment.** It tests whether discarding features judged uninformative on clean data changes how the
models behave when the input is degraded.

---

## 5. Image Degradation Methodology

Four degradations are implemented in `src/degradations.py` (`DEGRADATION_VERSION = 1`). All arithmetic is
done in float32 on the [0, 1] scale; the result is **clipped to [0, 1], multiplied by 255, rounded to the
nearest integer (ties to even) and stored as `uint8`**. Severity 0 returns an exact copy of the clean image.
There are five severity levels per degradation, named mild, moderate, high, very high and extreme.

| Degradation | Represents | Why relevant | Operation |
|---|---|---|---|
| **Gaussian noise** | Additive sensor / electronic noise | Common, and attacks pixel values and gradients directly | $x' = x + \sigma z,\ z\sim\mathcal N(0,1)$, drawn independently per pixel **and channel** |
| **Gaussian blur** | Defocus or loss of resolution | Removes high-frequency detail that edge and texture features depend on | Separable Gaussian low-pass filter |
| **Contrast reduction** | Washed-out, low-contrast capture | Compresses the intensity range without changing the mean | $x' = m + \alpha(x-m)$ |
| **Brightness increase** | Over-exposure / illumination shift | Shifts and saturates intensities | $x' = x + \beta$ |

### 5.1 Implementation details

* **Noise.** One standard-normal field per image from `numpy.random.default_rng([seed, realisation, stream_id, image_id])` (PCG64); `stream_id` is 0 for test, 1 for train, 2 for validation. The field is drawn once per image and scaled by σ, so the five severities share the same noise pattern. The noise is therefore independent of batch size and order, and `realisation` selects an independent field.
* **Blur.** Separable 1-D Gaussian kernel applied along height and width only (never across colour channels or images), radius $\lceil 3\sigma\rceil$, kernel size $2r+1$, weights normalised to sum 1, mirrored borders (reflect-101), so flat regions keep their value.
* **Contrast.** $m$ is **one scalar per image**: the mean of the clean float image over all pixels and all channels. For RGB images, colour differences are therefore scaled by α as well, and α → 0 gives a uniform grey. For $0\le\alpha\le1$ the result stays in [0, 1].
* **Brightness.** Positive offsets only (brightening); values above 1 are clipped.

### 5.2 Exact severity values used in the code

| Severity | Name | Gaussian noise σ (intensity scale 0–1) | Gaussian blur σ (pixels) / kernel size | Contrast α (1 = unchanged) | Brightness β (added, 0–1 scale) |
|---|---|---|---|---|---|
| 0 | clean | — | — | — | — |
| 1 | mild | 0.05 | 0.5 / 5 | 0.75 | 0.10 |
| 2 | moderate | 0.10 | 1.0 / 7 | 0.50 | 0.20 |
| 3 | high | 0.20 | 1.5 / 11 | 0.30 | 0.30 |
| 4 | very high | 0.35 | 2.0 / 13 | 0.15 | 0.40 |
| 5 | extreme | 0.50 | 3.0 / 19 | 0.05 | 0.50 |

### 5.3 Severity audit (model-independent)

Each ladder was audited before any model existed, on a fixed subset of **1,000 clean training images per
dataset** (100 per class), using PSNR (on the uint8 images; MSE over all pixels and channels) and SSIM
(`data_range=255`, Gaussian window σ = 1.5, window 11, no sample covariance; averaged over channels for
RGB). Three criteria were declared in advance: **C1** median PSNR and mean SSIM strictly decrease from
severity 1 to 5 for every dataset and family; **C2** at severity 1, at least 99% of images differ from the
clean image; **C3** at severity 5, at most 1% of images are constant. All three passed for both datasets
and the ladder was frozen unchanged (source: `results/metrics/severity_calibration.csv`,
`results/metrics/severity_ladder_frozen.json`).

Median PSNR (dB) / mean SSIM:

| Family | Sev | Fashion-MNIST PSNR / SSIM | ciFAIR-10 PSNR / SSIM |
|---|---|---|---|
| Gaussian noise | 1 | 27.5 / 0.85 | 26.1 / 0.85 |
| | 2 | 21.6 / 0.73 | 20.2 / 0.66 |
| | 3 | 16.0 / 0.54 | 14.7 / 0.40 |
| | 4 | 11.7 / 0.37 | 10.9 / 0.22 |
| | 5 | 9.3 / 0.26 | 9.1 / 0.15 |
| Gaussian blur | 1 | 27.6 / 0.96 | 34.5 / 0.98 |
| | 2 | 19.1 / 0.72 | 25.1 / 0.84 |
| | 3 | 17.1 / 0.55 | 22.3 / 0.69 |
| | 4 | 15.9 / 0.43 | 20.7 / 0.57 |
| | 5 | 14.3 / 0.30 | 18.9 / 0.41 |
| Contrast | 1 | 21.7 / 0.88 | 25.7 / 0.95 |
| | 2 | 15.7 / 0.71 | 19.7 / 0.80 |
| | 3 | 12.8 / 0.49 | 16.7 / 0.58 |
| | 4 | 11.1 / 0.28 | 15.0 / 0.37 |
| | 5 | 10.1 / 0.14 | 14.1 / 0.21 |
| Brightness | 1 | 19.9 / 0.86 | 20.0 / 0.96 |
| | 2 | 14.1 / 0.78 | 14.1 / 0.89 |
| | 3 | 10.9 / 0.69 | 10.8 / 0.79 |
| | 4 | 8.7 / 0.60 | 8.6 / 0.68 |
| | 5 | 7.0 / 0.51 | 7.1 / 0.55 |

Blur −3 dB cut-off frequencies of the discrete kernels at severities 1–5 are 0.311, 0.133, 0.088, 0.066 and
0.044 cycles per pixel. PSNR and SSIM measure image similarity, not recognition difficulty, so equal
values across families do **not** mean equal difficulty.

---

## 6. Experimental Design

### 6.1 Procedure

1. **Clean data only for learning.** Every model is trained on the clean training split. Hyperparameters (and, for the CNN, the epoch) are chosen on the clean validation split. No degraded image is used for training, tuning or model selection.
2. **Freeze.** Each trained model is saved and hashed; the training and validation data digests are recorded with it. Models are never retrained or re-tuned afterwards.
3. **Degrade the test set only.** The clean test images are degraded once, with the frozen ladder, and cached; every model receives the identical arrays (checked by SHA-256 digest).
4. **Evaluate.** Each frozen model predicts the clean test set and every degraded test set; per-image predictions are stored.

### 6.2 Size of the experiment (as defined by the code)

| Factor | Count | Detail |
|---|---|---|
| Datasets | 2 | Fashion-MNIST, ciFAIR-10 |
| Models | 3 main + 3 | LDA, SVM + HOG, CNN; plus LDA + FS, SVM + HOG + FS (ANOVA F-test feature selection) and the LDA + HOG ablation |
| Degradation types | 4 | Gaussian noise, Gaussian blur, contrast, brightness |
| Severity levels | 5 per type | plus the clean condition |
| Conditions per dataset and model | 21 | 4 × 5 + 1 clean |
| Evaluation cells | 252 | 2 datasets × 6 models × 21 conditions (126 for the three main models alone) |
| CNN training seeds | 3 | 42, 43, 44 |
| Noise realisations | 3 by default in `src/run_robustness.py` | independent noise fields for each noise condition, so each model-dataset pair has 31 evaluation runs (1 clean + 15 noise + 15 for the other three families) |

### 6.3 Why this setup measures robustness

Robustness here means how well a model that has only seen clean data keeps its performance when the input
distribution shifts. Holding everything else fixed isolates that shift: the training data, tuning
procedure and test images are the same for the clean and degraded evaluations, the degraded images are
identical across models, and severity is varied in five ordered steps within each degradation type, so
performance can be read as a function of severity. Because no model sees degraded data during training,
any difference in how performance falls is a property of the model and its input representation, not of
having been trained for the degradation.

---

## 7. Evaluation Metrics

Implemented in `src/metrics.py` on integer label vectors over the fixed label set $0,\dots,K-1$
($K=10$). A class that is never predicted, or absent, still has a row and column and enters the macro
averages with a value of 0 (`zero_division=0`); it is never dropped.

| Metric | Definition |
|---|---|
| Confusion matrix | $C_{ij}$ = number of test images of true class $i$ predicted as class $j$ (rows true, columns predicted) |
| Accuracy | $\dfrac{1}{N}\sum_{n}\mathbf 1[\hat y_n=y_n]=\dfrac{\sum_i C_{ii}}{N}$ |
| Error rate | $1-\text{accuracy}$ |
| Precision (class $k$) | $P_k=\dfrac{C_{kk}}{\sum_i C_{ik}}$ |
| Recall (class $k$) | $R_k=\dfrac{C_{kk}}{\sum_j C_{kj}}$ |
| F1-score (class $k$) | $F1_k=\dfrac{2P_kR_k}{P_k+R_k}$ |
| Macro precision / recall / F1 | unweighted mean over the $K$ classes, e.g. $\text{macro-F1}=\frac1K\sum_k F1_k$ |

Because the filtered test sets are not perfectly class-balanced (§2.1), macro recall is close to but not exactly equal to accuracy.

**Performance drop.** The drop of a condition relative to the clean condition is
$\Delta = \text{Acc}_{\text{clean}} - \text{Acc}_{\text{condition}}$ (in accuracy points); it follows by subtraction from the
accuracy column of the results table (`results/metrics/robustness_results.csv`, one row per dataset, model,
seed, condition and noise realisation) and is not stored as a separate column. The results table also records
the SHA-256 digest of each degraded array (`degraded_sha256`).

### 7.1 Statistical analysis (`src/analysis.py`)

* **Confidence intervals.** Non-parametric bootstrap over the **test images** (2,000 resamples, seed 42, 95% percentile interval). The same resampled image set is used for every model and condition, so differences are paired. This captures test-set sampling uncertainty only; CNN training-seed variance is reported as the min–max over the 3 seeds, and the deterministic models were fitted once.
* **McNemar test.** For two models on the same condition, the exact two-sided binomial test on the discordant images (right for one model and wrong for the other). Holm step-down correction over all pairwise tests of a dataset; significant means Holm $p<0.05$. Seven pairs are compared (SVM + HOG vs LDA, CNN vs LDA, CNN vs SVM + HOG, each feature-selection model vs its base model, LDA + HOG vs LDA, SVM + HOG vs LDA + HOG). Pairs with the CNN are tested for each of the 3 seeds, and a result is called robust only if all 3 seeds are significant in the same direction. **For Gaussian-noise conditions the McNemar test (a test of binary per-image outcomes) is run on noise realisation 0 only, whereas the accuracies shown elsewhere average the 3 realisations;** the pairwise table therefore also contains the difference of the realisation-averaged accuracies with its own paired bootstrap interval (`diff_mean_*`), and that is what the feature-selection table in §9.5 displays. Averaging changes a pairwise difference by at most 0.006, changes the sign of 2 of the 110 noise comparisons, and every noise comparison that is significant has an averaged-difference interval that excludes zero. The intervals are not adjusted for multiplicity; only the Holm-corrected tests are.
* **Chance-adjusted retention** at severity $s$: $(\text{Acc}_s-c)/(\text{Acc}_{\text{clean}}-c)$ with chance $c=0.1$; 1 means no loss, 0 chance level, negative below chance.
* **Area under the severity curve (AUSC):** trapezoid rule over severity 0 (clean) to 5, divided by 5, i.e. a mean accuracy; it summarises one family in one number. Intervals for both summaries come from the same bootstrap resamples.
* **Prediction collapse:** the share of predictions that fall in the single most-predicted class.
* The accuracy tables and summaries average the 3 noise realisations and, for the CNN, the 3 seeds.

---

## 8. Graphical Analysis

All figures are in `results/plots/`. The degradation grids and audit curves are regenerated by
`python -m src.visualization`; the leakage figures were produced by calling `save_closest_pairs` in
`src/leakage.py` directly (no command-line entry point regenerates them); the result figures (§8.5–§8.13) are
regenerated from the saved result files by `python -m src.plot_results` and `python -m src.plot_selection`. Model
colours come from a categorical palette checked for colour-blind separation; a model with feature selection keeps the
hue of its base model and is drawn dashed or hatched, and the LDA + HOG ablation is grey.

### 8.1 Degradation grids — `degradation_grid_<dataset>_<family>.png` (8 files)

* **Content.** One row per class (the first six classes), one clean-training calibration image per row. Columns are severity 0 (clean) to 5, labelled with the parameter value (e.g. "3 high, σ = 1.5").
* **Axes.** Rows: class; columns: severity. Images are shown after clipping and rounding to uint8.
* **Use.** A visual check that each severity ladder is monotone and what the degraded images look like, so that numerical results can be tied to a visible amount of degradation.
* **Reading.** Move along a row to see the progression; at the extreme levels (e.g. blur σ = 3 on ciFAIR-10) object shape is largely lost, as visible in `degradation_grid_cifair10_gaussian_blur.png`.

### 8.2 Severity-audit curves — `severity_calibration_<dataset>.png` (2 files)

* **Content.** Two panels per dataset: median PSNR (dB) and mean SSIM versus severity, with one line per degradation family (colour **and** marker shape, direct labels and a legend).
* **Axes.** X: severity 1 (mild) to 5 (extreme); Y: median PSNR in dB (left) or mean SSIM (right). Data: `severity_calibration.csv` (1,000 clean-training images per dataset).
* **Use.** Shows the frozen ladders are monotone and how the families compare in similarity loss.
* **Reading.** Lower values mean a larger departure from the clean image. The two panels rank families differently (for example, brightness has the lowest PSNR at severity 5 but the highest SSIM, visible in `severity_calibration_fashion_mnist.png`), so neither similarity measure should be read as difficulty for a classifier. These curves are model-independent.

### 8.3 Near-duplicate figures — `leakage_<dataset>_closest_pairs.png` (2 files; `fashion_mnist` and `cifar10`)

* **Content.** The 12 test images closest to the training pool (top row, with their RMS distance *d*) above their nearest training/validation image (bottom row).
* **Use.** Visual basis for the near-duplicate thresholds (§2.2). In `leakage_fashion_mnist_closest_pairs.png`, pairs with d from 0.003 to 0.030 show the same garment pattern, which is why such images are removed.

### 8.4 Confusion matrices

Confusion matrices are implemented (`src/metrics.py: confusion`). `src/evaluate_clean.py` writes the clean-test
matrix of each model as CSV (`results/confusion_matrices/clean_<dataset>_<model>_seed<k>.csv`: rows true class, columns
predicted class, class names as labels), and `src/run_robustness.py` stores one 10 × 10 matrix per condition
inside `results/predictions/robustness_<dataset>_<model>_seed<k>.npz`. A confusion matrix shows which classes a
model confuses; off-diagonal mass shows systematic errors, and comparing matrices between conditions shows
which classes a degradation moves into which others. The clean-test matrices (one per dataset, model and
seed) are in `results/confusion_matrices/`; the per-condition matrices are in the `.npz` files. They are drawn in
§8.9 and not interpreted class by class beyond that.


### 8.5 Pipelines — `pipelines.png`

* **Content.** One row per implemented solution, boxes for each processing step, hatched boxes for the feature-selection stage.
* **Use.** Shows exactly what each model does to a (degraded) test image and where HOG and feature selection sit.

### 8.6 Clean accuracy — `clean_accuracy_comparison.png`

* **Content.** Grouped bars of clean test accuracy per model for both datasets, with the three CNN training seeds as dots; hatched bars are the feature-selection models.
* **Axes.** X: dataset, bars by model; Y: accuracy, 0–1.
* **Use / reading.** Baseline performance before degradation: the CNN is highest on both datasets, and the gap to the other models is much larger on ciFAIR-10 than on Fashion-MNIST.

### 8.7 Robustness curves — `robustness_curves_<dataset>.png` (2 files)

* **Content.** One panel per degradation family; one line per model from clean (severity 0) to severity 5; the CNN is the mean of 3 seeds with its min–max range shaded; a dotted line marks chance (0.10).
* **Use / reading.** How fast each model loses accuracy; crossing lines mean the ranking changes with severity (for example on Fashion-MNIST under brightness the CNN is highest on clean data but is already below both LDA and SVM + HOG at severity 1).

### 8.8 Heatmaps and summaries — `robustness_heatmap_<dataset>.png`, `degraded_accuracy_summary.png`

* **Content.** The heatmaps show the accuracy of every model in every one of the 21 conditions (darker = higher; the numbers are printed); the summary figure shows each model's mean accuracy over severities 1–5 per family.
* **Use / reading.** A compact view of where each model is strong or weak; dark rows under contrast and brightness for the HOG-based models, dark under noise only for pixel-based LDA.

### 8.9 Confusion matrices — `confusion_<dataset>.png` (2 files)

* **Content.** Row-normalised confusion matrices (rows true, columns predicted) of LDA, SVM + HOG and the CNN (seed 42) for the clean data and for noise (severity 3), blur (severity 3) and contrast (severity 4).
* **Use / reading.** Clean matrices are diagonal with a few expected confusions (for example Shirt with T-shirt/top and Pullover); degraded failures show a vertical bar, i.e. most images are assigned to one or two predicted classes (Shirt and Bag on Fashion-MNIST).

### 8.10 Statistical summaries — `robustness_summary_ci.png`, `retention_ci.png`

* **Content.** Area under the severity curve and mean chance-adjusted retention per model and family, with 95% bootstrap intervals; shaded bars show the CNN seed range.
* **Use / reading.** The intervals are narrower than the markers, so test-set sampling noise is small compared with the differences between models; the CNN seed range is the larger source of uncertainty.

### 8.11 Significance map — `significance_<dataset>.png` (2 files)

* **Content.** One row per compared pair, one column per condition; the colour shows which model is significantly more accurate (paired exact McNemar test, Holm-corrected), light grey no significant difference, dark grey a CNN comparison that is significant for only some seeds.
* **Use / reading.** With about 10,000 test images almost every gap is significant, so the informative cells are the grey ones (for example SVM + HOG + FS against SVM + HOG on ciFAIR-10 is almost entirely grey).

### 8.12 Feature selection — `feature_selection_effect.png`, `feature_selection_robustness_delta.png`, `feature_selection_maps.png`

* **Content.** (1) Best clean validation accuracy against the fraction of features kept, with the no-selection model as the 100% point; (2) the test-accuracy difference between each `+ FS` model and its base model across severities, per family and dataset (note that the Y scales differ between the two rows); (3) maps of which pixels (LDA + FS) and how many HOG features per block (SVM + HOG + FS) were kept.
* **Use / reading.** Validation accuracy rises with the fraction kept, so selection does not help on clean data; the difference curves show where it helps or hurts under degradation (large positive for LDA under brightness on Fashion-MNIST; small for SVM + HOG); the Fashion-MNIST pixel map drops mostly border pixels.

### 8.13 CNN training — `cnn_training_curves.png`

* **Content.** Clean validation accuracy after each epoch for the three seeds of the selected configuration.
* **Use / reading.** On Fashion-MNIST the best epoch is the last one (epoch 10) for all three seeds, i.e. accuracy is still rising when training stops (§12.1).

---

## 9. Results

Everything below is read from files in the repository: dataset sizes (§2), the severity audit (§5.3), the
**validation** scores recorded when each model was tuned (`results/metrics/models/<dataset>__<model>.json`),
the **clean test** scores (`results/metrics/clean_test_metrics.csv`), the **degraded test** scores
(`results/metrics/robustness_results.csv`) and the statistical analysis (`results/metrics/analysis_*.csv`).
Validation scores were used to select hyperparameters (and the best epoch for the CNN), so they are optimistic;
the test scores were produced once, after training was frozen, and no model was retrained or re-tuned after its
test score was seen.

### 9.1 Tuned models: clean validation performance

Validation size: 9,986 (Fashion-MNIST), 4,928 (ciFAIR-10). Accuracy and macro-F1 of the final fitted model on clean validation images. "FS" = ANOVA F-test feature selection (§4.4).

| Dataset | Model | Selected hyperparameters | Val. accuracy | Val. macro-F1 |
|---|---|---|---|---|
| Fashion-MNIST | LDA | shrinkage 0.05 | 0.8298 | 0.8306 |
| Fashion-MNIST | LDA + FS | shrinkage 0.001, keep 75% of features | 0.8275 | 0.8286 |
| Fashion-MNIST | LDA + HOG (ablation) | shrinkage 0.001 | 0.8707 | 0.8714 |
| Fashion-MNIST | SVM + HOG | C = 10, γ-multiplier 2 | 0.9153 | 0.9152 |
| Fashion-MNIST | SVM + HOG + FS | C = 10, γ-multiplier 2, keep 75% of features | 0.9123 | 0.9121 |
| Fashion-MNIST | CNN, seed 42 | lr 0.003, weight decay 0, 10 epochs | 0.9340 | 0.9340 |
| Fashion-MNIST | CNN, seed 43 | same | 0.9363 | 0.9366 |
| Fashion-MNIST | CNN, seed 44 | same | 0.9365 | 0.9368 |
| ciFAIR-10 | LDA | shrinkage 0.1 | 0.4138 | 0.4119 |
| ciFAIR-10 | LDA + FS | shrinkage 0.3, keep 75% of features | 0.3991 | 0.3962 |
| ciFAIR-10 | LDA + HOG (ablation) | shrinkage 0.1 | 0.5515 | 0.5514 |
| ciFAIR-10 | SVM + HOG | C = 10, γ-multiplier 2 | 0.6378 | 0.6388 |
| ciFAIR-10 | SVM + HOG + FS | C = 10, γ-multiplier 1, keep 75% of features | 0.6376 | 0.6385 |
| ciFAIR-10 | CNN, seed 42 | lr 0.003, weight decay 0, 10 epochs | 0.8176 | 0.8155 |
| ciFAIR-10 | CNN, seed 43 | same | 0.8184 | 0.8165 |
| ciFAIR-10 | CNN, seed 44 | same | 0.8198 | 0.8180 |

### 9.2 Hyperparameter search results (clean validation accuracy)

LDA and LDA + HOG, shrinkage (columns 0.001, 0.01, 0.05, 0.1, 0.3, auto):

| Shrinkage | 0.001 | 0.01 | 0.05 | 0.1 | 0.3 | auto |
|---|---|---|---|---|---|---|
| Fashion-MNIST LDA | 0.8297 | 0.8297 | **0.8298** | 0.8275 | 0.8224 | 0.8282 |
| Fashion-MNIST LDA + HOG (ablation) | **0.8707** | 0.8697 | 0.8691 | 0.8676 | 0.8614 | 0.8697 |
| ciFAIR-10 LDA | 0.3896 | 0.3955 | 0.4125 | **0.4138** | 0.4107 | 0.3918 |
| ciFAIR-10 LDA + HOG (ablation) | 0.5455 | 0.5483 | 0.5503 | **0.5515** | 0.5501 | 0.5503 |

SVM + HOG on Fashion-MNIST (each fitted on the 10,000-image tuning subset; the selected setting refitted on the full training set reaches 0.9153):

| C \ γ-multiplier | 0.5 | 1 | 2 |
|---|---|---|---|
| 1 | 0.8793 | 0.8854 | 0.8876 |
| 10 | 0.8824 | 0.8881 | **0.8928** |

SVM + HOG on ciFAIR-10 (same protocol; refitted on the full training set: 0.6378):

| C \ γ-multiplier | 0.5 | 1 | 2 |
|---|---|---|---|
| 1 | 0.5457 | 0.5595 | 0.5582 |
| 10 | 0.5509 | 0.5680 | **0.5686** |

Feature selection: the best validation accuracy reached at each fraction of features kept (maximum over the other
hyperparameters of that model; the SVM rows come from the 10,000-image tuning subset, the LDA rows from the full
training set; the last column is the best of the corresponding model without selection, i.e. 100% of the features):

| Dataset | Model | 10% | 25% | 50% | 75% | 100% (no selection) |
|---|---|---|---|---|---|---|
| Fashion-MNIST | LDA + FS | 0.6529 | 0.7776 | 0.8159 | 0.8275 | 0.8298 |
| Fashion-MNIST | SVM + HOG + FS | 0.8304 | 0.8666 | 0.8884 | 0.8902 | 0.8928 |
| ciFAIR-10 | LDA + FS | 0.2285 | 0.3202 | 0.3746 | 0.3991 | 0.4138 |
| ciFAIR-10 | SVM + HOG + FS | 0.4432 | 0.5095 | 0.5473 | 0.5619 | 0.5686 |

For every feature-selection model the selected fraction is 75%, the largest value in the grid (§12.1).

CNN on Fashion-MNIST (full training set, 10 epochs, best-validation epoch kept):

| Learning rate | Weight decay | Best val. accuracy | Best epoch |
|---|---|---|---|
| 0.001 | 0 | 0.9294 | 10 |
| 0.001 | 0.0005 | 0.9301 | 9 |
| 0.003 | 0 | **0.9340** | 10 |
| 0.003 | 0.0005 | 0.9329 | 10 |

CNN on ciFAIR-10 (same protocol):

| Learning rate | Weight decay | Best val. accuracy | Best epoch |
|---|---|---|---|
| 0.001 | 0 | 0.7812 | 10 |
| 0.001 | 0.0005 | 0.7853 | 9 |
| 0.003 | 0 | **0.8176** | 9 |
| 0.003 | 0.0005 | 0.8174 | 10 |

Validation accuracy of the selected CNN configuration after each epoch on Fashion-MNIST (epochs 1–10):

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.8005 | 0.8683 | 0.8810 | 0.8941 | 0.9151 | 0.9202 | 0.9270 | 0.9301 | 0.9339 | 0.9340 |
| 43 | 0.7966 | 0.8679 | 0.8736 | 0.9101 | 0.9080 | 0.9238 | 0.9240 | 0.9318 | 0.9358 | 0.9363 |
| 44 | 0.7681 | 0.8484 | 0.8831 | 0.8996 | 0.9174 | 0.9185 | 0.9303 | 0.9252 | 0.9363 | 0.9365 |

### 9.3 Clean test performance

Test size: 9,982 (Fashion-MNIST), 9,901 (ciFAIR-10); the models are frozen and hash-verified before scoring.

| Dataset | Model | Seed | Test images | Accuracy | Error rate | Macro-precision | Macro-recall | Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | LDA | 42 | 9982 | 0.8166 | 0.1834 | 0.8188 | 0.8165 | 0.8174 |
| Fashion-MNIST | LDA + FS | 42 | 9982 | 0.8140 | 0.1860 | 0.8161 | 0.8139 | 0.8148 |
| Fashion-MNIST | SVM + HOG | 42 | 9982 | 0.9112 | 0.0888 | 0.9109 | 0.9112 | 0.9109 |
| Fashion-MNIST | SVM + HOG + FS | 42 | 9982 | 0.9103 | 0.0897 | 0.9101 | 0.9102 | 0.9101 |
| Fashion-MNIST | CNN | 42 | 9982 | 0.9278 | 0.0722 | 0.9280 | 0.9277 | 0.9276 |
| Fashion-MNIST | CNN | 43 | 9982 | 0.9283 | 0.0717 | 0.9288 | 0.9282 | 0.9283 |
| Fashion-MNIST | CNN | 44 | 9982 | 0.9270 | 0.0730 | 0.9274 | 0.9269 | 0.9270 |
| Fashion-MNIST | LDA + HOG (ablation) | 42 | 9982 | 0.8662 | 0.1338 | 0.8687 | 0.8661 | 0.8671 |
| ciFAIR-10 | LDA | 42 | 9901 | 0.4046 | 0.5954 | 0.4031 | 0.4045 | 0.4027 |
| ciFAIR-10 | LDA + FS | 42 | 9901 | 0.3940 | 0.6060 | 0.3904 | 0.3939 | 0.3908 |
| ciFAIR-10 | SVM + HOG | 42 | 9901 | 0.6249 | 0.3751 | 0.6267 | 0.6248 | 0.6249 |
| ciFAIR-10 | SVM + HOG + FS | 42 | 9901 | 0.6274 | 0.3726 | 0.6281 | 0.6273 | 0.6272 |
| ciFAIR-10 | CNN | 42 | 9901 | 0.8056 | 0.1944 | 0.8053 | 0.8054 | 0.8032 |
| ciFAIR-10 | CNN | 43 | 9901 | 0.7983 | 0.2017 | 0.7959 | 0.7982 | 0.7958 |
| ciFAIR-10 | CNN | 44 | 9901 | 0.8031 | 0.1969 | 0.8016 | 0.8029 | 0.8006 |
| ciFAIR-10 | LDA + HOG (ablation) | 42 | 9901 | 0.5369 | 0.4631 | 0.5369 | 0.5368 | 0.5359 |

### 9.4 Degraded test performance

Test accuracy of each frozen model on every degraded test condition; every model received bit-identical
degraded arrays (verified by SHA-256 digest). Gaussian-noise entries are the mean over 3 independent noise
realisations; CNN entries are the mean over the 3 training seeds. Per-run values (every seed and realisation) are in
`results/metrics/robustness_results.csv`, with predictions and confusion matrices per condition in
`results/predictions/robustness_<dataset>_<model>_seed<k>.npz`. Chance level for 10 roughly balanced classes is about 0.10.
The 95% bootstrap interval of each entry (`analysis_accuracy_ci.csv`) has a half-width of 0.008 on average
and 0.010 at most.

**Fashion-MNIST**

| Degradation | Model | Clean | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| Gaussian noise | LDA | 0.817 | 0.813 | 0.807 | 0.787 | 0.719 | 0.580 |
| Gaussian noise | LDA + FS | 0.814 | 0.812 | 0.803 | 0.776 | 0.694 | 0.568 |
| Gaussian noise | SVM + HOG | 0.911 | 0.801 | 0.602 | 0.276 | 0.170 | 0.155 |
| Gaussian noise | SVM + HOG + FS | 0.910 | 0.810 | 0.623 | 0.305 | 0.172 | 0.153 |
| Gaussian noise | CNN | 0.928 | 0.850 | 0.582 | 0.267 | 0.146 | 0.118 |
| Gaussian noise | LDA + HOG (ablation) | 0.866 | 0.549 | 0.381 | 0.202 | 0.165 | 0.148 |
| Gaussian blur | LDA | 0.817 | 0.811 | 0.770 | 0.704 | 0.651 | 0.490 |
| Gaussian blur | LDA + FS | 0.814 | 0.810 | 0.770 | 0.709 | 0.663 | 0.566 |
| Gaussian blur | SVM + HOG | 0.911 | 0.903 | 0.835 | 0.736 | 0.601 | 0.366 |
| Gaussian blur | SVM + HOG + FS | 0.910 | 0.901 | 0.827 | 0.726 | 0.593 | 0.340 |
| Gaussian blur | CNN | 0.928 | 0.915 | 0.701 | 0.472 | 0.381 | 0.246 |
| Gaussian blur | LDA + HOG (ablation) | 0.866 | 0.860 | 0.809 | 0.726 | 0.597 | 0.375 |
| Contrast reduction | LDA | 0.817 | 0.796 | 0.638 | 0.302 | 0.200 | 0.176 |
| Contrast reduction | LDA + FS | 0.814 | 0.793 | 0.667 | 0.417 | 0.232 | 0.178 |
| Contrast reduction | SVM + HOG | 0.911 | 0.912 | 0.910 | 0.909 | 0.906 | 0.892 |
| Contrast reduction | SVM + HOG + FS | 0.910 | 0.910 | 0.910 | 0.910 | 0.908 | 0.893 |
| Contrast reduction | CNN | 0.928 | 0.848 | 0.485 | 0.207 | 0.126 | 0.101 |
| Contrast reduction | LDA + HOG (ablation) | 0.866 | 0.865 | 0.865 | 0.864 | 0.862 | 0.853 |
| Brightness increase | LDA | 0.817 | 0.813 | 0.792 | 0.698 | 0.424 | 0.188 |
| Brightness increase | LDA + FS | 0.814 | 0.814 | 0.806 | 0.773 | 0.681 | 0.433 |
| Brightness increase | SVM + HOG | 0.911 | 0.907 | 0.890 | 0.864 | 0.847 | 0.820 |
| Brightness increase | SVM + HOG + FS | 0.910 | 0.907 | 0.890 | 0.866 | 0.848 | 0.824 |
| Brightness increase | CNN | 0.928 | 0.686 | 0.444 | 0.335 | 0.279 | 0.223 |
| Brightness increase | LDA + HOG (ablation) | 0.866 | 0.864 | 0.856 | 0.846 | 0.834 | 0.822 |

**ciFAIR-10**

| Degradation | Model | Clean | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| Gaussian noise | LDA | 0.405 | 0.397 | 0.384 | 0.348 | 0.287 | 0.241 |
| Gaussian noise | LDA + FS | 0.394 | 0.393 | 0.388 | 0.370 | 0.332 | 0.294 |
| Gaussian noise | SVM + HOG | 0.625 | 0.333 | 0.166 | 0.123 | 0.105 | 0.100 |
| Gaussian noise | SVM + HOG + FS | 0.627 | 0.331 | 0.166 | 0.120 | 0.104 | 0.101 |
| Gaussian noise | CNN | 0.802 | 0.520 | 0.276 | 0.163 | 0.131 | 0.121 |
| Gaussian noise | LDA + HOG (ablation) | 0.537 | 0.287 | 0.156 | 0.119 | 0.104 | 0.100 |
| Gaussian blur | LDA | 0.405 | 0.406 | 0.410 | 0.407 | 0.400 | 0.384 |
| Gaussian blur | LDA + FS | 0.394 | 0.395 | 0.399 | 0.397 | 0.391 | 0.378 |
| Gaussian blur | SVM + HOG | 0.625 | 0.614 | 0.495 | 0.367 | 0.283 | 0.197 |
| Gaussian blur | SVM + HOG + FS | 0.627 | 0.614 | 0.503 | 0.378 | 0.291 | 0.202 |
| Gaussian blur | CNN | 0.802 | 0.765 | 0.323 | 0.208 | 0.188 | 0.180 |
| Gaussian blur | LDA + HOG (ablation) | 0.537 | 0.528 | 0.447 | 0.354 | 0.291 | 0.217 |
| Contrast reduction | LDA | 0.405 | 0.394 | 0.367 | 0.293 | 0.190 | 0.147 |
| Contrast reduction | LDA + FS | 0.394 | 0.389 | 0.362 | 0.287 | 0.193 | 0.150 |
| Contrast reduction | SVM + HOG | 0.625 | 0.625 | 0.624 | 0.625 | 0.619 | 0.548 |
| Contrast reduction | SVM + HOG + FS | 0.627 | 0.625 | 0.627 | 0.629 | 0.622 | 0.553 |
| Contrast reduction | CNN | 0.802 | 0.785 | 0.676 | 0.401 | 0.164 | 0.103 |
| Contrast reduction | LDA + HOG (ablation) | 0.537 | 0.540 | 0.538 | 0.538 | 0.529 | 0.438 |
| Brightness increase | LDA | 0.405 | 0.387 | 0.362 | 0.320 | 0.260 | 0.192 |
| Brightness increase | LDA + FS | 0.394 | 0.380 | 0.348 | 0.296 | 0.237 | 0.175 |
| Brightness increase | SVM + HOG | 0.625 | 0.624 | 0.613 | 0.591 | 0.561 | 0.503 |
| Brightness increase | SVM + HOG + FS | 0.627 | 0.624 | 0.615 | 0.599 | 0.561 | 0.509 |
| Brightness increase | CNN | 0.802 | 0.792 | 0.766 | 0.717 | 0.631 | 0.501 |
| Brightness increase | LDA + HOG (ablation) | 0.537 | 0.538 | 0.530 | 0.516 | 0.493 | 0.452 |

**Variability.** Across the 3 independent noise realisations the accuracy of a single model varied by at most
0.012. Across the 3 CNN training seeds the accuracy under a single degraded condition varied by up to
0.124 (Fashion-MNIST) and 0.059 (ciFAIR-10), although on clean test data the three seeds
differ by only 0.0013 and 0.0073. LDA, SVM + HOG and LDA + HOG and the two feature-selection models are single deterministic fits.

### 9.5 Statistical analysis (Phase 6)

Method and definitions: §7.1. Intervals are 95% percentile bootstrap intervals over test images and describe
test-set sampling uncertainty only; for the CNN, training-seed variance is separate (shown in §9.4 and by the
seed-disagreement column below).

**Area under the severity curve** (AUSC; trapezoid rule over severity 0–5 divided by 5, i.e. a mean accuracy including the clean point), point estimate [95% CI]:

**Fashion-MNIST**

| Degradation | LDA | LDA + FS | SVM + HOG | SVM + HOG + FS | CNN | LDA + HOG (ablation) |
|---|---|---|---|---|---|---|
| Gaussian noise | 0.765 [0.759, 0.772] | 0.755 [0.749, 0.762] | 0.476 [0.471, 0.482] | 0.488 [0.483, 0.494] | 0.474 [0.469, 0.479] | 0.361 [0.355, 0.367] |
| Gaussian blur | 0.718 [0.711, 0.725] | 0.728 [0.722, 0.736] | 0.743 [0.737, 0.748] | 0.735 [0.729, 0.740] | 0.611 [0.606, 0.617] | 0.722 [0.716, 0.729] |
| Contrast reduction | 0.486 [0.480, 0.493] | 0.521 [0.515, 0.527] | 0.908 [0.903, 0.913] | 0.908 [0.903, 0.913] | 0.436 [0.432, 0.441] | 0.863 [0.857, 0.870] |
| Brightness increase | 0.646 [0.639, 0.652] | 0.740 [0.733, 0.746] | 0.875 [0.869, 0.880] | 0.875 [0.870, 0.881] | 0.464 [0.458, 0.470] | 0.849 [0.842, 0.855] |

**ciFAIR-10**

| Degradation | LDA | LDA + FS | SVM + HOG | SVM + HOG + FS | CNN | LDA + HOG (ablation) |
|---|---|---|---|---|---|---|
| Gaussian noise | 0.348 [0.340, 0.355] | 0.366 [0.358, 0.374] | 0.218 [0.212, 0.223] | 0.217 [0.212, 0.222] | 0.310 [0.305, 0.315] | 0.197 [0.192, 0.202] |
| Gaussian blur | 0.404 [0.395, 0.413] | 0.394 [0.385, 0.403] | 0.434 [0.427, 0.441] | 0.440 [0.433, 0.448] | 0.395 [0.390, 0.400] | 0.399 [0.392, 0.407] |
| Contrast reduction | 0.304 [0.297, 0.311] | 0.301 [0.294, 0.308] | 0.616 [0.607, 0.625] | 0.619 [0.610, 0.628] | 0.496 [0.490, 0.501] | 0.526 [0.517, 0.535] |
| Brightness increase | 0.326 [0.318, 0.333] | 0.309 [0.301, 0.317] | 0.591 [0.582, 0.600] | 0.593 [0.585, 0.603] | 0.712 [0.705, 0.719] | 0.514 [0.505, 0.523] |


**Chance-adjusted retention**, $(\text{acc}_s - 0.1)/(\text{acc}_{\text{clean}} - 0.1)$ averaged over severities 1–5, point estimate [95% CI]. A value of 1 means no loss, 0 means chance level, and negative values mean below chance:

**Fashion-MNIST**

| Degradation | LDA | LDA + FS | SVM + HOG | SVM + HOG + FS | CNN | LDA + HOG (ablation) |
|---|---|---|---|---|---|---|
| Gaussian noise | 0.895 [0.890, 0.900] | 0.883 [0.878, 0.888] | 0.371 [0.364, 0.378] | 0.386 [0.378, 0.393] | 0.354 [0.348, 0.360] | 0.247 [0.238, 0.255] |
| Gaussian blur | 0.817 [0.809, 0.824] | 0.845 [0.838, 0.853] | 0.725 [0.718, 0.732] | 0.713 [0.706, 0.720] | 0.535 [0.528, 0.542] | 0.748 [0.740, 0.756] |
| Contrast reduction | 0.450 [0.442, 0.458] | 0.501 [0.492, 0.509] | 0.993 [0.991, 0.996] | 0.995 [0.992, 0.997] | 0.306 [0.300, 0.312] | 0.995 [0.992, 0.997] |
| Brightness increase | 0.674 [0.668, 0.680] | 0.842 [0.837, 0.849] | 0.944 [0.939, 0.948] | 0.946 [0.941, 0.951] | 0.354 [0.346, 0.363] | 0.971 [0.967, 0.976] |

**ciFAIR-10**

| Degradation | LDA | LDA + FS | SVM + HOG | SVM + HOG + FS | CNN | LDA + HOG (ablation) |
|---|---|---|---|---|---|---|
| Gaussian noise | 0.759 [0.745, 0.772] | 0.870 [0.856, 0.883] | 0.124 [0.114, 0.133] | 0.122 [0.112, 0.131] | 0.202 [0.196, 0.209] | 0.122 [0.109, 0.133] |
| Gaussian blur | 0.990 [0.973, 1.007] | 0.993 [0.978, 1.008] | 0.555 [0.543, 0.566] | 0.564 [0.552, 0.576] | 0.332 [0.323, 0.340] | 0.612 [0.597, 0.626] |
| Contrast reduction | 0.585 [0.567, 0.603] | 0.599 [0.580, 0.618] | 0.968 [0.961, 0.974] | 0.970 [0.962, 0.977] | 0.464 [0.458, 0.470] | 0.953 [0.944, 0.962] |
| Brightness increase | 0.671 [0.651, 0.689] | 0.637 [0.617, 0.656] | 0.911 [0.903, 0.920] | 0.913 [0.904, 0.922] | 0.828 [0.821, 0.835] | 0.928 [0.917, 0.939] |


**Paired comparisons.** For each pair, the number of the 21 conditions (clean plus 4 families × 5 severities) in which an exact McNemar test, Holm-corrected over all 462 tests, finds the first or the second model significantly more accurate (Holm p < 0.05). For pairs with the CNN each of the 3 seeds is tested; a condition counts for a model only if all 3 seeds are significant in the same direction, and "CNN seeds disagree" counts conditions that are significant for only some seeds.

| Dataset | First model | Second model | First more accurate | Second more accurate | No significant difference | CNN seeds disagree |
|---|---|---|---|---|---|---|
| Fashion-MNIST | SVM + HOG | LDA | 14 | 6 | 1 | – |
| Fashion-MNIST | CNN | LDA | 3 | 16 | 0 | 2 |
| Fashion-MNIST | CNN | SVM + HOG | 2 | 16 | 0 | 3 |
| Fashion-MNIST | LDA + FS | LDA | 9 | 3 | 9 | – |
| Fashion-MNIST | SVM + HOG + FS | SVM + HOG | 3 | 3 | 15 | – |
| Fashion-MNIST | LDA + HOG (ablation) | LDA | 14 | 7 | 0 | – |
| Fashion-MNIST | SVM + HOG | LDA + HOG (ablation) | 15 | 0 | 6 | – |
| ciFAIR-10 | SVM + HOG | LDA | 13 | 8 | 0 | – |
| ciFAIR-10 | CNN | LDA | 11 | 9 | 0 | 1 |
| ciFAIR-10 | CNN | SVM + HOG | 11 | 6 | 0 | 4 |
| ciFAIR-10 | LDA + FS | LDA | 3 | 9 | 9 | – |
| ciFAIR-10 | SVM + HOG + FS | SVM + HOG | 1 | 0 | 20 | – |
| ciFAIR-10 | LDA + HOG (ablation) | LDA | 13 | 8 | 0 | – |
| ciFAIR-10 | SVM + HOG | LDA + HOG (ablation) | 15 | 1 | 5 | – |

**Effect of feature selection on test accuracy** (difference of the displayed accuracies, i.e. averaged over the 3 noise realisations for noise, with its paired bootstrap 95% CI, which is not adjusted for multiplicity; `*` = the exact McNemar test is significant after Holm correction — for noise conditions that test uses noise realisation 0 only, see §7.1). Absence of `*` means no significant difference was found, not that the models are equivalent:

| Dataset | Comparison | Family | Severity 3 | Severity 5 |
|---|---|---|---|---|
| Fashion-MNIST | LDA + FS − LDA | Gaussian noise | -0.011 [-0.014, -0.008] * | -0.013 [-0.017, -0.008] * |
| Fashion-MNIST | LDA + FS − LDA | Gaussian blur | +0.005 [+0.001, +0.008] | +0.075 [+0.068, +0.082] * |
| Fashion-MNIST | LDA + FS − LDA | Contrast reduction | +0.115 [+0.108, +0.123] * | +0.002 [-0.001, +0.006] |
| Fashion-MNIST | LDA + FS − LDA | Brightness increase | +0.075 [+0.069, +0.082] * | +0.245 [+0.237, +0.254] * |
| Fashion-MNIST | SVM + HOG + FS − SVM + HOG | Gaussian noise | +0.028 [+0.026, +0.031] * | -0.001 [-0.003, +0.000] |
| Fashion-MNIST | SVM + HOG + FS − SVM + HOG | Gaussian blur | -0.010 [-0.014, -0.006] * | -0.026 [-0.031, -0.021] * |
| Fashion-MNIST | SVM + HOG + FS − SVM + HOG | Contrast reduction | +0.000 [-0.003, +0.003] | +0.001 [-0.002, +0.004] |
| Fashion-MNIST | SVM + HOG + FS − SVM + HOG | Brightness increase | +0.002 [-0.001, +0.006] | +0.004 [+0.000, +0.008] |
| ciFAIR-10 | LDA + FS − LDA | Gaussian noise | +0.022 [+0.018, +0.028] * | +0.054 [+0.048, +0.059] * |
| ciFAIR-10 | LDA + FS − LDA | Gaussian blur | -0.010 [-0.015, -0.004] * | -0.006 [-0.010, -0.001] |
| ciFAIR-10 | LDA + FS − LDA | Contrast reduction | -0.005 [-0.011, -0.000] | +0.003 [+0.001, +0.005] |
| ciFAIR-10 | LDA + FS − LDA | Brightness increase | -0.024 [-0.030, -0.019] * | -0.017 [-0.022, -0.014] * |
| ciFAIR-10 | SVM + HOG + FS − SVM + HOG | Gaussian noise | -0.003 [-0.005, -0.001] | +0.001 [-0.001, +0.003] |
| ciFAIR-10 | SVM + HOG + FS − SVM + HOG | Gaussian blur | +0.011 [+0.005, +0.017] * | +0.005 [-0.000, +0.011] |
| ciFAIR-10 | SVM + HOG + FS − SVM + HOG | Contrast reduction | +0.005 [-0.001, +0.011] | +0.005 [-0.001, +0.012] |
| ciFAIR-10 | SVM + HOG + FS − SVM + HOG | Brightness increase | +0.008 [+0.002, +0.014] | +0.005 [-0.001, +0.011] |

**Prediction collapse.** Share of the test predictions that fall in the single most-predicted class (about 0.10 when predictions are spread over 10 balanced classes; 1.00 when the model outputs one label):

| Dataset | Model | Clean | Severity 5 (mean over the 4 families) |
|---|---|---|---|
| Fashion-MNIST | LDA | 0.11 | 0.56 |
| Fashion-MNIST | LDA + FS | 0.10 | 0.40 |
| Fashion-MNIST | SVM + HOG | 0.10 | 0.35 |
| Fashion-MNIST | SVM + HOG + FS | 0.10 | 0.35 |
| Fashion-MNIST | CNN | 0.11 | 0.78 |
| Fashion-MNIST | LDA + HOG (ablation) | 0.11 | 0.27 |
| ciFAIR-10 | LDA | 0.11 | 0.40 |
| ciFAIR-10 | LDA + FS | 0.12 | 0.40 |
| ciFAIR-10 | SVM + HOG | 0.11 | 0.42 |
| ciFAIR-10 | SVM + HOG + FS | 0.11 | 0.37 |
| ciFAIR-10 | CNN | 0.12 | 0.59 |
| ciFAIR-10 | LDA + HOG (ablation) | 0.12 | 0.42 |

### 9.6 Severity audit

See the PSNR/SSIM table in §5.3 (model-independent; clean-training calibration subset).

---

## 10. Results Comparison and Discussion

This section separates **observations** (read directly from the tables in §9) from **explanations**
(theoretical, **not tested here**). The statistics (§9.5) cover test-set sampling uncertainty and the CNN's three
training seeds; there is one test set per dataset and one fit of each deterministic model, so conclusions apply to
these data and these fitted models, not to the model families in general.

### 10.1 Observations

**Clean performance.** On both datasets the clean-test ranking is CNN > SVM + HOG > LDA + HOG (ablation) > LDA
(Fashion-MNIST: 0.927–0.928, 0.911, 0.866, 0.817; ciFAIR-10: 0.798–0.806, 0.625, 0.537, 0.405). The CNN's
advantage over SVM + HOG is about 0.017 on Fashion-MNIST and about 0.17–0.18 on ciFAIR-10. Test accuracy is
0.4–2.0 points below validation accuracy for every model (the largest gap is CNN seed 43 on ciFAIR-10), the expected
direction for scores used to choose hyperparameters.

**Effect of HOG features on LDA.** Replacing raw pixels by HOG features raised clean test accuracy from 0.817 to
0.866 on Fashion-MNIST (+0.050) and from 0.405 to 0.537 on ciFAIR-10 (+0.132); replacing the LDA classifier by an
RBF-SVM on the same features raised it further to 0.911 (+0.045) and 0.625 (+0.088).

**Clean accuracy does not predict robustness.** On Fashion-MNIST the CNN, the most accurate clean model, is
significantly *less* accurate than SVM + HOG in 16 of the 21 conditions and than LDA in 16 (more accurate in 2 and 3 conditions
respectively; for 3 and 2 conditions the three training seeds disagree). At severity 5 the best of the three main
models is LDA for Gaussian noise (0.580) and blur (0.490) and SVM + HOG for contrast (0.892) and brightness (0.820);
the CNN is lowest of the three at severity 5 for noise (0.118), blur (0.246) and contrast (0.101), and second lowest
for brightness (0.223, above LDA's 0.188). On ciFAIR-10 the CNN is significantly more accurate than SVM + HOG in 11
conditions and less accurate in 6 (seeds disagree in 4); it is the best main model at severity 1 under all four families,
but by severity 5 LDA is best for noise (0.241) and blur (0.384), SVM + HOG for contrast (0.548), and the CNN and
SVM + HOG are level for brightness (0.501 and 0.503).

**The ranking depends on the degradation family.**
* *Contrast reduction.* The HOG-based models are nearly unaffected on Fashion-MNIST (SVM + HOG 0.911 clean, 0.892 at severity 5; LDA + HOG 0.866 → 0.853) and degrade mainly at the last level on ciFAIR-10 (SVM + HOG 0.625 → 0.619 → 0.548 at severities 3, 4, 5). LDA on pixels falls to 0.176 and the CNN to 0.101 (Fashion-MNIST) at severity 5.
* *Brightness (additive).* SVM + HOG and LDA + HOG lose about 0.09 and 0.04 at severity 5 on Fashion-MNIST (0.911 → 0.820; 0.866 → 0.822) and about 0.12 and 0.09 on ciFAIR-10 (0.625 → 0.503; 0.537 → 0.452). The CNN already falls from 0.928 to 0.686 at severity 1 on Fashion-MNIST (seed range 0.622–0.723), whereas on ciFAIR-10 it degrades gradually (0.802 → 0.792 → … → 0.501) and stays above SVM + HOG through severity 4.
* *Gaussian noise.* The HOG-based models are the most sensitive: SVM + HOG falls from 0.625 to 0.333 at severity 1 on ciFAIR-10 and from 0.911 to 0.276 at severity 3 on Fashion-MNIST, and LDA + HOG to 0.549 at severity 1 on Fashion-MNIST. Pixel-based LDA is the most tolerant (0.817 → 0.580 at severity 5 on Fashion-MNIST; 0.405 → 0.241 on ciFAIR-10).
* *Gaussian blur.* On Fashion-MNIST all models decline steadily; the CNN declines fastest (0.701 at severity 2 versus 0.835 for SVM + HOG and 0.770 for LDA). On ciFAIR-10 LDA is almost unchanged (0.405 → 0.384) while the CNN drops from 0.802 to 0.323 at severity 2 and SVM + HOG to 0.495.

**Summaries over the whole severity range.** The area under the severity curve (mean accuracy over severity 0–5) ranks the models
differently in each family (§9.5): among the three main models, on Fashion-MNIST LDA is highest under noise (0.765) and SVM + HOG under blur, contrast and
brightness (0.743, 0.908, 0.875), and the CNN is lowest in all four families (under noise 0.474 against 0.476 for SVM + HOG); on ciFAIR-10 the CNN is
highest under brightness (0.712), SVM + HOG under contrast (0.616) and blur (0.434), and LDA under noise (0.348). Chance-adjusted retention
and absolute accuracy can disagree: on ciFAIR-10 LDA retains 0.99 of its above-chance accuracy under blur, but from a clean accuracy of only 0.405, so its
absolute blur accuracy is below that of SVM + HOG over the curve. Both views are therefore needed.

**HOG changes the robustness profile of LDA, not just its accuracy.** LDA with and without HOG differ only in the
input representation. HOG raised clean accuracy (above) and made LDA nearly insensitive to contrast and brightness,
but made it far more sensitive to Gaussian noise (Fashion-MNIST severity 1: 0.813 for LDA, 0.549 for LDA + HOG). SVM + HOG
is significantly more accurate than LDA + HOG in 15 of 21 conditions on both datasets and never significantly worse on Fashion-MNIST (once on ciFAIR-10);
most of the remaining conditions are high-noise ones where both are near chance.

**Failures are collapses onto a few classes.** On clean data every model spreads its predictions over the classes (the most-predicted class
receives 0.10–0.12 of the predictions). At severity 5 the share of predictions in the single most-predicted class (mean over the four families) is 0.78 for the CNN on Fashion-MNIST
(0.56 for LDA, 0.35 for SVM + HOG, 0.27 for LDA + HOG) and 0.59 on ciFAIR-10 (0.40–0.42 for LDA, SVM + HOG and LDA + HOG). The confusion matrices show the
same: under severe contrast reduction the CNN labels most Fashion-MNIST images "Bag" or "Shirt".

**Floor effects.** Several conditions end near chance (about 0.10): for example the CNN under severity-5 contrast
(0.101 on Fashion-MNIST, 0.103 on ciFAIR-10), and SVM + HOG and LDA + HOG under severity-5 noise on ciFAIR-10 (0.100). Beyond
that point the models are indistinguishable, so differences between them at the extreme levels carry no information.

**Feature selection (ANOVA F-test, keeping 75% of the features in all four models).**
* *Clean data.* Selection gave no consistent, statistically significant improvement in clean accuracy: validation accuracy increased monotonically with the fraction of features kept (e.g. Fashion-MNIST LDA 0.653, 0.778, 0.816, 0.828 at 10%, 25%, 50%, 75% against 0.830 without selection), and the selected fraction was the largest in the grid. Clean test accuracy was LDA 0.817 → 0.814 and SVM + HOG 0.911 → 0.910 on Fashion-MNIST, LDA 0.405 → 0.394 (a significant decrease) and SVM + HOG 0.625 → 0.627 (+0.003, not significant) on ciFAIR-10.
* *LDA + FS on Fashion-MNIST* is significantly more accurate than LDA in 9 of the 21 conditions and less accurate in 3 (no difference in 9): +0.245 [+0.237, +0.254] under severity-5 brightness, +0.075 at severity 3, +0.115 [+0.108, +0.123] under severity-3 contrast, +0.075 under severity-5 blur, but −0.011 and −0.013 under severity-3 and 5 noise.
* *LDA + FS on ciFAIR-10* goes the other way in most conditions: more accurate in 3 conditions (noise: +0.022 at severity 3 and +0.054 [+0.048, +0.059] at severity 5) and less accurate in 9 (brightness −0.024 and −0.018 at severities 3 and 5, blur about −0.010, clean −0.011).
* *SVM + HOG + FS* is close to SVM + HOG but not identical: no significant difference in 15 of 21 conditions on Fashion-MNIST, while 6 differ significantly (3 better, all under noise at severities 1–3, up to +0.028; 3 worse, all under blur, down to −0.026 at severity 5), and no significant difference in 20 of 21 on ciFAIR-10 (1 better: blur at severity 3, +0.011). The effects are small in absolute terms, but a non-significant result does not show that the two models are equivalent.
* So the effect of the selection stage on robustness is not consistent across datasets for LDA, and is small and mixed in sign for SVM + HOG. The selection was fitted on clean data only and was not designed to improve robustness.

**Variability.** Re-drawing the Gaussian noise (3 realisations) changed a model's accuracy by at most 0.012.
Training seed matters far more for the CNN: under a single degraded condition the three seeds differ by up to 0.124
(Fashion-MNIST) and 0.059 (ciFAIR-10), although on clean test data they differ by only 0.001 and 0.007. Clean accuracy
therefore does not reveal how a particular CNN seed will behave under degradation. The test-set sampling uncertainty is small by comparison: the 95% bootstrap
interval of an accuracy has a half-width of 0.008 on average (0.010 at most), so the differences between models in §9.4 are mostly far larger than test-set sampling noise.
In the CNN comparisons the result depends on the training seed in 5 of the 42 condition-pair cells on each dataset (§9.5).

**Tuning boundaries.** SVM + HOG improved with both larger C and larger γ-multiplier and its selected setting lies
at the largest value of both grids (both datasets); the CNN's selected learning rate (largest) and weight decay
(smallest) are also at the edge of their grids, and the selected configuration's best validation epoch was epoch 10 for all three Fashion-MNIST seeds and epoch 9 for ciFAIR-10 (seed 42), i.e. validation accuracy was still rising when training stopped. The feature-selection fraction (75%) is also at the edge of its grid.
The recorded optima may therefore not be the true optima of these model families.

**Degradation audit.** The audit (§5.3) shows that every ladder lowers PSNR and SSIM monotonically, that
the two measures rank the families differently (brightness has the lowest PSNR but the highest SSIM at
severity 5 on both datasets), and that identical parameters reduce similarity more on Fashion-MNIST than on ciFAIR-10 for blur
(severity 2: 19.1 dB versus 25.1 dB). On Fashion-MNIST, noise clipping raises the mean intensity (+2.6 grey levels at σ = 0.05 to
+22.3 at σ = 0.50) because about half of the pixels are exactly black; brightness at β = 0.5 clips 31% (Fashion-MNIST) and 44%
(ciFAIR-10) of pixel values. Equal severity numbers are therefore not equal difficulty across families.

### 10.2 Explanations (theoretical, not tested here)

* The CNN's clean advantage is consistent with learned, translation-tolerant features, versus a linear model on raw pixels and fixed hand-crafted features.
* The gain from HOG over raw pixels for LDA is consistent with HOG encoding local edge structure that a linear model on pixel intensities cannot express; the further gain from the SVM is consistent with the RBF kernel's non-linear boundary.
* The insensitivity of the HOG-based models to contrast reduction is consistent with HOG's block normalisation (L2-Hys) removing the overall scale of local gradients; their tolerance of additive brightness is consistent with an intensity offset leaving gradients unchanged except where pixels clip.
* The sensitivity of the HOG-based models to Gaussian noise is consistent with gradient computation amplifying independent pixel noise, while a linear classifier on raw pixels averages noise over many pixels.
* The CNN's sharp loss under brightness on Fashion-MNIST is consistent with a network trained without augmentation on a clean background that is exactly black: an additive offset shifts every activation away from the statistics that its batch-normalisation layers stored from clean training data. This was not tested (for example by examining activations).
* The collapse of failing models onto one or two classes is consistent with degraded inputs moving every sample into a region of the learned feature space that was assigned to a single class.
* The large gain of LDA + FS under brightness on Fashion-MNIST is consistent with the weakly regularised covariance estimate that was selected there (shrinkage 0.001): a nearly unregularised discriminant can put large weights on low-variance pixels such as the always-dark border, and an additive offset then moves the score strongly; the F-test drops 25% of the pixels, which the map in `feature_selection_maps.png` shows are mostly border pixels. Whether this explains the effect was not tested, and it does not explain why the same selection hurts LDA under brightness on ciFAIR-10.
* The small effect of selection on SVM + HOG is consistent with the F-test removing only the weakest quarter of the HOG features, which changes the RBF kernel's distances little; the few significant changes (better under moderate noise, worse under blur) were not investigated.
* The low LDA accuracy on ciFAIR-10 is consistent with colour natural images having larger within-class variation (object position, background, colour) than centred, similar-looking Fashion-MNIST items, which a single linear boundary on raw pixels cannot handle.
* The audit's disagreement between PSNR and SSIM is consistent with the metrics measuring different things: PSNR is dominated by squared intensity error (large for a brightness shift), while SSIM compares local structure, which an additive shift leaves largely intact.

---

## 11. Conclusion

This project implements a controlled comparison of LDA, SVM + HOG and a CNN (plus an LDA + HOG ablation and
variants of LDA and SVM + HOG with ANOVA F-test feature selection), trained on clean Fashion-MNIST and ciFAIR-10 images and evaluated on test
images degraded by Gaussian noise, Gaussian blur, contrast reduction and brightness increase at five severities
each (21 conditions per dataset). The pipeline is designed for validity: leakage-filtered test sets, clean-only
training and tuning, a severity ladder audited and frozen before any model evaluation, identical degraded arrays for
all models (checked by digest), frozen hash-verified models, and paired bootstrap intervals and exact McNemar tests with Holm correction.
On clean test data the CNN is the most accurate model on both datasets, followed by SVM + HOG and LDA. Under degradation that order does not
hold: the ranking depends on the degradation family and severity. The HOG-based models are nearly unaffected by contrast reduction and
additive brightness but sensitive to Gaussian noise; pixel-based LDA is the most tolerant of noise and blur, from a low clean accuracy; and on
Fashion-MNIST the CNN falls furthest by severity 5 under noise, blur and contrast and loses 0.24 already under severity-1 brightness, with large
differences between training seeds. Failing models collapse onto a few predicted classes. Feature selection gave no consistent significant improvement in clean accuracy;
its effect on robustness was large for LDA on Fashion-MNIST (up to +0.25 under brightness), mixed or negative for LDA on ciFAIR-10, and small and mixed
for SVM + HOG. These are findings for one test set per dataset and three CNN seeds; the explanations offered are consistent with the results but untested.

---

## 12. Limitations and Future Scope

### 12.1 Limitations

* **Synthetic degradations.** Noise, blur, contrast and brightness are simulated on small images (28 × 28 grayscale, 32 × 32 colour); they are not real-world degradations. Gaussian noise is i.i.d. and independent across channels, unlike real sensor noise; blur is isotropic; contrast uses one RGB-wide mean; brightness is positive only (darkening is not tested); the uint8 storage quantises every degraded image, which is coarse at the most extreme contrast level.
* **Severity levels are not equally hard across families.** The ladders are front-loaded in PSNR (e.g. blur loses 8.5–9.3 dB from severity 1 to 2 and 1.2–2.8 dB per step afterwards), and PSNR/SSIM do not measure recognition difficulty. The audit used 1,000 images per dataset, one noise realisation and no confidence intervals.
* **Distance-filtered benchmarks.** The near-duplicate thresholds were chosen by inspecting test/train pairs, so the test set was inspected during construction; on CIFAR-style images a pixel threshold cannot separate duplicates from look-alikes, and filtering changes the evaluated population in a way that has not been measured.
* **Few training seeds; intervals cover the test set only.** The CNN has 3 seeds and its accuracy under one degraded condition varies by up to 0.124 between them; the deterministic models have one fit each; the bootstrap intervals and McNemar tests describe test-set sampling only and treat test images as independent, and the Holm correction covers the tests listed in §7.1 only. Results are for one test set per dataset.
* **Feature selection is one filter method on two models.** The ANOVA F-test scores features one at a time on clean data; the selected fraction was the largest in the grid (75%) for all four models, so a larger fraction might score better on clean validation; the CNN has no separate selection stage; other selection methods (mutual information, embedded or wrapper methods) were not tried.
* **Tuning is small and sometimes at the grid edge.** The grids have 4–6 configurations; the SVM, the CNN's learning rate and weight decay, and LDA + HOG selected values at the edge of their grids; the CNN was still improving at epoch 10; the SVM was tuned on a 10,000-image subset; the HOG geometry is fixed and untuned. Validation scores used for selection are optimistic.
* **Models differ in determinism.** LDA and SVM are deterministic given the data; the CNN depends on the training seed, which is why three seeds are trained. PyTorch determinism is best-effort and may differ across hardware and library versions.
* **Pickled model files.** Sklearn models are stored with `joblib` (pickle) and loaded after a hash check; they should not be loaded from untrusted sources.
* **Caches and fingerprints.** Degradation and split caches are fingerprinted but their contents are not hashed; the frozen-ladder fingerprint covers the declared specification, not the code, so changing a transformation's behaviour requires a manual version bump and re-freeze.
* **Scope of the ciFAIR duplicate figure.** The ciFAIR page states 3.25% duplicates; 308 changed images (3.08%) were measured and the difference was not reconciled.

### 12.2 Future scope

* More CNN seeds, so that seed variance can be included in the intervals, and analyses stratified by class or by distance to the training set.
* Sensitivity analyses reusing the same frozen models: the original CIFAR-10 test set, ciFAIR-10 and Fashion-MNIST without the pixel filter, and performance stratified by distance to the nearest training image (the loader supports these variants through flags; the experiment runner does not run them yet).
* Wider tuning: larger SVM and CNN grids, longer CNN training, tuned HOG geometry, and other feature-selection methods and larger kept fractions, feature selection on the CNN's learned features, and dimensionality reduction (PCA is not used).
* Additional degradations (darkening, motion blur, JPEG compression, salt-and-pepper noise, combined degradations) and additional datasets or higher-resolution images.
* Comparison with models trained with augmentation or noise-aware training, to separate inherent robustness from training-time exposure.

---

## Appendix: source files for this document

| Topic | Source |
|---|---|
| Datasets, splits, thresholds, checksums | `src/config.py`, `src/data_loader.py`, `src/leakage.py`, `README.md` |
| Preprocessing | `src/preprocessing.py` |
| Degradations and ladder | `src/degradations.py`, `results/metrics/severity_ladder_frozen.json`, `docs/degradations.md` |
| Severity audit | `src/calibration.py`, `results/metrics/severity_calibration.csv` |
| Models and HOG | `src/models/classical.py`, `src/models/cnn.py`, `src/models/__init__.py`, `docs/models.md` |
| Tuning and training | `src/train.py`, `results/metrics/models/*.json`, `logs/train_full.log` |
| Evaluation and metrics | `src/evaluate_clean.py`, `src/run_robustness.py`, `src/metrics.py` |
| Results | `results/metrics/clean_test_metrics.csv`, `results/metrics/robustness_results.csv`, `results/metrics/robustness_run.json`, `results/metrics/degraded_test_digests.json`, `results/predictions/`, `results/confusion_matrices/` |
| Figures | `src/visualization.py`, `src/leakage.py`, `results/plots/*.png` |

> **Superseded for submission:** the IEEE-format paper is `paper/main.tex` (compiled to `paper/main.pdf`). This draft is kept as a longer working version that
> discusses feature selection at more length; its reference numbers are its own.

# Paper draft — Robust Pattern Recognition Under Noise and Image Degradation

Working draft of every section requested for Phase 7. All numbers come from the saved result files
(`results/metrics/`, `results/paper_tables/`) and are the same as in `docs/project_documentation.md`; nothing here
comes from experiments that were not run. Citations [1]–[10] are the references verified against web sources
(listed at the end). Statements that are explanations rather than observations are labelled as such. Section "Notes
for the team" at the end is not paper text.

**Candidate titles.** (1) *Robustness of LDA, SVM + HOG and a CNN to Test-Time Image Degradations: A Controlled
Comparison.* (2) *How Do Classical and Deep Classifiers Fail Under Noise, Blur, Contrast and Brightness Changes?*

---

## Abstract

Image classifiers are usually trained and tested on clean images, but deployed on images degraded by noise, blur,
poor contrast or exposure changes. We present a controlled comparison of three pattern-recognition approaches —
linear discriminant analysis (LDA) on pixels, a support vector machine on histogram-of-oriented-gradients features
(SVM + HOG), and a small convolutional neural network (CNN) — together with versions of LDA and SVM + HOG that use
ANOVA F-test feature selection. All models are trained and tuned on clean Fashion-MNIST and ciFAIR-10 images only,
frozen, and then scored on test images degraded by Gaussian noise, Gaussian blur, contrast reduction and brightness
increase at five audited severity levels each, with bit-identical degraded images for every model. We report accuracy
with paired bootstrap intervals, exact McNemar tests with Holm correction, and chance-adjusted retention.
On clean data the CNN is the most accurate model (0.928 on Fashion-MNIST, about 0.80 on ciFAIR-10), but the ranking
does not persist under degradation: on Fashion-MNIST the CNN is significantly less accurate than SVM + HOG in 16 of 21
conditions. HOG-based models are nearly unaffected by contrast reduction and additive brightness but sensitive to
noise, pixel-based LDA is the most tolerant of noise and blur, and failing models collapse onto one or two predicted
classes. Feature selection gave no consistent significant improvement in clean accuracy; its effect on robustness was large for LDA
on Fashion-MNIST (up to +0.25 accuracy under brightness), mixed for LDA on ciFAIR-10 and small and mixed in sign for SVM + HOG. The CNN's accuracy
under a given degradation varied by up to 0.12 between training seeds even though clean accuracy varied by about 0.001.

**Keywords:** robustness, image degradation, pattern recognition, HOG, convolutional neural network, feature selection.

---

## I. Introduction

Pattern-recognition systems are normally developed on clean benchmark images, whereas images in use are degraded by
sensor noise, defocus, poor illumination or low contrast. A model that scores well on clean data can lose accuracy
when input quality changes, and different families of classifiers may fail in different ways. Knowing how each family
degrades, and under which degradation, helps in choosing a method when input quality is not controlled.

This work is a **controlled comparison**; it does not propose a new algorithm. We ask:

* **RQ1.** How does the accuracy of a linear statistical classifier (LDA), a hand-crafted-feature kernel classifier
  (SVM + HOG) and a CNN, all trained on clean images only, change as test images are degraded by four types of
  degradation at five severities?
* **RQ2.** Does the ranking of the models by clean accuracy persist under degradation?
* **RQ3.** Does a feature-selection stage fitted on clean data change a classical model's robustness?

The contributions are methodological: (i) near-duplicate-filtered versions of two benchmarks; (ii) a degradation ladder
that is audited and frozen before any model is evaluated; (iii) a freeze-then-evaluate protocol in which models are
trained and tuned on clean data, hashed, and scored on degraded test images that are bit-identical across models;
(iv) paired statistical comparisons with intervals and multiple-comparison correction; and (v) an empirical account of
how each model family fails. *Figures: Fig. 1 (pipelines, `pipelines.png`).*

---

## II. Literature Review

**Robustness benchmarks and image quality.** Hendrycks and Dietterich introduced ImageNet-C and ImageNet-P to measure
classifier robustness to common corruptions and perturbations rather than worst-case adversarial changes, and reported
that relative corruption robustness changed little between AlexNet and ResNet [4]. Dodge and Karam evaluated four deep
networks on ImageNet under five quality distortions (blur, noise, contrast, JPEG and JPEG2000 compression) and found
the networks susceptible to them, particularly to blur and noise [5]. Both studies concern large natural-image
datasets and deep networks; our study uses small datasets so that classical and deep models can be compared with the
same degraded inputs and tuned with the same discipline.

**Benchmark hygiene.** Barz and Denzler found that about 3.3% of CIFAR-10 test images have near-duplicates in the
training set and released ciFAIR, in which the duplicated test images are replaced [2]. Geier reported near-duplicates
between Fashion-MNIST training and test images and proposed a cleaned version [3]. We use ciFAIR-10, and we apply a
pixel-distance filter to Fashion-MNIST (Section III-A).

**Classical and deep models.** Fashion-MNIST was introduced as a drop-in replacement for MNIST with 70,000 28 × 28
grayscale images of ten clothing categories [1]. HOG descriptors, computed from oriented gradients with contrast
normalisation in overlapping blocks and classified by a linear SVM, were introduced for human detection [6]. Gradient-trained
convolutional networks were demonstrated for document recognition by LeCun et al. [7].

**Feature selection.** Guyon and Elisseeff describe variable selection as serving three goals: improving predictor
performance, providing faster and more cost-effective predictors, and giving a better understanding of the process
that generated the data [8]. Filter methods such as the ANOVA F-test, used here, score each feature on its own.

**Statistical comparison.** Dietterich reviewed approximate tests for comparing classifiers and discussed McNemar's
test for settings where each algorithm can be run only once [9]; Holm's step-down procedure controls the family-wise
error rate over several tests [10].

**Position of this study.** We combine a classical-versus-deep comparison with (a) a frozen, audited degradation
ladder, (b) leakage-filtered test sets, (c) identical degraded inputs for every model, and (d) paired tests with
multiple-comparison correction. This review is limited to the references above; it is not a systematic survey (see
Notes for the team).

---

## III. Methodology

### A. Data and leakage control

Fashion-MNIST (28 × 28, grayscale, 10 classes) is the primary dataset and ciFAIR-10 (32 × 32, RGB, 10 classes) the
replication dataset. Each official training set is split into training and validation parts with a stratified split
(seed 42; validation size 10,000 for Fashion-MNIST, 5,000 for ciFAIR-10). Near-duplicates are removed by RMS pixel
distance on the [0, 1] scale: validation images against the retained training images, test images against the full
official training set, with thresholds 0.04 (Fashion-MNIST) and 0.08 (ciFAIR-10). This removed 14 validation and
18 test images from Fashion-MNIST and 72 and 99 from ciFAIR-10, leaving 50,000 / 9,986 / 9,982 and 45,000 / 4,928 /
9,901 training / validation / test images. The thresholds were chosen by inspecting ranked image pairs before any model
existed, so the test set was inspected during benchmark construction; on CIFAR-style images a pixel threshold cannot
separate duplicates from look-alikes, so these are distance-filtered benchmarks. *Table: dataset summary (from
`project_documentation.md` §2.1).*

### B. Degradations

Four degradations are applied to test images only, at five severities each (Table I): additive Gaussian noise
$x' = x + \sigma z$ (independent per pixel and channel), Gaussian blur (separable kernel, mirrored borders),
contrast reduction $x' = m + \alpha(x - m)$ with $m$ the image mean, and additive brightness $x' = x + \beta$.
Results are clipped to [0, 1] and stored as 8-bit integers. Noise is generated per image from
(seed, realisation, split, image id), so it is independent of batch size and order, and three independent noise
realisations are evaluated. The severity values were audited on 1,000 clean training images per dataset using PSNR and
SSIM against three criteria declared in advance (monotone decrease with severity, at least 99% of images altered at
severity 1, at most 1% constant at severity 5), which all passed; the ladder was then frozen, and the evaluation code
refuses to run if the frozen definition changes. PSNR and SSIM measure similarity, not recognition difficulty, so equal
severity numbers are not equal difficulty across families.

*Table I (severity values):* noise σ = 0.05, 0.10, 0.20, 0.35, 0.50; blur σ = 0.5, 1, 1.5, 2, 3 pixels; contrast α =
0.75, 0.50, 0.30, 0.15, 0.05; brightness β = 0.1, 0.2, 0.3, 0.4, 0.5. *Figures: degradation grids
(`degradation_grid_*`), audit curves (`severity_calibration_*`).*

### C. Models

* **LDA.** Frozen scalar normalisation (one mean and one standard deviation from the clean training images), flatten,
  shrinkage LDA (`solver="lsqr"`).
* **SVM + HOG.** HOG features from the unit-scale image (9 orientations, 4 × 4-pixel cells, 2 × 2-cell blocks, L2-Hys
  block normalisation; 1,296 features for 28 × 28 and 1,764 for 32 × 32 images; fixed in advance, not tuned),
  followed by an RBF-kernel SVM. HOG is computed after degradation, inside the model.
* **CNN.** Five 3 × 3 convolution layers (32, 32, 64, 64, 128 channels) with batch normalisation and ReLU, two max-pool
  and two dropout (0.25) layers, global average pooling and a linear classifier (about 140,000 parameters); Adam with a
  one-cycle schedule, batch size 128, 10 epochs, no data augmentation, best-validation epoch kept; three training
  seeds.
* **LDA + FS and SVM + HOG + FS.** The same models with an ANOVA F-test feature-selection stage (`SelectKBest`,
  `f_classif`): each feature (pixel or HOG feature) is scored by its one-way ANOVA F statistic against the class
  label on the clean training split, and only the top fraction is kept. The fraction (10%, 25%, 50% or 75%) is tuned on
  clean validation accuracy together with the classifier's hyperparameters, and the selected set is frozen with the
  model; it is never refitted on degraded data. A fraction of 100% is deliberately not in the grid, so a "+FS" model
  always selects, and the model without selection is the 100% reference.
* **LDA + HOG (ablation).** HOG features with LDA, to separate the effect of the features from that of the classifier.

*Figure: Fig. 1 `pipelines.png`; feature-selection maps `feature_selection_maps.png`.*

### D. Freeze-then-evaluate protocol

All models are trained and tuned on clean data only. Each model file is hashed (SHA-256) together with the digests of
the training and validation arrays and, for HOG and selection models, the HOG settings and the selected feature set;
evaluation refuses a model whose file, training data, HOG settings or selection differ from that record. The clean
test set is scored once per model; the degraded test images are generated once, their digests are verified to be
identical for every model, and the clean condition must reproduce the clean-test predictions exactly. No degraded image
is used for training, tuning, epoch selection or model selection.

---

## IV. Experimental Setup

**Tuning.** Selection metric: clean validation accuracy. LDA and LDA + HOG: shrinkage ∈ {0.001, 0.01, 0.05, 0.1, 0.3,
auto}. SVM + HOG: C ∈ {1, 10} × γ multiplier ∈ {0.5, 1, 2} (γ = multiplier / (D · var(X))), tuned on a stratified
10,000-image training subset and then refitted on the full training set. CNN: learning rate ∈ {0.001, 0.003} × weight
decay ∈ {0, 0.0005}, each trained for 10 epochs; the best configuration was then retrained with two further seeds. The
feature-selection models use the same grids crossed with the four fractions. Several selected values lie at the edge
of their grids (SVM C and γ, the CNN learning rate and weight decay, and the selected fraction, 75% for all four
selection models), so the optima may lie outside the grids.

**Evaluation.** Each model is scored on the clean test set and on 20 degraded conditions (4 families × 5 severities),
with 3 noise realisations: 31 runs per model and dataset. Metrics: accuracy, error rate, macro-averaged precision,
recall and F1, and confusion matrices per condition.

**Statistics.** (i) 95% percentile bootstrap intervals over test images (2,000 resamples), with the same resampled
images for every model so that differences are paired; these cover test-set sampling uncertainty, and CNN training-seed
variance is reported separately as the range over three seeds. (ii) Exact two-sided McNemar tests on the discordant
test images for seven model pairs in every condition, with Holm correction over all tests of a dataset; for the CNN each
of the three seeds is tested and a result is called robust only if all three agree. **For Gaussian-noise conditions the McNemar
tests use noise realisation 0 only, whereas reported accuracies and AUSC average the three realisations;** differences of the
realisation-averaged accuracies are reported with their own paired bootstrap intervals (not adjusted for multiplicity). Averaging changes
a pairwise difference by at most 0.006 and flips the sign of 2 of 110 noise comparisons, and every Holm-significant noise comparison has
an averaged-difference interval excluding zero. (iii) Area under the severity curve
(AUSC; mean accuracy over severity 0–5) and chance-adjusted retention $(\text{Acc}_s - 0.1)/(\text{Acc}_{\text{clean}} - 0.1)$.
(iv) Prediction collapse: the share of predictions in the most-predicted class.

**Implementation and reproducibility.** Python 3.12.16, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, scikit-image
0.26.0, PyTorch 2.15.0; all runs on an Apple M5 CPU; seed 42 throughout (CNN seeds 42–44). Code, frozen model
records, predictions, result tables and figures are in the repository; model weights are not committed, but their
hashes are recorded.

---

## V. Results

*Tables: `results/paper_tables/table_clean.tex`, `table_ausc.tex`, `table_drop.tex`, `table_rank.tex`, `table_fs.tex`;
the full tables are in `project_documentation.md` §9.*

### A. Clean performance

On clean test data the ranking is CNN > SVM + HOG > LDA + HOG > LDA on both datasets (Fashion-MNIST 0.927–0.928,
0.911, 0.866, 0.817; ciFAIR-10 0.798–0.806, 0.625, 0.537, 0.405). The CNN's lead over SVM + HOG is about 0.017 on
Fashion-MNIST and 0.17–0.18 on ciFAIR-10. HOG features raise LDA from 0.817 to 0.866 and from 0.405 to 0.537, and the RBF-SVM
adds a further 0.045 and 0.088. Test accuracy is 0.4–2.0 points below validation accuracy for every model.
*Figure: `clean_accuracy_comparison.png`.*

### B. Accuracy under degradation (RQ1, RQ2)

*Figures: `robustness_curves_<dataset>.png`, `robustness_heatmap_<dataset>.png`, `degraded_accuracy_summary.png`.*

**The clean ranking does not persist.** On Fashion-MNIST the CNN is significantly less accurate than SVM + HOG in 16 of
the 21 conditions and than LDA in 16 (more accurate in 2 and 3; the three seeds disagree in 3 and 2). At severity 5 the
best main model is LDA under noise (0.580) and blur (0.490) and SVM + HOG under contrast (0.892) and brightness (0.820);
the CNN is lowest of the three under noise (0.118), blur (0.246) and contrast (0.101). On ciFAIR-10 the CNN is the best
main model at severity 1 in all four families and is significantly more accurate than SVM + HOG in 11 conditions and
less accurate in 6 (seeds disagree in 4); by severity 5 LDA is best under noise (0.241) and blur (0.384), SVM + HOG
under contrast (0.548), and the CNN and SVM + HOG are level under brightness (0.501 and 0.503).

**The ranking depends on the degradation family.** Table "rank" orders the main models by AUSC. On Fashion-MNIST SVM + HOG
ranks first in blur, contrast and brightness and LDA in noise, and the CNN last throughout (under noise the CNN and
SVM + HOG differ by only 0.002, 0.474 against 0.476). On ciFAIR-10 the order changes with the family (CNN first only
under brightness) and the mean ranks are close (LDA 2.25, SVM + HOG 1.75, CNN 2.00), so no model is uniformly most robust.

* *Contrast reduction.* HOG-based models are nearly unaffected on Fashion-MNIST (SVM + HOG 0.911 clean, 0.892 at
  severity 5) and degrade mainly at severity 5 on ciFAIR-10 (0.625 → 0.548); LDA falls to 0.176 and the CNN to 0.101
  on Fashion-MNIST.
* *Brightness.* SVM + HOG loses about 0.09 at severity 5 on Fashion-MNIST (0.911 → 0.820); the CNN already falls
  from 0.928 to 0.686 at severity 1 (seed range 0.622–0.723), but on ciFAIR-10 it degrades gradually and stays above
  SVM + HOG through severity 4.
* *Noise.* HOG-based models are the most sensitive (SVM + HOG 0.625 → 0.333 at severity 1 on ciFAIR-10; LDA + HOG
  0.549 at severity 1 on Fashion-MNIST), while pixel-based LDA is the most tolerant (0.817 → 0.580 at severity 5 on
  Fashion-MNIST).
* *Blur.* On ciFAIR-10 LDA is almost unchanged (0.405 → 0.384) while the CNN drops from 0.802 to 0.323 at severity 2.

**Retention versus absolute accuracy.** Chance-adjusted retention can rank models differently from accuracy: on ciFAIR-10
LDA retains 0.99 of its above-chance accuracy under blur but from a clean accuracy of 0.405, so its AUSC (0.404) is below
that of SVM + HOG (0.434). *Figures: `robustness_summary_ci.png`, `retention_ci.png`.*

**Statistical support.** The 95% bootstrap interval of an accuracy has a half-width of 0.008 on average (0.010 at
most), so most differences between models are much larger than test-set sampling noise; consequently nearly every
pairwise comparison is significant after Holm correction, and the informative cells of `significance_<dataset>.png` are
the non-significant ones. The result depends on the CNN training seed in 5 of the 42 CNN comparison cells on each dataset.

**Failure mode.** On clean data every model spreads its predictions (the most-predicted class receives 0.10–0.12 of
them). At severity 5 the share in the single most-predicted class is 0.78 for the CNN on Fashion-MNIST (0.56 for LDA,
0.35 for SVM + HOG) and 0.59 on ciFAIR-10 (0.40–0.42 for the others): failing models label most images with one or two
classes. *Figure: `confusion_<dataset>.png`.* Several conditions end near chance (0.10), where models are
indistinguishable.

### C. Feature selection (RQ3)

*Figures: `feature_selection_effect.png`, `feature_selection_robustness_delta.png`, `feature_selection_maps.png`;
Table "fs".* Clean validation accuracy rose monotonically with the fraction of features kept, and every selection model
chose the largest fraction (75%); selection gave no consistent significant improvement in clean test accuracy (LDA 0.817 → 0.814 and
SVM + HOG 0.911 → 0.910 on Fashion-MNIST; LDA 0.405 → 0.394, a significant decrease, and SVM + HOG 0.625 → 0.627, not significant, on ciFAIR-10).

Under degradation the effect was dataset-dependent for LDA and small and mixed in sign for SVM + HOG. LDA + FS on Fashion-MNIST was
significantly more accurate than LDA in 9 of 21 conditions and less accurate in 3, with +0.245 [+0.237, +0.254] under
severity-5 brightness, +0.115 [+0.108, +0.123] under severity-3 contrast and −0.011 / −0.013 under severity-3 / 5 noise. On
ciFAIR-10 LDA + FS was more accurate in 3 conditions (noise, up to +0.054 [+0.048, +0.059] at severity 5) and less
accurate in 9 (brightness −0.024 and −0.018 at severities 3 and 5, blur about −0.010, clean −0.011). SVM + HOG + FS did not
differ significantly from SVM + HOG in 15 of 21 conditions on Fashion-MNIST, but 6 differed (3 better, under noise at severities 1–3, up to +0.028;
3 worse, under blur, down to −0.026 at severity 5), and in 20 of 21 on ciFAIR-10 (1 better: blur at severity 3, +0.011). The effects are small
in absolute terms; a non-significant result does not show equivalence.

---

## VI. Discussion

**Observations.** The central finding is that clean accuracy is a poor guide to robustness here. The CNN, best on clean
data, is among the weakest models under several degradations on Fashion-MNIST, while SVM + HOG is the most stable under
contrast and brightness and pixel-based LDA under noise and blur. The same model families rank differently on ciFAIR-10,
where the CNN is clearly best under brightness and the mean ranks are close, so the Fashion-MNIST ordering should not be
generalised. Failing models collapse onto one or two classes, and the CNN's behaviour under degradation varies strongly
with the training seed even though its clean accuracy does not.

**Explanations (consistent with the results; not tested).** The insensitivity of HOG-based models to contrast
reduction is consistent with block normalisation removing the scale of local gradients, and their tolerance of additive
brightness with an offset leaving gradients unchanged except where pixels clip. Their sensitivity to noise is consistent with gradient
computation amplifying independent pixel noise, while a linear classifier on raw pixels averages noise over many
pixels. The CNN's loss under brightness on Fashion-MNIST is consistent with a network trained without augmentation on an
exactly-black background, whose stored batch-normalisation statistics no longer match shifted activations. The large
brightness gain of LDA + FS on Fashion-MNIST is consistent with the weakly regularised discriminant selected there
(shrinkage 0.001) putting large weights on low-variance border pixels, which the F-test mostly drops; this does not explain
why the same selection hurts LDA under brightness on ciFAIR-10. The small effect on SVM + HOG is consistent with
the selection removing only the weakest quarter of the HOG features; its few significant changes were not investigated. None of these mechanisms was tested (for example by
activation analysis or ablations).

**Practical implication.** If the type of degradation in use is unknown, a model should not be chosen by clean accuracy
alone; for the degradations studied, HOG-based features offered the most stable accuracy under contrast and brightness
changes, at the price of noise sensitivity.

---

## VII. Limitations

* **Synthetic, simple degradations on small images.** Noise is i.i.d. and independent across channels, blur is isotropic,
  contrast uses one RGB-wide mean, brightness is positive only, and storage in 8 bits quantises the most extreme contrast level.
* **Severity levels are not equally hard across families**, and PSNR/SSIM do not measure recognition difficulty; the audit
  used 1,000 images per dataset and one noise realisation.
* **Distance-filtered benchmarks.** Duplicate thresholds were chosen by inspecting test/train pairs, and filtering changes the evaluated
  population in a way that was not measured.
* **Small tuning budgets, often at the grid edge**; the SVM was tuned on a 10,000-image subset; the HOG geometry is fixed and
  untuned; the CNN was still improving at epoch 10; validation scores used for selection are optimistic.
* **Few seeds; intervals cover the test set only.** The CNN has three seeds and varies by up to 0.124 under one condition;
  the deterministic models were fitted once; the bootstrap and McNemar tests treat images as independent and the Holm
  correction covers the listed tests only. One test set per dataset.
* **Feature selection** is one filter method on two models, fitted on clean data only; the selected fraction was at the grid edge; the CNN
  has no separate selection stage.
* **No component ablation or mechanism test** supports the explanations; no real-world degraded data were used.

---

## VIII. Conclusion

Under a controlled protocol in which every model is trained and tuned on clean data only and scored on bit-identical
degraded test images, the clean-accuracy ranking CNN > SVM + HOG > LDA did not persist under degradation. HOG-based models were
the most stable under contrast reduction and additive brightness but the most sensitive to Gaussian noise; pixel-based
LDA was the most tolerant of noise and blur from a low clean accuracy; on Fashion-MNIST the CNN fell furthest under
noise, blur and contrast and varied strongly across training seeds; failing models collapsed onto a few classes. ANOVA
feature selection gave no consistent significant improvement in clean accuracy, and its effect on robustness was large for LDA on Fashion-MNIST, mixed on
ciFAIR-10 and small and mixed in sign for SVM + HOG. These conclusions hold for the two datasets, four degradations, fitted models
and test sets studied.

---

## IX. Future Work

* More CNN training seeds, so that seed variance enters the intervals, and a bootstrap that resamples both images and seeds.
* Other feature-selection methods (mutual information, embedded or wrapper methods), kept fractions above 75%, selection on the CNN's learned features, and PCA.
* Tests of the proposed mechanisms: activation statistics of the batch-normalisation layers, HOG without block normalisation, LDA with different shrinkage.
* Augmentation or noise-aware training, to separate inherent robustness from exposure during training.
* Additional degradations (darkening, motion blur, JPEG compression, salt-and-pepper noise, combined degradations), real degraded data, higher-resolution images and more datasets.
* Sensitivity analyses reusing the frozen models: the original CIFAR-10 test set, unfiltered test sets, and accuracy stratified by distance to the training set.

---

## References (verified against web sources in this session)

[1] H. Xiao, K. Rasul and R. Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms," arXiv:1708.07747, 2017.

[2] B. Barz and J. Denzler, "Do We Train on Test Data? Purging CIFAR of Near-Duplicates," *Journal of Imaging*, vol. 6, no. 6, art. 41, 2020 (arXiv:1902.00423).

[3] C. Geier, "Training on test data: Removing near duplicates in Fashion-MNIST," arXiv:1906.08255, 2019.

[4] D. Hendrycks and T. Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," in *Proc. ICLR*, 2019 (arXiv:1903.12261).

[5] S. Dodge and L. Karam, "Understanding How Image Quality Affects Deep Neural Networks," in *Proc. Int. Conf. Quality of Multimedia Experience (QoMEX)*, 2016 (arXiv:1604.04004).

[6] N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2005, pp. 886–893, doi:10.1109/CVPR.2005.177.

[7] Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, vol. 86, no. 11, pp. 2278–2324, Nov. 1998, doi:10.1109/5.726791.

[8] I. Guyon and A. Elisseeff, "An Introduction to Variable and Feature Selection," *J. Machine Learning Research*, vol. 3, pp. 1157–1182, Mar. 2003.

[9] T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," *Neural Computation*, vol. 10, no. 7, 1998 (page range differs between sources: confirm against the original before submission).

[10] S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," *Scandinavian J. Statistics*, vol. 6, no. 2, pp. 65–70, 1979.

---

## Notes for the team (not paper text)

**Open items before submission**
1. **Literature.** No 2023–2026 IEEE references have been identified or verified; the literature review above is limited to the
   ten verified sources and is not a systematic survey. A proper search (robustness of classical versus deep classifiers under
   image degradation, HOG/SVM and CNN comparisons, feature selection for image classification) is still needed, and every new reference
   must be checked against its source. Standard references for LDA (Fisher, 1936), SVM (Cortes and Vapnik, 1995), the bootstrap (Efron, 1979)
   and CIFAR (Krizhevsky, 2009) were *not* verified in this session and are not cited above; add them only after checking.
2. **Reference [9]**: confirm the page range and that the McNemar recommendation is stated as summarised in Section II.
3. **Course requirement.** Feature selection is applied to LDA and SVM + HOG; ask the instructor whether the CNN needs its own selection step.
4. **Format.** Convert to the target IEEE template (IEEEtran) and fit the page limit; the five generated LaTeX tables need `\usepackage{booktabs}`.
5. **Figures to use** (suggested, three-model set): `main_robustness_curves.png` (the core comparison, both datasets in one figure),
   `main_clean_accuracy.png`, `main_degraded_summary.png`, `main_confusion_fashion_mnist.png` (clean against contrast severity 4),
   one degradation grid (for example `degradation_grid_fashion_mnist_gaussian_blur.png`), and `feature_selection_robustness_delta.png`;
   `architecture_system.pdf` as the methodology overview figure (the CNN diagram `architecture_cnn.pdf` can go in the model description or the appendix); optionally `pipelines.png`. Everything else (heatmaps, `robustness_summary_ci.png`, `retention_ci.png`, `significance_*.png`,
   `cnn_training_curves.png`, `feature_selection_effect.png`, `feature_selection_maps.png`, the six-model curves and the ciFAIR-10 confusion matrices)
   is better placed in an appendix or supplementary material.
6. The README in the repository is out of date (it still says no models exist) and should be refreshed before the repository is shared.

**Claim register — each claim, where it comes from, and its strength**

| Claim | Evidence | Strength / caveat |
|---|---|---|
| Clean ranking CNN > SVM + HOG > LDA on both datasets | `clean_test_metrics.csv` | Single test set; CNN over 3 seeds |
| CNN significantly less accurate than SVM + HOG in 16/21 Fashion-MNIST conditions | `analysis_pairwise.csv` | Holm-corrected; 3 seeds agree; CIs cover test-set noise only |
| HOG-based models nearly unaffected by contrast and brightness on Fashion-MNIST | `robustness_results.csv`, `analysis_accuracy_ci.csv` | Descriptive; mechanism untested |
| HOG-based models most sensitive to Gaussian noise | same | Descriptive; mechanism untested |
| Failing models collapse onto a few classes | `analysis_prediction_collapse.csv`, confusion matrices | Observed; cause untested |
| CNN accuracy under degradation varies up to 0.124 across seeds | `robustness_results.csv` | Only 3 seeds |
| Feature selection gives no consistent significant clean improvement | `results/metrics/models/*_fs.json` (tuning tables), `clean_test_metrics.csv` | Fraction at grid edge (75%) |
| LDA + FS gains up to +0.25 under brightness on Fashion-MNIST | `analysis_pairwise.csv` | Dataset-specific (opposite sign on ciFAIR-10); explanation untested |
| SVM + HOG + FS is close to SVM + HOG, with small mixed effects | `analysis_pairwise.csv` | 15/21 (Fashion-MNIST) and 20/21 (ciFAIR-10) conditions not significant; non-significance is not equivalence |
| Noise-condition tests use realisation 0; displayed differences are realisation averages | `analysis_pairwise.csv` (`diff_mean_*`) | Averaging changes differences by ≤ 0.006 |

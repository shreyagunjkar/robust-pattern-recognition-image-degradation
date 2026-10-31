# Literature survey — robustness of image classifiers to image degradation

Scope: work on how trained image classifiers behave when test images are degraded (noise, blur, contrast, brightness and
similar corruptions), with emphasis on recent (2023–2026) and IEEE-published papers. Every entry below was checked against
a source page in this project (arXiv abstract pages, CVF open-access listings or publisher listings); descriptions repeat only
what those pages state. The survey is **not systematic** and the search tool could not open IEEE Xplore directly, so the
set of IEEE papers is small; the limits are stated at the end. Reference numbers match the compiled paper (`paper/main.pdf`, numbered by order of first citation).

## 1. Evidence table

| Ref. | Work | Venue / year | Verified status | What it studies | Relevance here |
|---|---|---|---|---|---|
| [1] | Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* | ICLR 2019 | confirmed (arXiv 1903.12261) | ImageNet-C / ImageNet-P benchmarks of common corruptions, not adversarial changes | Defines the severity-graded corruption protocol that this study follows on small datasets |
| [2] | Mu & Gilmer, *MNIST-C: A Robustness Benchmark for Computer Vision* | arXiv 1906.02337 (2019) | confirmed | 15 corruptions applied to the MNIST test set | Corruption benchmark for a small grayscale dataset |
| [3] | Dodge & Karam, *Understanding How Image Quality Affects Deep Neural Networks* | QoMEX 2016 (IEEE) | confirmed (arXiv 1604.04004) | Four deep networks under blur, noise, contrast, JPEG, JPEG2000 | Earliest closely related study; networks are most sensitive to blur and noise |
| [4] | Vasiljevic, Chakrabarti & Shakhnarovich, *Examining the Impact of Blur on Recognition by Convolutional Networks* | arXiv 1611.05760 (2016) | confirmed | Blur harms networks trained on clean images; fine-tuning on blurred images recovers much accuracy | Motivates why blur is one of the four degradations |
| [5] | Dodge & Karam, *Can the Early Human Visual System Compete with Deep Neural Networks?* | ICCV Workshops 2017 (IEEE), pp. 2798–2804, doi 10.1109/ICCVW.2017.329 | confirmed (CVF listing) | Human versus deep-network classification of distorted images | Context: comparisons under distortion |
| [6] | Drenkow et al., *A Systematic Review of Robustness in Deep Learning for Computer Vision: Mind the gap?* | arXiv 2112.00639 (2021, rev. 2022) | confirmed | Systematic review of non-adversarial robustness; a causal framing | Background for the definition of corruption robustness |
| [7] | Sarkar et al., *Benchmark Generation Framework With Customizable Distortions for Image Classifier Robustness* | IEEE/CVF WACV 2024, pp. 4418–4427 | confirmed (search listing of the proceedings; the CVF page itself returned 403) | A framework that generates robustness benchmarks with adjustable distortion levels | Recent IEEE work on graded distortion benchmarks |
| [8] | Shen et al., *Assessing Visually-Continuous Corruption Robustness of Neural Networks Relative to Human Performance* | arXiv 2402.19401 (2024) | confirmed; no venue stated on the page | Continuous corruption range, 14 corruptions, 7,718 human participants; the gap between networks and humans is larger than previously known | Recent evidence that fixed severity grids can hide weaknesses |
| [9] | Liu et al., *A Comprehensive Study on Robustness of Image Classification Models: Benchmarking and Rethinking* | arXiv 2302.14301 (2023; listed as submitted to IJCV) | confirmed | ARES-Bench: 55 ImageNet models; adversarial and out-of-distribution robustness; trade-off between adversarial and natural robustness within an architecture; pre-training improves natural robustness | Large-scale recent benchmark |
| [10] | Wang, Veldhuis, Brune & Strisciuglio, *A Survey on the Robustness of Computer Vision Models against Common Corruptions* | arXiv 2305.06024 (v4, Sept. 2024) | confirmed (no journal reference on the page) | Survey of methods (augmentation, learning strategies, network components) and a unified benchmark; corruption robustness does not reliably grow with model or data size | Most relevant recent survey |
| [11] | Delanois et al., *Improving Robustness of Convolutional Networks Through Sleep-Like Replay* | ICMLA 2023 (IEEE) | authors, title and venue confirmed from search listings; pages and DOI not obtained | Unsupervised sleep-replay consolidation to make convolutional filters more robust to perturbations such as noise and blur, evaluated on MNIST and CIFAR-10 | Recent IEEE work on CNN robustness to noise and blur |
| [12] | Camuffo et al., *FFT-based Selection and Optimization of Statistics for Robust Recognition of Severely Corrupted Images* | ICASSP 2024 (IEEE); arXiv 2403.14335 | confirmed from the arXiv page (comments say ICASSP 2024; its journal-reference field says 2023); IEEE Xplore page not opened | FROST: detects the corruption type from high-frequency features and selects matching normalisation statistics; reported up to 37.1% relative gain on ImageNet-C | Recent IEEE signal-processing work on severely corrupted images |
| [13] | Mitra, Schwalbe & Klein, *Investigating Calibration and Corruption Robustness of Post-hoc Pruned Perception CNNs: An Image Classification Benchmark Study* | IEEE/CVF CVPR Workshops 2024, pp. 3542–3552, doi 10.1109/CVPRW63382.2024.00358 | confirmed (CVF listing and search summary) | Effect of post-hoc pruning on calibration and natural-corruption robustness of CNNs | Recent IEEE work on corruption robustness of CNNs |
| [14] | Ramírez-Agudelo et al., *The Role of Noisy Data in Improving CNN Robustness for Image Classification* | arXiv 2601.08043 (Jan. 2026); presented at SPIE Applications of Machine Learning 2025 | confirmed | ResNet-18 on CIFAR-10; Gaussian noise, salt-and-pepper noise, Gaussian blur; about 10% noisy training data improves accuracy on corrupted test images with little effect on clean accuracy | Training-time remedy (outside this study's scope) |
| [17] | Ezuma & Ugwu, *Enhancing Histopathological Image Classification via Integrated HOG and Deep Features with Robust Noise Performance* | Proc. SPIE 13932 (2026); arXiv 2601.01056 | confirmed | Classifiers on deep features were more resilient to noise; adding HOG improved accuracy, but less so in noisy settings | One of the few recent studies that combines HOG features and noise |

Foundational references used for methods and statistics (also verified): Xiao et al. 2017 (Fashion-MNIST) [22]; Barz & Denzler 2020 (ciFAIR) [18];
Geier 2019 (near-duplicates in Fashion-MNIST) [19]; Dalal & Triggs, CVPR 2005 (HOG) [15]; LeCun et al., Proc. IEEE 1998 [16]; Guyon & Elisseeff, JMLR 2003 [23];
Dietterich, Neural Computation 1998 [20]; Holm, Scand. J. Statistics 1979 [21].

## 2. Synthesis

**Corruption benchmarks and degradation studies.** The standard approach is to apply graded, model-agnostic corruptions to a test set and
measure the loss in accuracy [1], [2]. Early work found that networks trained on clean images lose accuracy under blur and noise [3], [4], and
later surveys organise the field into augmentation-based, learning-strategy and architectural remedies [6], [10]. Recent benchmark papers
extend this in three directions: customisable distortion generators [7], continuous severity ranges compared with human performance [8],
and large-scale comparisons across architectures and training methods [9], [10]. A common observation in [9] and [10] is that robustness does not follow
accuracy or scale in a simple way, which is the same pattern this study finds on small datasets.

**Remedies.** Many recent IEEE-published papers propose a remedy rather than a measurement: replay-based consolidation [11], corruption-aware
selection of normalisation statistics [12], and pruning [13]; others add noisy images to the training set [14]. These works start from a trained deep
network. They do not compare it with classical models on identical degraded inputs.

**Classical versus deep models.** Work that places hand-crafted features beside deep features under noise is scarce in what we found. The HOG descriptor
was designed with contrast normalisation in overlapping blocks [15], and one recent study reports that adding HOG to deep features helps less in noisy settings [17].
None of the reviewed papers reports a controlled comparison of a linear statistical classifier, a HOG-feature kernel classifier and a CNN under a frozen
degradation ladder with paired statistical tests.

**Evaluation hygiene.** Duplicates between training and test images inflate scores and are documented for CIFAR-10 [18] and Fashion-MNIST [19]. Paired tests on the same
test images [20] with multiple-comparison correction [21] are standard in classifier comparison but are rarely reported in the corruption-robustness papers above.

**Gap addressed by this paper (within the reviewed set):** a controlled, leakage-filtered comparison of LDA, SVM + HOG and a CNN, trained on clean images only and
evaluated on bit-identical degraded test images, reported with paired intervals and corrected tests.

## 3. Limits of this survey and leads still to be checked

* The search tool could not open IEEE Xplore pages; IEEE status for [7], [11]–[13] rests on the venue statements in the listings named above. Confirm each entry (pages, DOI) on IEEE Xplore before submission.
* A lead not cited in the paper because authors and venue could not be confirmed: IEEE Xplore document 11098661, *Robustness Benchmarking of Convolutional and Transformer Architectures for Image Classification* (a lab page reports 18 networks, 5 datasets and 15 corruption types, with ViTs more robust to noise and CNNs more robust to environmental corruptions). Open the record and add it if it is relevant.
* Arguably relevant 2023–2026 IEEE journals (TIP, TNNLS, Access) were searched without an identifiable match for a classical-versus-deep robustness comparison; a manual IEEE Xplore search with the terms "HOG", "SVM", "CNN", "Gaussian noise", "robustness" is recommended.
* The ICMLA 2023 entry [11] lacks pages and DOI in this survey.
* This survey is not exhaustive; claims of novelty should be worded as "among the works we reviewed".

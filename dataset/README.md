# Dataset files for the case study

The case study compares LDA, SVM + HOG and a CNN on two image datasets.

## Fashion-MNIST (primary dataset) — included in this folder

`fashion_mnist/` holds the four original files (70,000 grayscale 28 x 28 images, 10 clothing classes: 60,000 training and 10,000 test):

| File | SHA-256 |
|---|---|
| `train-images-idx3-ubyte.gz` | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| `train-labels-idx1-ubyte.gz` | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` |
| `t10k-images-idx3-ubyte.gz` | `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` |
| `t10k-labels-idx1-ubyte.gz` | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` |

Source: Zalando Research, Fashion-MNIST (http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/; project page
https://github.com/zalandoresearch/fashion-mnist). Please see the source page for its licence and citation: H. Xiao, K. Rasul and R. Vollgraf,
"Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms", arXiv:1708.07747, 2017.

## ciFAIR-10 (replication dataset) — not stored in this repository

The archive (`ciFAIR-10.zip`, about 161 MB) is larger than GitHub's 100 MB per-file limit. It is CIFAR-10 with the duplicated test images replaced
(B. Barz and J. Denzler, "Do we train on test data? Purging CIFAR of near-duplicates", J. Imaging 6(6):41, 2020).
Download: https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip . The code downloads and checksum-verifies it automatically.

## Using these files with the code

Either let the code download and verify everything: `python -m src.data_loader --dataset fashion_mnist` and `python -m src.data_loader --dataset cifair10`,
or copy the four Fashion-MNIST files into `data/raw/fashion_mnist/` first (the loader verifies their checksums and then reuses them).

"""LaTeX tables for the paper, generated from the saved result files (nothing is computed from the models).

Writes results/paper_tables/*.tex (booktabs; add \\usepackage{booktabs} to the paper):
  table_clean.tex     clean test accuracy and macro-F1 per model (CNN: mean and range over 3 seeds)
  table_ausc.tex      area under the severity curve with 95% bootstrap CI, per family and model
  table_drop.tex      accuracy drop (clean - degraded) at severities 3 and 5
  table_rank.tex      robustness ranking of the three main models by area under the severity curve
  table_fs.tex        effect of the feature-selection stage (accuracy difference with CI, Holm-significant marked)
"""
from __future__ import annotations

import pandas as pd

from . import config

OUT = config.RESULTS_DIR / "paper_tables"
ORDER = ("lda", "lda_fs", "svm_hog", "svm_hog_fs", "cnn", "lda_hog")
MAIN = ("lda", "svm_hog", "cnn")
NAME = {"lda": "LDA", "lda_fs": "LDA+FS", "svm_hog": "SVM+HOG", "svm_hog_fs": "SVM+HOG+FS", "cnn": "CNN", "lda_hog": "LDA+HOG"}
DS = {"fashion_mnist": "Fashion-MNIST", "cifair10": "ciFAIR-10"}
FAM = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "contrast": "Contrast", "brightness": "Brightness"}


def table(caption: str, label: str, header: list, rows: list, spec: str) -> str:
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(r if r == "\\midrule" else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(config.METRICS_DIR / "clean_test_metrics.csv")
    ci = pd.read_csv(config.METRICS_DIR / "analysis_accuracy_ci.csv")
    summ = pd.read_csv(config.METRICS_DIR / "analysis_robustness_summary.csv")
    pair = pd.read_csv(config.METRICS_DIR / "analysis_pairwise.csv")

    rows = []
    for ds, title in DS.items():
        for m in ORDER:
            q = clean[(clean.dataset == ds) & (clean.model == m)]
            acc = f"{q.accuracy.mean():.3f}" + (f" ({q.accuracy.min():.3f}--{q.accuracy.max():.3f})" if len(q) > 1 else "")
            rows.append([title, NAME[m], acc, f"{q.macro_f1.mean():.3f}"])
        if ds != list(DS)[-1]:
            rows.append("\\midrule")
    (OUT / "table_clean.tex").write_text(table(
        "Clean test accuracy and macro-F1 of the frozen models (CNN: mean over 3 seeds, range in parentheses).",
        "tab:clean", ["Dataset", "Model", "Accuracy", "Macro-F1"], rows, "llcc"))

    rows = []
    for ds, title in DS.items():
        for fam, ftitle in FAM.items():
            q = summ[(summ.dataset == ds) & (summ.family == fam)].set_index("model")
            rows.append([title if fam == "gaussian_noise" else "", ftitle] +
                        [f"{q.loc[m, 'ausc']:.3f} [{q.loc[m, 'ausc_ci_low']:.3f}, {q.loc[m, 'ausc_ci_high']:.3f}]" for m in MAIN])
        if ds != list(DS)[-1]:
            rows.append("\\midrule")
    (OUT / "table_ausc.tex").write_text(table(
        "Area under the severity curve (mean accuracy over severity 0--5) with 95\\% bootstrap intervals over test images; "
        "CNN: mean of 3 seeds.", "tab:ausc", ["Dataset", "Degradation"] + [NAME[m] for m in MAIN], rows, "llccc"))

    rows = []
    for ds, title in DS.items():
        for fam, ftitle in FAM.items():
            cells = []
            for m in ORDER:
                q = ci[(ci.dataset == ds) & (ci.model == m)].set_index("condition").accuracy
                cells.append(" / ".join(f"{q['clean'] - q[f'{fam}_s{s}']:.2f}" for s in (3, 5)))
            rows.append([title if fam == "gaussian_noise" else "", ftitle] + cells)
        if ds != list(DS)[-1]:
            rows.append("\\midrule")
    (OUT / "table_drop.tex").write_text(table(
        "Accuracy drop (clean minus degraded accuracy) at severity 3 / severity 5.", "tab:drop",
        ["Dataset", "Degradation"] + [NAME[m] for m in ORDER], rows, "ll" + "c" * len(ORDER)))

    rows = []
    for ds, title in DS.items():
        ranks = {m: [] for m in MAIN}
        for fam, ftitle in FAM.items():
            q = summ[(summ.dataset == ds) & (summ.family == fam) & summ.model.isin(MAIN)].set_index("model").ausc
            rk = q.rank(ascending=False, method="min").astype(int)
            for m in MAIN:
                ranks[m].append(rk[m])
            rows.append([title if fam == "gaussian_noise" else "", ftitle] + [str(rk[m]) for m in MAIN])
        rows.append(["", "Mean rank"] + [f"{sum(ranks[m]) / len(ranks[m]):.2f}" for m in MAIN])
        if ds != list(DS)[-1]:
            rows.append("\\midrule")
    (OUT / "table_rank.tex").write_text(table(
        "Robustness ranking of the three main models (1 = highest area under the severity curve) per degradation family.",
        "tab:rank", ["Dataset", "Degradation"] + [NAME[m] for m in MAIN], rows, "llccc"))

    rows = []
    for ds, title in DS.items():
        for a, b in (("lda_fs", "lda"), ("svm_hog_fs", "svm_hog")):
            for fam, ftitle in FAM.items():
                cells = []
                for sev in (3, 5):
                    r = pair[(pair.dataset == ds) & (pair.model_a == a) & (pair.model_b == b) & (pair.condition == f"{fam}_s{sev}")].iloc[0]
                    cells.append(f"{r.diff_a_minus_b:+.3f}" + ("$^{*}$" if r.significant else ""))
                rows.append([title if (a, fam) == ("lda_fs", "gaussian_noise") else "", f"{NAME[a]} $-$ {NAME[b]}" if fam == "gaussian_noise" else "",
                             ftitle] + cells)
        if ds != list(DS)[-1]:
            rows.append("\\midrule")
    (OUT / "table_fs.tex").write_text(table(
        "Effect of ANOVA feature selection on test accuracy (model with selection minus model without) at severities 3 and 5; "
        "$^{*}$ = significant (exact McNemar test, Holm-corrected over all tests of the dataset, $p<0.05$).",
        "tab:fs", ["Dataset", "Comparison", "Degradation", "Sev. 3", "Sev. 5"], rows, "lllcc"))
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables to {OUT}")


if __name__ == "__main__":
    main()

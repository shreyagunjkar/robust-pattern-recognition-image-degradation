"""Compact LaTeX tables (LDA, SVM + HOG, CNN) for the IEEE-format paper in paper/, from the saved result files.

Writes paper/tables/{clean,ausc,pairs}.tex and prints the numbers quoted in the text, so the prose can be checked against them.
"""
from __future__ import annotations

import pandas as pd

from . import config

OUT = config.PROJECT_ROOT / "paper" / "tables"
MAIN = ("lda", "svm_hog", "cnn")
NAME = {"lda": "LDA", "svm_hog": "SVM+HOG", "cnn": "CNN"}
DS = {"fashion_mnist": "Fashion-MNIST", "cifair10": "ciFAIR-10"}
FAM = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "contrast": "Contrast reduction", "brightness": "Brightness increase"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(config.METRICS_DIR / "clean_test_metrics.csv")
    summ = pd.read_csv(config.METRICS_DIR / "analysis_robustness_summary.csv")
    pair = pd.read_csv(config.METRICS_DIR / "analysis_pairwise.csv")

    rows = []
    for ds, title in DS.items():
        for m in MAIN:
            q = clean[(clean.dataset == ds) & (clean.model == m)]
            acc = f"{q.accuracy.mean():.3f}" + (f" ({q.accuracy.min():.3f}--{q.accuracy.max():.3f})" if len(q) > 1 else "")
            rows.append(f"{title if m == 'lda' else ''} & {NAME[m]} & {acc} & {q.macro_f1.mean():.3f} \\\\")
        if ds != "cifair10":
            rows.append("\\midrule")
    (OUT / "clean.tex").write_text("\n".join(["\\begin{tabular}{llcc}", "\\toprule", "Dataset & Model & Accuracy & Macro-F1 \\\\", "\\midrule", *rows,
                                              "\\bottomrule", "\\end{tabular}", ""]))

    rows = []
    for fam, ftitle in FAM.items():
        cells = []
        for ds in DS:
            q = summ[(summ.dataset == ds) & (summ.family == fam)].set_index("model")
            best = q.loc[list(MAIN), "ausc"].idxmax()
            cells += [(f"\\textbf{{{q.loc[m, 'ausc']:.3f}}}" if m == best else f"{q.loc[m, 'ausc']:.3f}") for m in MAIN]
        rows.append(f"{ftitle} & " + " & ".join(cells) + " \\\\")
    (OUT / "ausc.tex").write_text("\n".join(["\\begin{tabular}{lcccccc}", "\\toprule",
                                             " & \\multicolumn{3}{c}{Fashion-MNIST} & \\multicolumn{3}{c}{ciFAIR-10} \\\\",
                                             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                                             "Degradation & LDA & SVM+HOG & CNN & LDA & SVM+HOG & CNN \\\\", "\\midrule", *rows,
                                             "\\bottomrule", "\\end{tabular}", ""]))
    half = ((summ.ausc_ci_high - summ.ausc_ci_low) / 2)
    print("AUSC CI half-width (all models): mean %.4f max %.4f; main models max %.4f" %
          (half.mean(), half.max(), half[summ.model.isin(MAIN)].max()))

    def state(g):
        a, b = g.model_a.iloc[0], g.model_b.iloc[0]
        if "cnn" in (a, b):
            if g.significant.all() and g.direction.nunique() == 1:
                return "A" if g.direction.iloc[0] > 0 else "B"
            return "seed" if g.significant.any() else "ns"
        return ("A" if g.direction.iloc[0] > 0 else "B") if g.significant.iloc[0] else "ns"

    st = pd.DataFrame([(ds, a, b, c, state(g)) for (ds, a, b, c), g in pair.groupby(["dataset", "model_a", "model_b", "condition"])],
                      columns=["dataset", "a", "b", "cond", "state"])
    rows = []
    for ds, title in DS.items():
        for a, b in (("svm_hog", "lda"), ("cnn", "lda"), ("cnn", "svm_hog")):
            r = st[(st.dataset == ds) & (st.a == a) & (st.b == b)].state.value_counts()
            rows.append(f"{title if (a, b) == ('svm_hog', 'lda') else ''} & {NAME[a]} vs {NAME[b]} & {r.get('A', 0)} & {r.get('B', 0)} & "
                        f"{r.get('ns', 0)} & {r.get('seed', 0) if 'cnn' in (a, b) else '--'} \\\\")
        if ds != "cifair10":
            rows.append("\\midrule")
    (OUT / "pairs.tex").write_text("\n".join(["\\begin{tabular}{llcccc}", "\\toprule",
                                              "Dataset & Pair (first vs second) & First & Second & n.s. & Seeds \\\\", "\\midrule", *rows,
                                              "\\bottomrule", "\\end{tabular}", ""]))
    print(st[st.a.isin(["svm_hog", "cnn"]) & st.b.isin(["lda", "svm_hog"])].groupby(["dataset", "a", "b"]).state.value_counts().unstack(fill_value=0))
    print(summ[summ.model.isin(MAIN)].pivot_table(index=["dataset", "family"], columns="model", values="ausc").round(3))


if __name__ == "__main__":
    main()

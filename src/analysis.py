"""Phase 6: statistical analysis of the saved Phase 5 predictions (nothing is trained or scored here).

Inputs: results/predictions/robustness_<dataset>_<model>_seed<k>.npz (per-image predictions for every condition).
Outputs (results/metrics/):
  analysis_accuracy_ci.csv          accuracy per model and condition with a 95% bootstrap CI
  analysis_pairwise.csv             paired comparisons: accuracy difference, bootstrap CI, exact McNemar test, Holm p
  analysis_robustness_summary.csv   area under the severity curve and chance-adjusted retention per family, with CIs
  analysis_prediction_collapse.csv  share of predictions falling in the single most-predicted class
  analysis_run.json                 settings and provenance

Method (all choices fixed here, before the numbers were looked at):
* Uncertainty = resampling the TEST IMAGES (non-parametric bootstrap, B resamples, percentile 95% CI). The same
  resampled image set is used for every model and condition, so differences are paired. It captures test-set
  sampling noise only; training-seed variance (CNN) is reported separately as the min-max over the 3 seeds, and the
  deterministic models were fitted once.
* Gaussian-noise conditions: the accuracy table and the robustness summary average the 3 independent noise
  realisations. In the pairwise table the McNemar test (a test of binary per-image outcomes) and the columns
  `diff_a_minus_b`, `diff_ci_*` use noise realisation 0 only; the columns `diff_mean_*` give the difference of the
  realisation-AVERAGED accuracies (the quantity displayed elsewhere) with its own paired bootstrap interval, so the
  displayed differences are never presented as if the test had been run on them. Other conditions: identical columns.
* CNN: accuracy tables and the robustness summary use the mean of the 3 seeds (the bootstrap is applied to the
  seed mean); every pairwise comparison involving the CNN is run separately for each seed, and `robust_across_seeds`
  is true only if all three seeds are significant in the same direction.
* McNemar: exact two-sided binomial test on the discordant pairs (images one model gets right and the other wrong).
  Holm step-down correction over ALL pairwise tests of a dataset; significant means Holm p < alpha.
* Chance-adjusted retention at severity s = (acc_s - c) / (acc_clean - c) with chance c = 1/10; area under the
  severity curve (AUSC) = trapezoid rule over severity 0 (clean) .. 5, divided by 5, so it is a mean accuracy.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import config
from .models import MODEL_NAMES
from .run_robustness import DIGESTS_JSON, RUN_RECORD, predictions_path

ALPHA = 0.05
CHANCE = 0.10
BOOTSTRAP = 2000
FAMILIES = ("gaussian_noise", "gaussian_blur", "contrast", "brightness")
# (first model, second model): a positive difference means the FIRST model is more accurate
PAIRS = (("svm_hog", "lda"), ("cnn", "lda"), ("cnn", "svm_hog"),
         ("lda_fs", "lda"), ("svm_hog_fs", "svm_hog"), ("lda_hog", "lda"), ("svm_hog", "lda_hog"))


def cond_of(run: str) -> str:
    return re.sub(r"_r\d+$", "", run)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def load_predictions(dataset: str, log=print):
    """{(model, seed): npz arrays} for every model whose results exist; all must share images and runs.

    Every file must carry the same experiment context (test pixels / ids / labels, ladder fingerprint, noise seed,
    numpy version, realisation count) and the same per-condition degraded-image digests, and both must agree with
    the run record (robustness_run.json) and the digest record (degraded_test_digests.json); otherwise predictions
    from different experiments could be paired.
    """
    record = json.loads(RUN_RECORD.read_text())
    expected = json.loads(DIGESTS_JSON.read_text())[dataset]
    found, ref, ref_context = {}, None, None
    for model in MODEL_NAMES:
        meta_path = config.MODEL_META_DIR / f"{dataset}__{model}.json"
        if not meta_path.exists():
            continue
        entries = json.loads(meta_path.read_text())["files"]
        paths = [predictions_path(dataset, model, e["seed"]) for e in entries]
        if not all(p.exists() for p in paths):
            log(f"[{dataset}] {model}: robustness results not complete, skipped")
            continue
        for entry, path in zip(entries, paths):
            with np.load(path) as blob:
                if blob["model_sha256"].item() != entry["sha256"]:
                    raise RuntimeError(f"{path.name}: predictions do not belong to the frozen model")
                data = {k: blob[k] for k in ("runs", "preds", "confusion", "y_true", "test_idx")}
                context = json.loads(blob["context"].item())
                digests = {str(r): str(d) for r, d in zip(blob["runs"], blob["digests"])}
            if digests != expected:
                raise RuntimeError(f"{path.name}: degraded-image digests differ from {DIGESTS_JSON.name}")
            if ref_context is None:
                ref_context = context
            elif context != ref_context:
                raise RuntimeError(f"{path.name}: experiment context differs from the other models' files")
            for key, want in (("ladder_fingerprint", record["ladder_fingerprint"]), ("noise_realisations", record["noise_realisations"]),
                              ("seed", record["seed"]), ("numpy", record.get("numpy")), ("dataset", dataset)):
                if context.get(key) != want:
                    raise RuntimeError(f"{path.name}: context {key}={context.get(key)!r} does not match the run record ({want!r})")
            if ref is None:
                ref = data
            elif not (np.array_equal(ref["y_true"], data["y_true"]) and np.array_equal(ref["test_idx"], data["test_idx"])
                      and list(ref["runs"]) == list(data["runs"])):
                raise RuntimeError(f"{path.name}: test images or conditions differ between models")
            found[(model, entry["seed"])] = data
    return found, ref


def bootstrap_counts(n: int, resamples: int, seed: int) -> np.ndarray:
    """(B, n) matrix: how often each image is drawn in each resample (shared by all models)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=resamples).astype(np.float32)


def ci(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def analyse(dataset: str, resamples: int, log=print) -> Dict[str, List[Dict[str, object]]]:
    found, ref = load_predictions(dataset, log)
    y, runs = ref["y_true"], [str(r) for r in ref["runs"]]
    n = len(y)
    conds = list(dict.fromkeys(cond_of(r) for r in runs))
    cols = {c: [i for i, r in enumerate(runs) if cond_of(r) == c] for c in conds}
    counts = bootstrap_counts(n, resamples, config.SEED)

    correct, acc_run, boot_run = {}, {}, {}
    for key, data in found.items():
        correct[key] = (data["preds"].astype(np.int64) == y[None, :]).T  # (n, runs)
        acc_run[key] = correct[key].mean(axis=0)
        boot_run[key] = counts @ correct[key].astype(np.float32) / n  # (B, runs)

    def by_cond(x):  # (..., runs) -> (..., conditions): mean over the noise realisations
        return np.stack([x[..., cols[c]].mean(axis=-1) for c in conds], axis=-1)

    models = [m for m in MODEL_NAMES if any(k[0] == m for k in found)]
    seeds = {m: sorted(s for mm, s in found if mm == m) for m in models}
    acc, boot, lo_seed, hi_seed = {}, {}, {}, {}
    for m in models:
        per_seed = np.stack([by_cond(acc_run[(m, s)]) for s in seeds[m]])  # (S, C)
        acc[m] = per_seed.mean(axis=0)
        lo_seed[m], hi_seed[m] = per_seed.min(axis=0), per_seed.max(axis=0)
        boot[m] = np.mean([by_cond(boot_run[(m, s)]) for s in seeds[m]], axis=0)  # (B, C)

    cond_acc = {k: by_cond(v) for k, v in acc_run.items()}  # (C,) per model-seed, realisations averaged
    cond_boot = {k: by_cond(v) for k, v in boot_run.items()}  # (B, C)
    out: Dict[str, List[Dict[str, object]]] = {"ci": [], "pairwise": [], "summary": [], "collapse": []}

    for m in models:
        low, high = ci(boot[m])
        for j, c in enumerate(conds):
            family, severity = _split(c)
            out["ci"].append({"dataset": dataset, "model": m, "condition": c, "family": family, "severity": severity,
                              "accuracy": acc[m][j], "ci_low": low[j], "ci_high": high[j],
                              "seed_min": lo_seed[m][j], "seed_max": hi_seed[m][j], "n_seeds": len(seeds[m])})

    for a, b in PAIRS:
        if a not in models or b not in models:
            continue
        for sa in (seeds[a] if a == "cnn" else [config.SEED]):
            for sb in (seeds[b] if b == "cnn" else [config.SEED]):
                if a == "cnn" and b == "cnn":
                    continue
                for c in conds:
                    col = cols[c][0]  # noise: realisation 0
                    ca, cb = correct[(a, sa)][:, col], correct[(b, sb)][:, col]
                    only_a, only_b = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
                    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
                    diff_boot = boot_run[(a, sa)][:, col] - boot_run[(b, sb)][:, col]
                    low, high = ci(diff_boot)
                    j = conds.index(c)
                    mean_low, mean_high = ci(cond_boot[(a, sa)][:, j] - cond_boot[(b, sb)][:, j])
                    family, severity = _split(c)
                    out["pairwise"].append({
                        "dataset": dataset, "model_a": a, "model_b": b, "seed_a": sa, "seed_b": sb, "condition": c,
                        "family": family, "severity": severity, "acc_a": ca.mean(), "acc_b": cb.mean(),
                        "diff_a_minus_b": ca.mean() - cb.mean(), "diff_ci_low": low, "diff_ci_high": high,
                        "tested_noise_realisation": 0 if family == "gaussian_noise" else None,
                        "diff_mean_realisations": cond_acc[(a, sa)][j] - cond_acc[(b, sb)][j],
                        "diff_mean_ci_low": mean_low, "diff_mean_ci_high": mean_high,
                        "only_a_correct": only_a, "only_b_correct": only_b, "p_mcnemar": p})
    if out["pairwise"]:
        frame = pd.DataFrame(out["pairwise"])
        frame["p_holm"] = holm(frame.p_mcnemar.to_numpy())
        frame["significant"] = frame.p_holm < ALPHA
        frame["direction"] = np.sign(frame.diff_a_minus_b).astype(int)
        grouped = frame.groupby(["model_a", "model_b", "condition"])
        ok = grouped.apply(lambda g: bool(g.significant.all() and g.direction.nunique() == 1 and g.direction.iloc[0] != 0),
                           include_groups=False)
        frame["robust_across_seeds"] = [
            ok[(r.model_a, r.model_b, r.condition)] if "cnn" in (r.model_a, r.model_b) else None
            for r in frame.itertuples()]
        out["pairwise"] = frame.to_dict("records")

    for m in models:
        for family in FAMILIES:
            idx = [conds.index("clean")] + [conds.index(f"{family}_s{s}") for s in range(1, 6)]
            for key, source in (("point", acc[m]), ("boot", boot[m])):
                curve = source[..., idx]  # (6,) or (B, 6)
                ausc = np.trapezoid(curve, dx=1.0, axis=-1) / 5.0
                retention = (curve[..., 1:] - CHANCE) / (curve[..., :1] - CHANCE)
                stats = {"ausc": ausc, "ret_sev3": retention[..., 2], "ret_sev5": retention[..., 4],
                         "mean_retention": retention.mean(axis=-1)}
                if key == "point":
                    point = stats
                else:
                    boots = stats
            per_seed_ausc = [np.trapezoid(by_cond(acc_run[(m, s)])[idx], dx=1.0) / 5.0 for s in seeds[m]]
            row = {"dataset": dataset, "model": m, "family": family, "clean_accuracy": acc[m][idx[0]],
                   "ausc_seed_min": min(per_seed_ausc), "ausc_seed_max": max(per_seed_ausc)}
            for name, value in point.items():
                low, high = ci(boots[name])
                row.update({name: value, f"{name}_ci_low": low, f"{name}_ci_high": high})
            out["summary"].append(row)

    for (m, s), data in found.items():
        for c in conds:
            share = np.mean([data["confusion"][i].sum(axis=0).max() / data["confusion"][i].sum() for i in cols[c]])
            classes = np.mean([(data["confusion"][i].sum(axis=0) > 0).sum() for i in cols[c]])
            family, severity = _split(c)
            out["collapse"].append({"dataset": dataset, "model": m, "seed": s, "condition": c, "family": family,
                                    "severity": severity, "top_class_share": share, "classes_predicted": classes})
    return out


def _split(condition: str) -> Tuple[str, int]:
    if condition == "clean":
        return "clean", 0
    family, severity = condition.rsplit("_s", 1)
    return family, int(severity)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CIs, McNemar tests and robustness summaries.")
    parser.add_argument("--dataset", nargs="+", default=["fashion_mnist", "cifair10"])
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()
    tables: Dict[str, List[Dict[str, object]]] = {"ci": [], "pairwise": [], "summary": [], "collapse": []}
    models_used = {}
    for dataset in args.dataset:
        result = analyse(dataset, args.bootstrap)
        for k in tables:
            tables[k] += result[k]
        models_used[dataset] = sorted({r["model"] for r in result["ci"]})
    names = {"ci": "analysis_accuracy_ci.csv", "pairwise": "analysis_pairwise.csv",
             "summary": "analysis_robustness_summary.csv", "collapse": "analysis_prediction_collapse.csv"}
    for key, name in names.items():
        pd.DataFrame(tables[key]).to_csv(config.METRICS_DIR / name, index=False, float_format="%.6g")
        print(f"wrote {config.METRICS_DIR / name} ({len(tables[key])} rows)")
    record = {"date": datetime.date.today().isoformat(), "bootstrap_resamples": args.bootstrap, "seed": config.SEED,
              "alpha": ALPHA, "chance": CHANCE, "pairs": [list(p) for p in PAIRS], "models": models_used,
              "robustness_run_sha256": hashlib.sha256(RUN_RECORD.read_bytes()).hexdigest()}
    (config.METRICS_DIR / "analysis_run.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

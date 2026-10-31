"""Phase 5: score the frozen, clean-trained models on every degraded test condition.

Nothing is trained or tuned here. For each dataset the clean test images are degraded once through
`degradations.get_degraded` (which enforces the frozen ladder and caches by fingerprint), so every
model receives bit-identical arrays; the SHA-256 digest of every degraded array is recorded and
cross-checked between models. Per (dataset, model, seed) one file is written:

  results/predictions/robustness_<dataset>_<model>_seed<k>.npz
      runs (condition_rN names), preds (runs x n_test, int8), confusion (runs x 10 x 10), digests,
      test_idx, y_true, model_sha256
and results/metrics/robustness_results.csv collects one row per (dataset, model, seed, condition,
noise realisation); a partial run replaces only the rows it recomputed. Each file also stores a
context (test pixels, ids and labels digests, ladder fingerprint, noise realisations, model SHA-256).
A finished file is reused only if that context matches the current run exactly, so an interrupted run
resumes safely and stale predictions can never be aggregated. The clean condition must reproduce the
Phase 4 clean-test predictions exactly.
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config, degradations
from .data_loader import ImageDataset, array_digest, load_dataset
from .evaluate_clean import check_preprocessing, check_record
from .metrics import classification_metrics, confusion
from .models import MODEL_NAMES, load_model
from .train import N_CLASSES, file_sha256

RESULTS_CSV = config.METRICS_DIR / "robustness_results.csv"
RUN_RECORD = config.METRICS_DIR / "robustness_run.json"
DIGESTS_JSON = config.METRICS_DIR / "degraded_test_digests.json"


def run_list(realisations: int) -> List[Tuple[str, int, int]]:
    """(family, severity, noise realisation); only Gaussian noise is stochastic, so only it repeats."""
    if isinstance(realisations, bool) or not isinstance(realisations, int) or realisations < 1:
        raise ValueError(f"noise realisations must be a positive integer, got {realisations!r}")
    runs = []
    for family, severity in degradations.conditions():
        for r in range(realisations if family == "gaussian_noise" else 1):
            runs.append((family, severity, r))
    return runs


def run_name(family: str, severity: int, realisation: int) -> str:
    return degradations.condition_name(family, severity) + (f"_r{realisation}" if family == "gaussian_noise" else "")


def predictions_path(dataset: str, model: str, seed: int) -> Path:
    return config.PREDICTIONS_DIR / f"robustness_{dataset}_{model}_seed{seed}.npz"


def run_context(ds: ImageDataset, x_digest: str, realisations: int) -> str:
    """Everything a saved prediction file depends on besides the model: test data, ladder, run layout."""
    return json.dumps({"dataset": ds.name, "x_test_sha256": x_digest, "test_idx_sha256": array_digest(ds.test_idx),
                       "y_test_sha256": array_digest(ds.y_test), "n_test": int(len(ds.y_test)),
                       "ladder_fingerprint": degradations.ladder_fingerprint(),
                       "degradation_version": degradations.DEGRADATION_VERSION,
                       "noise_realisations": realisations,
                       # complete noise-generation identity (the degradation cache is keyed on the same items)
                       "seed": config.SEED, "stream_ids": degradations.STREAM_IDS, "numpy": np.__version__}, sort_keys=True)


def saved_matches(blob, entry: Dict[str, object], names: List[str], ds: ImageDataset, context: str) -> bool:
    """True iff a saved results file was produced by this model on exactly this test data and run layout."""
    try:
        return (blob["model_sha256"].item() == entry["sha256"] and blob["context"].item() == context
                and list(blob["runs"]) == names and np.array_equal(blob["test_idx"], ds.test_idx)
                and np.array_equal(blob["y_true"], ds.y_test) and blob["preds"].shape == (len(names), len(ds.y_test)))
    except KeyError:  # a file from before the context was stored
        return False


def score_model(ds: ImageDataset, model_name: str, entry: Dict[str, object], runs, digests: Dict[str, str],
                x_digest: str, context: str, meta: Dict[str, object], log=print) -> Path:
    """Predict on every run for one frozen model; returns the path of its results file."""
    path = predictions_path(ds.name, model_name, entry["seed"])
    names = [run_name(*r) for r in runs]
    if path.exists():
        with np.load(path) as old:
            if saved_matches(old, entry, names, ds, context):
                log(f"[{ds.name}] {model_name} seed {entry['seed']}: reusing {path.name}")
                for name, digest in zip(names, old["digests"]):  # still enforce identical inputs
                    if digests.setdefault(name, str(digest)) != str(digest):
                        raise RuntimeError(f"{name}: degraded array differs from the one other models saw")
                return path
            log(f"[{ds.name}] {model_name} seed {entry['seed']}: {path.name} is stale, recomputing")
    model = load_model(model_name, config.PROJECT_ROOT / entry["file"])
    check_preprocessing(model, meta, config.MODEL_META_DIR / f"{ds.name}__{model_name}.json")
    preds = np.empty((len(runs), len(ds.y_test)), dtype=np.int8)
    cms = np.empty((len(runs), N_CLASSES, N_CLASSES), dtype=np.int32)
    run_digests = []
    t0 = time.time()
    for i, (family, severity, realisation) in enumerate(runs):
        degraded = degradations.get_degraded(ds.x_test, ds.test_idx, family, severity, dataset=ds.name,
                                             split="test", realisation=realisation, source_digest=x_digest)
        digest = array_digest(degraded)
        if digests.setdefault(names[i], digest) != digest:
            raise RuntimeError(f"{names[i]}: degraded array differs from the one other models saw")
        run_digests.append(digest)
        preds[i] = model.predict(degraded)
        cms[i] = confusion(ds.y_test, preds[i].astype(np.int64), N_CLASSES)
    clean_file = config.PREDICTIONS_DIR / f"clean_{ds.name}_{model_name}_seed{entry['seed']}.npz"
    if clean_file.exists():  # the clean run must reproduce Phase 4 exactly
        with np.load(clean_file) as clean:
            if not np.array_equal(clean["y_pred"], preds[names.index("clean")]):
                raise RuntimeError(f"{path.name}: clean predictions differ from the Phase 4 clean-test predictions")
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, runs=np.array(names), preds=preds, confusion=cms, digests=np.array(run_digests),
                        test_idx=ds.test_idx, y_true=ds.y_test, model_sha256=entry["sha256"],
                        context=np.array(context))
    log(f"[{ds.name}] {model_name} seed {entry['seed']}: {len(runs)} runs in {time.time() - t0:.0f}s")
    return path


def collect_results(datasets: Dict[str, ImageDataset], contexts: Dict[str, str], models, runs) -> pd.DataFrame:
    rows = []
    names = [run_name(*r) for r in runs]
    for dataset, ds in datasets.items():
        for model in models:
            meta = json.loads((config.MODEL_META_DIR / f"{dataset}__{model}.json").read_text())
            for entry in meta["files"]:
                with np.load(predictions_path(dataset, model, entry["seed"])) as blob:
                    if not saved_matches(blob, entry, names, ds, contexts[dataset]):
                        raise RuntimeError(f"{predictions_path(dataset, model, entry['seed']).name} does not match "
                                           "the current model, test data, ladder and run configuration")
                    y_true = blob["y_true"]
                    for name, pred, digest, (family, severity, realisation) in zip(
                            blob["runs"], blob["preds"], blob["digests"], runs):
                        rows.append({"dataset": dataset, "model": model, "seed": entry["seed"],
                                     "condition": degradations.condition_name(family, severity),
                                     "family": "clean" if severity == 0 else family, "severity": severity,
                                     "realisation": realisation if family == "gaussian_noise" else 0,
                                     **classification_metrics(y_true, pred.astype(np.int64), N_CLASSES),
                                     "model_sha256": entry["sha256"], "degraded_sha256": str(digest)})
    return pd.DataFrame(rows)


def upgrade_contexts(datasets, models, realisations: int, log=print) -> None:
    """Re-stamp existing results files with the current context WITHOUT recomputing predictions.

    Used when the context definition gained fields (here: the noise seed, stream ids and numpy version). A file is
    re-stamped only if the degraded images generated NOW (current seed, numpy and ladder) have exactly the
    digests the file recorded for every condition, and its model hash, test ids and labels match; otherwise the
    file is left untouched and an error is raised, so a file made from different inputs is never blessed.
    """
    degradations.verify_frozen_ladder()
    runs = run_list(realisations)
    names = [run_name(*r) for r in runs]
    for dataset in datasets:
        ds = load_dataset(dataset, debug=False)
        x_digest = array_digest(ds.x_test)
        context = run_context(ds, x_digest, realisations)
        fresh = [array_digest(degradations.get_degraded(ds.x_test, ds.test_idx, f, s, dataset=ds.name, split="test",
                                                         realisation=r, source_digest=x_digest)) for f, s, r in runs]
        for model in models:
            meta = json.loads((config.MODEL_META_DIR / f"{dataset}__{model}.json").read_text())
            for entry in meta["files"]:
                path = predictions_path(dataset, model, entry["seed"])
                with np.load(path) as old:
                    data = {k: old[k] for k in old.files}
                ok = (data["model_sha256"].item() == entry["sha256"] and [str(r) for r in data["runs"]] == names
                      and [str(d) for d in data["digests"]] == fresh and np.array_equal(data["test_idx"], ds.test_idx)
                      and np.array_equal(data["y_true"], ds.y_test))
                if not ok:
                    raise RuntimeError(f"{path.name}: recorded inputs differ from the degraded images generated now; "
                                       "not re-stamped (re-run this model)")
                data["context"] = np.array(context)
                np.savez_compressed(path, **data)
                log(f"[{dataset}] {model} seed {entry['seed']}: inputs verified, context updated")
    record = json.loads(RUN_RECORD.read_text())
    if record["ladder_fingerprint"] != degradations.ladder_fingerprint() or record["noise_realisations"] != realisations \
            or record["seed"] != config.SEED:
        raise RuntimeError(f"{RUN_RECORD.name} does not match the current ladder, seed or noise-realisation count")
    record["numpy"] = np.__version__
    RUN_RECORD.write_text(json.dumps(record, indent=2) + "\n")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


KEY = ["dataset", "model", "seed", "condition", "realisation"]


def merge_records(old: Dict[str, object], new: Dict[str, object], realisations: int) -> Dict[str, object]:
    """Combine the run record of an earlier partial run with this one (same ladder and noise count only)."""
    same = (old["ladder_fingerprint"] == new["ladder_fingerprint"] and old["noise_realisations"] == realisations
            and old.get("seed") == new["seed"] and old.get("numpy") == new["numpy"])
    if not same:
        raise RuntimeError(f"{RESULTS_CSV.name} was made with a different ladder, noise-realisation count, seed or "
                           "numpy version; delete the results and re-run every model to compare like with like")
    models = {d: dict(m) for d, m in old.get("models", {}).items()}
    for dataset, used in new["models"].items():
        models.setdefault(dataset, {}).update(used)
    return {**new, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen models on all degraded test conditions.")
    parser.add_argument("--dataset", nargs="+", default=["fashion_mnist", "cifair10"])
    parser.add_argument("--model", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--noise-realisations", type=_positive_int, default=3,
                        help="independent noise fields per noise condition (realisations 0..n-1)")
    parser.add_argument("--upgrade-contexts", action="store_true",
                        help="re-stamp existing results files with the current context after verifying their degraded-image "
                             "digests (no predictions are recomputed)")
    args = parser.parse_args()
    if args.upgrade_contexts:
        upgrade_contexts(args.dataset, args.model, args.noise_realisations)
        return
    runs = run_list(args.noise_realisations)
    degradations.verify_frozen_ladder()  # before anything is reused or computed
    all_digests: Dict[str, Dict[str, str]] = {}
    models_used: Dict[str, Dict[str, str]] = {}
    loaded: Dict[str, ImageDataset] = {}
    contexts: Dict[str, str] = {}
    for dataset in args.dataset:
        ds = loaded[dataset] = load_dataset(dataset, debug=False)
        x_digest, digests = array_digest(ds.x_test), {}
        contexts[dataset] = run_context(ds, x_digest, args.noise_realisations)
        for model in args.model:
            meta_path = config.MODEL_META_DIR / f"{dataset}__{model}.json"
            meta = json.loads(meta_path.read_text())
            check_record(ds, meta, meta_path)
            for entry in meta["files"]:
                if file_sha256(config.PROJECT_ROOT / entry["file"]) != entry["sha256"]:
                    raise RuntimeError(f"{entry['file']}: file changed since it was frozen")
                score_model(ds, model, entry, runs, digests, x_digest, contexts[dataset], meta)
                models_used.setdefault(dataset, {})[f"{model}_seed{entry['seed']}"] = entry["sha256"]
        all_digests[dataset] = digests
    results = collect_results(loaded, contexts, args.model, runs)
    record = {"date": datetime.date.today().isoformat(), "noise_realisations": args.noise_realisations,
              "ladder_fingerprint": degradations.ladder_fingerprint(),
              "degradation_version": degradations.DEGRADATION_VERSION, "seed": config.SEED, "numpy": np.__version__,
              "models": models_used}
    digests_out = all_digests
    if RESULTS_CSV.exists():  # a partial run replaces only the rows (and records) it recomputed
        if not RUN_RECORD.exists():
            raise RuntimeError(f"{RESULTS_CSV.name} exists without {RUN_RECORD.name}; delete both and re-run")
        record = merge_records(json.loads(RUN_RECORD.read_text()), record, args.noise_realisations)
        old = pd.read_csv(RESULTS_CSV)
        gone = old.set_index(KEY).index.isin(results.set_index(KEY).index)
        results = pd.concat([old[~gone], results], ignore_index=True)
        if DIGESTS_JSON.exists():
            digests_out = {**json.loads(DIGESTS_JSON.read_text()), **all_digests}
    results = results.sort_values(KEY).reset_index(drop=True)
    results.to_csv(RESULTS_CSV, index=False, float_format="%.6g")
    DIGESTS_JSON.write_text(json.dumps(digests_out, indent=1, sort_keys=True) + "\n")
    RUN_RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {RESULTS_CSV} ({len(results)} rows)")


if __name__ == "__main__":
    main()

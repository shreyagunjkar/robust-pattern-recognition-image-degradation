"""Train and tune the models on CLEAN data, then freeze them.

Test-set discipline: every function here receives only the training and validation arrays; no test
image is loaded into the tuning code. Hyperparameters are chosen by clean-VALIDATION accuracy
(ties: the earlier grid entry). The frozen model files are hashed (SHA-256) in a metadata record, and
src/evaluate_clean.py later scores them on the clean test set exactly as they are.

Tuning budgets (small, explicit and comparable; the grids are in GRIDS):
  lda, lda_hog  6 shrinkage values, each fitted on the full training set
  lda_fs        6 shrinkage values x 4 feature fractions k_frac (ANOVA F-test selection), full training set
  svm_hog       6 configurations (C x RBF gamma multiplier), tuned on a stratified 10,000-image
                training subset (kernel SVM cost grows super-linearly), then the chosen
                configuration is refitted on the FULL training set
  svm_hog_fs    the same 6 configurations x 4 feature fractions k_frac, same protocol
  cnn           4 configurations (learning rate x weight decay), each trained for CNN_EPOCHS epochs on
                the full training set with the best-validation epoch kept; the winner is then
                retrained with CNN_EXTRA_SEEDS further seeds to expose training-seed variance
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import config
from .data_loader import ImageDataset, array_digest, load_dataset, stratified_subsample
from .metrics import classification_metrics
from .preprocessing import fit_norm_stats
from .models import (CNNModel, HOG_PARAMS, MODEL_NAMES, MODEL_TITLES, SklearnModel, hog_features)
from .utils import environment_info, get_device

N_CLASSES = 10  # both datasets
SVM_TUNE_PER_CLASS = 1000
CNN_EPOCHS = 10
CNN_EXTRA_SEEDS = 2  # CNN seeds: SEED, SEED + 1, SEED + 2
# fraction of the features kept by the ANOVA F-test selection; 'keep everything' is deliberately not in the grid,
# so a selection model always selects (the no-selection models lda / svm_hog are the 100% reference)
K_FRACS = (0.1, 0.25, 0.5, 0.75)
GRIDS: Dict[str, List[Dict[str, object]]] = {
    "lda": [{"shrinkage": s} for s in (0.001, 0.01, 0.05, 0.1, 0.3, "auto")],
    "lda_hog": [{"shrinkage": s} for s in (0.001, 0.01, 0.05, 0.1, 0.3, "auto")],
    "svm_hog": [{"C": c, "gamma_mult": g} for c, g in itertools.product((1.0, 10.0), (0.5, 1.0, 2.0))],
    "lda_fs": [{"shrinkage": s, "k_frac": f} for s, f in
               itertools.product((0.001, 0.01, 0.05, 0.1, 0.3, "auto"), K_FRACS)],
    "svm_hog_fs": [{"C": c, "gamma_mult": g, "k_frac": f} for c, g, f in
                   itertools.product((1.0, 10.0), (0.5, 1.0, 2.0), K_FRACS)],
    "cnn": [{"lr": lr, "weight_decay": wd} for lr, wd in itertools.product((1e-3, 3e-3), (0.0, 5e-4))],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def at_grid_edge(grid: Sequence[Dict[str, object]], selected: Dict[str, object]) -> Dict[str, bool]:
    """For each numeric hyperparameter with several values: is the selected one the smallest or largest?"""
    edge = {}
    for key in grid[0]:
        values = sorted({g[key] for g in grid if isinstance(g[key], (int, float))})
        if len(values) > 1 and isinstance(selected[key], (int, float)):
            edge[key] = bool(selected[key] in (values[0], values[-1]))
    return edge


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = classification_metrics(y_true, y_pred, N_CLASSES)
    return {"val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"]}


def tune_sklearn(name: str, x_train, y_train, x_val, y_val, log=print) -> Tuple[Dict, List[Dict]]:
    """Grid search for lda / lda_hog / svm_hog using train and validation data only."""
    grid = GRIDS[name]
    if name.startswith("svm_hog"):
        chosen = stratified_subsample(y_train, SVM_TUNE_PER_CLASS, config.SEED)
        x_fit, y_fit = x_train[chosen], y_train[chosen]
    else:
        x_fit, y_fit = x_train, y_train
    probe = SklearnModel(name, grid[0])
    if probe.use_hog:  # features do not depend on the hyperparameters: compute once
        f_fit, f_val = hog_features(x_fit), hog_features(x_val)
    else:
        probe.stats = fit_norm_stats(x_train)  # clean training images only
        f_fit, f_val = probe.transform(x_fit), probe.transform(x_val)
    table = []
    for hp in grid:
        model = SklearnModel(name, hp).fit_features(f_fit, y_fit)
        row = {**hp, **_scores(y_val, model.predict_features(f_val)), "fit_seconds": round(model.fit_seconds, 2),
               "n_fit": int(len(y_fit))}
        table.append(row)
        log(f"    {name} {hp}: val acc {row['val_accuracy']:.4f}  ({row['fit_seconds']}s)")
    best = max(range(len(table)), key=lambda i: (table[i]["val_accuracy"], -i))
    return grid[best], table


def tune_cnn(x_train, y_train, x_val, y_val, epochs: int, log=print) -> Tuple[Dict, List[Dict], CNNModel]:
    """Train every CNN configuration (full training set, best-validation epoch kept); keep the winner."""
    table, best_model, best_acc, best_i = [], None, -1.0, -1
    for i, hp in enumerate(GRIDS["cnn"]):
        full = {**hp, "epochs": epochs}
        model = CNNModel(full, seed=config.SEED).fit(x_train, y_train, x_val, y_val)
        acc = max(h["val_accuracy"] for h in model.history)
        table.append({**hp, "epochs": epochs, "val_accuracy": acc, "best_epoch": model.best_epoch,
                      "fit_seconds": round(model.fit_seconds, 1), "n_fit": int(len(y_train))})
        log(f"    cnn {hp}: best val acc {acc:.4f} at epoch {model.best_epoch}  ({model.fit_seconds:.0f}s)")
        if acc > best_acc:
            best_acc, best_model, best_i = acc, model, i
    return {**GRIDS["cnn"][best_i], "epochs": epochs}, table, best_model


def _model_path(root: Path, dataset: str, name: str, seed: int = config.SEED) -> Path:
    suffix = "" if seed == config.SEED else f"_seed{seed}"
    return root / dataset / f"{name}{suffix}{'.pt' if name == 'cnn' else '.joblib'}"


def train_one(ds: ImageDataset, name: str, *, epochs: int = CNN_EPOCHS, extra_seeds: int = CNN_EXTRA_SEEDS,
              log=print) -> Dict[str, object]:
    """Tune, fit, save and describe one model on one dataset. Test data are never passed on."""
    root = (config.MODELS_DIR / "debug") if ds.debug else config.MODELS_DIR
    meta_dir = (config.MODELS_DIR / "debug") if ds.debug else config.MODEL_META_DIR
    x_tr, y_tr, x_val, y_val = ds.x_train, ds.y_train, ds.x_val, ds.y_val
    log(f"[{ds.name}] {name}: tuning on {len(y_tr)} train / {len(y_val)} validation images")
    files: List[Dict[str, object]] = []
    feature_selection = None

    def record(model, seed: int) -> None:
        path = _model_path(root, ds.name, name, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        scores = _scores(y_val, model.predict(x_val))
        files.append({"file": os.path.relpath(path, config.PROJECT_ROOT), "sha256": file_sha256(path),
                      "seed": seed, **scores, "fit_seconds": round(float(model.fit_seconds), 1),
                      **({"best_epoch": model.best_epoch, "val_accuracy_by_epoch":
                          [round(h["val_accuracy"], 4) for h in model.history]} if name == "cnn" else {})})
        log(f"    saved {path.name}: val acc {scores['val_accuracy']:.4f}")

    if name == "cnn":
        selected, table, winner = tune_cnn(x_tr, y_tr, x_val, y_val, epochs, log)
        record(winner, config.SEED)
        for seed in range(config.SEED + 1, config.SEED + 1 + extra_seeds):
            log(f"    retraining the selected configuration with seed {seed}")
            record(CNNModel(selected, seed=seed).fit(x_tr, y_tr, x_val, y_val), seed)
    else:
        selected, table = tune_sklearn(name, x_tr, y_tr, x_val, y_val, log)
        model = SklearnModel(name, selected).fit(x_tr, y_tr)  # refit on the FULL training set
        if name.startswith("svm_hog"):
            log(f"    refitted on the full training set: gamma={model.gamma:.4g}, "
                f"{int(model.estimator.n_support_.sum())} support vectors")
        feature_selection = model.feature_selection_record()
        if feature_selection is not None:
            log(f"    feature selection kept {feature_selection['k']} of {feature_selection['n_features_in']} features")
            meta_dir.mkdir(parents=True, exist_ok=True)
            np.save(meta_dir / f"{ds.name}__{name}__selected_mask.npy", model.selector.get_support())
        record(model, config.SEED)

    meta = {
        "dataset": ds.name, "model": name, "title": MODEL_TITLES[name], "debug": ds.debug,
        "trained_on": datetime.date.today().isoformat(), "device": get_device(),
        "selection": {"metric": "clean validation accuracy", "ties": "earlier grid entry",
                      "grid": table, "selected": selected, "selected_at_grid_edge": at_grid_edge(GRIDS[name], selected)},
        "hog_params": {k: list(v) if isinstance(v, tuple) else v for k, v in HOG_PARAMS.items()}
        if name in ("svm_hog", "lda_hog", "svm_hog_fs") else None,
        "feature_selection": feature_selection,
        "data": {"train_n": int(len(y_tr)), "val_n": int(len(y_val)),
                 "x_train_sha256": array_digest(x_tr), "y_train_sha256": array_digest(y_tr),
                 "x_val_sha256": array_digest(x_val), "y_val_sha256": array_digest(y_val),
                 "train_idx_sha256": array_digest(ds.train_idx)},
        "files": files, "test_set_used_in_training_or_tuning": False,
        "environment": {k: v for k, v in environment_info().items() if k in ("python", "numpy", "scipy", "sklearn", "skimage", "torch", "platform")},
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{ds.name}__{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and train the models on clean data (no test data).")
    parser.add_argument("--dataset", nargs="+", default=["fashion_mnist", "cifair10"], choices=["fashion_mnist", "cifair10"])
    parser.add_argument("--model", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--debug", action="store_true", help="tiny subsets and 2 CNN epochs (smoke test; separate output dir)")
    args = parser.parse_args()
    for dataset in args.dataset:
        ds = load_dataset(dataset, debug=True if args.debug else False)
        for name in args.model:
            train_one(ds, name, epochs=2 if args.debug else CNN_EPOCHS, extra_seeds=1 if args.debug else CNN_EXTRA_SEEDS)


if __name__ == "__main__":
    main()

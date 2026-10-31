"""Score the FROZEN models on the clean test set (the only place Phase 4 touches test data).

Before a model is scored, (1) its file must match the SHA-256 recorded when it was trained and (2) the
training/validation arrays recorded in its metadata must equal the ones the current split produces,
so a model trained on a different split cannot be evaluated by accident. Nothing here fits or tunes
anything. Outputs:
  results/metrics/clean_test_metrics.csv     accuracy, error rate, macro P/R/F1, validation accuracy
  results/confusion_matrices/clean_*.csv     test confusion matrices (rows true, columns predicted)
  results/predictions/clean_*.npz            test ids, labels, predictions (for paired tests later)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config
from .data_loader import ImageDataset, array_digest, load_dataset
from .metrics import classification_metrics, confusion
from .models import MODEL_NAMES, load_model
from .models.classical import normalized_hog_params
from .train import N_CLASSES, file_sha256

CLEAN_CSV = config.METRICS_DIR / "clean_test_metrics.csv"


class FrozenModelError(RuntimeError):
    """A model file or its training data does not match the record made at training time."""


def check_record(ds: ImageDataset, meta: Dict[str, object], meta_path: Path) -> None:
    data = meta["data"]
    current = {"x_train_sha256": array_digest(ds.x_train), "y_train_sha256": array_digest(ds.y_train),
               "x_val_sha256": array_digest(ds.x_val), "y_val_sha256": array_digest(ds.y_val),
               "train_idx_sha256": array_digest(ds.train_idx)}
    stale = [k for k, v in current.items() if data[k] != v]
    if stale:
        raise FrozenModelError(f"{meta_path.name}: model was trained on different data ({stale})")
    if meta.get("test_set_used_in_training_or_tuning") is not False:
        raise FrozenModelError(f"{meta_path.name}: record does not state that the test set was unused")


def check_preprocessing(model, meta: Dict[str, object], meta_path: Path) -> None:
    """The HOG settings the loaded model will use must equal those recorded when it was trained."""
    recorded = meta.get("hog_params")
    used = getattr(model, "frozen_hog_params", lambda: None)()
    used = None if used is None else normalized_hog_params(used)
    if used != recorded:
        raise FrozenModelError(f"{meta_path.name}: HOG settings {used} differ from the recorded {recorded}")
    selection = getattr(model, "feature_selection_record", lambda: None)()
    if selection != meta.get("feature_selection"):
        raise FrozenModelError(f"{meta_path.name}: feature selection {selection} differs from the recorded "
                               f"{meta.get('feature_selection')}")


def evaluate_models(ds: ImageDataset, meta_dir: Path, models=MODEL_NAMES, log=print) -> List[Dict[str, object]]:
    rows = []
    for name in models:
        meta_path = meta_dir / f"{ds.name}__{name}.json"
        meta = json.loads(meta_path.read_text())
        check_record(ds, meta, meta_path)
        for entry in meta["files"]:
            path = config.PROJECT_ROOT / entry["file"]
            if file_sha256(path) != entry["sha256"]:
                raise FrozenModelError(f"{path.name}: file changed since it was frozen (SHA-256 mismatch)")
            model = load_model(name, path)
            check_preprocessing(model, meta, meta_path)
            y_pred = model.predict(ds.x_test)
            scores = classification_metrics(ds.y_test, y_pred, N_CLASSES)
            tag = f"clean_{ds.name}_{name}_seed{entry['seed']}"
            for folder, suffix in ((config.CONFUSION_DIR, "csv"), (config.PREDICTIONS_DIR, "npz")):
                folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(confusion(ds.y_test, y_pred, N_CLASSES), index=ds.class_names,
                         columns=ds.class_names).to_csv(config.CONFUSION_DIR / f"{tag}.csv")
            np.savez_compressed(config.PREDICTIONS_DIR / f"{tag}.npz", test_idx=ds.test_idx,
                                y_true=ds.y_test, y_pred=y_pred)
            rows.append({"dataset": ds.name, "model": name, "seed": entry["seed"], **scores,
                         "val_accuracy": entry["val_accuracy"], "model_sha256": entry["sha256"],
                         "x_test_sha256": array_digest(ds.x_test)})
            log(f"[{ds.name}] {name} seed {entry['seed']}: test acc {scores['accuracy']:.4f} "
                f"macro-F1 {scores['macro_f1']:.4f} (validation acc {entry['val_accuracy']:.4f})")
    return rows


def write_rows(rows: List[Dict[str, object]], path: Path) -> None:
    """Merge into the CSV by (dataset, model, seed): evaluating one dataset keeps the others' rows."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keep = ~old.set_index(["dataset", "model", "seed"]).index.isin(new.set_index(["dataset", "model", "seed"]).index)
        new = pd.concat([old[keep], new], ignore_index=True)
    new = new.sort_values(["dataset", "model", "seed"]).reset_index(drop=True)
    new.to_csv(path, index=False, float_format="%.6g")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen models on the clean test set.")
    parser.add_argument("--dataset", nargs="+", default=["fashion_mnist", "cifair10"])
    parser.add_argument("--model", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    args = parser.parse_args()
    rows: List[Dict[str, object]] = []
    for dataset in args.dataset:
        rows += evaluate_models(load_dataset(dataset, debug=False), config.MODEL_META_DIR, args.model)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(rows, CLEAN_CSV)
    print(f"wrote {CLEAN_CSV}")


if __name__ == "__main__":
    main()

"""The convolutional network (see the package docstring in src/models/__init__.py)."""
from __future__ import annotations

import copy
import time
from typing import Dict, Optional

import numpy as np

from .. import config
from ..preprocessing import NormStats, fit_norm_stats, normalize
from ..utils import get_device, set_seed

# --------------------------------------------------------------------------- #
# CNN
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    from torch import nn
    return torch, nn


def build_cnn(in_channels: int, n_classes: int = 10, width: int = 32):
    """Five 3x3 conv layers (BatchNorm + ReLU), two max-pools, global average pooling, one linear layer."""
    torch, nn = _torch()

    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

    w = width
    return nn.Sequential(
        *block(in_channels, w), *block(w, w), nn.MaxPool2d(2), nn.Dropout(0.25),
        *block(w, 2 * w), *block(2 * w, 2 * w), nn.MaxPool2d(2), nn.Dropout(0.25),
        *block(2 * w, 4 * w), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, n_classes))


class CNNModel:
    """Trained once on clean data (no augmentation: it would confound the robustness comparison),
    with the epoch of best clean-VALIDATION accuracy kept; evaluated in eval mode (BatchNorm uses
    its running statistics, never statistics of the degraded batch)."""

    name = "cnn"

    def __init__(self, hparams: Dict[str, object], seed: int = config.SEED):
        self.hparams, self.seed = dict(hparams), int(seed)
        self.stats: Optional[NormStats] = None
        self.net = None
        self.in_channels: Optional[int] = None
        self.history: list = []
        self.best_epoch: Optional[int] = None
        self.fit_seconds: Optional[float] = None

    def _tensor(self, x: np.ndarray):
        torch, _ = _torch()
        return torch.from_numpy(np.ascontiguousarray(normalize(x, self.stats).transpose(0, 3, 1, 2)))

    def fit(self, x_train, y_train, x_val, y_val) -> "CNNModel":
        torch, nn = _torch()
        t0 = time.time()
        set_seed(self.seed)
        self.stats = fit_norm_stats(x_train)  # clean training images only, then frozen
        self.in_channels = x_train.shape[-1]
        device = get_device()
        net = build_cnn(self.in_channels).to(device)
        hp, bs = self.hparams, int(self.hparams.get("batch_size", 128))
        opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
        epochs = int(hp["epochs"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(hp["lr"]), epochs=epochs,
                                                    steps_per_epoch=-(-len(x_train) // bs))
        xt, yt = self._tensor(x_train), torch.from_numpy(y_train)
        generator = torch.Generator().manual_seed(self.seed)
        best_acc, best_state = -1.0, None
        for epoch in range(1, epochs + 1):
            net.train()
            order = torch.randperm(len(xt), generator=generator)
            total = 0.0
            for start in range(0, len(order), bs):
                idx = order[start:start + bs]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(net(xt[idx].to(device)), yt[idx].to(device))
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            self.net = net
            val_acc = float((self.predict(x_val) == y_val).mean())
            self.history.append({"epoch": epoch, "train_loss": total / len(xt), "val_accuracy": val_acc})
            if val_acc > best_acc:
                best_acc, self.best_epoch = val_acc, epoch
                best_state = copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()})
        net.load_state_dict(best_state)
        self.net = net.eval()
        self.fit_seconds = time.time() - t0
        return self

    def predict(self, x: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        torch, _ = _torch()
        device = next(self.net.parameters()).device
        self.net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                logits = self.net(self._tensor(x[start:start + batch_size]).to(device))
                out.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(out).astype(np.int64)

    def save(self, path) -> None:
        torch, _ = _torch()
        torch.save({"state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
                    "stats": {"mean": self.stats.mean, "std": self.stats.std},
                    "in_channels": self.in_channels, "hparams": self.hparams, "seed": self.seed,
                    "best_epoch": self.best_epoch, "history": self.history}, path)

    @staticmethod
    def load(path) -> "CNNModel":
        torch, _ = _torch()
        blob = torch.load(path, map_location=get_device(), weights_only=True)
        model = CNNModel(blob["hparams"], blob["seed"])
        model.stats = NormStats(**blob["stats"])
        model.in_channels, model.best_epoch, model.history = blob["in_channels"], blob["best_epoch"], blob["history"]
        model.net = build_cnn(model.in_channels).to(get_device())
        model.net.load_state_dict(blob["state_dict"])
        model.net.eval()
        return model

"""Seeding, device selection and environment reporting."""
from __future__ import annotations

import importlib
import platform
import random
import sys
from typing import Dict

import numpy as np

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed every RNG the project uses (Python, NumPy and, if installed, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """'cuda' if an NVIDIA GPU is available, otherwise 'cpu'. Never required."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def environment_info() -> Dict[str, str]:
    """Software versions for the reproducibility section of the paper."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": get_device(),
        "seed": str(config.SEED),
    }
    for module in ("numpy", "pandas", "scipy", "sklearn", "skimage", "cv2",
                   "matplotlib", "seaborn", "statsmodels", "torch"):
        try:
            info[module] = importlib.import_module(module).__version__
        except ImportError:
            info[module] = "not installed"
    return info


def main() -> None:
    """Write the software versions used for a run to results/metrics/environment.json."""
    import json

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.METRICS_DIR / "environment.json"
    path.write_text(json.dumps(environment_info(), indent=2) + "\n")
    print(path.read_text(), end="")


if __name__ == "__main__":
    main()

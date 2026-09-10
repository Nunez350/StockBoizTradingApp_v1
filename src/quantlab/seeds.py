"""Global seeding so every run is reproducible bit-for-bit where the libraries allow it."""

from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:  # pragma: no cover
        pass


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

"""Target portfolio for the next open from the registered strategy config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import StrategyConfig
from quantlab.data.panel import Panel
from quantlab.registry.experiment import run_pipeline


def compute_targets(cfg: StrategyConfig, root: Path, panel: Panel, features: dict, equity: float) -> pd.DataFrame:
    """Weights decided at the latest close -> target shares at the latest raw close."""
    out = run_pipeline(cfg, root, panel=panel, features=features)
    w_all = out.weights
    if w_all.empty:
        raise RuntimeError("strategy produced no weights")
    t = w_all.index[-1]
    if t != panel.sessions[-1] and (panel.sessions[-1] - t).days > 8:
        raise RuntimeError(f"latest decision date {t.date()} is stale vs panel end {panel.sessions[-1].date()}")
    w = w_all.loc[t]
    w = w[w.abs() > 1e-9]
    px = panel.close.iloc[-1].reindex(w.index)
    df = pd.DataFrame({"target_w": w, "ref_price": px})
    df["target_notional"] = df["target_w"] * equity
    df["target_shares"] = np.floor(np.abs(df["target_notional"]) / df["ref_price"]) * np.sign(df["target_w"])
    df.attrs["decision_date"] = str(t.date())
    return df.dropna()

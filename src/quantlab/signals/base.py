"""Signal = DataFrame (decision dates x symbols) of scores; higher = more attractive.

Dispatch by ``SignalConfig.kind``. Signals only read features/panel data at or before
each decision date (the features already guarantee this; signals must not shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.calendar import rebalance_dates
from quantlab.config import StrategyConfig
from quantlab.data.panel import Panel
from quantlab.features.base import cs_rank, cs_zscore, mask


def decision_dates(panel: Panel, rule: str) -> pd.DatetimeIndex:
    d = rebalance_dates(panel.sessions, rule)
    return d[d < panel.sessions[-1]]  # last session cannot be executed


def momentum_signal(features: dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
    name = params.get("feature", "mom_12_1")
    s = features[name]
    if params.get("vol_scale", False):
        s = s / features["rvol_60"].replace(0, np.nan)
    return s


def meanrev_signal(features: dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
    name = params.get("feature", "rev_5d")
    s = -features[name]
    if params.get("vol_scale", False):
        s = s / features["rvol_20"].replace(0, np.nan)
    return s


def combine_signal(features: dict[str, pd.DataFrame], params: dict, tradable: pd.DataFrame) -> pd.DataFrame:
    """Rank-average of named components with signs, e.g. {mom_12_1: 1, rev_1m: -1}."""
    comps = params["components"]
    acc = None
    for name, sign in comps.items():
        r = cs_rank(mask(features[name], tradable)) * float(sign)
        acc = r if acc is None else acc.add(r, fill_value=0.0)
    return acc


def random_signal(panel: Panel, seed: int) -> pd.DataFrame:
    rs = np.random.default_rng(seed)
    return pd.DataFrame(rs.normal(size=panel.adj_close.shape), index=panel.sessions, columns=panel.symbols)


def make_signal(cfg: StrategyConfig, panel: Panel, features: dict[str, pd.DataFrame],
                ml_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    kind, params = cfg.signal.kind, cfg.signal.params
    dates = decision_dates(panel, cfg.rebalance)
    if kind == "benchmark":
        return pd.DataFrame(index=dates, columns=panel.symbols, dtype=float)  # weights come from fixed_weights
    if kind == "zero":
        return pd.DataFrame(0.0, index=dates, columns=panel.symbols)
    if kind == "random":
        s = random_signal(panel, cfg.seed)
    elif kind == "momentum":
        s = momentum_signal(features, params)
    elif kind == "meanrev":
        s = meanrev_signal(features, params)
    elif kind == "combine":
        s = combine_signal(features, params, panel.tradable)
    elif kind == "ml":
        if ml_predictions is None:
            raise ValueError("ml signal needs predictions")
        s = ml_predictions
    else:  # pragma: no cover
        raise ValueError(kind)
    s = mask(s.reindex(index=panel.sessions, columns=panel.symbols), panel.tradable)
    if params.get("zscore", False):
        s = cs_zscore(s)
    return s.loc[s.index.intersection(dates)]

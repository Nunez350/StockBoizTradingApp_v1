"""Feature contract, registry, and per-date scalers.

THE NO-LOOKAHEAD CONTRACT
-------------------------
A feature is a function ``Panel -> DataFrame (sessions x symbols)`` whose value at
row ``t`` depends only on panel data with session date <= t (close of t inclusive).

* Never call ``.shift(-k)`` in a feature. Forward returns are computed only in
  ``ml/dataset.py`` and ``stats/`` and are always named ``fwd_*``.
* Cross-sectional scaling is per date (``cs_zscore``, ``cs_rank``); time-series
  scaling uses rolling/expanding windows that end at ``t``.
* The backtest engine, not the feature or the signal, applies the execution lag:
  a score at decision date ``t`` is traded at the open of ``t+1``.

Every feature must be registered with ``@feature`` so ``tests/test_no_lookahead.py``
covers it automatically (it recomputes each feature on ``panel.slice(end=t)`` and
asserts equality at ``t``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantlab.data.panel import Panel

FeatureFn = Callable[[Panel], pd.DataFrame]
FEATURES: dict[str, FeatureFn] = {}


def feature(name: str) -> Callable[[FeatureFn], FeatureFn]:
    def deco(fn: FeatureFn) -> FeatureFn:
        if name in FEATURES:
            raise KeyError(f"feature {name!r} already registered")
        FEATURES[name] = fn
        fn.feature_name = name  # type: ignore[attr-defined]
        return fn

    return deco


def compute(name: str, panel: Panel) -> pd.DataFrame:
    return FEATURES[name](panel)


# ------------------------------------------------------------------ per-date scalers
def cs_winsorize(df: pd.DataFrame, lo: float = 0.01, hi: float = 0.99) -> pd.DataFrame:
    q_lo = df.quantile(lo, axis=1)
    q_hi = df.quantile(hi, axis=1)
    return df.clip(lower=q_lo, upper=q_hi, axis=0)


def cs_zscore(df: pd.DataFrame, winsor: bool = True) -> pd.DataFrame:
    x = cs_winsorize(df) if winsor else df
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank within each date, centered on zero: in (-0.5, 0.5)."""
    return df.rank(axis=1, pct=True) - 0.5 - 0.5 / df.notna().sum(axis=1).clip(lower=1).values[:, None]


def mask(df: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    """NaN-out entries outside the tradable universe (per date)."""
    return df.where(tradable.reindex_like(df).fillna(False))


# ------------------------------------------------------------------ helpers
def returns(panel: Panel) -> pd.DataFrame:
    return panel.adj_close.pct_change(fill_method=None)


def benchmark_returns(panel: Panel) -> pd.Series:
    b = panel.meta.get("benchmark", "SPY")
    return panel.adj_close[b].pct_change(fill_method=None)

"""Newey-West (Bartlett kernel) HAC mean and t-statistic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def nw_variance(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    g0 = float(e @ e) / n
    v = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        v += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    return max(v, 1e-18)


def nw_mean_tstat(x, lags: int | None = None) -> dict:
    """Mean, HAC standard error, t and two-sided p. ``lags`` default = floor(4 (n/100)^(2/9))."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": n, "lags": 0}
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    var = nw_variance(x, lags)
    se = np.sqrt(var / n)
    t = x.mean() / se if se > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
    return {"mean": float(x.mean()), "se": float(se), "t": float(t), "p": float(p), "n": n, "lags": lags}

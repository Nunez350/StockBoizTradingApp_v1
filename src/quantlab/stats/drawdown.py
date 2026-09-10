from __future__ import annotations

import numpy as np
import pandas as pd


def dd_series(returns: pd.Series) -> pd.Series:
    eq = (1 + returns.fillna(0)).cumprod()
    return eq / eq.cummax() - 1.0


def max_dd(returns: pd.Series) -> float:
    return float(dd_series(returns).min())


def dd_duration(returns: pd.Series) -> int:
    """Longest run (periods) below a prior peak."""
    dd = dd_series(returns).to_numpy()
    best = cur = 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return int(best)


def cagr(returns: pd.Series, periods: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (periods / len(r)) - 1)

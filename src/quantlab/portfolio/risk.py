"""Risk helpers: strictly-lagged vol targeting and beta estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_scalar(past_returns: pd.Series | np.ndarray, target: float, lookback: int, cap: float,
                      floor: float = 0.0) -> float:
    """Leverage multiplier from an EWMA of *past* daily returns (must exclude the current day)."""
    r = pd.Series(np.asarray(past_returns, dtype=float)).dropna()
    if len(r) < max(20, lookback // 3):
        return 1.0
    ew = r.ewm(span=lookback, min_periods=20).std().iloc[-1] * np.sqrt(252)
    if not np.isfinite(ew) or ew <= 1e-8:
        return 1.0
    return float(np.clip(target / ew, floor, cap))


def rolling_betas(returns: pd.DataFrame, bench: pd.Series, window: int = 252, min_periods: int = 120) -> pd.DataFrame:
    cov = returns.rolling(window, min_periods=min_periods).cov(bench)
    var = bench.rolling(window, min_periods=min_periods).var()
    return cov.div(var, axis=0)

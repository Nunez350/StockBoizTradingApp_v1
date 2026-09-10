"""Quantile portfolio analysis: equal-weight decile returns, top-bottom spread, monotonicity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.stats.newey_west import nw_mean_tstat


def decile_returns(signal: pd.DataFrame, fwd: pd.DataFrame, n: int = 10, min_names: int = 50) -> pd.DataFrame:
    fwd = fwd.reindex(index=signal.index, columns=signal.columns)
    both = signal.notna() & fwd.notna()
    s, f = signal.where(both), fwd.where(both)
    q = s.rank(axis=1, pct=True)
    bucket = np.ceil(q * n).clip(1, n)
    out = {}
    for d in range(1, n + 1):
        out[d] = f.where(bucket == d).mean(axis=1)
    df = pd.DataFrame(out)
    return df.where(both.sum(axis=1) >= min_names)


def spread_stats(dec: pd.DataFrame, lags: int = 0, periods_per_year: float = 12.0, n_boot: int = 1000, seed: int = 0) -> dict:
    n = dec.shape[1]
    spread = (dec[n] - dec[1]).dropna()
    nw = nw_mean_tstat(spread, lags=lags)
    means = dec.mean()
    rho, _ = stats.spearmanr(np.arange(1, n + 1), means.values)
    rs = np.random.default_rng(seed)
    vals = dec.dropna().to_numpy()
    boots = []
    for _ in range(n_boot):
        idx = rs.integers(0, len(vals), len(vals))
        m = vals[idx].mean(axis=0)
        boots.append(stats.spearmanr(np.arange(1, n + 1), m)[0])
    p_mono = float(np.mean(np.asarray(boots) <= 0))
    sd = spread.std(ddof=1)
    return {
        "spread_mean": nw["mean"], "spread_t_nw": nw["t"], "spread_p": nw["p"], "n": nw["n"],
        "spread_sharpe_ann": float(nw["mean"] / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
        "monotonicity_rho": float(rho), "monotonicity_p_boot": p_mono, "decile_means": means.to_dict(),
    }

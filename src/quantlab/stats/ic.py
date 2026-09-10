"""Information coefficient analysis. Forward returns here are the ONLY forward-looking
quantities in the research code and are always named ``fwd``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.panel import Panel
from quantlab.stats.newey_west import nw_mean_tstat


def forward_returns(panel: Panel, horizon: int, execution: str = "next_open") -> pd.DataFrame:
    """Return earned by a decision at close t: open(t+1) -> open(t+1+h) (or close(t+1) -> close(t+1+h))."""
    px = panel.adj_open if execution == "next_open" else panel.adj_close
    return px.shift(-(1 + horizon)) / px.shift(-1) - 1.0


def _rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int) -> pd.Series:
    both = a.notna() & b.notna()
    a = a.where(both)
    b = b.where(both)
    n = both.sum(axis=1)
    am = a.sub(a.mean(axis=1), axis=0)
    bm = b.sub(b.mean(axis=1), axis=0)
    num = (am * bm).sum(axis=1)
    den = np.sqrt((am**2).sum(axis=1) * (bm**2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.where(n >= min_names)


def ic_series(signal: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman", min_names: int = 30) -> pd.Series:
    fwd = fwd.reindex(index=signal.index, columns=signal.columns)
    both = signal.notna() & fwd.notna()
    s, f = signal.where(both), fwd.where(both)
    if method == "spearman":
        s, f = s.rank(axis=1), f.rank(axis=1)
    return _rowwise_corr(s, f, min_names).rename("ic")


def ic_summary(ic: pd.Series, lags: int = 0) -> dict:
    ic = ic.dropna()
    nw = nw_mean_tstat(ic, lags=lags)
    half = len(ic) // 2
    return {
        "mean": nw["mean"], "std": float(ic.std()), "ir": float(nw["mean"] / ic.std()) if ic.std() > 0 else np.nan,
        "t_nw": nw["t"], "p_nw": nw["p"], "n": nw["n"], "lags": lags, "hit": float((ic > 0).mean()),
        "mean_first_half": float(ic.iloc[:half].mean()), "mean_second_half": float(ic.iloc[half:].mean()),
    }


def ic_decay(signal: pd.DataFrame, panel: Panel, horizons=(1, 2, 5, 10, 21, 42, 63), execution="next_open",
             lag_spacing: int = 1) -> pd.DataFrame:
    rows = []
    for h in horizons:
        ic = ic_series(signal, forward_returns(panel, h, execution))
        lags = max(0, int(np.ceil(h / max(lag_spacing, 1))) - 1)
        s = ic_summary(ic, lags=lags)
        rows.append({"horizon": h, "mean_ic": s["mean"], "t_nw": s["t_nw"], "ir": s["ir"], "n": s["n"]})
    return pd.DataFrame(rows).set_index("horizon")


def decision_spacing(index: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> int:
    """Median number of sessions between consecutive decision dates (for NW lag choice)."""
    if len(index) < 2:
        return 1
    pos = sessions.get_indexer(index)
    return int(max(1, np.median(np.diff(pos))))

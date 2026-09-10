"""Fama-MacBeth cross-sectional regressions with per-date winsorizing and standardization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.features.base import cs_zscore
from quantlab.stats.newey_west import nw_mean_tstat


def fama_macbeth(fwd: pd.DataFrame, signal: pd.DataFrame, controls: dict[str, pd.DataFrame] | None = None,
                 lags: int = 0, min_names: int = 50) -> dict:
    """Regress fwd returns on standardized signal (+controls) date by date; report mean gammas and NW t."""
    controls = controls or {}
    dates = signal.index
    X = {"signal": cs_zscore(signal)}
    for k, v in controls.items():
        X[k] = cs_zscore(v.reindex(index=dates, columns=signal.columns))
    y = fwd.reindex(index=dates, columns=signal.columns)
    names = ["const", "signal", *controls.keys()]
    gammas, r2s = [], []
    for t in dates:
        cols = [X[k].loc[t] for k in names[1:]]
        df = pd.concat([y.loc[t].rename("y"), *[c.rename(k) for c, k in zip(cols, names[1:])]], axis=1).dropna()
        if len(df) < min_names:
            gammas.append([np.nan] * len(names))
            r2s.append(np.nan)
            continue
        A = np.column_stack([np.ones(len(df)), df[names[1:]].to_numpy()])
        yy = df["y"].to_numpy()
        beta, res, *_ = np.linalg.lstsq(A, yy, rcond=None)
        ss_res = float(((yy - A @ beta) ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        gammas.append(beta.tolist())
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    G = pd.DataFrame(gammas, index=dates, columns=names)
    out = {"gammas": G, "r2_mean": float(np.nanmean(r2s)), "n_dates": int(G["signal"].notna().sum())}
    for k in names:
        nw = nw_mean_tstat(G[k], lags=lags)
        out[k] = {"mean": nw["mean"], "t_nw": nw["t"], "p": nw["p"]}
    return out

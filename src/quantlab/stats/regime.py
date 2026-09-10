"""Performance conditional on (lagged, filtered) regime labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.stats.drawdown import max_dd


def conditional_performance(returns: pd.Series, labels: pd.Series, periods: int = 252) -> pd.DataFrame:
    df = pd.DataFrame({"r": returns, "lab": labels.reindex(returns.index)}).dropna()
    rows = []
    for lab, g in df.groupby("lab"):
        r = g["r"]
        sd = r.std(ddof=1)
        rows.append({"regime": lab, "n": len(r), "share": len(r) / len(df), "ann_ret": r.mean() * periods,
                     "ann_vol": sd * np.sqrt(periods), "sharpe": r.mean() / sd * np.sqrt(periods) if sd > 0 else np.nan,
                     "hit": (r > 0).mean(), "max_dd": max_dd(r)})
    return pd.DataFrame(rows).set_index("regime")

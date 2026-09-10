"""Long (date, symbol) dataset for ML: lagged, per-date-ranked features and a forward-return label.

Labels are the ONLY forward-looking column and carry the interval [t0, t1] needed for purging:
t0 = decision date, t1 = session at which the label is fully known (t0 + 1 + horizon).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.data.panel import Panel
from quantlab.features.base import cs_rank, cs_zscore, mask
from quantlab.signals.base import decision_dates
from quantlab.stats.ic import forward_returns


@dataclass
class Dataset:
    X: pd.DataFrame  # index (date, symbol)
    y: pd.Series
    t0: pd.DatetimeIndex
    t1: pd.DatetimeIndex
    feature_names: list[str]

    def rows_for_dates(self, dates) -> np.ndarray:
        return np.flatnonzero(self.X.index.get_level_values("date").isin(dates))


def build_dataset(panel: Panel, features: dict[str, pd.DataFrame], names: list[str], horizon: int,
                  rebalance: str = "W-FRI", execution: str = "next_open") -> Dataset:
    dates = decision_dates(panel, rebalance)
    sess = panel.sessions
    fwd = forward_returns(panel, horizon, execution)
    y_wide = cs_zscore(mask(fwd, panel.tradable)).loc[dates]
    cols = {}
    for n in names:
        f = cs_rank(mask(features[n].reindex(index=sess, columns=panel.symbols), panel.tradable)).loc[dates]
        cols[n] = f.stack(future_stack=True)
    X = pd.DataFrame(cols)
    X.index.names = ["date", "symbol"]
    y = y_wide.stack(future_stack=True).rename("y")
    y.index.names = ["date", "symbol"]
    df = X.join(y, how="inner").dropna(subset=["y"])
    df = df[df[names].notna().sum(axis=1) >= max(3, len(names) // 2)]  # drop rows with mostly-missing features
    t0 = df.index.get_level_values("date")
    pos = sess.get_indexer(t0) + 1 + horizon
    pos = np.minimum(pos, len(sess) - 1)
    t1 = sess[pos]
    return Dataset(df[names], df["y"], pd.DatetimeIndex(t0), pd.DatetimeIndex(t1), names)


def shuffle_labels_within_date(y: pd.Series, seed: int = 0) -> pd.Series:
    rs = np.random.default_rng(seed)
    out = y.copy()
    for d, idx in y.groupby(level="date").indices.items():
        out.iloc[idx] = rs.permutation(y.iloc[idx].to_numpy())
    return out

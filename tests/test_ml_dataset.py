import numpy as np
import pandas as pd

from quantlab.features.registry import build_features
from quantlab.ml.dataset import build_dataset, shuffle_labels_within_date


def test_dataset_labels_and_causality(panel):
    feats = build_features(panel, ["mom_6_1", "rev_5d", "rvol_20"])
    ds = build_dataset(panel, feats, ["mom_6_1", "rev_5d", "rvol_20"], horizon=5, rebalance="W-FRI")
    assert (ds.t1 > ds.t0).all()
    pos0, pos1 = panel.sessions.get_indexer(ds.t0), panel.sessions.get_indexer(ds.t1)
    assert ((pos1 - pos0) == 6).all()  # 1 (next open) + horizon
    # feature rows at date t are identical when the panel is truncated at t
    t = ds.t0.unique()[len(ds.t0.unique()) // 2]
    feats_cut = build_features(panel.slice(end=t), ["mom_6_1", "rev_5d", "rvol_20"])
    ds_cut = build_dataset(panel.slice(end=t + pd.Timedelta(days=20)), feats_cut, ["mom_6_1", "rev_5d", "rvol_20"], 5, "W-FRI")
    a = ds.X.xs(t, level="date"); b = ds_cut.X.xs(t, level="date")
    common = a.index.intersection(b.index)
    np.testing.assert_allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy(), rtol=1e-9)
    # label is z-scored per date
    z = ds.y.groupby(level="date").agg(["mean", "std"])
    assert (z["mean"].abs() < 1e-8).all() and (np.abs(z["std"] - 1) < 1e-6).all()


def test_shuffle_preserves_per_date_distribution(panel):
    feats = build_features(panel, ["rev_5d"])
    ds = build_dataset(panel, feats, ["rev_5d"], horizon=5)
    ys = shuffle_labels_within_date(ds.y, 0)
    assert not np.allclose(ys.to_numpy(), ds.y.to_numpy())
    d = ds.t0[0]
    assert np.allclose(np.sort(ys.xs(d, level="date")), np.sort(ds.y.xs(d, level="date")))

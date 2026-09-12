"""Meta-labeling (Lopez de Prado): the primary strategy proposes trades; a secondary classifier
predicts whether each proposed position will cover its round-trip cost over the holding horizon.

Label for candidate (t, i):  1 if side_i * fwd_ret_i(t, h) - roundtrip_cost_i(t) > 0
Features: primary-signal strength, liquidity/vol, earnings timing, market stress, plus anything
listed in ``meta.features``. Trained with purged walk-forward exactly like the primary ML model.
Output P(profit) is used to FILTER (drop low-confidence trades -> less turnover) or SCALE weights.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import StrategyConfig
from quantlab.data.panel import Panel
from quantlab.features.base import cs_rank, mask
from quantlab.portfolio.costs import CostModel
from quantlab.stats.cv import walk_forward_splits
from quantlab.stats.ic import forward_returns

DEFAULT_META_FEATURES = ["rvol_20", "log_adv", "amihud_20", "turnover_20", "mom_12_1", "max_ret_21",
                         "days_since_earn", "earn_soon", "sue", "vix", "vix_term"]


def roundtrip_cost_frac(cfg: StrategyConfig, panel: Panel, features: dict, weights: pd.DataFrame) -> pd.DataFrame:
    """Round-trip cost as a fraction of position notional, per (decision date, symbol)."""
    cm = CostModel(cfg.costs)
    sig = (features["rvol_20"] / np.sqrt(252)).reindex(weights.index)
    adv = panel.adv_usd.reindex(weights.index)
    px = panel.close.reindex(weights.index)
    notional = weights.abs() * cfg.initial_capital
    out = pd.DataFrame(np.nan, index=weights.index, columns=weights.columns)
    for t in weights.index:
        n = notional.loc[t].to_numpy()
        act = n > 0
        if not act.any():
            continue
        c = cm.trade_cost(n[act], px.loc[t].to_numpy()[act], sig.loc[t].to_numpy()[act], adv.loc[t].to_numpy()[act])
        row = np.full(len(n), np.nan)
        row[act] = 2.0 * c / n[act]
        out.loc[t] = row
    return out


def build_meta_dataset(cfg: StrategyConfig, panel: Panel, features: dict, signal: pd.DataFrame, weights: pd.DataFrame):
    from quantlab.features.fundamental import TIME_SERIES

    names = cfg.meta.features or [f for f in DEFAULT_META_FEATURES if f in features]
    dates = weights.index
    sess = panel.sessions
    fwd = forward_returns(panel, cfg.horizon, cfg.execution).reindex(index=dates, columns=weights.columns)
    side = np.sign(weights)
    cost = roundtrip_cost_frac(cfg, panel, features, weights)
    pnl = side * fwd - cost
    y_wide = (pnl > 0).astype(float).where(side != 0).where(pnl.notna())
    cols = {}
    srank = cs_rank(mask(signal.reindex(index=dates, columns=weights.columns), panel.tradable.reindex(dates)))
    cols["signal_abs_rank"] = srank.abs().stack(future_stack=True)
    cols["side"] = side.replace(0, np.nan).stack(future_stack=True)
    for n in names:
        raw = features[n].reindex(index=sess, columns=weights.columns)
        if n in TIME_SERIES:
            v = raw.iloc[:, 0]
            z = (v - v.expanding(min_periods=250).mean()) / v.expanding(min_periods=250).std()
            f = pd.DataFrame(np.tile(z.to_numpy()[:, None], (1, raw.shape[1])), index=sess, columns=raw.columns).loc[dates]
        else:
            f = cs_rank(mask(raw, panel.tradable)).loc[dates]
        cols[n] = f.stack(future_stack=True)
    X = pd.DataFrame(cols)
    X.index.names = ["date", "symbol"]
    y = y_wide.stack(future_stack=True).rename("y")
    y.index.names = ["date", "symbol"]
    df = X.join(y, how="inner").dropna(subset=["y", "side"])
    t0 = pd.DatetimeIndex(df.index.get_level_values("date"))
    pos = np.minimum(sess.get_indexer(t0) + 1 + cfg.horizon, len(sess) - 1)
    return df.drop(columns="y"), df["y"], t0, pd.DatetimeIndex(sess[pos]), list(X.columns)


def train_meta(cfg: StrategyConfig, panel: Panel, features: dict, signal: pd.DataFrame, weights: pd.DataFrame, log=print):
    import lightgbm as lgb

    X, y, t0, t1, names = build_meta_dataset(cfg, panel, features, signal, weights)
    m = cfg.meta
    params = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 300,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0} | m.params
    preds, folds = [], []
    t_start = time.time()
    for k, (tr, te, cutoff) in enumerate(walk_forward_splits(t0, t1, panel.sessions, int(m.train_min_years * 252), m.retrain_every, cfg.horizon)):
        clf = lgb.LGBMClassifier(**params, random_state=cfg.seed, deterministic=True, force_col_wise=True, num_threads=24, verbose=-1)
        clf.fit(X.iloc[tr].fillna(0.0).to_numpy(), y.iloc[tr].to_numpy())
        p = pd.Series(clf.predict_proba(X.iloc[te].fillna(0.0).to_numpy())[:, 1], index=X.index[te])
        preds.append(p)
        yy = y.iloc[te]
        base = float(yy.mean())
        sel = p >= m.threshold
        prec = float(yy[sel].mean()) if sel.any() else np.nan
        rc = float(pd.Series(p.to_numpy()).corr(pd.Series(yy.to_numpy()), method="spearman"))
        folds.append({"fold": k, "cutoff": str(cutoff.date()), "n_train": len(tr), "n_test": len(te), "base_rate": base,
                      "precision_at_thr": prec, "kept_frac": float(sel.mean()), "rank_corr": rc})
        if k % 10 == 0:
            log(f"  meta fold {k} {cutoff.date()} base {base:.3f} precision@{m.threshold} {prec:.3f} kept {sel.mean():.0%} ({time.time() - t_start:.0f}s)")
    if not preds:
        raise RuntimeError("no meta folds")
    P = pd.concat(preds).sort_index().unstack("symbol").reindex(columns=weights.columns)
    imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
    return P, pd.DataFrame(folds), imp


def apply_meta(weights: pd.DataFrame, P: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    P = P.reindex(index=weights.index, columns=weights.columns)
    known = P.notna()
    if mode == "filter":
        keep = (P >= threshold) | ~known  # before the first meta fold, trade as the primary model would
        w = weights.where(keep, 0.0)
    else:
        w = weights * P.where(known, 0.5).clip(0, 1) * 2.0
        w = w.clip(lower=-weights.abs(), upper=weights.abs())  # never above the primary weight
    # keep dollar-neutral by shrinking the larger leg to the smaller one (never scale up)
    long, short = w.clip(lower=0).sum(axis=1), -w.clip(upper=0).sum(axis=1)
    tgt = np.minimum(long, short)
    fl = (tgt / long.replace(0, np.nan)).fillna(0.0)
    fs = (tgt / short.replace(0, np.nan)).fillna(0.0)
    w = w.where(w <= 0, w.mul(fl, axis=0)).where(w >= 0, w.mul(fs, axis=0))
    return w.fillna(0.0)


def meta_cache(root: Path, cfg: StrategyConfig) -> Path:
    d = root / "ml"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"meta_{cfg.hash()}.parquet"


def get_or_train_meta(cfg, root: Path, panel, features, signal, weights, log=print) -> pd.DataFrame:
    p = meta_cache(root, cfg)
    if p.exists():
        return pd.read_parquet(p)
    P, folds, imp = train_meta(cfg, panel, features, signal, weights, log=log)
    P.to_parquet(p)
    folds.to_csv(p.with_suffix(".folds.csv"), index=False)
    imp.to_csv(p.with_suffix(".importances.csv"))
    return P

"""Purged walk-forward training, CPCV path distribution, and the shuffled-label leakage test."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import StrategyConfig, load_config
from quantlab.data.panel import Panel
from quantlab.ml.dataset import Dataset, build_dataset, shuffle_labels_within_date
from quantlab.ml.models import make_model
from quantlab.stats.cv import CombinatorialPurgedCV, walk_forward_splits
from quantlab.stats.ic import ic_summary


def _fold_ic(pred: pd.Series, y: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": pred, "y": y})
    return df.groupby(level="date").apply(lambda g: g["p"].corr(g["y"], method="spearman") if len(g) >= 30 else np.nan)


def walk_forward_predictions(cfg: StrategyConfig, panel: Panel, features: dict, ds: Dataset | None = None,
                             shuffle_seed: int | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (predictions wide frame dates x symbols, per-fold metrics, feature importances by fold)."""
    ml = cfg.ml
    names = ml.features or sorted(features)
    ds = ds or build_dataset(panel, features, names, ml.horizon, cfg.rebalance, cfg.execution)
    y = shuffle_labels_within_date(ds.y, shuffle_seed) if shuffle_seed is not None else ds.y
    embargo = ml.embargo if ml.embargo is not None else ml.horizon
    preds, folds, imps = [], [], {}
    t0 = time.time()
    for k, (tr, te, cutoff) in enumerate(walk_forward_splits(ds.t0, ds.t1, panel.sessions, int(ml.train_min_years * 252), ml.retrain_every, embargo)):
        model = make_model(ml.model, cfg.seed, ml.params).fit(ds.X.iloc[tr], y.iloc[tr])
        p = pd.Series(model.predict(ds.X.iloc[te]), index=ds.X.index[te])
        preds.append(p)
        ic = _fold_ic(p, y.iloc[te])
        folds.append({"fold": k, "cutoff": str(cutoff.date()), "n_train": len(tr), "n_test": len(te), "ic_mean": float(ic.mean()), "n_dates": int(ic.notna().sum())})
        if hasattr(model, "importances"):
            imps[k] = model.importances(names)
        if k % 10 == 0:
            log(f"  fold {k} cutoff {cutoff.date()} train {len(tr)} test {len(te)} ic {ic.mean():.4f} ({time.time() - t0:.0f}s)")
    if not preds:
        raise RuntimeError("no walk-forward folds; not enough history for train_min_years")
    P = pd.concat(preds).sort_index()
    wide = P.unstack("symbol").reindex(columns=panel.symbols)
    return wide, pd.DataFrame(folds), pd.DataFrame(imps)


def cpcv_paths(cfg: StrategyConfig, root: Path, panel: Panel, features: dict, log=print) -> pd.DataFrame:
    """Train/test on every CPCV combination; stitch each path's OOS predictions; backtest each path."""
    from quantlab.registry.experiment import run_pipeline
    from quantlab.stats.sharpe import sharpe

    ml = cfg.ml
    names = ml.features or sorted(features)
    ds = build_dataset(panel, features, names, ml.horizon, cfg.rebalance, cfg.execution)
    embargo = ml.embargo if ml.embargo is not None else ml.horizon
    cv = CombinatorialPurgedCV(ml.cpcv_groups, ml.cpcv_test, embargo)
    path_preds: dict[int, list] = {p: [] for p in range(cv.n_paths)}
    t0 = time.time()
    for si, tr, te, grp in cv.split(ds.t0, ds.t1, panel.sessions):
        model = make_model(ml.model, cfg.seed, ml.params).fit(ds.X.iloc[tr], ds.y.iloc[tr])
        p = pd.Series(model.predict(ds.X.iloc[te]), index=ds.X.index[te])
        for g in set(grp[te]):
            m = grp[te] == g
            path_preds[cv.path_of(si, g)].append(p[m])
        log(f"  cpcv split {si + 1}/{len(cv.combos)} ({time.time() - t0:.0f}s)")
    rows = []
    for pi, parts in path_preds.items():
        wide = pd.concat(parts).sort_index().unstack("symbol").reindex(columns=panel.symbols)
        r = run_pipeline(cfg, root, panel=panel, features=features, ml_predictions=wide).result.returns
        rows.append({"path": pi, "sharpe": sharpe(r), "n_obs": len(r)})
        log(f"  path {pi} sharpe {rows[-1]['sharpe']:.2f}")
    return pd.DataFrame(rows)


def predictions_cache(root: Path, cfg: StrategyConfig) -> Path:
    d = root / "ml"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"pred_{cfg.hash()}.parquet"


def get_or_train_predictions(cfg: StrategyConfig, root: Path, panel: Panel, features: dict, log=print) -> pd.DataFrame:
    p = predictions_cache(root, cfg)
    if p.exists():
        return pd.read_parquet(p)
    wide, folds, imps = walk_forward_predictions(cfg, panel, features, log=log)
    wide.to_parquet(p)
    folds.to_csv(p.with_suffix(".folds.csv"), index=False)
    if len(imps):
        imps.to_csv(p.with_suffix(".importances.csv"))
    return wide


def cli_ml(a) -> None:
    from quantlab.registry.experiment import prepare_panel, run_pipeline
    from quantlab.features.registry import load_or_build
    from quantlab.data.panel import load_panel
    from quantlab.registry.runs import RunRegistry

    cfg = load_config(a.config)
    if cfg.ml is None:
        raise SystemExit("config has no `ml:` section")
    root = Path(a.root or cfg.data.root)
    panel = prepare_panel(cfg, root)
    full = load_panel(root, cfg.data.snapshot)
    features = {k: v.reindex(index=panel.sessions, columns=panel.symbols) for k, v in load_or_build(root, full, snapshot=cfg.data.snapshot).items()}
    reg = RunRegistry(root)
    if a.action == "train":
        wide, folds, imps = walk_forward_predictions(cfg, panel, features)
        wide.to_parquet(predictions_cache(root, cfg))
        print(folds.to_string(index=False))
        ics = folds["ic_mean"]
        print(f"mean fold IC {ics.mean():.4f}, share of folds with IC>0: {(ics > 0).mean():.0%}")
        if len(imps):
            print("feature importance (mean over folds):")
            print(imps.mean(axis=1).sort_values(ascending=False).round(1).to_string())
        out = run_pipeline(cfg, root, panel=panel, features=features, ml_predictions=wide)
        s = out.result.summary()
        print(f"walk-forward OOS backtest: net sharpe {s['sharpe']:.2f} gross {s['sharpe_gross']:.2f} cagr {s['cagr']:.2%}")
        reg.log_trial(reg.new_run_id(cfg.name + "_hp"), cfg.name, cfg.family, "ml_hp", cfg.hash(), cfg.to_json(),
                      panel.meta.get("snapshot_id"), out.result.meta.get("git_sha", "nogit"), cfg.seed, out.result.returns,
                      {"fold_ic_mean": float(ics.mean()), "sharpe": s["sharpe"]})
        try:
            import torch

            if cfg.ml.model in ("mlp", "ensemble") and torch.cuda.is_available():
                print(f"GPU used: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass
    elif a.action == "cpcv":
        paths = cpcv_paths(cfg, root, panel, features)
        print(paths.to_string(index=False))
        print(f"CPCV OOS Sharpe: mean {paths['sharpe'].mean():.2f} sd {paths['sharpe'].std():.2f} min {paths['sharpe'].min():.2f} "
              f"P(path Sharpe<0) {(paths['sharpe'] < 0).mean():.0%}")
    elif a.action == "shuffle-test":
        _, folds, _ = walk_forward_predictions(cfg, panel, features, shuffle_seed=cfg.seed)
        ics = folds["ic_mean"].dropna()
        t = ics.mean() / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else np.nan
        print(f"shuffled-label OOS IC mean {ics.mean():.4f}, t = {t:.2f} over {len(ics)} folds -> {'OK (no leakage)' if abs(t) < 2 else 'LEAK SUSPECTED'}")

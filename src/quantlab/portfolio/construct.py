"""Scores -> target weights under explicit constraints.

All methods take a per-date score vector already masked to the tradable universe.
Output weights are fractions of NAV; sum(|w|) == gross (before overlays).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import PortfolioConfig


def _cap_and_renorm(w: pd.Series, max_pos: float, gross: float, iters: int = 5) -> pd.Series:
    w = w.copy()
    for _ in range(iters):
        w = w.clip(-max_pos, max_pos)
        long, short = w[w > 0], w[w < 0]
        tot = w.abs().sum()
        if tot == 0:
            return w
        # keep the long/short split, rescale each leg toward its share of gross
        target_long = gross * (long.sum() / tot) if tot else 0
        target_short = gross * (-short.sum() / tot) if tot else 0
        if long.sum() > 0:
            w[w > 0] = long * target_long / long.sum()
        if short.sum() < 0:
            w[w < 0] = short * target_short / (-short.sum())
        if (w.abs() <= max_pos + 1e-12).all():
            break
    return w


def _dollar_neutral(w: pd.Series, gross: float) -> pd.Series:
    long, short = w[w > 0], w[w < 0]
    out = w.copy()
    if long.sum() > 0:
        out[w > 0] = long / long.sum() * gross / 2
    if short.sum() < 0:
        out[w < 0] = short / (-short.sum()) * gross / 2
    return out


def scores_to_weights(scores: pd.Series, cfg: PortfolioConfig, betas: pd.Series | None = None,
                      bl_fn=None, date=None) -> pd.Series:
    """One date. ``scores`` indexed by tradable symbols (NaN dropped)."""
    s = scores.dropna()
    s = s[np.isfinite(s)]
    if cfg.method == "fixed":
        return pd.Series(cfg.fixed_weights, dtype=float)
    if len(s) < 10:
        return pd.Series(dtype=float)

    if cfg.method == "rank_ls":
        r = s.rank(pct=True) - 0.5 - 0.5 / len(s)
        w = r - r.mean()
    elif cfg.method == "zscore":
        z = (s - s.mean()) / (s.std() + 1e-12)
        w = z.clip(-3, 3)
    elif cfg.method == "decile_ls":
        q = s.rank(pct=True)
        top = q > 1 - 1 / cfg.n_deciles
        bot = q <= 1 / cfg.n_deciles
        w = pd.Series(0.0, index=s.index)
        w[top] = 1.0
        w[bot] = -1.0
    elif cfg.method == "long_only":
        q = s.rank(pct=True)
        top = q > 1 - 1 / cfg.n_deciles
        w = pd.Series(0.0, index=s.index)
        w[top] = 1.0
        w = w / w.sum() * cfg.gross
    elif cfg.method == "bl":
        if bl_fn is None:
            raise ValueError("bl method needs bl_fn")
        w = bl_fn(s, date)
    else:  # pragma: no cover
        raise ValueError(cfg.method)

    w = w[w != 0]
    if cfg.method == "long_only":
        pass  # already normalized; long-only never goes dollar neutral
    elif cfg.neutral == "dollar":
        w = _dollar_neutral(w, cfg.gross)
    else:
        w = w / w.abs().sum() * cfg.gross
    w = _cap_and_renorm(w, cfg.max_pos, cfg.gross)

    if cfg.neutral == "beta":
        if betas is None:
            raise ValueError("beta neutral needs betas")
        b = betas.reindex(w.index).fillna(1.0)
        hedge = -(w * b).sum()
        w[cfg.hedge_symbol] = w.get(cfg.hedge_symbol, 0.0) + hedge
    return w


def build_target_weights(scores: pd.DataFrame, tradable: pd.DataFrame, cfg: PortfolioConfig,
                         betas: pd.DataFrame | None = None, bl_fn=None) -> pd.DataFrame:
    """Apply ``scores_to_weights`` on every row of a (rebalance dates x symbols) score frame."""
    rows = {}
    for t, row in scores.iterrows():
        ok = tradable.loc[t] if t in tradable.index else pd.Series(True, index=row.index)
        s = row.where(ok.reindex(row.index).fillna(False))
        b = betas.loc[t] if betas is not None and t in betas.index else None
        rows[t] = scores_to_weights(s, cfg, b, bl_fn, t)
    w = pd.DataFrame(rows).T.fillna(0.0)
    w.index.name = "date"
    return w

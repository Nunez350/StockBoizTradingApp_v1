"""Stationary (Politis-Romano) bootstrap for dependent returns with Politis-White block length."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def optimal_block_len(x: np.ndarray) -> float:
    """Politis & White (2004) automatic block length for the stationary bootstrap (with 2009 correction)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 50:
        return max(2.0, n ** (1 / 3))
    e = x - x.mean()
    kn = max(5, int(np.sqrt(np.log10(n))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    acf = np.array([(e[k:] @ e[:-k]) / (e @ e) if k > 0 else 1.0 for k in range(mmax + 1)])
    band = 2 * np.sqrt(np.log10(n) / n)
    m_hat = None
    for m in range(1, mmax - kn):
        if np.all(np.abs(acf[m + 1 : m + 1 + kn]) < band):
            m_hat = m
            break
    if m_hat is None:
        m_hat = max(1, int(np.argmax(np.abs(acf[1:]) < band)))
    M = min(2 * m_hat, mmax)
    ks = np.arange(-M, M + 1)
    lam = np.where(np.abs(ks) <= M / 2, 1.0, 2 * (1 - np.abs(ks) / M))
    R = acf[np.abs(ks)] * (e @ e) / n
    G = float(np.sum(lam * np.abs(ks) * R))
    g0 = float(np.sum(lam * R))
    D = 2 * g0**2
    if D <= 0 or G <= 0:
        return max(2.0, n ** (1 / 3))
    b = (2 * G**2 / D) ** (1 / 3) * n ** (1 / 3)
    return float(np.clip(b, 2.0, n / 4))


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(0, n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                i = rng.integers(0, n)
            idx[t] = i
            i = (i + 1) % n
        out[b] = idx
    return out


def bootstrap_ci(x, stat_fn: Callable[[np.ndarray], float], n_boot: int = 2000, mean_block: float | None = None,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if mean_block is None:
        mean_block = optimal_block_len(x)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, rng)
    vals = np.array([stat_fn(x[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(stat_fn(x)), "lo": float(lo), "hi": float(hi), "se": float(vals.std(ddof=1)),
            "mean_block": float(mean_block), "n_boot": int(len(vals)), "p_le_0": float((vals <= 0).mean())}


def bootstrap_report(returns: pd.Series, periods: int = 252, n_boot: int = 2000, seed: int = 0) -> dict:
    from quantlab.stats.drawdown import max_dd

    r = returns.dropna()
    blk = optimal_block_len(r.to_numpy())
    sh = bootstrap_ci(r, lambda a: a.mean() / a.std(ddof=1) * np.sqrt(periods) if a.std() > 0 else np.nan, n_boot, blk, seed=seed)
    cg = bootstrap_ci(r, lambda a: (1 + a).prod() ** (periods / len(a)) - 1, n_boot, blk, seed=seed)
    dd = bootstrap_ci(r, lambda a: max_dd(pd.Series(a)), n_boot, blk, seed=seed)
    return {"sharpe": sh, "cagr": cg, "max_dd": dd, "mean_block": blk,
            "note": "max_dd CI is biased toward shallower drawdowns for short samples (bootstrap paths are shorter-memory)"}

"""Sharpe ratio inference: Lo SE, Probabilistic and Deflated Sharpe (Bailey & Lopez de Prado 2012/2014),
minimum track record length, and the effective number of trials via correlation clustering.

Conventions: ``sr`` inputs to psr/dsr/min_trl are PER-PERIOD (not annualized); ``kurt`` is
raw (non-excess) kurtosis. ``annualize`` converts for reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

EULER_GAMMA = 0.5772156649015329


def sharpe(r, periods: int = 252) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def moments(r) -> dict:
    r = pd.Series(r).dropna()
    return {"n": len(r), "sr": float(r.mean() / r.std(ddof=1)) if r.std() > 0 else np.nan,
            "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False))}


def sharpe_se(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of the per-period SR estimate (Lo 2002 / Mertens 2002)."""
    return float(np.sqrt((1 - skew * sr + (kurt - 1) / 4.0 * sr**2) / (n - 1)))


def psr(sr: float, sr0: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probability that the true SR exceeds ``sr0`` given the observed per-period ``sr``."""
    se = sharpe_se(sr, n, skew, kurt)
    return float(stats.norm.cdf((sr - sr0) / se)) if se > 0 else np.nan


def expected_max_sr(n_trials: int, var_trials: float) -> float:
    """E[max SR] over ``n_trials`` independent trials with SR variance ``var_trials`` (per-period)."""
    if n_trials <= 1 or var_trials <= 0:
        return 0.0
    n = float(n_trials)
    z1 = stats.norm.ppf(1 - 1 / n)
    z2 = stats.norm.ppf(1 - 1 / (n * np.e))
    return float(np.sqrt(var_trials) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def dsr(sr: float, n: int, skew: float, kurt: float, n_trials: int, var_trials: float) -> dict:
    sr0 = expected_max_sr(n_trials, var_trials)
    return {"sr0": sr0, "dsr": psr(sr, sr0, n, skew, kurt), "n_trials": n_trials, "var_trials": var_trials}


def min_trl(sr: float, sr0: float, skew: float = 0.0, kurt: float = 3.0, alpha: float = 0.05) -> float:
    """Minimum track record length (periods) to reject SR <= sr0 at level alpha."""
    if sr <= sr0:
        return np.inf
    z = stats.norm.ppf(1 - alpha)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4.0 * sr**2) * (z / (sr - sr0)) ** 2)


def annualize(sr_per_period: float, periods: int = 252) -> float:
    return sr_per_period * np.sqrt(periods)


def effective_trials(returns: pd.DataFrame, corr_threshold: float = 0.9) -> int:
    """Number of clusters of trials whose return correlation exceeds ``corr_threshold`` (single linkage)."""
    if returns.shape[1] <= 1:
        return returns.shape[1]
    c = returns.corr().fillna(0.0).clip(-1, 1).to_numpy()
    d = np.sqrt(0.5 * (1 - c))
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="single")
    labels = fcluster(Z, t=np.sqrt(0.5 * (1 - corr_threshold)), criterion="distance")
    return int(len(np.unique(labels)))


def sharpe_report(r, periods: int = 252, n_trials: int = 1, var_trials: float | None = None,
                  trial_srs: list[float] | None = None) -> dict:
    """All-in-one: annualized SR, Lo SE, PSR(0), DSR, MinTRL. Trial SRs are per-period."""
    m = moments(r)
    sr, n, sk, ku = m["sr"], m["n"], m["skew"], m["kurt"]
    if var_trials is None:
        var_trials = float(np.var(trial_srs, ddof=1)) if trial_srs is not None and len(trial_srs) > 1 else 0.0
    d = dsr(sr, n, sk, ku, n_trials, var_trials)
    return {
        "n": n, "sharpe_ann": annualize(sr, periods), "sharpe_se_ann": annualize(sharpe_se(sr, n, sk, ku), periods),
        "skew": sk, "kurt": ku, "psr_0": psr(sr, 0.0, n, sk, ku),
        "n_trials": n_trials, "sr0_ann": annualize(d["sr0"], periods), "dsr": d["dsr"],
        "min_trl_years": min_trl(sr, 0.0, sk, ku) / periods,
    }

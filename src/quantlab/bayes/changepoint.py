"""Bayesian online changepoint detection (Adams & MacKay 2007) with a Normal-Inverse-Gamma
conjugate predictive (Student-t). Returns the run-length posterior and P(changepoint) per step.
Everything at step t uses observations <= t."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bocpd(x, hazard: float = 1 / 250, mu0: float = 0.0, kappa0: float = 0.01, alpha0: float = 1.0, beta0: float | None = None,
          max_run: int = 2000, recent: int = 10) -> dict:
    """Returns p_changepoint (mass at run length 0), p_recent = P(run length <= ``recent``), and the MAP run length.
    ``p_recent`` is the practical alert statistic: it accumulates over the steps after a break."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if beta0 is None:
        beta0 = alpha0 * np.var(x[: min(60, n)]) if n > 5 else 1.0
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0
    mu = np.array([mu0]); kappa = np.array([kappa0]); alpha = np.array([alpha0]); beta = np.array([beta0])
    p_cp = np.zeros(n)
    p_recent = np.zeros(n)
    rl_map = np.zeros(n, dtype=int)
    for t in range(n):
        df_ = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred = stats.t.pdf(x[t], df_, loc=mu, scale=scale)
        growth = R[t, : t + 1] * pred * (1 - hazard)
        cp = (R[t, : t + 1] * pred * hazard).sum()
        R[t + 1, 1 : t + 2] = growth
        R[t + 1, 0] = cp
        R[t + 1] /= R[t + 1].sum() + 1e-300
        p_cp[t] = R[t + 1, 0]
        p_recent[t] = R[t + 1, : min(recent, t + 1) + 1].sum()
        rl_map[t] = int(np.argmax(R[t + 1, : t + 2]))
        # update sufficient statistics (prepend prior for the new run length 0)
        mu_n = (kappa * mu + x[t]) / (kappa + 1)
        beta_n = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_n]); kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5]); beta = np.concatenate([[beta0], beta_n])
        if len(mu) > max_run:  # truncate very long run lengths
            mu, kappa, alpha, beta = mu[:max_run], kappa[:max_run], alpha[:max_run], beta[:max_run]
    return {"p_changepoint": p_cp, "p_recent": p_recent, "run_length_map": rl_map}


def changepoint_alerts(returns: pd.Series, threshold: float = 0.5, hazard: float = 1 / 250) -> pd.DataFrame:
    r = returns.dropna()
    res = bocpd(r.to_numpy(), hazard=hazard)
    df = pd.DataFrame({"p_changepoint": res["p_changepoint"], "p_recent": res["p_recent"], "run_length": res["run_length_map"]}, index=r.index)
    df["alert"] = (df["p_recent"] > threshold) & (df.index > r.index[30])
    return df

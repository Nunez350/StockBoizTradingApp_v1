"""Black-Litterman posterior expected returns and a constrained mean-variance weighting.

prior:   pi = delta * Sigma * w_mkt          (reverse optimization; Sigma = Ledoit-Wolf shrinkage)
views:   P mu = Q + eps, eps ~ N(0, Omega)   (here: one absolute view per stock from the signal)
posterior: mu_bl = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 Q]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrink_cov(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.dropna(how="all").fillna(0.0)
    lw = LedoitWolf().fit(r.to_numpy())
    return pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)


def bl_posterior(pi: pd.Series, Sigma: pd.DataFrame, Q: pd.Series, omega_diag: pd.Series, tau: float = 0.05) -> pd.Series:
    syms = pi.index
    S = Sigma.loc[syms, syms].to_numpy()
    tS_inv = np.linalg.pinv(tau * S)
    P = np.eye(len(syms))
    Om_inv = np.diag(1.0 / omega_diag.reindex(syms).to_numpy())
    A = tS_inv + P.T @ Om_inv @ P
    b = tS_inv @ pi.to_numpy() + P.T @ Om_inv @ Q.reindex(syms).to_numpy()
    return pd.Series(np.linalg.solve(A, b), index=syms, name="mu_bl")


def bl_weights_from_scores(scores: pd.Series, past_returns: pd.DataFrame, ic: float, gross: float, max_pos: float,
                           delta: float = 2.5, tau: float = 0.05, horizon_days: int = 21) -> pd.Series:
    """Signal-implied views: expected excess return = IC * z(score) * sigma_i (Grinold), confidence from IC."""
    syms = scores.dropna().index.intersection(past_returns.columns)
    R = past_returns[syms].tail(252)
    Sigma = shrink_cov(R) * horizon_days
    vol = pd.Series(np.sqrt(np.diag(Sigma)), index=syms)
    w_mkt = pd.Series(1.0 / len(syms), index=syms)
    pi = delta * Sigma @ w_mkt
    z = (scores[syms] - scores[syms].mean()) / (scores[syms].std() + 1e-12)
    Q = ic * z * vol
    omega = (vol**2) * (1.0 - ic**2) / max(ic**2, 1e-4) * tau  # less confident views when IC is small
    mu = bl_posterior(pi, Sigma, Q, omega, tau)
    # long-short weights ∝ Sigma^-1 (mu - mean); capped and dollar-neutral via the constructor conventions
    excess = mu - mu.mean()
    w = pd.Series(np.linalg.pinv(Sigma.to_numpy()) @ excess.to_numpy(), index=syms)
    w = w - w.mean()
    w = w / w.abs().sum() * gross
    w = w.clip(-max_pos, max_pos)
    return w / w.abs().sum() * gross

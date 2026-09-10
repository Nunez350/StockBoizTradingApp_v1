"""Gaussian HMM regimes with a hand-rolled FORWARD filter.

``hmmlearn.predict_proba`` runs forward-backward (smoothing) and therefore uses future
observations. For anything that feeds a trading decision we need filtered probabilities
P(state_t | x_1..x_t). We fit with hmmlearn on an expanding window (refit monthly) and
run our own forward recursion on the fitted parameters. States are relabeled by
increasing variance of the first feature so "high vol" is always the last state.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from quantlab.data.panel import Panel


def fit_hmm(X: np.ndarray, n_states: int = 2, seed: int = 0):
    from hmmlearn.hmm import GaussianHMM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=seed, tol=1e-4)
        m.fit(X)
    order = np.argsort([m.covars_[k][0, 0] for k in range(n_states)])
    return {"startprob": m.startprob_[order], "transmat": m.transmat_[np.ix_(order, order)],
            "means": m.means_[order], "covars": m.covars_[order]}


def forward_filter(params: dict, X: np.ndarray) -> np.ndarray:
    """Filtered state probabilities, shape (T, K). Row t uses observations 0..t only."""
    K = len(params["startprob"])
    dens = np.column_stack([multivariate_normal(params["means"][k], params["covars"][k], allow_singular=True).pdf(X) for k in range(K)])
    dens = np.atleast_2d(dens).reshape(len(X), K)
    A = params["transmat"]
    out = np.empty((len(X), K))
    a = params["startprob"] * dens[0]
    a = a / a.sum() if a.sum() > 0 else np.full(K, 1.0 / K)
    out[0] = a
    for t in range(1, len(X)):
        a = (a @ A) * dens[t]
        s = a.sum()
        a = a / s if s > 0 else np.full(K, 1.0 / K)
        out[t] = a
    return out


def regime_features(panel: Panel) -> pd.DataFrame:
    b = panel.meta.get("benchmark", "SPY")
    r = panel.adj_close[b].pct_change(fill_method=None)
    rv = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    disp = panel.adj_close.pct_change(fill_method=None).where(panel.tradable).std(axis=1)
    X = pd.DataFrame({"ret": r, "rvol": rv, "dispersion": disp}).dropna()
    return X


def rolling_hmm_high_vol_prob(panel: Panel, n_states: int = 2, refit: str = "ME", min_obs: int = 750) -> pd.Series:
    """P(high-vol state at t | data <= t), refitting the HMM at each period boundary on data <= t."""
    X = regime_features(panel)
    vals = X.to_numpy()
    idx = X.index
    periods = idx.to_period("M" if refit in ("M", "ME") else "Q")
    out = pd.Series(np.nan, index=idx, name="p_high_vol")
    params = None
    start = min_obs
    for i in range(min_obs, len(vals)):
        if params is None or periods[i] != periods[i - 1]:
            params = fit_hmm(vals[:i], n_states)
            start = i
            filt = forward_filter(params, vals[: i + 1])  # includes today's observation
        else:
            filt = forward_filter(params, vals[: i + 1])
        out.iloc[i] = filt[-1, -1]
    return out


def hmm_labels(p_high: pd.Series, threshold: float = 0.5) -> pd.Series:
    lab = pd.Series(np.where(p_high > threshold, "high vol (HMM)", "low vol (HMM)"), index=p_high.index)
    lab[p_high.isna()] = np.nan
    return lab.shift(1)

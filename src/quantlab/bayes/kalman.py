"""Random-walk Kalman filter for time-varying regression coefficients (numpy).

y_t = x_t' beta_t + e_t,  e ~ N(0, r);  beta_t = beta_{t-1} + w_t,  w ~ N(0, q I).
Filtered (not smoothed) estimates are returned so beta_t only uses data <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class KalmanRegression:
    def __init__(self, q: float = 1e-5, r: float | None = None, beta0: np.ndarray | None = None, p0: float = 1.0):
        self.q, self.r, self.beta0, self.p0 = q, r, beta0, p0

    def filter(self, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n, k = X.shape
        r = self.r if self.r is not None else float(np.nanvar(y[: min(60, n)])) * 0.5  # early data only: causal
        beta = np.zeros(k) if self.beta0 is None else np.asarray(self.beta0, dtype=float)
        P = np.eye(k) * self.p0
        out = np.full((n, k), np.nan)
        var = np.full((n, k), np.nan)
        Q = np.eye(k) * self.q
        for t in range(n):
            P = P + Q  # predict
            if np.isfinite(y[t]) and np.all(np.isfinite(X[t])):
                x = X[t]
                S = x @ P @ x + r
                K = P @ x / S
                beta = beta + K * (y[t] - x @ beta)
                P = P - np.outer(K, x) @ P
            out[t] = beta
            var[t] = np.diag(P)
        return out, var


def kalman_beta(returns: pd.Series, bench: pd.Series, q: float = 1e-5) -> pd.Series:
    """Time-varying market beta (with intercept), filtered."""
    df = pd.DataFrame({"y": returns, "x": bench}).dropna()
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    b, _ = KalmanRegression(q=q, beta0=np.array([0.0, 1.0])).filter(df["y"].to_numpy(), X)
    return pd.Series(b[:, 1], index=df.index, name="kalman_beta")


def kalman_panel_betas(returns: pd.DataFrame, bench: pd.Series, q: float = 1e-5) -> pd.DataFrame:
    return pd.DataFrame({c: kalman_beta(returns[c], bench, q) for c in returns.columns}).reindex(returns.index)


def dynamic_fama_macbeth(gammas: pd.Series, q: float = 1e-4) -> pd.DataFrame:
    """Kalman-filtered level of a Fama-MacBeth coefficient series: tracks signal decay over time."""
    g = gammas.dropna()
    b, v = KalmanRegression(q=q, beta0=np.array([g.iloc[: min(24, len(g))].mean()]), r=float(g.var())).filter(g.to_numpy(), np.ones((len(g), 1)))
    return pd.DataFrame({"level": b[:, 0], "sd": np.sqrt(v[:, 0])}, index=g.index)

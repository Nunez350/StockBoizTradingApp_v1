"""GARCH(1,1)-t volatility forecasts, refit on an expanding window, strictly one-step-ahead.

``arch`` numerical gotcha: fit on returns scaled by 100 and rescale the forecast back.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from quantlab.compat import to_numpy_returns


def fit_garch(returns_pct: np.ndarray):
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(returns_pct, vol="GARCH", p=1, q=1, dist="t", rescale=False, mean="Constant")
        return am.fit(disp="off", show_warning=False)


def rolling_garch_forecast(returns: pd.Series, refit: str = "ME", min_obs: int = 750) -> pd.Series:
    """Annualized 1-step-ahead vol forecast for each session t, using data <= t only.

    The model is refit at the end of every ``refit`` period; between refits the fitted
    parameters are carried forward and the variance recursion is updated with each new
    return, so every forecast at t uses only information up to t.
    """
    r = returns.dropna()
    x = to_numpy_returns(r) * 100.0
    idx = r.index
    out = np.full(len(x), np.nan)
    periods = idx.to_period("M" if refit in ("M", "ME") else "Q")
    params = None
    for i in range(min_obs, len(x)):
        new_period = i == min_obs or periods[i] != periods[i - 1]
        if new_period:
            res = fit_garch(x[:i])
            params = res.params
            omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
            mu = params["mu"]
            # initialize the recursion on the fit window
            e2 = (x[:i] - mu) ** 2
            h = np.empty(i)
            h[0] = e2.mean()
            for k in range(1, i):
                h[k] = omega + alpha * e2[k - 1] + beta * h[k - 1]
            h_prev, e2_prev = h[-1], e2[-1]
        else:
            e2_prev = (x[i - 1] - mu) ** 2
        h_next = omega + alpha * e2_prev + beta * h_prev  # forecast for session i made with data <= i-1
        out[i] = np.sqrt(h_next) / 100.0 * np.sqrt(252)
        h_prev = h_next
    return pd.Series(out, index=idx, name="garch_vol_ann").shift(0)

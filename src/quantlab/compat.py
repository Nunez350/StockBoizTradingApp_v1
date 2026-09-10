"""Single shim point for third-party incompatibilities with pandas 3.

pandas 3 enables copy-on-write and a default ``str`` dtype. If a library chokes on
those, the workaround goes here and nowhere else. ``smoke_test`` fits tiny models
with each library and is run by the M0 test suite.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def to_numpy_returns(x: pd.Series | np.ndarray) -> np.ndarray:
    """Float64 contiguous array; strips pandas extension dtypes before handing to C/Cython libs."""
    return np.ascontiguousarray(np.asarray(x, dtype="float64"))


def smoke_test() -> dict[str, str]:
    """Fit a tiny model with each fragile dependency. Returns {lib: 'ok' | error}."""
    out: dict[str, str] = {}
    rs = np.random.default_rng(0)
    r = pd.Series(rs.standard_t(4, 600) * 0.01, index=pd.date_range("2020-01-01", periods=600, freq="B"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import statsmodels.api as sm

            x = sm.add_constant(np.arange(600.0))
            sm.OLS(to_numpy_returns(r), x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            out["statsmodels"] = "ok"
        except Exception as e:  # pragma: no cover
            out["statsmodels"] = f"{type(e).__name__}: {e}"

        try:
            from arch import arch_model

            am = arch_model(100 * r, vol="GARCH", p=1, q=1, dist="t", rescale=False)
            am.fit(disp="off", show_warning=False)
            out["arch"] = "ok"
        except Exception as e:  # pragma: no cover
            out["arch"] = f"{type(e).__name__}: {e}"

        try:
            from hmmlearn.hmm import GaussianHMM

            X = np.column_stack([to_numpy_returns(r), np.abs(to_numpy_returns(r))])
            GaussianHMM(n_components=2, covariance_type="full", n_iter=20, random_state=0).fit(X)
            out["hmmlearn"] = "ok"
        except Exception as e:  # pragma: no cover
            out["hmmlearn"] = f"{type(e).__name__}: {e}"

        try:
            import lightgbm as lgb

            X = rs.normal(size=(300, 4))
            y = X[:, 0] + rs.normal(size=300)
            lgb.LGBMRegressor(n_estimators=10, verbose=-1, deterministic=True, force_col_wise=True).fit(X, y)
            out["lightgbm"] = "ok"
        except Exception as e:  # pragma: no cover
            out["lightgbm"] = f"{type(e).__name__}: {e}"

        try:
            import exchange_calendars as xc

            xc.get_calendar("XNYS", start="2024-01-01", end="2024-12-31")
            out["exchange_calendars"] = "ok"
        except Exception as e:  # pragma: no cover
            out["exchange_calendars"] = f"{type(e).__name__}: {e}"
    return out

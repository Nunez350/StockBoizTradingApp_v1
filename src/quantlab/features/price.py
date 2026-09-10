"""Price/volume features. All causal (see features/base.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.panel import Panel
from quantlab.features.base import benchmark_returns, feature, returns

ANN = 252


@feature("mom_12_1")
def mom_12_1(p: Panel) -> pd.DataFrame:
    """12-month momentum skipping the most recent month."""
    c = p.adj_close
    return c.shift(21) / c.shift(252) - 1.0


@feature("mom_6_1")
def mom_6_1(p: Panel) -> pd.DataFrame:
    c = p.adj_close
    return c.shift(21) / c.shift(126) - 1.0


@feature("rev_1m")
def rev_1m(p: Panel) -> pd.DataFrame:
    """Trailing 1-month return (short-term reversal uses its negative)."""
    c = p.adj_close
    return c / c.shift(21) - 1.0


@feature("rev_5d")
def rev_5d(p: Panel) -> pd.DataFrame:
    c = p.adj_close
    return c / c.shift(5) - 1.0


@feature("rvol_20")
def rvol_20(p: Panel) -> pd.DataFrame:
    return returns(p).rolling(20, min_periods=15).std() * np.sqrt(ANN)


@feature("rvol_60")
def rvol_60(p: Panel) -> pd.DataFrame:
    return returns(p).rolling(60, min_periods=40).std() * np.sqrt(ANN)


def _rolling_beta_corr(p: Panel, window: int, min_periods: int):
    r = returns(p)
    m = benchmark_returns(p)
    cov = r.rolling(window, min_periods=min_periods).cov(m)
    var_m = m.rolling(window, min_periods=min_periods).var()
    var_r = r.rolling(window, min_periods=min_periods).var()
    beta = cov.div(var_m, axis=0)
    corr2 = (cov**2).div(var_m, axis=0).div(var_r)
    return beta, corr2, var_r


@feature("beta_252")
def beta_252(p: Panel) -> pd.DataFrame:
    beta, _, _ = _rolling_beta_corr(p, 252, 120)
    return beta


@feature("idio_vol_60")
def idio_vol_60(p: Panel) -> pd.DataFrame:
    """Residual vol vs benchmark over 60d: sqrt(var_r * (1 - corr^2)) (exact for in-window OLS)."""
    _, corr2, var_r = _rolling_beta_corr(p, 60, 40)
    return np.sqrt((var_r * (1.0 - corr2.clip(0, 1))).clip(lower=0)) * np.sqrt(ANN)


@feature("log_adv")
def log_adv(p: Panel) -> pd.DataFrame:
    return np.log(p.adv_usd.where(p.adv_usd > 0))


@feature("amihud_20")
def amihud_20(p: Panel) -> pd.DataFrame:
    dv = (p.close * p.volume).where(lambda x: x > 0)
    return (returns(p).abs() / dv * 1e6).rolling(20, min_periods=15).mean()


@feature("max_ret_21")
def max_ret_21(p: Panel) -> pd.DataFrame:
    return returns(p).rolling(21, min_periods=15).max()


@feature("high_52w")
def high_52w(p: Panel) -> pd.DataFrame:
    c = p.adj_close
    return c / c.rolling(252, min_periods=120).max()


@feature("skew_60")
def skew_60(p: Panel) -> pd.DataFrame:
    return returns(p).rolling(60, min_periods=40).skew()


@feature("turnover_20")
def turnover_20(p: Panel) -> pd.DataFrame:
    """Dollar volume relative to its 120d mean (volume shock)."""
    dv = p.close * p.volume
    return dv.rolling(20, min_periods=15).mean() / dv.rolling(120, min_periods=60).mean()

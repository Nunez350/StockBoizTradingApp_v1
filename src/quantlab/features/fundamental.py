"""Fundamental, earnings-event and market features.

All fundamental inputs are point-in-time frames (known at their SEC filing date, forward-filled),
built by ``data/edgar.py`` and stored in the snapshot under ``fundamentals/``. Market-level series
(VIX) are broadcast to every symbol; they are flagged ``TIME_SERIES`` so the ML dataset scales
them over time (expanding z-score) instead of cross-sectionally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.panel import Panel
from quantlab.features.base import feature

TIME_SERIES: set[str] = set()


def _need(p: Panel, *names: str) -> bool:
    return all(n in p.frames for n in names)


def _nan_like(p: Panel) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=p.sessions, columns=p.symbols)


def _mktcap(p: Panel) -> pd.DataFrame:
    return p.close * p.frames["shares"]


@feature("earnings_yield")
def earnings_yield(p: Panel) -> pd.DataFrame:
    if not _need(p, "ni_ttm", "shares"):
        return _nan_like(p)
    return p.frames["ni_ttm"] / _mktcap(p).where(lambda x: x > 0)


@feature("book_to_market")
def book_to_market(p: Panel) -> pd.DataFrame:
    if not _need(p, "equity", "shares"):
        return _nan_like(p)
    return p.frames["equity"] / _mktcap(p).where(lambda x: x > 0)


@feature("gross_profitability")
def gross_profitability(p: Panel) -> pd.DataFrame:
    if not _need(p, "gp_ttm", "assets"):
        return _nan_like(p)
    return p.frames["gp_ttm"] / p.frames["assets"].where(lambda x: x > 0)


@feature("roe")
def roe(p: Panel) -> pd.DataFrame:
    if not _need(p, "ni_ttm", "equity"):
        return _nan_like(p)
    return p.frames["ni_ttm"] / p.frames["equity"].where(lambda x: x > 0)


@feature("accruals")
def accruals(p: Panel) -> pd.DataFrame:
    if not _need(p, "ni_ttm", "ocf_ttm", "assets"):
        return _nan_like(p)
    return (p.frames["ni_ttm"] - p.frames["ocf_ttm"]) / p.frames["assets"].where(lambda x: x > 0)


@feature("asset_growth")
def asset_growth(p: Panel) -> pd.DataFrame:
    if not _need(p, "assets"):
        return _nan_like(p)
    a = p.frames["assets"]
    return a / a.shift(252).where(lambda x: x > 0) - 1.0


@feature("rev_growth")
def rev_growth(p: Panel) -> pd.DataFrame:
    if not _need(p, "rev_ttm"):
        return _nan_like(p)
    r = p.frames["rev_ttm"]
    return r / r.shift(252).abs().where(lambda x: x > 0) - 1.0


@feature("sue")
def sue(p: Panel) -> pd.DataFrame:
    """Standardized unexpected earnings: YoY change in quarterly EPS / trailing std of those changes."""
    if not _need(p, "eps_q_yoy"):
        return _nan_like(p)
    d = p.frames["eps_q_yoy"]
    # changes only update at filings; std over ~8 quarters of distinct values
    changed = d.ne(d.shift(1))
    sd = d.where(changed).rolling(252 * 2, min_periods=4).std()
    return d / sd.ffill().where(lambda x: x > 0)


@feature("log_mktcap")
def log_mktcap(p: Panel) -> pd.DataFrame:
    if not _need(p, "shares"):
        return _nan_like(p)
    return np.log(_mktcap(p).where(lambda x: x > 0))


@feature("days_since_earn")
def days_since_earn(p: Panel) -> pd.DataFrame:
    return p.frames["days_since_earn"] if _need(p, "days_since_earn") else _nan_like(p)


@feature("earn_soon")
def earn_soon(p: Panel) -> pd.DataFrame:
    return p.frames["earn_soon"] if _need(p, "earn_soon") else _nan_like(p)


@feature("pead_2d")
def pead_2d(p: Panel) -> pd.DataFrame:
    """Two-day price reaction to the last earnings release (post-earnings-announcement drift input)."""
    return p.frames["pead_2d"] if _need(p, "pead_2d") else _nan_like(p)


def _broadcast(p: Panel, s: pd.Series) -> pd.DataFrame:
    v = s.reindex(p.sessions).ffill()
    return pd.DataFrame(np.tile(v.to_numpy()[:, None], (1, len(p.symbols))), index=p.sessions, columns=p.symbols)


@feature("vix")
def vix(p: Panel) -> pd.DataFrame:
    return _broadcast(p, p.frames["market"]["vix"]) if "market" in p.frames else _nan_like(p)


@feature("vix_term")
def vix_term(p: Panel) -> pd.DataFrame:
    """VIX / VIX3M: > 1 means backwardation (stress)."""
    if "market" not in p.frames:
        return _nan_like(p)
    m = p.frames["market"]
    return _broadcast(p, m["vix"] / m["vix3m"])


@feature("vix_chg_5d")
def vix_chg_5d(p: Panel) -> pd.DataFrame:
    if "market" not in p.frames:
        return _nan_like(p)
    v = p.frames["market"]["vix"]
    return _broadcast(p, v / v.shift(5) - 1.0)


TIME_SERIES |= {"vix", "vix_term", "vix_chg_5d"}

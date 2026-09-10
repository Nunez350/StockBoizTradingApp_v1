"""Vectorized daily backtest engine with a rebalance-date loop.

Conventions
-----------
* ``target_weights``: rows = decision dates (a subset of panel sessions), columns =
  symbols, values = fraction of NAV. Decided at the close of ``t``.
* Execution ``next_open``: fills at ``adj_open`` of the next session, cost-adjusted.
  ``next_close``: fills at ``adj_close`` of the next session (used for sanity checks).
* Positions are held as shares in *adjusted* units, so dividends and splits need no
  special handling; raw prices enter only through per-share commission and ADV.
* Between rebalances holdings are constant; weights drift with prices.
* Borrow cost accrues daily on short notional.
* A held symbol whose price disappears is liquidated at its last available close
  (``reason='delisting'``).
* ``overlay``: optional callable ``(t, past_net_returns: pd.Series) -> float`` that
  scales the gross exposure decided at ``t``; it only ever sees returns up to ``t``.

The engine is the ONLY place the execution lag is applied.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantlab.backtest.result import BacktestResult
from quantlab.data.panel import Panel
from quantlab.portfolio.costs import CostModel

Overlay = Callable[[pd.Timestamp, pd.Series], float]


def run_backtest(panel: Panel, target_weights: pd.DataFrame, cost_model: CostModel,
                 overlay: Overlay | None = None, execution: str = "next_open",
                 initial_capital: float = 1_000_000.0, sigma_daily: pd.DataFrame | None = None) -> BacktestResult:
    sess = panel.sessions
    syms = list(target_weights.columns.union(panel.frames["adj_close"].columns.intersection(target_weights.columns)))
    syms = [s for s in target_weights.columns]  # keep declared order
    missing = [s for s in syms if s not in panel.frames["adj_close"].columns]
    if missing:
        raise KeyError(f"target weights reference symbols not in panel: {missing[:5]}")
    bad_dates = target_weights.index.difference(sess)
    if len(bad_dates):
        raise ValueError(f"target weight dates not in panel sessions: {list(bad_dates[:3])}")

    adj_close = panel.adj_close[syms].to_numpy(dtype=float)
    adj_open = panel.adj_open[syms].to_numpy(dtype=float)
    raw_open = panel.open[syms].to_numpy(dtype=float)
    adv = panel.adv_usd[syms].to_numpy(dtype=float)
    if sigma_daily is None:
        sig = (panel.adj_close[syms].pct_change(fill_method=None).rolling(20, min_periods=10).std()).to_numpy(dtype=float)
    else:
        sig = sigma_daily.reindex(index=sess, columns=syms).to_numpy(dtype=float)

    n, k = adj_close.shape
    pos_by_date = {sess[i]: i for i in range(n)}
    rebal_idx = {pos_by_date[t]: t for t in target_weights.index}

    shares = np.zeros(k)
    cash = float(initial_capital)
    nav = np.full(n, np.nan)
    daily_cost = np.zeros(n)
    weights_close = np.zeros((n, k))
    trades: list[dict] = []
    turnover: dict[pd.Timestamp, float] = {}
    pending: tuple[np.ndarray, pd.Timestamp] | None = None  # (target weights, decision date)
    last_close = np.full(k, np.nan)
    prev_nav = float(initial_capital)
    net_ret = np.full(n, np.nan)

    for i in range(n):
        # ---- execute pending rebalance at this session's open
        if pending is not None:
            w_t, t_dec = pending
            pending = None
            px = adj_open[i] if execution == "next_open" else adj_close[i]
            have_px = np.isfinite(px) & (px > 0)
            nav_pre = cash + np.nansum(np.where(have_px, shares * px, shares * np.nan_to_num(last_close)))
            target_shares = np.where(have_px, w_t * nav_pre / np.where(have_px, px, 1.0), shares)
            d_sh = target_shares - shares
            notional = np.where(have_px, d_sh * px, 0.0)
            act = np.abs(notional) > 1e-6
            if act.any():
                j = i - 1 if i > 0 else i  # decision-date data for cost inputs
                cost = np.zeros(k)
                cost[act] = cost_model.trade_cost(notional[act], raw_open[i][act], sig[j][act], adv[j][act])
                cash -= notional[act].sum() + cost[act].sum()
                daily_cost[i] += cost[act].sum()
                shares = np.where(act, target_shares, shares)
                turnover[t_dec] = float(np.abs(notional[act]).sum() / nav_pre) if nav_pre > 0 else 0.0
                for jj in np.flatnonzero(act):
                    trades.append({"date": sess[i], "decision_date": t_dec, "symbol": syms[jj], "notional": notional[jj],
                                   "shares": notional[jj] / raw_open[i][jj] if raw_open[i][jj] > 0 else np.nan,
                                   "price": px[jj], "cost": cost[jj], "reason": "rebalance"})
            else:
                turnover[t_dec] = 0.0

        # ---- mark to market at close; handle delistings; borrow
        px_c = adj_close[i]
        have = np.isfinite(px_c) & (px_c > 0)
        gone = (~have) & (np.abs(shares) > 0)
        if gone.any():
            liq = shares[gone] * np.nan_to_num(last_close[gone])
            cash += liq.sum()
            for jj in np.flatnonzero(gone):
                trades.append({"date": sess[i], "decision_date": sess[i], "symbol": syms[jj], "notional": -shares[jj] * last_close[jj],
                               "shares": np.nan, "price": last_close[jj], "cost": 0.0, "reason": "delisting"})
            shares = np.where(gone, 0.0, shares)
        last_close = np.where(have, px_c, last_close)
        pos_val = np.where(have, shares * px_c, 0.0)
        short_notional = pos_val[pos_val < 0].sum()
        if short_notional < 0:
            b = cost_model.borrow_cost_daily(short_notional)
            cash -= b
            daily_cost[i] += b
        nav[i] = cash + pos_val.sum()
        weights_close[i] = pos_val / nav[i] if nav[i] > 0 else 0.0
        net_ret[i] = nav[i] / prev_nav - 1.0 if i > 0 else 0.0
        prev_nav = nav[i]

        # ---- decision at close: queue the next rebalance
        if i in rebal_idx and i < n - 1:
            t = rebal_idx[i]
            w = target_weights.loc[t].to_numpy(dtype=float)
            w = np.nan_to_num(w)
            if overlay is not None:
                m = overlay(t, pd.Series(net_ret[1 : i + 1], index=sess[1 : i + 1]))
                w = w * float(m)
            pending = (w, t)

    idx = sess
    equity = pd.Series(nav, index=idx, name="equity")
    returns = pd.Series(net_ret, index=idx, name="net")
    costs = pd.Series(daily_cost, index=idx, name="costs")
    prev = equity.shift(1)
    gross = (returns + costs / prev).rename("gross")
    gross.iloc[0] = 0.0
    positions = pd.DataFrame(weights_close, index=idx, columns=syms)
    long = positions.clip(lower=0).sum(axis=1)
    short = positions.clip(upper=0).sum(axis=1)
    exposure = pd.DataFrame({"long": long, "short": short, "net": long + short, "gross": long - short})
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["date", "decision_date", "symbol", "notional", "shares", "price", "cost", "reason"])
    to = pd.Series(turnover, dtype=float).sort_index()
    to.index.name = "date"
    return BacktestResult(returns, gross, equity, positions, trades_df, to, exposure, costs,
                          meta={"execution": execution, "initial_capital": initial_capital})

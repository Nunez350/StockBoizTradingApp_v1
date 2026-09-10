import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.backtest.engine import run_backtest
from quantlab.calendar import rebalance_dates
from quantlab.config import CostConfig, PortfolioConfig
from quantlab.portfolio.construct import build_target_weights, scores_to_weights
from quantlab.portfolio.costs import CostModel

ZERO = CostModel(CostConfig().scaled(0))
FULL = CostModel(CostConfig())


def _rebal(panel, rule="ME"):
    d = rebalance_dates(panel.sessions, rule)
    return d[d < panel.sessions[-1]]


def test_zero_weights_zero_returns(panel):
    tw = pd.DataFrame(0.0, index=_rebal(panel), columns=panel.symbols)
    res = run_backtest(panel, tw, FULL)
    assert np.allclose(res.returns.fillna(0), 0) and res.trades.empty


def test_spy_buy_and_hold_matches_total_return(panel):
    tw = pd.DataFrame(0.0, index=_rebal(panel), columns=panel.symbols)
    tw["SPY"] = 1.0
    res = run_backtest(panel, tw, ZERO)
    first = res.trades["date"].min()
    spy = panel.adj_close["SPY"]
    # from the first fill (at open) the NAV must track SPY's adjusted close exactly
    nav_path = res.equity.loc[first:] / res.equity.loc[first]
    spy_path = spy.loc[first:] / spy.loc[first]
    np.testing.assert_allclose(nav_path.values, spy_path.values, rtol=1e-8)
    # only one real trade (the initial buy); monthly re-targeting 100% is a no-op
    assert len(res.trades) == 1


def test_more_costs_never_help(panel):
    rs = np.random.default_rng(1)
    dates = _rebal(panel, "W-FRI")
    scores = pd.DataFrame(rs.normal(size=(len(dates), len(panel.symbols))), index=dates, columns=panel.symbols)
    tw = build_target_weights(scores, panel.tradable, PortfolioConfig(method="rank_ls", max_pos=0.1))
    r0 = run_backtest(panel, tw, ZERO).equity.iloc[-1]
    r1 = run_backtest(panel, tw, FULL).equity.iloc[-1]
    r2 = run_backtest(panel, tw, CostModel(CostConfig().scaled(3))).equity.iloc[-1]
    assert r0 > r1 > r2


def test_execution_lag_is_next_open(panel):
    dates = _rebal(panel, "ME")
    tw = pd.DataFrame(0.0, index=dates, columns=panel.symbols)
    tw.loc[dates[0], "S000"] = 0.5
    res = run_backtest(panel, tw, ZERO)
    t_fill = res.trades["date"].iloc[0]
    assert t_fill == panel.sessions[panel.sessions.get_loc(dates[0]) + 1]
    assert np.isclose(res.trades["price"].iloc[0], panel.adj_open.loc[t_fill, "S000"])


def test_dollar_neutral_and_caps():
    s = pd.Series(np.arange(50, dtype=float), index=[f"S{i}" for i in range(50)])
    w = scores_to_weights(s, PortfolioConfig(method="rank_ls", gross=1.0, max_pos=0.05, neutral="dollar"))
    assert np.isclose(w[w > 0].sum(), 0.5) and np.isclose(w[w < 0].sum(), -0.5)
    assert (w.abs() <= 0.05 + 1e-9).all()


def test_beta_neutral_hedge():
    s = pd.Series(np.arange(40, dtype=float), index=[f"S{i}" for i in range(40)])
    betas = pd.Series(1.0, index=s.index)
    w = scores_to_weights(s, PortfolioConfig(method="long_only", gross=1.0, max_pos=0.5, neutral="beta", n_deciles=4), betas)
    assert np.isclose(w["SPY"], -1.0)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 10_000))
def test_symbol_order_invariance(panel, seed):
    rs = np.random.default_rng(seed)
    dates = _rebal(panel, "ME")
    scores = pd.DataFrame(rs.normal(size=(len(dates), len(panel.symbols))), index=dates, columns=panel.symbols)
    tw = build_target_weights(scores, panel.tradable, PortfolioConfig(method="rank_ls", max_pos=0.1))
    a = run_backtest(panel, tw, FULL).equity.iloc[-1]
    perm = list(rs.permutation(tw.columns))
    b = run_backtest(panel, tw[perm], FULL).equity.iloc[-1]
    assert np.isclose(a, b, rtol=1e-10)


def test_delisting_liquidation():
    from tests.conftest import make_synthetic_panel

    p = make_synthetic_panel(n_sym=5, start="2020-01-01", end="2020-12-31")
    cut = pd.Timestamp("2020-06-15")
    for k in ("adj_close", "adj_open", "close", "open", "high", "low", "adj_high", "adj_low"):
        p.frames[k].loc[cut:, "S000"] = np.nan
    dates = _rebal(p, "ME")
    tw = pd.DataFrame(0.0, index=dates, columns=p.symbols)
    tw["S000"] = 0.3
    res = run_backtest(p, tw, ZERO)
    d = res.trades[res.trades["reason"] == "delisting"]
    assert len(d) == 1 and d["date"].iloc[0] == cut
    assert (res.positions.loc[cut:, "S000"].abs() < 1e-12).all()

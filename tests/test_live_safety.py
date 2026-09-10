import pandas as pd
import pytest

from quantlab.config import LiveConfig
from quantlab.live.broker import DryRunBroker, Order
from quantlab.live.reconcile import diff_positions
from quantlab.live.safety import SafetyRails, SafetyViolation, kill_switch_present


def _cfg(tmp_path, **kw):
    return LiveConfig(strategy_ref="x", strategy_config="configs/strategies/meanrev_5d.yaml", kill_file=tmp_path / "KILL",
                      state_db=tmp_path / "live.sqlite", report_dir=tmp_path / "rep", **kw)


def _targets(n=5, w=0.02, price=100.0, equity=100_000):
    idx = [f"S{i}" for i in range(n)]
    df = pd.DataFrame({"target_w": [w] * n, "ref_price": [price] * n}, index=idx)
    df["target_notional"] = df["target_w"] * equity
    df["target_shares"] = (df["target_notional"] / price).round()
    return df


def test_diff_positions_and_min_notional():
    t = _targets()
    cur = pd.DataFrame({"qty": [20.0, 0.0], "market_value": [2000.0, 0.0], "avg_entry": [100.0, 0.0]}, index=["S0", "ZZZ"])
    orders = diff_positions(t, cur, min_trade_notional=200)
    syms = {o.symbol: o for o in orders}
    assert "S0" not in syms  # already at target (20 shares)
    assert syms["S1"].qty == 20 and syms["S1"].side == "buy"
    # tiny rebalance below min notional is skipped, but a full close is not
    t2 = t.copy(); t2.loc["S0", "target_shares"] = 21
    assert "S0" not in {o.symbol for o in diff_positions(t2, cur, 200)}
    t3 = t.drop("S0")
    cur2 = pd.DataFrame({"qty": [1.0], "market_value": [100.0], "avg_entry": [100.0]}, index=["S0"])
    assert any(o.symbol == "S0" and o.side == "sell" for o in diff_positions(t3, cur2, 200))


def test_safety_rails_each_rule(tmp_path):
    acct = {"equity": 100_000.0, "status": "ACTIVE"}
    uni = {f"S{i}" for i in range(10)}
    t = _targets()
    orders = [Order(s, 20, "buy", 2000.0) for s in t.index]
    SafetyRails(_cfg(tmp_path)).check(orders, t, acct, uni, 0)  # baseline passes
    with pytest.raises(SafetyViolation, match="gross"):
        SafetyRails(_cfg(tmp_path, max_gross_notional=5000)).check(orders, t, acct, uni, 0)
    with pytest.raises(SafetyViolation, match="max position"):
        SafetyRails(_cfg(tmp_path, max_position_pct=0.01)).check(orders, t, acct, uni, 0)
    with pytest.raises(SafetyViolation, match="turnover"):
        SafetyRails(_cfg(tmp_path, max_turnover_pct=0.05)).check(orders, t, acct, uni, 0)
    with pytest.raises(SafetyViolation, match="orders >"):
        SafetyRails(_cfg(tmp_path, max_orders=2)).check(orders, t, acct, uni, 0)
    with pytest.raises(SafetyViolation, match="per-order"):
        SafetyRails(_cfg(tmp_path, max_order_notional=1000)).check(orders, t, acct, uni, 0)
    with pytest.raises(SafetyViolation, match="outside universe"):
        SafetyRails(_cfg(tmp_path)).check(orders, t, acct, {"S0"}, 0)
    with pytest.raises(SafetyViolation, match="stale"):
        SafetyRails(_cfg(tmp_path)).check(orders, t, acct, uni, 5)
    with pytest.raises(SafetyViolation, match="status"):
        SafetyRails(_cfg(tmp_path)).check(orders, t, {"equity": 1e5, "status": "SUSPENDED"}, uni, 0)


def test_kill_switch(tmp_path):
    c = _cfg(tmp_path)
    assert not kill_switch_present(c)
    (tmp_path / "KILL").touch()
    assert kill_switch_present(c)


def test_dry_run_broker_records():
    b = DryRunBroker()
    out = b.submit([Order("AAA", 5, "buy", 500.0)])
    assert out[0]["status"] == "dry_run" and len(b.submitted) == 1

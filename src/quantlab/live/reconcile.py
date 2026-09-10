"""Diff target vs current positions -> orders; post-trade reconciliation report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.live.broker import Order


def diff_positions(targets: pd.DataFrame, current: pd.DataFrame, min_trade_notional: float) -> list[Order]:
    cur = current["qty"] if len(current) else pd.Series(dtype=float)
    syms = targets.index.union(cur.index)
    tgt = targets["target_shares"].reindex(syms).fillna(0.0)
    have = cur.reindex(syms).fillna(0.0)
    px = targets["ref_price"].reindex(syms)
    if len(current):
        px = px.fillna(current["market_value"].reindex(syms) / cur.reindex(syms).replace(0, np.nan))
    d = (tgt - have).round()
    orders = []
    for s in syms:
        q = float(d[s])
        if q == 0 or not np.isfinite(px[s]):
            continue
        notional = abs(q) * float(px[s])
        if notional < min_trade_notional and tgt[s] != 0:  # always allow full closes
            continue
        orders.append(Order(symbol=s, qty=q, side="buy" if q > 0 else "sell", notional_est=notional))
    return orders


def post_trade_report(targets: pd.DataFrame, fills: pd.DataFrame, positions: pd.DataFrame, prior_close: pd.Series) -> str:
    lines = ["# Reconciliation", ""]
    if len(fills):
        f = fills[fills["filled_qty"] > 0].copy()
        f["prior_close"] = f["symbol"].map(prior_close)
        f["slippage_bps"] = np.where(f["side"].str.contains("buy"), 1, -1) * (f["filled_avg_price"] / f["prior_close"] - 1) * 1e4
        lines += [f"fills: {len(f)} of {len(fills)} orders", f"mean slippage vs prior close: {f['slippage_bps'].mean():.1f} bps",
                  f"worst: {f['slippage_bps'].max():.1f} bps", ""]
        unfilled = fills[fills["filled_qty"] == 0]
        if len(unfilled):
            lines += [f"UNFILLED: {list(unfilled['symbol'])}", ""]
    tgt = targets["target_shares"]
    have = positions["qty"].reindex(tgt.index).fillna(0.0) if len(positions) else pd.Series(0.0, index=tgt.index)
    dev = (have - tgt)
    lines += [f"positions matching target: {(dev.abs() < 1).sum()} / {len(tgt)}"]
    bad = dev[dev.abs() >= 1]
    if len(bad):
        lines += ["deviations (shares):", bad.to_string()]
    extra = [s for s in (positions.index if len(positions) else []) if s not in tgt.index]
    if extra:
        lines += [f"positions outside target: {extra}"]
    return "\n".join(lines)

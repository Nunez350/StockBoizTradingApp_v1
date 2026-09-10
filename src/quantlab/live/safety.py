"""Safety rails: every check raises before any order is submitted."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.config import LiveConfig
from quantlab.live.broker import Order


class SafetyViolation(RuntimeError):
    pass


def kill_switch_present(cfg: LiveConfig) -> bool:
    return Path(cfg.kill_file).exists()


class SafetyRails:
    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg

    def check(self, orders: list[Order], targets: pd.DataFrame, account: dict, universe: set[str],
              price_age_sessions: int) -> None:
        c = self.cfg
        eq = account["equity"]
        gross = float(targets["target_notional"].abs().sum())
        if gross > c.max_gross_notional:
            raise SafetyViolation(f"gross target notional {gross:,.0f} > max {c.max_gross_notional:,.0f}")
        mx = float((targets["target_notional"].abs() / eq).max()) if len(targets) else 0.0
        if mx > c.max_position_pct:
            raise SafetyViolation(f"max position {mx:.1%} of equity > {c.max_position_pct:.1%}")
        turnover = sum(o.notional_est for o in orders) / eq if eq > 0 else 0.0
        if turnover > c.max_turnover_pct:
            raise SafetyViolation(f"turnover {turnover:.1%} > max {c.max_turnover_pct:.1%}")
        if len(orders) > c.max_orders:
            raise SafetyViolation(f"{len(orders)} orders > max {c.max_orders}")
        big = [o for o in orders if o.notional_est > c.max_order_notional]
        if big:
            raise SafetyViolation(f"{len(big)} orders exceed per-order cap {c.max_order_notional:,.0f}: {[o.symbol for o in big[:5]]}")
        outside = [o.symbol for o in orders if o.symbol not in universe and o.qty > 0]
        if outside:
            raise SafetyViolation(f"buy orders outside universe: {outside[:5]}")
        if price_age_sessions > c.max_price_staleness_sessions:
            raise SafetyViolation(f"prices are {price_age_sessions} sessions stale > {c.max_price_staleness_sessions}")
        if account.get("status") not in ("ACTIVE", "DRY", "AccountStatus.ACTIVE"):
            raise SafetyViolation(f"account status {account.get('status')}")

"""Transaction-cost model: commission + half spread + square-root market impact + borrow.

impact_bps = impact_coef * sigma_daily * sqrt(|notional| / ADV$) * 1e4
where sigma_daily and ADV$ are measured with data available at the decision date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.config import CostConfig


@dataclass
class CostModel:
    cfg: CostConfig

    def trade_cost(self, notional: np.ndarray, raw_price: np.ndarray, sigma_daily: np.ndarray, adv_usd: np.ndarray) -> np.ndarray:
        """Per-symbol dollar cost for a vector of signed trade notionals."""
        c = self.cfg
        absn = np.abs(notional)
        shares = np.where(raw_price > 0, absn / np.where(raw_price > 0, raw_price, 1.0), 0.0)
        commission = np.where(absn > 0, np.maximum(c.min_commission, c.commission_per_share * shares), 0.0)
        spread = absn * c.half_spread_bps / 1e4
        sig = np.nan_to_num(sigma_daily, nan=0.02)
        adv = np.where(np.isfinite(adv_usd) & (adv_usd > 0), adv_usd, 1e6)
        impact = absn * c.impact_coef * sig * np.sqrt(absn / adv)
        return commission + spread + impact

    def borrow_cost_daily(self, short_notional: float) -> float:
        return abs(short_notional) * self.cfg.borrow_bps_annual / 1e4 / 252.0

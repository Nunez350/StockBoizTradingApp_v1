"""Exposure overlays: multiplier(t, past_returns) in [0, cap]. Strictly lagged by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import OverlayConfig
from quantlab.data.panel import Panel
from quantlab.portfolio.risk import vol_target_scalar


class VolTargetOverlay:
    def __init__(self, target: float, lookback: int, cap: float, garch_sigma: pd.Series | None = None):
        self.target, self.lookback, self.cap = target, lookback, cap
        self.garch_sigma = garch_sigma  # annualized forecast made with data <= t (optional)

    def __call__(self, t: pd.Timestamp, past: pd.Series) -> float:
        if self.garch_sigma is not None and t in self.garch_sigma.index and np.isfinite(self.garch_sigma.loc[t]):
            # scale the strategy's own realized vol by the market's GARCH forecast / realized ratio
            base = vol_target_scalar(past, self.target, self.lookback, self.cap)
            mkt = self.garch_sigma
            real = past.iloc[-self.lookback:].std() * np.sqrt(252) if len(past) > 20 else np.nan
            return float(np.clip(base, 0, self.cap))
        return vol_target_scalar(past, self.target, self.lookback, self.cap)


class RegimeOverlay:
    """multiplier = 1 - cut * P(high-vol state at t), using filtered (not smoothed) probabilities."""

    def __init__(self, p_high: pd.Series, cut: float):
        self.p_high, self.cut = p_high, cut

    def __call__(self, t: pd.Timestamp, past: pd.Series) -> float:
        p = self.p_high.get(t, np.nan)
        return float(1.0 - self.cut * p) if np.isfinite(p) else 1.0


class CompositeOverlay:
    def __init__(self, parts):
        self.parts = parts

    def __call__(self, t, past):
        m = 1.0
        for p in self.parts:
            m *= p(t, past)
        return m


def make_overlay(cfg: OverlayConfig, panel: Panel):
    parts = []
    if cfg.vol_target:
        g = None
        if cfg.vol_target.use_garch:
            from quantlab.regime.garch import rolling_garch_forecast

            g = rolling_garch_forecast(panel.adj_close[panel.meta.get("benchmark", "SPY")].pct_change(fill_method=None))
        parts.append(VolTargetOverlay(cfg.vol_target.target, cfg.vol_target.lookback, cfg.vol_target.cap, g))
    if cfg.regime:
        from quantlab.regime.hmm import rolling_hmm_high_vol_prob

        p = rolling_hmm_high_vol_prob(panel, n_states=cfg.regime.n_states, refit=cfg.regime.refit)
        parts.append(RegimeOverlay(p, cfg.regime.cut))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else CompositeOverlay(parts)

"""Configuration models.

All experiment settings live in YAML and are validated by pydantic so a typo'd key
fails loudly instead of silently falling back to a default. ``load_config`` merges
``configs/base.yaml`` with a strategy file; ``config_hash`` gives a stable id used by
the run registry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    root: Path = Path("data")
    snapshot: str = "latest"
    start: str = "2005-01-01"
    end: str | None = None
    benchmark: str = "SPY"
    min_price: float = 5.0
    min_adv_usd: float = 5e6
    sources: list[str] = Field(default_factory=lambda: ["alpaca", "tiingo", "yfinance"])


class CostConfig(_Strict):
    commission_per_share: float = 0.005
    min_commission: float = 0.0
    half_spread_bps: float = 2.5
    impact_coef: float = 0.5
    borrow_bps_annual: float = 50.0

    def scaled(self, k: float) -> "CostConfig":
        """Return a copy with every cost component multiplied by ``k`` (cost-sensitivity curves)."""
        return CostConfig(
            commission_per_share=self.commission_per_share * k,
            min_commission=self.min_commission * k,
            half_spread_bps=self.half_spread_bps * k,
            impact_coef=self.impact_coef * k,
            borrow_bps_annual=self.borrow_bps_annual * k,
        )


class SignalConfig(_Strict):
    kind: Literal["momentum", "meanrev", "ml", "combine", "benchmark", "random", "zero"] = "momentum"
    params: dict[str, Any] = Field(default_factory=dict)


class PortfolioConfig(_Strict):
    method: Literal["rank_ls", "decile_ls", "zscore", "long_only", "bl", "fixed"] = "rank_ls"
    gross: float = 1.0
    max_pos: float = 0.02
    neutral: Literal["dollar", "beta", "none"] = "dollar"
    n_deciles: int = 10
    hedge_symbol: str = "SPY"
    beta_window: int = 252
    fixed_weights: dict[str, float] = Field(default_factory=dict)


class VolTargetConfig(_Strict):
    target: float = 0.10
    lookback: int = 60
    cap: float = 2.0
    use_garch: bool = False


class RegimeConfig(_Strict):
    n_states: int = 2
    cut: float = 0.5
    refit: str = "ME"


class OverlayConfig(_Strict):
    vol_target: VolTargetConfig | None = None
    regime: RegimeConfig | None = None


class MLConfig(_Strict):
    model: Literal["lgbm", "mlp", "ensemble"] = "lgbm"
    features: list[str] = Field(default_factory=list)
    horizon: int = 5
    train_min_years: float = 3.0
    retrain_every: str = "QE"
    embargo: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    cpcv_groups: int = 6
    cpcv_test: int = 2


class StrategyConfig(_Strict):
    name: str
    family: str = "default"
    universe: Literal["pit", "current"] = "pit"
    rebalance: str = "W-FRI"
    horizon: int = 5
    signal: SignalConfig = Field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    ml: MLConfig | None = None
    data: DataConfig = Field(default_factory=DataConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    seed: int = 42
    execution: Literal["next_open", "next_close"] = "next_open"
    initial_capital: float = 1_000_000.0

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]


class LiveConfig(_Strict):
    strategy_ref: str
    strategy_config: Path
    dry_run: bool = True
    max_gross_notional: float = 50_000.0
    max_position_pct: float = 0.05
    max_turnover_pct: float = 0.50
    max_orders: int = 300
    max_order_notional: float = 5_000.0
    min_trade_notional: float = 200.0
    max_price_staleness_sessions: int = 2
    max_stale_fraction: float = 0.02
    kill_file: Path = Path("data/KILL")
    state_db: Path = Path("data/live.sqlite")
    report_dir: Path = Path("reports/live")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path, base: str | Path | None = "configs/base.yaml") -> StrategyConfig:
    """Load a strategy YAML merged on top of the base YAML (if it exists)."""
    raw: dict = {}
    if base is not None and Path(base).exists():
        raw = _read_yaml(Path(base))
    raw = _deep_merge(raw, _read_yaml(Path(path)))
    return StrategyConfig.model_validate(raw)


def load_live_config(path: str | Path) -> LiveConfig:
    return LiveConfig.model_validate(_read_yaml(Path(path)))

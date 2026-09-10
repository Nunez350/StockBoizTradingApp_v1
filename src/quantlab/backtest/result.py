"""Standardized backtest output consumed by stats and reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series  # daily net returns of NAV
    gross_returns: pd.Series  # before transaction/borrow costs
    equity: pd.Series  # NAV
    positions: pd.DataFrame  # daily weights at close (sessions x symbols)
    trades: pd.DataFrame  # date, symbol, notional, shares, price, cost, reason
    turnover: pd.Series  # per rebalance: sum|trade notional| / NAV
    exposure: pd.DataFrame  # long, short, net, gross per day
    costs: pd.Series  # daily dollar costs
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ summaries
    def summary(self, periods: int = 252) -> dict:
        r = self.returns.dropna()
        n = len(r)
        if n < 2:
            return {"n_obs": n}
        mu, sd = r.mean(), r.std(ddof=1)
        sharpe = mu / sd * np.sqrt(periods) if sd > 0 else np.nan
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        yrs = n / periods
        cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
        g = self.gross_returns.dropna()
        return {
            "n_obs": n,
            "start": str(r.index[0].date()),
            "end": str(r.index[-1].date()),
            "cagr": float(cagr),
            "ann_vol": float(sd * np.sqrt(periods)),
            "sharpe": float(sharpe),
            "sharpe_gross": float(g.mean() / g.std(ddof=1) * np.sqrt(periods)) if g.std() > 0 else np.nan,
            "max_dd": float(dd.min()),
            "skew": float(r.skew()),
            "kurt": float(r.kurt()),
            "hit_rate": float((r > 0).mean()),
            "avg_turnover": float(self.turnover.mean()) if len(self.turnover) else 0.0,
            "avg_gross_exposure": float(self.exposure["gross"].mean()),
            "avg_net_exposure": float(self.exposure["net"].mean()),
            "total_costs_pct_nav": float(self.costs.sum() / self.equity.iloc[0]),
            "n_trades": int(len(self.trades)),
        }

    # ------------------------------------------------------------ persistence
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"net": self.returns, "gross": self.gross_returns, "equity": self.equity, "costs": self.costs}).to_parquet(d / "returns.parquet")
        self.positions.to_parquet(d / "positions.parquet")
        self.trades.to_parquet(d / "trades.parquet")
        self.turnover.to_frame("turnover").to_parquet(d / "turnover.parquet")
        self.exposure.to_parquet(d / "exposure.parquet")
        (d / "meta.json").write_text(json.dumps(self.meta, default=str, indent=2))

    @classmethod
    def load(cls, d: Path) -> "BacktestResult":
        r = pd.read_parquet(d / "returns.parquet")
        return cls(
            returns=r["net"], gross_returns=r["gross"], equity=r["equity"], costs=r["costs"],
            positions=pd.read_parquet(d / "positions.parquet"),
            trades=pd.read_parquet(d / "trades.parquet"),
            turnover=pd.read_parquet(d / "turnover.parquet")["turnover"],
            exposure=pd.read_parquet(d / "exposure.parquet"),
            meta=json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {},
        )

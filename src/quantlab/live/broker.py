"""Alpaca broker adapter (paper by default). Thin wrapper so the daily job can be unit-tested with a fake."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class Order:
    symbol: str
    qty: float
    side: str  # buy | sell
    notional_est: float


class AlpacaBroker:
    def __init__(self, paper: bool | None = None):
        from alpaca.trading.client import TradingClient

        key, sec = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
        self.paper = paper if paper is not None else os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self.client = TradingClient(key, sec, paper=self.paper)

    def account(self) -> dict:
        a = self.client.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash), "buying_power": float(a.buying_power),
                "status": str(a.status), "paper": self.paper}

    def positions(self) -> pd.DataFrame:
        rows = [{"symbol": p.symbol.replace("-", "."), "qty": float(p.qty), "market_value": float(p.market_value),
                 "avg_entry": float(p.avg_entry_price)} for p in self.client.get_all_positions()]
        return pd.DataFrame(rows, columns=["symbol", "qty", "market_value", "avg_entry"]).set_index("symbol")

    def clock(self) -> dict:
        c = self.client.get_clock()
        return {"is_open": c.is_open, "next_open": c.next_open, "next_close": c.next_close, "timestamp": c.timestamp}

    def submit(self, orders: list[Order], tif: str = "opg") -> list[dict]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        out = []
        for o in orders:
            req = MarketOrderRequest(symbol=o.symbol, qty=abs(o.qty), side=OrderSide.BUY if o.side == "buy" else OrderSide.SELL,
                                     time_in_force=TimeInForce(tif))
            r = self.client.submit_order(req)
            out.append({"symbol": o.symbol, "qty": o.qty, "side": o.side, "order_id": str(r.id), "status": str(r.status)})
        return out

    def orders(self, after: pd.Timestamp | None = None) -> pd.DataFrame:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=after.to_pydatetime() if after is not None else None, limit=500)
        rows = []
        for o in self.client.get_orders(req):
            rows.append({"order_id": str(o.id), "symbol": o.symbol, "side": str(o.side), "qty": float(o.qty or 0),
                         "filled_qty": float(o.filled_qty or 0), "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                         "status": str(o.status), "submitted_at": o.submitted_at, "filled_at": o.filled_at})
        return pd.DataFrame(rows)

    def cancel_all(self) -> int:
        return len(self.client.cancel_orders())


class DryRunBroker:
    """Stands in for Alpaca when no keys exist or dry_run is on: reports a fixed equity and no positions."""

    def __init__(self, equity: float = 100_000.0, positions: pd.DataFrame | None = None):
        self._equity = equity
        self._pos = positions if positions is not None else pd.DataFrame(columns=["qty", "market_value", "avg_entry"])
        self.submitted: list[Order] = []

    def account(self) -> dict:
        return {"equity": self._equity, "cash": self._equity, "buying_power": self._equity, "status": "DRY", "paper": True}

    def positions(self) -> pd.DataFrame:
        return self._pos

    def clock(self) -> dict:
        return {"is_open": False}

    def submit(self, orders: list[Order], tif: str = "opg") -> list[dict]:
        self.submitted.extend(orders)
        return [{"symbol": o.symbol, "qty": o.qty, "side": o.side, "order_id": f"dry-{i}", "status": "dry_run"} for i, o in enumerate(orders)]

    def orders(self, after=None) -> pd.DataFrame:
        return pd.DataFrame()

    def cancel_all(self) -> int:
        return 0

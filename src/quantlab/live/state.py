"""SQLite journal of live orders, fills and daily snapshots (idempotency + audit)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (date TEXT, ts TEXT, symbol TEXT, qty REAL, side TEXT, order_id TEXT, status TEXT, dry_run INTEGER);
CREATE TABLE IF NOT EXISTS snapshots (date TEXT PRIMARY KEY, ts TEXT, equity REAL, n_positions INTEGER, gross REAL, payload TEXT);
CREATE TABLE IF NOT EXISTS pnl (date TEXT PRIMARY KEY, equity REAL, ret REAL);
"""


class LiveState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.executescript(_SCHEMA)

    def already_submitted(self, date: str) -> bool:
        with sqlite3.connect(self.path) as c:
            n = c.execute("SELECT COUNT(*) FROM submissions WHERE date = ? AND dry_run = 0", (date,)).fetchone()[0]
        return n > 0

    def record_submissions(self, date: str, rows: list[dict], dry_run: bool) -> None:
        ts = pd.Timestamp.utcnow().isoformat()
        with sqlite3.connect(self.path) as c:
            c.executemany("INSERT INTO submissions VALUES (?,?,?,?,?,?,?,?)",
                          [(date, ts, r["symbol"], r["qty"], r["side"], r.get("order_id"), r.get("status"), int(dry_run)) for r in rows])

    def record_snapshot(self, date: str, account: dict, positions: pd.DataFrame) -> None:
        gross = float(positions["market_value"].abs().sum()) if len(positions) else 0.0
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?)",
                      (date, pd.Timestamp.utcnow().isoformat(), account["equity"], int(len(positions)), gross, json.dumps(account, default=str)))
            prev = c.execute("SELECT equity FROM snapshots WHERE date < ? ORDER BY date DESC LIMIT 1", (date,)).fetchone()
            ret = account["equity"] / prev[0] - 1 if prev and prev[0] else None
            c.execute("INSERT OR REPLACE INTO pnl VALUES (?,?,?)", (date, account["equity"], ret))

    def pnl_series(self) -> pd.Series:
        with sqlite3.connect(self.path) as c:
            df = pd.read_sql("SELECT date, ret FROM pnl ORDER BY date", c)
        return pd.Series(df["ret"].to_numpy(), index=pd.to_datetime(df["date"]), name="live_ret").dropna()

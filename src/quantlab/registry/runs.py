"""Run registry: every backtest trial is logged so multiple-testing corrections are honest.

SQLite table ``trials``. ``kind``:
  trial       - a candidate strategy configuration (counts toward DSR)
  ml_hp       - a hyper-parameter configuration evaluated during ML model selection (counts)
  grid        - a parameter-stability grid cell (counts)
  cv_fold     - an out-of-sample fold of a trial (does not count)
  permutation - a null draw (does not count)
  sanity      - SPY hold / random / zero checks (does not count)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

COUNTED = ("trial", "ml_hp", "grid")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
  run_id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  name TEXT,
  family TEXT NOT NULL,
  kind TEXT NOT NULL,
  config_hash TEXT,
  config_json TEXT,
  snapshot_id TEXT,
  git_sha TEXT,
  seed INTEGER,
  n_obs INTEGER,
  sharpe REAL,
  sharpe_pp REAL,
  skew REAL,
  kurt REAL,
  metrics_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_family_kind ON trials(family, kind);
"""


class RunRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.db = self.root / "registry.sqlite"
        self.runs_dir = self.root / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db) as c:
            c.executescript(_SCHEMA)

    @staticmethod
    def new_run_id(name: str) -> str:
        return f"{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%S')}_{name}_{uuid.uuid4().hex[:6]}"

    def log_trial(self, run_id: str, name: str, family: str, kind: str, cfg_hash: str, cfg_json: str,
                  snapshot_id: str | None, git_sha: str, seed: int, returns: pd.Series, metrics: dict) -> None:
        r = returns.dropna()
        sd = r.std(ddof=1)
        sr_pp = float(r.mean() / sd) if sd > 0 else None
        row = (run_id, pd.Timestamp.utcnow().isoformat(), name, family, kind, cfg_hash, cfg_json, snapshot_id, git_sha,
               seed, int(len(r)), sr_pp * np.sqrt(252) if sr_pp is not None else None, sr_pp,
               float(r.skew()) if len(r) > 2 else None, float(r.kurt() + 3) if len(r) > 3 else None,
               json.dumps(metrics, default=str))
        with sqlite3.connect(self.db) as c:
            c.execute("INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        returns.to_frame("net").to_parquet(d / "returns.parquet")

    def query(self, family: str | None = None, kinds: tuple[str, ...] | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM trials", []
        conds = []
        if family:
            conds.append("family = ?")
            args.append(family)
        if kinds:
            conds.append(f"kind IN ({','.join('?' * len(kinds))})")
            args.extend(kinds)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        with sqlite3.connect(self.db) as c:
            return pd.read_sql(q + " ORDER BY ts", c, params=args)

    def count_trials(self, family: str, kinds: tuple[str, ...] = COUNTED) -> int:
        return int(self.query(family, kinds)["config_hash"].nunique())

    def trial_srs(self, family: str, kinds: tuple[str, ...] = COUNTED) -> list[float]:
        df = self.query(family, kinds)
        return [float(x) for x in df["sharpe_pp"].dropna()]

    def load_returns(self, run_ids: list[str]) -> pd.DataFrame:
        cols = {}
        for r in run_ids:
            p = self.runs_dir / r / "returns.parquet"
            if p.exists():
                cols[r] = pd.read_parquet(p)["net"]
        return pd.DataFrame(cols)

    def trial_stats(self, family: str) -> dict:
        """Inputs for DSR: number of counted trials, variance of their per-period SRs, and the
        effective number after clustering highly correlated trials."""
        from quantlab.stats.sharpe import effective_trials

        df = self.query(family, COUNTED)
        df = df.sort_values("ts").drop_duplicates("config_hash", keep="last")  # re-running a config is not a new trial
        n = len(df)
        srs = df["sharpe_pp"].dropna().to_numpy()
        var = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
        eff = effective_trials(self.load_returns(list(df["run_id"]))) if n > 1 else n
        return {"n_trials": n, "var_trials": var, "n_effective": eff}


def cli_registry(a) -> None:
    reg = RunRegistry(Path(a.root))
    if a.action == "count":
        print(json.dumps(reg.trial_stats(a.family or "default"), indent=2))
        return
    df = reg.query(a.family)
    if a.action == "show":
        row = df[df["run_id"] == a.run_id]
        if row.empty:
            raise SystemExit(f"no run {a.run_id}")
        rec = row.iloc[0].to_dict()
        rec["metrics"] = json.loads(rec.pop("metrics_json") or "{}")
        rec.pop("config_json", None)
        print(json.dumps(rec, indent=2, default=str))
        return
    if df.empty:
        print("registry empty")
        return
    cols = ["run_id", "family", "kind", "n_obs", "sharpe", "snapshot_id", "git_sha"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df[cols].to_string(index=False))
    print(f"\n{len(df)} rows; counted trials by family: {df[df['kind'].isin(COUNTED)].groupby('family').size().to_dict()}")

"""FastAPI application serving the StockBoiz v3 web UI.

Everything here is read-only over artefacts the CLI already produces:

- ``data/curated/<snapshot>``  prices, membership, tradability, sector, feature cache
- ``data/universe``            current constituents (names, sectors, CIKs)
- ``data/registry.sqlite``     every registered backtest trial
- ``reports/<run_id>``         tearsheets
- ``reports/live/<date>``      the daily job's targets, orders, safety verdicts

The old StockBoiz (v1/v2) backend exposed a symbol lookup over MySQL and promised buy/sell
suggestions it never produced. ``/api/stocks`` replaces the lookup with the point-in-time
universe; ``/api/suggestions`` serves the orders the paper-trading job writes.

Frames are loaded lazily and cached per field so a request for one symbol's history does
not deserialize the whole panel.
"""

from __future__ import annotations

import json
import math
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from quantlab.data.panel import snapshot_dir

ORDER_FILES = ("orders_submitted.json", "orders_dryrun.json")


def _clean(v: Any) -> Any:
    """JSON-safe scalar: NaN/inf -> None, numpy/pandas scalars -> python."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp,)):
        return str(v.date())
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


class Store:
    """Lazy, cached access to the data root, the reports directory, and the strategy configs."""

    def __init__(self, root: Path, reports: Path, live_config: Path | None, strategies: Path,
                 snapshot: str = "latest"):
        self.root = Path(root)
        self.reports = Path(reports)
        self.live_config = Path(live_config) if live_config else None
        self.strategies = Path(strategies)
        self.snapshot = snapshot
        self._frames: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ panel
    @cached_property
    def snap_dir(self) -> Path:
        return snapshot_dir(self.root, self.snapshot)

    @cached_property
    def meta(self) -> dict:
        p = self.snap_dir / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @cached_property
    def sector(self) -> dict[str, str]:
        p = self.snap_dir / "sector.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def frame(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            p = self.snap_dir / f"{name}.parquet"
            if not p.exists():
                raise HTTPException(404, f"panel field {name!r} not in snapshot")
            df = pd.read_parquet(p)
            if name in ("membership", "tradable"):
                df = df.astype(bool)
            self._frames[name] = df
        return self._frames[name]

    @cached_property
    def sessions(self) -> pd.DatetimeIndex:
        return self.frame("close").index

    @cached_property
    def symbols(self) -> list[str]:
        return list(self.frame("close").columns)

    # --------------------------------------------------------------- universe
    @cached_property
    def names(self) -> dict[str, dict]:
        """symbol -> {name, sector, sub_industry} from the current-constituents table (if built)."""
        p = self.root / "universe" / "current.parquet"
        out: dict[str, dict] = {}
        if p.exists():
            cur = pd.read_parquet(p)
            for r in cur.to_dict(orient="records"):
                out[str(r["symbol"])] = {"name": r.get("security"), "sector": r.get("sector"),
                                         "sub_industry": r.get("sub_industry")}
        return out

    def describe(self, sym: str) -> dict:
        info = self.names.get(sym, {})
        return {"symbol": sym, "name": info.get("name"),
                "sector": info.get("sector") or self.sector.get(sym, "Unknown")}

    # --------------------------------------------------------------- features
    def feature_values(self, sym: str) -> dict[str, float | None]:
        d = self.snap_dir / "features"
        out: dict[str, float | None] = {}
        if not d.exists():
            return out
        for p in sorted(d.glob("*.parquet")):
            try:
                col = pd.read_parquet(p, columns=[sym])[sym].dropna()
            except (KeyError, ValueError):
                continue
            out[p.stem] = _clean(col.iloc[-1]) if len(col) else None
        return out

    # --------------------------------------------------------------- registry
    @cached_property
    def registry(self):
        from quantlab.registry.runs import RunRegistry

        return RunRegistry(self.root)

    # ------------------------------------------------------------------- live
    @cached_property
    def live_dir(self) -> Path:
        if self.live_config and self.live_config.exists():
            from quantlab.config import load_live_config

            return Path(load_live_config(self.live_config).report_dir)
        return self.reports / "live"

    @cached_property
    def kill_file(self) -> Path:
        if self.live_config and self.live_config.exists():
            from quantlab.config import load_live_config

            return Path(load_live_config(self.live_config).kill_file)
        return self.root / "KILL"

    def live_dates(self) -> list[str]:
        if not self.live_dir.exists():
            return []
        artefacts = ("targets.csv", "SAFETY_ABORT.txt", *ORDER_FILES)
        return sorted(p.name for p in self.live_dir.iterdir()
                      if p.is_dir() and any((p / f).exists() for f in artefacts))

    def suggestions(self, date: str | None) -> dict:
        dates = self.live_dates()
        if not dates:
            raise HTTPException(404, "no live reports yet; run `quantlab live run --dry-run`")
        date = date or dates[-1]
        d = self.live_dir / date
        if not d.is_dir():
            raise HTTPException(404, f"no live report for {date}")
        orders, mode = [], None
        for f in ORDER_FILES:
            if (d / f).exists():
                orders = json.loads((d / f).read_text())
                mode = "submitted" if f == "orders_submitted.json" else "dry_run"
                break
        for o in orders:
            o.update(self.describe(o["symbol"]))
        targets = []
        if (d / "targets.csv").exists():
            t = pd.read_csv(d / "targets.csv", index_col=0)
            t.index.name = "symbol"
            t = t.reset_index()
            targets = _records(t)
            for r in targets:
                r.update(self.describe(r["symbol"]))
        abort = (d / "SAFETY_ABORT.txt").read_text().strip() if (d / "SAFETY_ABORT.txt").exists() else None
        recon = (d / "reconciliation.md").read_text() if (d / "reconciliation.md").exists() else None
        gross = float(sum(abs(o.get("notional_est", 0.0)) for o in orders))
        return {
            "date": date, "dates": dates, "mode": mode, "safety_abort": abort,
            "kill_switch": self.kill_file.exists(),
            "n_buy": sum(o["side"] == "buy" for o in orders), "n_sell": sum(o["side"] == "sell" for o in orders),
            "gross_notional": round(gross, 2), "orders": orders, "targets": targets, "reconciliation": recon,
        }


def create_app(root: str | Path = "data", reports: str | Path = "reports",
               live_config: str | Path | None = "configs/paper.yaml",
               strategies: str | Path = "configs/strategies", web_dist: str | Path | None = "web/dist",
               snapshot: str = "latest") -> FastAPI:
    store = Store(Path(root), Path(reports), Path(live_config) if live_config else None, Path(strategies), snapshot)
    app = FastAPI(title="StockBoiz v3 API", version="3.0.0",
                  description="Read-only API over the quantlab research and paper-trading pipeline.")
    app.state.store = store

    # ------------------------------------------------------------------ health
    @app.get("/api/health")
    def health() -> dict:
        out: dict[str, Any] = {"status": "ok", "kill_switch": store.kill_file.exists(), "live_dates": store.live_dates()[-5:]}
        try:
            sess = store.sessions
            out.update(snapshot_id=store.meta.get("snapshot_id"), snapshot_dir=str(store.snap_dir),
                       first_session=str(sess[0].date()), last_session=str(sess[-1].date()),
                       n_symbols=len(store.symbols), n_tradable=int(store.frame("tradable").iloc[-1].sum()))
        except (FileNotFoundError, HTTPException) as e:
            out.update(status="no_snapshot", detail=str(e))
        return out

    # ------------------------------------------------------------- suggestions
    @app.get("/api/suggestions")
    def suggestions(date: str | None = None) -> dict:
        return store.suggestions(date)

    @app.get("/api/suggestions/dates")
    def suggestion_dates() -> list[str]:
        return store.live_dates()

    # ------------------------------------------------------------------ stocks
    @app.get("/api/stocks")
    def search_stocks(q: str = Query("", max_length=64), limit: int = Query(25, ge=1, le=200)) -> list[dict]:
        q = q.strip().upper()
        mem = store.frame("membership").iloc[-1]
        trad = store.frame("tradable").iloc[-1]
        rows = []
        for sym in store.symbols:
            info = store.describe(sym)
            name = (info["name"] or "").upper()
            if q and not (sym.startswith(q) or q in name):
                continue
            rows.append({**info, "member": bool(mem.get(sym, False)), "tradable": bool(trad.get(sym, False))})
        rows.sort(key=lambda r: (not r["symbol"].startswith(q) if q else 0, not r["member"], r["symbol"]))
        return rows[:limit]

    @app.get("/api/stocks/{symbol}")
    def stock_detail(symbol: str, sessions: int = Query(252, ge=5, le=10000)) -> dict:
        sym = symbol.upper()
        if sym not in store.symbols:
            raise HTTPException(404, f"{sym} is not in the panel")
        close, adj, vol = store.frame("close")[sym], store.frame("adj_close")[sym], store.frame("volume")[sym]
        hist = pd.DataFrame({"close": close, "adj_close": adj, "volume": vol}).dropna(subset=["close"]).tail(sessions)
        hist.index.name = "date"
        mem, trad = store.frame("membership")[sym], store.frame("tradable")[sym]
        first_member = mem.index[mem.to_numpy()].min() if mem.any() else None
        out = store.describe(sym)
        out.update(
            member=bool(mem.iloc[-1]), tradable=bool(trad.iloc[-1]),
            first_member=_clean(first_member), last_close=_clean(close.dropna().iloc[-1]) if close.notna().any() else None,
            last_session=_clean(close.dropna().index[-1]) if close.notna().any() else None,
            history=_records(hist.reset_index()), features=store.feature_values(sym),
        )
        return out

    # -------------------------------------------------------------------- runs
    @app.get("/api/runs")
    def runs(family: str | None = None) -> list[dict]:
        df = store.registry.query(family)
        if df.empty:
            return []
        df = df.drop(columns=["config_json"]).sort_values("ts", ascending=False)
        rows = _records(df)
        for r in rows:
            r["metrics"] = json.loads(r.pop("metrics_json") or "{}")
            r["has_tearsheet"] = (store.reports / r["run_id"] / "tearsheet.html").exists()
        return rows

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        df = store.registry.query()
        row = df[df["run_id"] == run_id]
        if row.empty:
            raise HTTPException(404, f"no run {run_id}")
        r = {k: _clean(v) for k, v in row.iloc[0].to_dict().items()}
        r["metrics"] = json.loads(r.pop("metrics_json") or "{}")
        r["config"] = json.loads(r.pop("config_json") or "{}")
        r["has_tearsheet"] = (store.reports / run_id / "tearsheet.html").exists()
        return r

    @app.get("/api/runs/{run_id}/tearsheet")
    def tearsheet(run_id: str):
        p = store.reports / run_id / "tearsheet.html"
        if not p.exists():
            raise HTTPException(404, f"no tearsheet for {run_id}")
        return FileResponse(p, media_type="text/html")

    # -------------------------------------------------------------- strategies
    @app.get("/api/strategies")
    def strategies() -> list[dict]:
        from quantlab.config import load_config

        out = []
        base = store.strategies.parent / "base.yaml"
        for p in sorted(store.strategies.glob("*.yaml")):
            try:
                cfg = load_config(p, base if base.exists() else None)
            except (ValueError, OSError, yaml.YAMLError) as e:  # a bad YAML should not take the listing down
                out.append({"file": str(p), "error": str(e)})
                continue
            out.append({"file": str(p), "name": cfg.name, "family": cfg.family, "signal": cfg.signal.kind,
                        "params": cfg.signal.params, "rebalance": cfg.rebalance, "horizon": cfg.horizon,
                        "portfolio": cfg.portfolio.method, "gross": cfg.portfolio.gross,
                        "overlay": cfg.overlay.model_dump(mode="json"), "meta": bool(cfg.meta and cfg.meta.enabled)})
        return out

    # ---------------------------------------------------------- static assets
    if store.reports.exists():
        app.mount("/reports", StaticFiles(directory=store.reports), name="reports")

    dist = Path(web_dist) if web_dist else None
    if dist and (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            """Serve the built React app; unknown paths fall back to index.html for client routing."""
            if path.startswith("api/"):
                raise HTTPException(404, "no such endpoint")
            f = dist / path
            if path and f.is_file():
                return FileResponse(f)
            return FileResponse(dist / "index.html")

    return app


def cli_api(a) -> None:
    import uvicorn

    app = create_app(root=a.root, reports=a.reports, live_config=a.live_config, strategies=a.strategies,
                     web_dist=a.web_dist, snapshot=a.snapshot)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")

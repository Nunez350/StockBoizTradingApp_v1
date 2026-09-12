"""The web API is a read-only view over files the CLI writes; build a tiny site on disk and read it back."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.api.app import create_app
from quantlab.data.universe import Universe
from quantlab.registry.runs import RunRegistry
from tests.conftest import make_synthetic_panel

RUN_ID = "20210101T000000_test_abc123"
STRATEGIES = Path(__file__).resolve().parent.parent / "configs" / "strategies"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    reports = tmp_path_factory.mktemp("reports")
    panel = make_synthetic_panel(n_sym=8)
    snap = root / "curated" / "20210101T000000"
    sid = panel.save(snap)
    (root / "curated" / "latest").symlink_to(snap)
    (snap / "features").mkdir()
    panel.adj_close.pct_change(5).to_parquet(snap / "features" / "rev_5d.parquet")

    cur = pd.DataFrame({"symbol": ["S000", "S001"], "security": ["Alpha Corp", "Beta Inc"],
                        "sector": ["Tech", "Energy"], "sub_industry": ["x", "y"],
                        "date_added": ["2000-01-01", "2000-01-01"], "cik": [1, 2]})
    Universe(cur, pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")}), panel.membership).save(root)

    reg = RunRegistry(root)
    rets = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 300), index=panel.sessions[:300])
    reg.log_trial(RUN_ID, "test", "fam", "trial", "h1", "{}", sid, "sha", 0, rets, {"sharpe": 0.5})
    (reports / RUN_ID).mkdir()
    (reports / RUN_ID / "tearsheet.html").write_text("<html>tearsheet</html>")

    day1 = reports / "live" / "2021-12-30"
    day1.mkdir(parents=True)
    (day1 / "orders_dryrun.json").write_text(json.dumps([
        {"symbol": "S000", "qty": 5.0, "side": "buy", "notional_est": 500.0},
        {"symbol": "S001", "qty": -2.0, "side": "sell", "notional_est": 200.0},
    ]))
    pd.DataFrame({"target_w": [0.01, -0.004], "ref_price": [100.0, 100.0], "target_notional": [500.0, -200.0],
                  "target_shares": [5.0, -2.0]}, index=["S000", "S001"]).to_csv(day1 / "targets.csv")
    day2 = reports / "live" / "2021-12-31"
    day2.mkdir()
    (day2 / "SAFETY_ABORT.txt").write_text("turnover 90.0% > max 60.0%")

    dist = tmp_path_factory.mktemp("dist")
    (dist / "assets").mkdir()
    (dist / "index.html").write_text("<html>spa</html>")
    (dist / "robots.txt").write_text("User-agent: *")

    app = create_app(root=root, reports=reports, live_config=None, strategies=STRATEGIES, web_dist=dist)
    return TestClient(app)


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["n_symbols"] == 9 and h["last_session"] == "2021-12-31"
    assert h["kill_switch"] is False and h["live_dates"] == ["2021-12-30", "2021-12-31"]


def test_suggestions_latest_is_safety_abort(client):
    s = client.get("/api/suggestions").json()
    assert s["date"] == "2021-12-31" and s["safety_abort"].startswith("turnover")
    assert s["orders"] == [] and s["mode"] is None


def test_suggestions_by_date_enriched(client):
    s = client.get("/api/suggestions", params={"date": "2021-12-30"}).json()
    assert s["mode"] == "dry_run" and s["n_buy"] == 1 and s["n_sell"] == 1 and s["gross_notional"] == 700.0
    buy = next(o for o in s["orders"] if o["side"] == "buy")
    assert buy["symbol"] == "S000" and buy["name"] == "Alpha Corp" and buy["sector"] == "Tech"
    assert {t["symbol"] for t in s["targets"]} == {"S000", "S001"}
    assert client.get("/api/suggestions", params={"date": "1999-01-01"}).status_code == 404
    assert client.get("/api/suggestions/dates").json() == ["2021-12-30", "2021-12-31"]


def test_stock_search(client):
    rows = client.get("/api/stocks", params={"q": "alp"}).json()
    assert rows and rows[0]["symbol"] == "S000" and rows[0]["name"] == "Alpha Corp"
    rows = client.get("/api/stocks", params={"q": "S00", "limit": 3}).json()
    assert [r["symbol"] for r in rows] == ["S000", "S001", "S002"]
    assert all(r["member"] for r in rows)
    everything = client.get("/api/stocks").json()
    assert len(everything) == 9 and everything[-1]["symbol"] == "SPY"  # the benchmark is not a member, sorts last


def test_stock_detail(client):
    d = client.get("/api/stocks/s000", params={"sessions": 50}).json()
    assert d["symbol"] == "S000" and d["name"] == "Alpha Corp" and d["member"] is True
    assert len(d["history"]) == 50 and set(d["history"][0]) == {"date", "close", "adj_close", "volume"}
    assert d["last_session"] == "2021-12-31" and d["last_close"] == pytest.approx(d["history"][-1]["close"])
    assert "rev_5d" in d["features"] and d["features"]["rev_5d"] is not None
    assert d["first_member"] == "2019-01-02"
    assert client.get("/api/stocks/NOPE").status_code == 404


def test_runs_and_tearsheet(client):
    rows = client.get("/api/runs").json()
    assert len(rows) == 1 and rows[0]["run_id"] == RUN_ID and rows[0]["has_tearsheet"] is True
    assert rows[0]["metrics"] == {"sharpe": 0.5} and "config_json" not in rows[0]
    assert client.get("/api/runs", params={"family": "other"}).json() == []
    d = client.get(f"/api/runs/{RUN_ID}").json()
    assert d["family"] == "fam" and d["config"] == {} and d["n_obs"] == 300
    assert client.get("/api/runs/missing").status_code == 404
    assert client.get(f"/api/runs/{RUN_ID}/tearsheet").text == "<html>tearsheet</html>"
    assert client.get(f"/reports/{RUN_ID}/tearsheet.html").status_code == 200


def test_strategies_listing(client):
    rows = client.get("/api/strategies").json()
    names = {r.get("name") for r in rows}
    assert "meanrev_5d" in names and not any("error" in r for r in rows)
    mr = next(r for r in rows if r.get("name") == "meanrev_5d")
    assert mr["signal"] == "meanrev" and mr["portfolio"] == "rank_ls"


def test_spa_fallback(client):
    assert client.get("/stocks/S000").text == "<html>spa</html>"
    assert client.get("/robots.txt").text == "User-agent: *"
    assert client.get("/api/nope").status_code == 404

"""Daily paper-trading job.

08:15 ET  ``quantlab live run``        update data -> targets -> reconcile -> safety -> submit (or dry-run)
10:00 ET  ``quantlab live post-trade`` fills, slippage, reconciliation, BOCPD alert on live P&L

Safety: kill-switch file, idempotency (one live submission per session), all rails checked
before any order, dry-run by default, every step journaled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from quantlab.calendar import is_session, prev_session, sessions
from quantlab.config import load_config, load_live_config
from quantlab.data.adjust import apply_adjustment
from quantlab.data.cache import ParquetCache
from quantlab.data.panel import build_panel, load_panel
from quantlab.data.quality import run_checks
from quantlab.data.sources import make_sources
from quantlab.data.universe import Universe
from quantlab.features.registry import build_features
from quantlab.live.broker import AlpacaBroker, DryRunBroker
from quantlab.live.reconcile import diff_positions, post_trade_report
from quantlab.live.safety import SafetyRails, SafetyViolation, kill_switch_present
from quantlab.live.state import LiveState
from quantlab.live.targets import compute_targets


def _log(msg: str) -> None:
    print(f"[{pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S')} ET] {msg}")


def _broker(live_cfg, force_live: bool):
    dry = live_cfg.dry_run and not force_live
    have_keys = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))
    if have_keys:
        return AlpacaBroker(), dry
    if not dry:
        raise SystemExit("live submission requested but no Alpaca keys in the environment")
    _log("no Alpaca keys: using DryRunBroker with synthetic $100k equity")
    return DryRunBroker(), True


def refresh_data(cfg, root: Path, log=_log) -> tuple:
    """Incremental update of the raw cache, quality check, and an in-memory panel through the last close."""
    u = Universe.load(root)
    syms = sorted(set(u.symbols) | {cfg.data.benchmark})
    cache = ParquetCache(root, cfg.data.sources)
    summ = cache.update(syms, make_sources(cfg.data.sources), cfg.data.start, workers=4, log=lambda m: None)
    log(f"data update: {summ}")
    frames = {}
    for s in syms:
        df = cache.get(s)
        if df is not None and len(df) > 30:
            frames[s] = apply_adjustment(df)
    end = max(df.index.max() for df in frames.values())
    sess = sessions(cfg.data.start, end)
    rep = run_checks(frames, sess, u.membership)
    last = prev_session(pd.Timestamp.today().normalize()) if not is_session(pd.Timestamp.today().normalize()) else pd.Timestamp.today().normalize()
    stale = [s for s, df in frames.items() if u.membership.iloc[-1].get(s, False) and (df.index.max() < prev_session(last, 1))]
    frac = len(stale) / max(1, int(u.membership.iloc[-1].sum()))
    panel = build_panel(frames, u.membership, u.sector_map(), cfg.data.start, str(end.date()), cfg.data.min_price,
                        cfg.data.min_adv_usd, rep.critical_symbols, cfg.data.benchmark)
    snap = load_panel(root, cfg.data.snapshot)
    panel.meta["snapshot_id"] = f"live-{snap.meta.get('snapshot_id')}"
    age = len(sessions(end, last)) - 1 if last > end else 0
    return panel, frac, age, rep


def run_daily(a) -> None:
    live_cfg = load_live_config(a.config)
    cfg = load_config(live_cfg.strategy_config)
    root = Path(cfg.data.root)
    today = pd.Timestamp.today().normalize()
    date = str(today.date())
    state = LiveState(live_cfg.state_db)
    rep_dir = Path(live_cfg.report_dir) / date
    rep_dir.mkdir(parents=True, exist_ok=True)

    if kill_switch_present(live_cfg):
        _log(f"kill switch present at {live_cfg.kill_file}: exiting without action")
        return
    if not is_session(today):
        _log(f"{date} is not a session: nothing to do")
        return
    broker, dry = _broker(live_cfg, a.live or a.dry_run is False and False)
    if a.live:
        dry = False
    if not dry and state.already_submitted(date):
        _log("orders already submitted today (idempotency guard): exiting")
        return

    panel, stale_frac, age, rep = refresh_data(cfg, root)
    if stale_frac > live_cfg.max_stale_fraction:
        raise SystemExit(f"{stale_frac:.1%} of the universe has stale prices > {live_cfg.max_stale_fraction:.1%}: aborting")
    features = build_features(panel)
    account = broker.account()
    _log(f"account equity {account['equity']:,.0f} status {account['status']} (dry_run={dry})")
    targets = compute_targets(cfg, root, panel, features, account["equity"])
    current = broker.positions()
    orders = diff_positions(targets, current, live_cfg.min_trade_notional)
    _log(f"targets: {len(targets)} names, gross {targets['target_notional'].abs().sum():,.0f}; orders: {len(orders)} "
         f"(decision date {targets.attrs['decision_date']})")
    universe = set(panel.tradable.iloc[-1][panel.tradable.iloc[-1]].index) | {cfg.portfolio.hedge_symbol}
    try:
        SafetyRails(live_cfg).check(orders, targets, account, universe, age)
    except SafetyViolation as e:
        (rep_dir / "SAFETY_ABORT.txt").write_text(str(e))
        raise SystemExit(f"SAFETY ABORT: {e}")

    targets.to_csv(rep_dir / "targets.csv")
    payload = [{"symbol": o.symbol, "qty": o.qty, "side": o.side, "notional_est": round(o.notional_est, 2)} for o in orders]
    (rep_dir / ("orders_dryrun.json" if dry else "orders_submitted.json")).write_text(json.dumps(payload, indent=2))
    if dry:
        _log(f"DRY RUN: wrote {len(orders)} orders to {rep_dir / 'orders_dryrun.json'}; nothing submitted")
        state.record_submissions(date, [{**p, "order_id": None, "status": "dry_run"} for p in payload], dry_run=True)
        return
    res = broker.submit(orders, tif="opg")
    state.record_submissions(date, res, dry_run=False)
    _log(f"submitted {len(res)} market-on-open orders")


def post_trade(a) -> None:
    live_cfg = load_live_config(a.config)
    cfg = load_config(live_cfg.strategy_config)
    today = pd.Timestamp.today().normalize()
    date = str(today.date())
    rep_dir = Path(live_cfg.report_dir) / date
    rep_dir.mkdir(parents=True, exist_ok=True)
    broker, _ = _broker(live_cfg, False)
    account, positions = broker.account(), broker.positions()
    state = LiveState(live_cfg.state_db)
    state.record_snapshot(date, account, positions)
    fills = broker.orders(after=today)
    tpath = rep_dir / "targets.csv"
    targets = pd.read_csv(tpath, index_col=0) if tpath.exists() else pd.DataFrame(columns=["target_shares", "ref_price"])
    prior = targets["ref_price"] if "ref_price" in targets else pd.Series(dtype=float)
    text = post_trade_report(targets, fills, positions, prior)
    pnl = state.pnl_series()
    if len(pnl) > 40:
        from quantlab.bayes.changepoint import changepoint_alerts

        cp = changepoint_alerts(pnl)
        text += f"\n\nBOCPD on live P&L: P(recent changepoint) = {cp['p_recent'].iloc[-1]:.2f}" + (" ALERT" if cp["alert"].iloc[-1] else "")
    (rep_dir / "reconciliation.md").write_text(text)
    _log(text)


def cli_live(a) -> None:
    if a.action == "run":
        run_daily(a)
    elif a.action == "post-trade":
        post_trade(a)
    elif a.action == "status":
        live_cfg = load_live_config(a.config)
        broker, dry = _broker(live_cfg, False)
        print(json.dumps(broker.account(), indent=2, default=str))
        print(broker.positions().to_string())
        print(f"kill switch: {kill_switch_present(live_cfg)}; dry_run: {dry}")
    elif a.action == "cancel-all":
        live_cfg = load_live_config(a.config)
        broker, _ = _broker(live_cfg, False)
        print(f"cancelled {broker.cancel_all()} open orders")

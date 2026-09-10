"""Experiment orchestration: panel -> features -> signal -> weights -> backtest (-> stats -> registry -> report).

``run_pipeline`` is the shared core used by the CLI, the permutation test, the
parameter grids and the live job, so every path trades exactly the same way.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.backtest.engine import run_backtest
from quantlab.backtest.result import BacktestResult
from quantlab.config import StrategyConfig, load_config
from quantlab.data.panel import Panel, load_panel
from quantlab.features.registry import load_or_build
from quantlab.portfolio.construct import build_target_weights
from quantlab.portfolio.costs import CostModel
from quantlab.portfolio.risk import rolling_betas
from quantlab.seeds import set_all_seeds
from quantlab.signals.base import make_signal


@dataclass
class PipelineOutput:
    cfg: StrategyConfig
    panel: Panel
    features: dict[str, pd.DataFrame]
    signal: pd.DataFrame
    weights: pd.DataFrame
    result: BacktestResult


def git_sha(root: Path = Path(".")) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def git_dirty(root: Path = Path(".")) -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False


def prepare_panel(cfg: StrategyConfig, root: Path | None = None) -> Panel:
    root = Path(root or cfg.data.root)
    panel = load_panel(root, cfg.data.snapshot)
    if cfg.universe == "current":
        # survivorship-biased comparison: today's members treated as always-members
        cur = panel.membership.iloc[-1]
        mem = pd.DataFrame(np.tile(cur.values, (len(panel.sessions), 1)), index=panel.sessions, columns=panel.symbols)
        panel.frames["membership"] = mem
        px_ok = (panel.close > cfg.data.min_price) & (panel.adv_usd > cfg.data.min_adv_usd)
        panel.frames["tradable"] = (mem & px_ok).fillna(False).astype(bool)
    end = cfg.data.end or str(panel.sessions[-1].date())
    return panel.slice(start=pd.Timestamp(cfg.data.start) - pd.Timedelta(days=400), end=pd.Timestamp(end))


def build_overlay(cfg: StrategyConfig, panel: Panel):
    """Compose vol-target and regime overlays into one callable (t, past_returns) -> multiplier."""
    from quantlab.regime.overlay import make_overlay

    return make_overlay(cfg.overlay, panel)


def run_pipeline(cfg: StrategyConfig, root: Path | None = None, panel: Panel | None = None,
                 features: dict[str, pd.DataFrame] | None = None, signal_override: pd.DataFrame | None = None,
                 cost_scale: float = 1.0, ml_predictions: pd.DataFrame | None = None,
                 overlay=None) -> PipelineOutput:
    set_all_seeds(cfg.seed)
    root = Path(root or cfg.data.root)
    panel = panel if panel is not None else prepare_panel(cfg, root)
    if features is None:
        full = load_panel(root, cfg.data.snapshot)
        features = {k: v.reindex(index=panel.sessions, columns=panel.symbols) for k, v in load_or_build(root, full, snapshot=cfg.data.snapshot).items()}
    if cfg.signal.kind == "ml" and signal_override is None and ml_predictions is None:
        from quantlab.ml.train import get_or_train_predictions

        ml_predictions = get_or_train_predictions(cfg, root, panel, features)
    signal = signal_override if signal_override is not None else make_signal(cfg, panel, features, ml_predictions)
    signal = signal.loc[signal.index >= pd.Timestamp(cfg.data.start)]
    betas = None
    if cfg.portfolio.neutral == "beta":
        betas = features["beta_252"].reindex(signal.index)
    weights = build_target_weights(signal, panel.tradable, cfg.portfolio, betas)
    weights = weights.reindex(columns=panel.symbols, fill_value=0.0)
    if overlay is None and (cfg.overlay.vol_target or cfg.overlay.regime):
        overlay = build_overlay(cfg, panel)
    sigma = features["rvol_20"] / np.sqrt(252) if "rvol_20" in features else None
    res = run_backtest(panel.slice(start=pd.Timestamp(cfg.data.start)), weights, CostModel(cfg.costs.scaled(cost_scale)),
                       overlay=overlay, execution=cfg.execution, initial_capital=cfg.initial_capital, sigma_daily=sigma)
    # trim the evaluation window to where the strategy is actually live (first non-zero weight)
    live = weights.abs().sum(axis=1)
    first_live = live[live > 0].index.min() if (live > 0).any() else res.returns.index[0]
    res.returns = res.returns.loc[first_live:]
    res.gross_returns = res.gross_returns.loc[first_live:]
    res.meta.update({"config_hash": cfg.hash(), "name": cfg.name, "family": cfg.family, "seed": cfg.seed,
                     "snapshot_id": panel.meta.get("snapshot_id"), "git_sha": git_sha(), "first_live": str(first_live.date())})
    return PipelineOutput(cfg, panel, features, signal, weights, res)


def _fmt(summary: dict) -> str:
    keys = ["start", "end", "n_obs", "cagr", "ann_vol", "sharpe", "sharpe_gross", "max_dd", "avg_turnover",
            "avg_gross_exposure", "avg_net_exposure", "total_costs_pct_nav", "n_trades"]
    return "\n".join(f"  {k:22s} {summary[k]:.4f}" if isinstance(summary.get(k), float) else f"  {k:22s} {summary.get(k)}" for k in keys)


def cli_backtest(a) -> None:
    cfg = load_config(a.config)
    t0 = time.time()
    out = run_pipeline(cfg, a.root)
    print(f"{cfg.name} [{cfg.family}] snapshot={out.panel.meta.get('snapshot_id')} ({time.time() - t0:.1f}s)")
    print(_fmt(out.result.summary()))


def cli_run(a) -> None:
    from quantlab.registry.runner import run_experiment

    run_experiment(a)

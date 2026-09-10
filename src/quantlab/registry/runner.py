"""Full experiment: pipeline -> statistics -> registry -> tearsheet."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import StrategyConfig, load_config
from quantlab.registry.experiment import git_dirty, run_pipeline
from quantlab.registry.runs import RunRegistry
from quantlab.report import plots
from quantlab.report.tearsheet import build_tearsheet
from quantlab.stats.bootstrap import bootstrap_report
from quantlab.stats.deciles import decile_returns, spread_stats
from quantlab.stats.fama_macbeth import fama_macbeth
from quantlab.stats.ic import decision_spacing, forward_returns, ic_decay, ic_series, ic_summary
from quantlab.stats.regime import conditional_performance
from quantlab.stats.sensitivity import cost_sensitivity
from quantlab.stats.sharpe import sharpe_report


def _sanity_kind(cfg: StrategyConfig) -> str:
    return "sanity" if cfg.family == "sanity" else "trial"


def vol_regime_labels(panel, window: int = 60) -> pd.Series:
    """Lagged realized-vol regime of the benchmark: 'high vol' when trailing vol > expanding median (data <= t-1)."""
    b = panel.meta.get("benchmark", "SPY")
    r = panel.adj_close[b].pct_change(fill_method=None)
    v = r.rolling(window, min_periods=40).std() * np.sqrt(252)
    med = v.expanding(min_periods=252).median()
    lab = pd.Series(np.where(v > med, "high vol", "low vol"), index=v.index)
    lab[v.isna() | med.isna()] = np.nan
    return lab.shift(1)  # known at the close of t-1 for the return on t


def run_experiment(a) -> str:
    t_all = time.time()
    cfg = load_config(a.config)
    if git_dirty() and not a.allow_dirty:
        raise SystemExit("working tree is dirty; commit first or pass --allow-dirty (git sha must identify the code)")
    root = Path(a.root or cfg.data.root)
    reg = RunRegistry(root)
    run_id = reg.new_run_id(cfg.name)
    quick = getattr(a, "quick", False)
    log = lambda m: print(f"[{time.time() - t_all:6.1f}s] {m}")  # noqa: E731

    log(f"pipeline {cfg.name}")
    out = run_pipeline(cfg, root)
    res, panel, sig = out.result, out.panel, out.signal
    s = res.summary()
    log(f"net sharpe {s['sharpe']:.2f} gross {s['sharpe_gross']:.2f} cagr {s['cagr']:.2%} n={s['n_obs']}")
    warnings = []

    # --- signal quality ------------------------------------------------------
    spacing = decision_spacing(sig.index, panel.sessions)
    lags = max(0, int(np.ceil(cfg.horizon / spacing)) - 1)
    ppy = 252.0 / spacing
    fwd = forward_returns(panel, cfg.horizon, cfg.execution)
    ic = ic_series(sig, fwd)
    ics = ic_summary(ic, lags=lags)
    decay = ic_decay(sig, panel, execution=cfg.execution, lag_spacing=spacing)
    controls = {k: out.features[k] for k in ("beta_252", "log_adv", "rvol_60") if k in out.features}
    fm = fama_macbeth(fwd, sig, controls, lags=lags)
    dec = decile_returns(sig, fwd, n=10)
    decs = spread_stats(dec, lags=lags, periods_per_year=ppy, n_boot=500)
    log(f"IC {ics['mean']:.4f} (t={ics['t_nw']:.1f}), decile spread sharpe {decs['spread_sharpe_ann']:.2f}")

    # --- Sharpe inference: trials from the registry (this run counts as one more) --
    tstats = reg.trial_stats(cfg.family)
    kind = _sanity_kind(cfg)
    n_trials = tstats["n_trials"] + (1 if kind in ("trial",) else 0)
    sh = sharpe_report(res.returns, n_trials=max(1, n_trials), var_trials=tstats["var_trials"])
    tstats = {**tstats, "n_trials": max(1, n_trials), "n_effective": max(1, tstats["n_effective"] + (1 if kind == "trial" else 0))}
    boot = bootstrap_report(res.returns, n_boot=200 if quick else a.n_boot)
    log(f"PSR {sh['psr_0']:.3f} DSR {sh['dsr']:.3f} (n_trials={tstats['n_trials']}) boot CI [{boot['sharpe']['lo']:.2f}, {boot['sharpe']['hi']:.2f}]")

    # --- robustness ---------------------------------------------------------
    def rerun_cost(m):
        return run_pipeline(cfg, root, panel=panel, features=out.features, signal_override=sig, cost_scale=m).result.returns

    cost_tbl, breakeven = cost_sensitivity(rerun_cost, (0.0, 0.5, 1.0, 1.5, 2.0, 3.0) if not quick else (0.0, 1.0, 2.0))
    log(f"cost curve done; breakeven x{breakeven}")

    alt = None
    if cfg.universe == "pit" and not quick:
        alt_cfg = cfg.model_copy(update={"universe": "current"})
        r_alt = run_pipeline(alt_cfg, root).result.summary()
        alt = {"label": "current constituents only (survivorship-biased)", **r_alt}
        log(f"survivorship comparison: pit {s['sharpe']:.2f} vs current-only {r_alt['sharpe']:.2f}")

    labels = vol_regime_labels(panel)
    regime = conditional_performance(res.returns, labels)

    bayes = perm = leak = None
    if not quick:
        try:
            from quantlab.bayes.sharpe_posterior import bayes_sharpe

            bayes = bayes_sharpe(res.returns)
            log(f"bayes P(SR>0) {bayes['p_sr_gt_0']:.3f}")
        except ImportError:
            pass
        if getattr(a, "n_perm", 0) and a.n_perm > 0:
            try:
                from quantlab.stats.permutation import permutation_test

                perm = permutation_test(cfg, root, panel, out.features, sig, n_perm=a.n_perm, workers=8, log=log)
                log(f"permutation p {perm['p_value']:.3f}")
            except ImportError:
                pass

    # --- persist + registry --------------------------------------------------
    run_dir = root / "runs" / run_id
    res.save(run_dir)
    metrics = {**s, "ic": ics, "decile": {k: v for k, v in decs.items() if k != "decile_means"}, "sharpe_inf": sh,
               "bootstrap": {k: v for k, v in boot.items() if k in ("sharpe", "cagr", "max_dd")}, "cost_curve": cost_tbl.to_dict("records"),
               "breakeven": breakeven, "alt_universe": alt, "bayes": bayes, "permutation": perm}
    reg.log_trial(run_id, cfg.name, cfg.family, kind, cfg.hash(), cfg.to_json(), panel.meta.get("snapshot_id"),
                  res.meta.get("git_sha", "nogit"), cfg.seed, res.returns, metrics)
    (run_dir / "config.yaml").write_text(cfg.to_json())

    if s["sharpe_gross"] > 0 and s["sharpe"] < 0.5 * s["sharpe_gross"]:
        warnings.append(f"Costs consume more than half of the gross Sharpe ({s['sharpe_gross']:.2f} → {s['sharpe']:.2f}); turnover {s['avg_turnover']:.0%} per rebalance.")
    if sh["dsr"] < 0.95:
        warnings.append(f"DSR {sh['dsr']:.2f}: after deflating for {tstats['n_trials']} registered trials, the Sharpe is not distinguishable from selection luck at 95%.")
    if ics["mean_first_half"] * ics["mean_second_half"] < 0:
        warnings.append("IC changes sign between the first and second half of the sample: signal is not stable.")
    if alt and alt["sharpe"] > s["sharpe"] + 0.15:
        warnings.append(f"Survivorship: using today's constituents inflates Sharpe by {alt['sharpe'] - s['sharpe']:.2f}. Point-in-time coverage is incomplete (see note).")

    # --- report -----------------------------------------------------------------
    if not getattr(a, "no_report", False):
        rep = Path("reports") / run_id
        rep.mkdir(parents=True, exist_ok=True)
        bench = panel.adj_close[panel.meta.get("benchmark", "SPY")].pct_change(fill_method=None)
        figs = {}
        plots.equity_curve(res.returns, bench, figs.setdefault("equity", rep / "equity.png"))
        plots.drawdown(res.returns, figs.setdefault("drawdown", rep / "drawdown.png"))
        plots.rolling_sharpe(res.returns, figs.setdefault("rolling_sharpe", rep / "rolling_sharpe.png"))
        plots.exposure_plot(res.exposure.loc[res.returns.index], figs.setdefault("exposure", rep / "exposure.png"))
        plots.ic_series_plot(ic, figs.setdefault("ic", rep / "ic.png"))
        plots.ic_decay_plot(decay, figs.setdefault("ic_decay", rep / "ic_decay.png"))
        plots.decile_bars(decs["decile_means"], figs.setdefault("deciles", rep / "deciles.png"), ppy)
        plots.cost_curve(cost_tbl, figs.setdefault("cost", rep / "cost.png"), breakeven)
        if bayes is not None and "samples" in bayes:
            plots.histogram(np.asarray(bayes["samples"]), figs.setdefault("bayes", rep / "bayes.png"),
                            "Posterior of annualized Sharpe (Student-t likelihood)", vline=0.0, xlabel="Sharpe")
        cov = json.loads((root / "curated" / "latest" / "quality.json").read_text()).get("coverage_by_year", {}) if (root / "curated" / "latest" / "quality.json").exists() else {}
        cov_note = "PIT price coverage (share of true members with prices): " + ", ".join(f"{y}: {v:.0%}" for y, v in cov.items() if int(y) % 5 == 0) if cov else ""
        ctx = {
            "name": cfg.name, "family": cfg.family, "run_id": run_id, "snapshot_id": panel.meta.get("snapshot_id"),
            "git_sha": res.meta.get("git_sha"), "seed": cfg.seed, "config_hash": cfg.hash(), "universe": cfg.universe,
            "rebalance": cfg.rebalance, "horizon": cfg.horizon, "trials": tstats, "s": s, "sh": sh, "boot": boot,
            "bayes": {k: v for k, v in bayes.items() if k != "samples"} if bayes else None, "perm": perm, "warnings": warnings,
            "ic": ics, "dec": decs, "fm": {k: fm[k] for k in fm if isinstance(fm[k], dict)}, "fm_r2": fm["r2_mean"], "fm_n": fm["n_dates"],
            "cost_rows": cost_tbl.to_dict("records"), "breakeven": f"{breakeven:.2f}×" if breakeven is not None and np.isfinite(breakeven) else ("never (positive at 3×)" if breakeven == np.inf else "n/a"),
            "costs": cfg.costs.model_dump(), "regime": regime.to_dict("index") if len(regime) else None,
            "regime_note": "Regime = benchmark trailing-60d realized vol above/below its expanding median, known at the prior close.",
            "alt": alt, "coverage_note": cov_note, "leak": leak, "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }
        html = build_tearsheet(ctx, figs, rep / "tearsheet.html")
        log(f"tearsheet -> {html}")
    log(f"registered run {run_id} ({kind}); family {cfg.family} now has {tstats['n_trials']} counted trials")
    return run_id

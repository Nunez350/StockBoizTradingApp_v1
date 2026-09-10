"""``quantlab stats <action> <target>`` standalone robustness tools."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantlab.config import load_config
from quantlab.registry.experiment import run_pipeline
from quantlab.registry.runs import RunRegistry


def _cfg_from_target(target: str, root: Path):
    p = Path(target)
    if p.exists():
        return load_config(p)
    reg = RunRegistry(root)
    df = reg.query()
    row = df[df["run_id"] == target]
    if row.empty:
        raise SystemExit(f"{target} is neither a config path nor a registered run id")
    from quantlab.config import StrategyConfig

    return StrategyConfig.model_validate(json.loads(row.iloc[0]["config_json"]))


def cli_stats(a) -> None:
    root = Path(a.root)
    cfg = _cfg_from_target(a.target, root)
    out = run_pipeline(cfg, root)
    panel, feats, sig = out.panel, out.features, out.signal
    print(f"{cfg.name}: net sharpe {out.result.summary()['sharpe']:.3f}")

    if a.action == "permute":
        from quantlab.stats.permutation import permutation_test

        r = permutation_test(cfg, root, panel, feats, sig, n_perm=a.n, workers=a.workers)
        print(json.dumps(r, indent=2))
        RunRegistry(root)  # permutation draws are not counted as trials
    elif a.action == "leak-canary":
        from quantlab.stats.leak import lookahead_canary

        print(json.dumps(lookahead_canary(cfg, root, panel, feats), indent=2))
    elif a.action == "cost-curve":
        from quantlab.stats.sensitivity import cost_sensitivity

        t, be = cost_sensitivity(lambda m: run_pipeline(cfg, root, panel=panel, features=feats, signal_override=sig, cost_scale=m).result.returns)
        print(t.to_string(index=False))
        print(f"breakeven multiplier: {be}")
    elif a.action == "param-grid":
        from quantlab.stats.sensitivity import param_stability

        grid = json.loads(a.grid)
        reg = RunRegistry(root)
        table, recs = param_stability(cfg, lambda c: run_pipeline(c, root, panel=panel, features=feats).result.returns, grid, log=print)
        for rec in recs:  # every cell is a counted trial
            c = rec["config"]
            reg.log_trial(reg.new_run_id(c.name + "_grid"), c.name, c.family, "grid", c.hash(), c.to_json(),
                          panel.meta.get("snapshot_id"), out.result.meta.get("git_sha", "nogit"), c.seed, rec["returns"],
                          {k: rec[k] for k in rec if k not in ("returns", "config")})
        print(table.round(3).to_string())
        print(f"logged {len(recs)} grid cells as counted trials in family {cfg.family}")
    elif a.action == "offsets":
        from quantlab.stats.sensitivity import rebalance_offset_dispersion

        def run_off(c, off):
            dates = sig.index
            pos = panel.sessions.get_indexer(dates) + off
            pos = pos[pos < len(panel.sessions) - 1]
            shifted = pd.DatetimeIndex(panel.sessions[pos])
            from quantlab.signals.base import make_signal

            full = make_signal(c.model_copy(update={"rebalance": "D"}), panel, feats)
            s2 = full.loc[full.index.intersection(shifted)]
            return run_pipeline(c, root, panel=panel, features=feats, signal_override=s2).result.returns

        t = rebalance_offset_dispersion(cfg, run_off, a.n if a.n < 30 else 4)
        print(t.to_string(index=False))
    elif a.action == "bayes":
        from quantlab.bayes.sharpe_posterior import bayes_sharpe

        r = bayes_sharpe(out.result.returns)
        print(json.dumps({k: v for k, v in r.items() if k != "samples"}, indent=2))

"""Register every meta threshold/mode cell as a counted trial, then the chosen one gets a full run."""
from pathlib import Path
import pandas as pd
from quantlab.config import load_config
from quantlab.registry.experiment import run_pipeline
from quantlab.registry.runs import RunRegistry
from quantlab.ml.meta import apply_meta, get_or_train_meta

root = Path("data")
cfg = load_config("configs/strategies/meanrev_5d_meta.yaml")
base = run_pipeline(cfg.model_copy(update={"meta": None}), root)
P = get_or_train_meta(cfg, root, base.panel, base.features, base.signal, base.weights)
reg = RunRegistry(root)
for mode, thr in [("filter", 0.50), ("filter", 0.52), ("filter", 0.55), ("filter", 0.58), ("scale", 0.5)]:
    c = cfg.model_copy(update={"meta": cfg.meta.model_copy(update={"threshold": thr, "mode": mode})})
    w = apply_meta(base.weights, P, mode, thr)
    out = run_pipeline(c, root, panel=base.panel, features=base.features, signal_override=base.signal, weights_override=w)
    s = out.result.summary()
    reg.log_trial(reg.new_run_id(f"{c.name}_{mode}{thr}"), c.name, c.family, "grid", c.hash(), c.to_json(),
                  base.panel.meta.get("snapshot_id"), out.result.meta.get("git_sha", "nogit"), c.seed, out.result.returns,
                  {"mode": mode, "threshold": thr, "sharpe": s["sharpe"], "turnover": s["avg_turnover"], "gross_exp": s["avg_gross_exposure"]})
    print(f"registered {mode} {thr}: net sharpe {s['sharpe']:.3f} gross exp {s['avg_gross_exposure']:.2f}", flush=True)
print(reg.trial_stats(cfg.family))

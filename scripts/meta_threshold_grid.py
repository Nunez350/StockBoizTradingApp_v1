"""Apply an already-trained meta-labeler at several thresholds/modes (engine only, no retraining)."""
import glob, os, sys
import pandas as pd
from quantlab.config import load_config
from quantlab.registry.experiment import run_pipeline
from quantlab.ml.meta import apply_meta

cfg = load_config("configs/strategies/meanrev_5d_meta.yaml")
base = run_pipeline(cfg.model_copy(update={"meta": None}), "data")
P = pd.read_parquet(sorted(glob.glob("data/ml/meta_*.parquet"), key=os.path.getmtime)[-1])
rows = []
for mode, thr in [("filter", 0.50), ("filter", 0.52), ("filter", 0.55), ("filter", 0.58), ("scale", 0.5)]:
    w = apply_meta(base.weights, P, mode, thr)
    r = run_pipeline(cfg, "data", panel=base.panel, features=base.features, signal_override=base.signal, weights_override=w).result.summary()
    rows.append({"mode": mode, "thr": thr, "net_sharpe": r["sharpe"], "gross_sharpe": r["sharpe_gross"], "cagr": r["cagr"],
                 "turnover": r["avg_turnover"], "gross_exp": r["avg_gross_exposure"], "max_dd": r["max_dd"]})
    print(rows[-1], flush=True)
b = base.result.summary()
rows.insert(0, {"mode": "primary", "thr": None, "net_sharpe": b["sharpe"], "gross_sharpe": b["sharpe_gross"], "cagr": b["cagr"],
                "turnover": b["avg_turnover"], "gross_exp": b["avg_gross_exposure"], "max_dd": b["max_dd"]})
df = pd.DataFrame(rows); print(df.round(3).to_string(index=False))
df.to_csv("reports/meta_threshold_grid.csv", index=False)

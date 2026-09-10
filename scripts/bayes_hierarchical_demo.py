"""Hierarchical shrinkage of per-stock rev_5d betas on the last 5 years (weekly decisions)."""
import sys
from pathlib import Path

import pandas as pd

from quantlab.config import load_config
from quantlab.bayes.hierarchical import hierarchical_signal_betas
from quantlab.registry.experiment import prepare_panel, run_pipeline
from quantlab.stats.ic import forward_returns

cfg = load_config("configs/strategies/meanrev_5d.yaml")
out = run_pipeline(cfg, "data")
panel, sig = out.panel, out.signal
start = panel.sessions[-1] - pd.Timedelta(days=5 * 365)
s = sig.loc[start:]
fwd = forward_returns(panel, cfg.horizon, cfg.execution).loc[s.index]
res = hierarchical_signal_betas(s, fwd, panel.sector, min_obs=100, num_warmup=400, num_samples=600)
Path("reports/bayes").mkdir(parents=True, exist_ok=True)
res.to_csv("reports/bayes/hierarchical_rev5d_betas.csv")
print("global beta", res.attrs["b_global"], "tau_sector", res.attrs["tau_sec"], "tau_stock", res.attrs["tau_stock"])
print("sector betas:", res.attrs["b_sector"])
print(f"stocks: {len(res)}; |beta_ols| mean {res['beta_ols'].abs().mean():.4f} -> |beta_shrunk| mean {res['beta_shrunk'].abs().mean():.4f}")
print(f"share of stocks with P(beta>0) > 0.9: {(res['p_positive'] > 0.9).mean():.1%}; < 0.1: {(res['p_positive'] < 0.1).mean():.1%}")
print(res.sort_values('beta_shrunk').head(5).round(4).to_string())
print(res.sort_values('beta_shrunk').tail(5).round(4).to_string())

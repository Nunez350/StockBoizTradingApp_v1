"""Permutation test through the full pipeline (construction, costs, execution lag included).

``within_date``: shuffle scores across symbols independently on each decision date.
Preserves the time structure, turnover level and universe; destroys the cross-sectional
information. The p-value is the share of null Sharpes >= the observed one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from quantlab.registry.experiment import run_pipeline
from quantlab.stats.sharpe import sharpe


def _shuffle_within_date(sig: pd.DataFrame, seed: int) -> pd.DataFrame:
    rs = np.random.default_rng(seed)
    out = sig.copy()
    v = out.to_numpy(copy=True)
    for i in range(v.shape[0]):
        ok = np.isfinite(v[i])
        if ok.sum() > 1:
            v[i, ok] = rs.permutation(v[i, ok])
    out[:] = v
    return out


def _time_shuffle(sig: pd.DataFrame, seed: int) -> pd.DataFrame:
    rs = np.random.default_rng(seed)
    return pd.DataFrame(sig.to_numpy()[rs.permutation(len(sig))], index=sig.index, columns=sig.columns)


def _one(cfg, root, panel, features, sig, seed, mode):
    s = _shuffle_within_date(sig, seed) if mode == "within_date" else _time_shuffle(sig, seed)
    r = run_pipeline(cfg, root, panel=panel, features=features, signal_override=s).result.returns
    return sharpe(r)


def permutation_test(cfg, root, panel, features, sig, n_perm: int = 500, mode: str = "within_date",
                     workers: int = 8, seed: int = 0, log=None) -> dict:
    obs = sharpe(run_pipeline(cfg, root, panel=panel, features=features, signal_override=sig).result.returns)
    null = Parallel(n_jobs=workers, prefer="threads")(delayed(_one)(cfg, root, panel, features, sig, seed + i, mode) for i in range(n_perm))
    null = np.asarray([x for x in null if np.isfinite(x)])
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    return {"observed": float(obs), "p_value": p, "n": int(len(null)), "mode": mode,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "null_q95": float(np.percentile(null, 95))}

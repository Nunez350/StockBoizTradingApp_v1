"""Hierarchical (partial-pooling) shrinkage of per-stock signal betas.

fwd_ret[i,t] = a_i + b_i * signal[i,t] + e,   b_i ~ N(b_sector(i), tau_sector),   b_sector ~ N(b_global, tau_global)

Shows how much per-name "alpha" survives shrinkage and gives shrunk expected returns for
portfolio construction. NUTS via numpyro; ~500 names x 11 sectors is small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hierarchical_signal_betas(signal: pd.DataFrame, fwd: pd.DataFrame, sector: dict[str, str],
                              min_obs: int = 60, num_warmup: int = 500, num_samples: int = 1000, seed: int = 0) -> pd.DataFrame:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    s = signal.stack(future_stack=True).rename("s")
    y = fwd.reindex_like(signal).stack(future_stack=True).rename("y")
    df = pd.concat([s, y], axis=1).dropna()
    df.index.names = ["date", "symbol"]
    df = df.reset_index()
    counts = df.groupby("symbol").size()
    df = df[df["symbol"].isin(counts[counts >= min_obs].index)]
    # standardize per date so betas are comparable
    df["s"] = df.groupby("date")["s"].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    df["y"] = df.groupby("date")["y"].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    syms = sorted(df["symbol"].unique())
    sec_names = sorted({sector.get(x, "Unknown") for x in syms})
    sym_id = {x: i for i, x in enumerate(syms)}
    sec_id = {x: i for i, x in enumerate(sec_names)}
    stock_sec = jnp.asarray([sec_id[sector.get(x, "Unknown")] for x in syms])
    si = jnp.asarray(df["symbol"].map(sym_id).to_numpy())
    S = jnp.asarray(df["s"].to_numpy()); Y = jnp.asarray(df["y"].to_numpy())
    # OLS per stock (unpooled) for comparison
    ols = df.groupby("symbol").apply(lambda g: np.polyfit(g["s"], g["y"], 1)[0], include_groups=False)

    def model(si, S, Y):
        b_global = numpyro.sample("b_global", dist.Normal(0.0, 0.1))
        tau_sec = numpyro.sample("tau_sec", dist.HalfNormal(0.05))
        tau_stock = numpyro.sample("tau_stock", dist.HalfNormal(0.05))
        with numpyro.plate("sec", len(sec_names)):
            b_sec = numpyro.sample("b_sec", dist.Normal(b_global, tau_sec))
        with numpyro.plate("stock", len(syms)):
            b_stock = numpyro.sample("b_stock", dist.Normal(b_sec[stock_sec], tau_stock))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        numpyro.sample("y", dist.Normal(b_stock[si] * S, sigma), obs=Y)

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), si, S, Y)
    sm = mcmc.get_samples()
    b = np.asarray(sm["b_stock"])
    out = pd.DataFrame({"beta_ols": ols.reindex(syms).to_numpy(), "beta_shrunk": b.mean(axis=0),
                        "beta_sd": b.std(axis=0), "p_positive": (b > 0).mean(axis=0),
                        "sector": [sector.get(x, "Unknown") for x in syms]}, index=syms)
    out.attrs = {"b_global": float(np.asarray(sm["b_global"]).mean()), "tau_sec": float(np.asarray(sm["tau_sec"]).mean()),
                 "tau_stock": float(np.asarray(sm["tau_stock"]).mean()),
                 "b_sector": dict(zip(sec_names, np.asarray(sm["b_sec"]).mean(axis=0).round(4).tolist()))}
    return out

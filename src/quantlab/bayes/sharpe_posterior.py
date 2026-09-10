"""Bayesian Sharpe ratio: Student-t likelihood on daily returns, NUTS via numpyro.

Reports the posterior of the annualized Sharpe, P(SR > 0), P(SR > 0.5), and the posterior
mean of the degrees of freedom (fat tails). A conjugate Normal-Inverse-Gamma closed form is
provided as a fast path (``conjugate_sharpe``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conjugate_sharpe(returns, periods: int = 252, n_draws: int = 20000, seed: int = 0) -> dict:
    """Normal likelihood, Jeffreys-type prior: mu | sigma^2 ~ N(xbar, sigma^2/n), sigma^2 ~ Inv-Gamma((n-1)/2, (n-1)s^2/2)."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n, xbar, s2 = len(r), r.mean(), r.var(ddof=1)
    rs = np.random.default_rng(seed)
    sigma2 = 1.0 / rs.gamma((n - 1) / 2.0, 2.0 / ((n - 1) * s2), n_draws)
    mu = rs.normal(xbar, np.sqrt(sigma2 / n))
    sr = mu / np.sqrt(sigma2) * np.sqrt(periods)
    return _summarize(sr, nu=np.nan, model="normal-conjugate")


def bayes_sharpe(returns, periods: int = 252, num_warmup: int = 500, num_samples: int = 1500, seed: int = 0) -> dict:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    scale = r.std(ddof=1)
    y = jnp.asarray(r / scale)  # standardize for sampler geometry; SR is scale-free

    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
        nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))  # mean 20, supports fat tails
        numpyro.sample("y", dist.StudentT(nu, mu, sigma), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), y)
    sm = mcmc.get_samples()
    mu, sigma, nu = np.asarray(sm["mu"]), np.asarray(sm["sigma"]), np.asarray(sm["nu"])
    # SR uses the sd of the predictive distribution: sigma * sqrt(nu/(nu-2)) for nu > 2
    sd = sigma * np.sqrt(np.where(nu > 2.05, nu / (nu - 2.0), np.nan))
    sr = mu / sd * np.sqrt(periods)
    sr = sr[np.isfinite(sr)]
    out = _summarize(sr, nu=float(nu.mean()), model="student-t NUTS")
    # the t-location is a robust center: under skew it differs from the sample mean, so report the
    # normal-likelihood answer next to it
    out["p_sr_gt_0_normal"] = conjugate_sharpe(r, periods)["p_sr_gt_0"]
    return out


def _summarize(sr: np.ndarray, nu: float, model: str) -> dict:
    lo, hi = np.percentile(sr, [2.5, 97.5])
    return {"model": model, "sr_mean": float(sr.mean()), "sr_lo": float(lo), "sr_hi": float(hi),
            "p_sr_gt_0": float((sr > 0).mean()), "p_sr_gt_0p5": float((sr > 0.5).mean()), "nu_mean": nu,
            "samples": sr.astype(float).tolist()}

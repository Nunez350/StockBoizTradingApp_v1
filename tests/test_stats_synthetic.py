import numpy as np
import pandas as pd
from scipy import stats as sps

from quantlab.calendar import sessions
from quantlab.stats.bootstrap import bootstrap_ci, optimal_block_len
from quantlab.stats.deciles import decile_returns, spread_stats
from quantlab.stats.fama_macbeth import fama_macbeth
from quantlab.stats.ic import forward_returns, ic_series, ic_summary
from quantlab.stats.newey_west import nw_mean_tstat
from quantlab.stats.sharpe import dsr, effective_trials, expected_max_sr, psr, sharpe_se


def _planted(rho=0.05, n_dates=400, n_sym=300, seed=0):
    rs = np.random.default_rng(seed)
    idx = sessions("2015-01-01", "2020-12-31")[:n_dates]
    cols = [f"S{i}" for i in range(n_sym)]
    sig = pd.DataFrame(rs.normal(size=(n_dates, n_sym)), index=idx, columns=cols)
    noise = pd.DataFrame(rs.normal(size=(n_dates, n_sym)), index=idx, columns=cols)
    fwd = (rho * sig + np.sqrt(1 - rho**2) * noise) * 0.02
    return sig, fwd


def test_ic_recovers_planted_correlation():
    sig, fwd = _planted(rho=0.05)
    ic = ic_series(sig, fwd, method="pearson")
    s = ic_summary(ic, lags=0)
    assert abs(s["mean"] - 0.05) < 3 * (ic.std() / np.sqrt(len(ic)))
    assert s["t_nw"] > 5


def test_ic_zero_when_no_signal():
    sig, fwd = _planted(rho=0.0, seed=3)
    s = ic_summary(ic_series(sig, fwd), lags=0)
    assert abs(s["t_nw"]) < 3


def test_fama_macbeth_recovers_slope():
    sig, fwd = _planted(rho=0.05)
    fm = fama_macbeth(fwd, sig, lags=0)
    # y = 0.02 * (rho*z + ...): slope on standardized signal ~ 0.02*rho = 0.001
    assert abs(fm["signal"]["mean"] - 0.001) < 0.0003
    assert fm["signal"]["t_nw"] > 5


def test_deciles_monotone():
    sig, fwd = _planted(rho=0.08)
    dec = decile_returns(sig, fwd, n=10)
    st = spread_stats(dec, lags=0, n_boot=200)
    assert st["spread_mean"] > 0 and st["spread_t_nw"] > 5
    assert st["monotonicity_rho"] > 0.9 and st["monotonicity_p_boot"] < 0.01


def test_newey_west_widens_se_for_autocorrelated_series():
    rs = np.random.default_rng(0)
    n = 2000
    e = rs.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + e[t]
    naive = x.std(ddof=1) / np.sqrt(n)
    nw = nw_mean_tstat(x, lags=20)
    assert nw["se"] > 1.8 * naive  # AR(1) rho=0.7 long-run variance factor (1+rho)/(1-rho) = 5.7 -> se x2.4


def test_psr_closed_form_gaussian():
    # skew 0, kurt 3 -> SE = sqrt((1 + SR^2/2)/(n-1)); PSR = Phi(SR / SE)
    sr, n = 0.05, 500
    assert np.isclose(psr(sr, 0.0, n), sps.norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 + sr**2 / 2)))
    assert np.isclose(sharpe_se(sr, n), np.sqrt((1 + sr**2 / 2) / (n - 1)))


def test_expected_max_sr_vs_monte_carlo():
    rs = np.random.default_rng(1)
    n_trials, n_obs = 200, 1000
    maxes = []
    for _ in range(300):
        r = rs.normal(size=(n_trials, n_obs))
        srs = r.mean(axis=1) / r.std(axis=1, ddof=1)
        maxes.append(srs.max())
    var_trials = 1.0 / n_obs  # SR sampling variance for iid normal
    formula = expected_max_sr(n_trials, var_trials)
    assert abs(np.mean(maxes) - formula) < 0.1 * formula


def test_dsr_deflates_with_more_trials():
    d1 = dsr(0.1, 1000, 0.0, 3.0, n_trials=1, var_trials=0.001)
    d100 = dsr(0.1, 1000, 0.0, 3.0, n_trials=100, var_trials=0.001)
    assert d1["dsr"] > d100["dsr"]


def test_effective_trials_clusters_duplicates():
    rs = np.random.default_rng(0)
    base = rs.normal(size=500)
    df = pd.DataFrame({"a": base, "b": base + 1e-3 * rs.normal(size=500), "c": rs.normal(size=500)})
    assert effective_trials(df) == 2


def test_block_length_and_bootstrap_ci_covers_truth():
    rs = np.random.default_rng(0)
    x = rs.normal(0.02, 1.0, 2000)
    b = optimal_block_len(x)
    assert 2 <= b <= 50
    ci = bootstrap_ci(x, lambda a: a.mean(), n_boot=300, mean_block=b)
    assert ci["lo"] < 0.02 < ci["hi"]


def test_forward_returns_are_forward(panel):
    fwd = forward_returns(panel, 5)
    t = panel.sessions[100]
    i = 100
    expect = panel.adj_open.iloc[i + 6]["S000"] / panel.adj_open.iloc[i + 1]["S000"] - 1
    assert np.isclose(fwd.loc[t, "S000"], expect)
    assert fwd.iloc[-6:].isna().all().all()

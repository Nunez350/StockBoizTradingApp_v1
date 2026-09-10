import numpy as np
import pandas as pd

from quantlab.bayes.changepoint import bocpd
from quantlab.bayes.kalman import KalmanRegression, kalman_beta
from quantlab.bayes.sharpe_posterior import bayes_sharpe, conjugate_sharpe


def test_conjugate_sharpe_recovers_truth():
    rs = np.random.default_rng(0)
    r = rs.normal(0.0005, 0.01, 8000)  # true annualized SR = 0.05*sqrt(252) ~ 0.79
    out = conjugate_sharpe(r)
    assert out["sr_lo"] < 0.79 < out["sr_hi"] and out["p_sr_gt_0"] > 0.95


def test_student_t_sharpe_nuts_runs_and_recovers():
    rs = np.random.default_rng(1)
    r = rs.standard_t(4, 1500) * 0.01 + 0.0006
    true_sr = 0.0006 / (0.01 * np.sqrt(4 / 2)) * np.sqrt(252)
    out = bayes_sharpe(r, num_warmup=300, num_samples=600)
    assert out["sr_lo"] < true_sr < out["sr_hi"]
    assert 2.5 < out["nu_mean"] < 8  # fat tails detected


def test_kalman_tracks_step_in_beta():
    rs = np.random.default_rng(2)
    n = 1000
    x = rs.normal(0, 0.01, n)
    beta = np.where(np.arange(n) < 500, 0.8, 1.6)
    y = beta * x + rs.normal(0, 0.005, n)
    b = kalman_beta(pd.Series(y), pd.Series(x), q=1e-3)
    assert abs(b.iloc[450] - 0.8) < 0.2 and abs(b.iloc[-1] - 1.6) < 0.2


def test_kalman_filter_is_causal():
    rs = np.random.default_rng(3)
    X = rs.normal(size=(300, 2)); y = X @ np.array([1.0, -0.5]) + rs.normal(0, 0.1, 300)
    a, _ = KalmanRegression(q=1e-4).filter(y[:200], X[:200])
    b, _ = KalmanRegression(q=1e-4).filter(y, X)
    np.testing.assert_allclose(a, b[:200])


def test_bocpd_flags_planted_changepoint():
    rs = np.random.default_rng(4)
    x = np.concatenate([rs.normal(0.0, 1.0, 400), rs.normal(2.5, 1.0, 200)])
    res = bocpd(x, hazard=1 / 200)
    p = res["p_recent"]
    hit = np.argmax(p[395:] > 0.5) + 395 if (p[395:] > 0.5).any() else None
    assert hit is not None and hit - 400 <= 10, f"changepoint detected at {hit}"
    assert (p[50:390] > 0.5).mean() < 0.05  # few false alarms before the break

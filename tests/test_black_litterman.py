import numpy as np
import pandas as pd

from quantlab.bayes.black_litterman import bl_posterior, bl_weights_from_scores, shrink_cov


def test_bl_posterior_moves_toward_views():
    syms = list("ABCD")
    Sigma = pd.DataFrame(np.eye(4) * 0.04, index=syms, columns=syms)
    pi = pd.Series(0.01, index=syms)
    Q = pd.Series([0.05, 0.01, 0.01, -0.03], index=syms)
    omega = pd.Series(0.001, index=syms)
    mu = bl_posterior(pi, Sigma, Q, omega, tau=0.05)
    assert mu["A"] > mu["B"] > mu["D"]
    assert (mu["A"] < Q["A"]) and (mu["A"] > pi["A"])


def test_bl_weights_are_dollar_neutral_and_capped(panel):
    rets = panel.adj_close.pct_change(fill_method=None).iloc[-300:]
    scores = pd.Series(np.random.default_rng(0).normal(size=len(panel.symbols) - 1), index=[s for s in panel.symbols if s != "SPY"])
    w = bl_weights_from_scores(scores, rets, ic=0.03, gross=1.0, max_pos=0.1)
    assert abs(w.sum()) < 1e-6 and abs(w.abs().sum() - 1.0) < 1e-6 and (w.abs() <= 0.1 + 1e-9).all()

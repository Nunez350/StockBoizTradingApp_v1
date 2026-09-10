import numpy as np

from quantlab.regime.hmm import fit_hmm, forward_filter


def _sim(n=1500, seed=0):
    rs = np.random.default_rng(seed)
    state = 0
    X = np.empty((n, 2))
    for t in range(n):
        if rs.random() < 0.03:
            state = 1 - state
        sd = 0.01 if state == 0 else 0.03
        X[t] = [rs.normal(0, sd), abs(rs.normal(0, sd)) + sd]
    return X


def test_forward_filter_is_causal():
    X = _sim()
    p = fit_hmm(X[:1000], n_states=2)
    a = forward_filter(p, X[:1200])
    b = forward_filter(p, X[:1500])
    np.testing.assert_allclose(a, b[:1200], atol=1e-12)


def test_smoothed_probs_differ_from_filtered():
    """Sanity: hmmlearn's predict_proba (forward-backward) is NOT causal - proves why we hand-roll."""
    from hmmlearn.hmm import GaussianHMM

    X = _sim()
    m = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=0).fit(X[:1000])
    a = m.predict_proba(X[:1200])
    b = m.predict_proba(X[:1500])
    assert not np.allclose(a, b[:1200], atol=1e-6)


def test_high_vol_state_is_last():
    X = _sim()
    p = fit_hmm(X[:1000], n_states=2)
    assert p["covars"][1][0, 0] > p["covars"][0][0, 0]

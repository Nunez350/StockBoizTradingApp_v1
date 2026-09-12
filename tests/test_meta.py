import numpy as np
import pandas as pd

from quantlab.ml.meta import apply_meta


def _w():
    idx = pd.to_datetime(["2024-01-05", "2024-01-12"])
    cols = ["A", "B", "C", "D"]
    return pd.DataFrame([[0.25, 0.25, -0.25, -0.25], [0.3, 0.2, -0.1, -0.4]], index=idx, columns=cols)


def test_filter_drops_low_confidence_and_stays_neutral():
    w = _w()
    P = pd.DataFrame([[0.9, 0.2, 0.9, 0.9], [np.nan] * 4], index=w.index, columns=w.columns)
    out = apply_meta(w, P, "filter", 0.5)
    assert out.loc[w.index[0], "B"] == 0
    assert abs(out.iloc[0].sum()) < 1e-12 and out.iloc[0].abs().sum() <= w.iloc[0].abs().sum() + 1e-12
    # unknown P (before first fold) -> primary weights untouched except neutralization
    assert np.allclose(out.iloc[1].abs().sum(), 2 * min(0.5, 0.5))


def test_scale_never_exceeds_primary():
    w = _w()
    P = pd.DataFrame(0.95, index=w.index, columns=w.columns)
    out = apply_meta(w, P, "scale", 0.5)
    assert (out.abs() <= w.abs() + 1e-12).all().all()
    assert abs(out.iloc[0].sum()) < 1e-12

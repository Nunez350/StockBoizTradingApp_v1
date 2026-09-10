"""Every registered feature must be causal: value at t is unchanged when data after t is removed."""

import numpy as np
import pandas as pd
import pytest

from quantlab.features.registry import FEATURES, compute, list_features

CHECK_DATES = ["2020-03-16", "2020-11-30", "2021-06-15"]


@pytest.mark.parametrize("name", list_features())
def test_feature_is_causal(panel, name):
    full = compute(name, panel)
    for d in CHECK_DATES:
        t = pd.Timestamp(d)
        part = compute(name, panel.slice(end=t))
        a, b = full.loc[t], part.loc[t]
        assert a.notna().sum() > 0, f"{name} all-NaN at {d}"
        np.testing.assert_allclose(a.values, b.values, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f"{name} @ {d}")


def test_registry_nonempty():
    assert len(FEATURES) >= 10


def test_no_negative_shift_in_feature_source():
    import inspect

    import quantlab.features.price as m

    src = inspect.getsource(m)
    assert ".shift(-" not in src, "features must never shift forward"

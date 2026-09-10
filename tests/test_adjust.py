import numpy as np
import pandas as pd

from quantlab.data.adjust import adjustment_factor, apply_adjustment


def _frame(split_adjusted):
    idx = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 51, 52, 53.0],
            "high": [101, 102, 103, 52, 53, 54.0],
            "low": [99, 100, 101, 50, 51, 52.0],
            "close": [100, 101, 102, 51, 52, 53.0],
            "volume": [1e6] * 6,
            "adj_close": np.nan,
            "dividend": [0, 0, 1.01, 0, 0, 0.0],  # ex-date day 3, 1% of prior close 101
            "split_ratio": [1, 1, 1, 2, 1, 1.0],  # 2:1 split on day 4
        },
        index=idx,
    )
    df.attrs = {"source": "t", "split_adjusted": split_adjusted}
    return df


def test_factor_raw_prices_split_and_dividend():
    df = _frame(split_adjusted=False)
    f = adjustment_factor(df, False)
    assert f.iloc[-1] == 1.0
    # last three days after split: factor 1; day 3 (split next day): 0.5; day 2: 0.5 * (1 - 1.01/101) = 0.495
    assert np.isclose(f.iloc[3], 1.0) and np.isclose(f.iloc[2], 0.5) and np.isclose(f.iloc[1], 0.495)
    out = apply_adjustment(df)
    # total-return adj close should show no artificial -50% jump across the split
    r = out["adj_close_ours"].pct_change()
    assert abs(r.iloc[3]) < 0.02


def test_factor_split_adjusted_source_ignores_splits():
    df = _frame(split_adjusted=True)
    f = adjustment_factor(df, True)
    assert np.isclose(f.iloc[2], 1.0) and np.isclose(f.iloc[1], 1 - 1.01 / 101)


def test_vendor_discrepancy_zero_when_consistent():
    df = _frame(split_adjusted=False)
    f = adjustment_factor(df, False)
    df["adj_close"] = df["close"] * f * 3.0  # vendor basis differs by a constant
    out = apply_adjustment(df)
    assert out["adj_disc"].max() < 1e-9

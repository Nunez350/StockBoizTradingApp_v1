import numpy as np
import pandas as pd
import pytest

from quantlab.calendar import sessions
from quantlab.data.panel import Panel


def make_synthetic_panel(n_sym: int = 30, start="2019-01-01", end="2021-12-31", seed: int = 0,
                         benchmark: str = "SPY") -> Panel:
    rs = np.random.default_rng(seed)
    sess = sessions(start, end)
    syms = [f"S{i:03d}" for i in range(n_sym)] + [benchmark]
    n, k = len(sess), len(syms)
    mkt = rs.normal(0.0003, 0.01, n)
    beta = rs.uniform(0.5, 1.5, k)
    idio = rs.normal(0, 0.015, (n, k))
    r = mkt[:, None] * beta[None, :] + idio
    r[:, -1] = mkt  # benchmark
    close = 100 * np.exp(np.cumsum(r, axis=0))
    close = pd.DataFrame(close, index=sess, columns=syms)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rs.normal(0, 0.003, (n, k)))
    high = np.maximum(open_, close) * (1 + np.abs(rs.normal(0, 0.003, (n, k))))
    low = np.minimum(open_, close) * (1 - np.abs(rs.normal(0, 0.003, (n, k))))
    vol = pd.DataFrame(rs.lognormal(14, 0.3, (n, k)), index=sess, columns=syms)
    fr = {
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        "adj_open": open_, "adj_high": high, "adj_low": low, "adj_close": close,
        "dividend": close * 0.0, "split_ratio": close * 0.0 + 1.0,
    }
    fr["adv_usd"] = (close * vol).rolling(20, min_periods=10).mean()
    mem = pd.DataFrame(True, index=sess, columns=syms)
    mem[benchmark] = False
    fr["membership"] = mem
    fr["tradable"] = mem.copy()
    return Panel(fr, {s: "Synthetic" for s in syms}, {"benchmark": benchmark, "start": start, "end": end})


@pytest.fixture(scope="session")
def panel() -> Panel:
    return make_synthetic_panel()

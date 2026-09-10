import pandas as pd

from quantlab.data.cache import ParquetCache
from quantlab.data.sources import SCHEMA


class FakeSource:
    name = "fake"
    split_adjusted = False

    def __init__(self):
        self.calls = []

    def available(self):
        return True

    def fetch_daily(self, symbol, start, end=None):
        self.calls.append((symbol, start))
        idx = pd.bdate_range(start, "2024-02-15")
        idx = idx[[d.weekday() < 5 for d in idx]]
        df = pd.DataFrame({c: 1.0 for c in SCHEMA}, index=idx)
        df["close"] = range(1, len(idx) + 1)
        df.attrs = {"source": "fake", "split_adjusted": False}
        return df


def test_incremental_update(tmp_path):
    src = FakeSource()
    cache = ParquetCache(tmp_path, ["fake"])
    s = cache.update(["AAA"], [src], "2024-01-02", workers=1, asof=pd.Timestamp("2024-02-15"), log=lambda *_: None)
    assert s.fetched == ["AAA"]
    df = cache.get("AAA")
    assert df.attrs["source"] == "fake" and df.index.max() == pd.Timestamp("2024-02-15")
    # second update: up to date -> skipped, no fetch
    n = len(src.calls)
    s2 = cache.update(["AAA"], [src], "2024-01-02", workers=1, asof=pd.Timestamp("2024-02-15"), log=lambda *_: None)
    assert s2.skipped == ["AAA"] and len(src.calls) == n
    # later asof -> refetch from overlap only
    s3 = cache.update(["AAA"], [src], "2024-01-02", workers=1, asof=pd.Timestamp("2024-02-20"), log=lambda *_: None)
    assert s3.fetched == ["AAA"]
    assert pd.Timestamp(src.calls[-1][1]) > pd.Timestamp("2024-02-01")

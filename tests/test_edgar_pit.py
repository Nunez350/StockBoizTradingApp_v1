import numpy as np
import pandas as pd

from quantlab.data.edgar import pit_series_stock, pit_series_ttm


def _facts():
    rows = []
    # four quarters of a fiscal year: Q1..Q3 quarterly (10-Q) and the annual total in the 10-K (no Q4 quarterly)
    q = [("2023-01-01", "2023-03-31", 100, "2023-05-01", "10-Q"), ("2023-04-01", "2023-06-30", 110, "2023-08-01", "10-Q"),
         ("2023-07-01", "2023-09-30", 120, "2023-11-01", "10-Q"), ("2023-01-01", "2023-12-31", 460, "2024-02-15", "10-K"),
         ("2024-01-01", "2024-03-31", 105, "2024-05-01", "10-Q")]
    for start, end, val, filed, form in q:
        rows.append({"start": start, "end": end, "val": val, "filed": filed, "form": form, "fp": "Q", "fy": 2023})
    # a restatement of Q1 filed later must NOT replace the originally reported value
    rows.append({"start": "2023-01-01", "end": "2023-03-31", "val": 999, "filed": "2024-02-15", "form": "10-K", "fp": "Q1", "fy": 2023})
    assets = [{"end": "2023-03-31", "val": 1000, "filed": "2023-05-01", "form": "10-Q"}, {"end": "2023-06-30", "val": 1100, "filed": "2023-08-01", "form": "10-Q"}]
    return {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": rows}}, "Assets": {"units": {"USD": assets}}}}}


def test_ttm_first_reported_and_q4_derived():
    t = pit_series_ttm(_facts(), "net_income")
    assert t is not None
    # Q4 = 460 - (100+110+120) = 130; TTM at FY end = 460, known when the 10-K is filed
    row = t[t["end"] == pd.Timestamp("2023-12-31")].iloc[0]
    assert row["ttm"] == 460 and row["filed"] == pd.Timestamp("2024-02-15")
    # next quarter TTM = 110+120+130+105 = 465 and uses the ORIGINAL Q1 (100), not the restated 999
    row2 = t[t["end"] == pd.Timestamp("2024-03-31")].iloc[0]
    assert row2["ttm"] == 465 and row2["q_yoy"] == 5


def test_stock_series_known_at_filing():
    s = pit_series_stock(_facts(), "assets")
    assert list(s["filed"]) == [pd.Timestamp("2023-05-01"), pd.Timestamp("2023-08-01")]

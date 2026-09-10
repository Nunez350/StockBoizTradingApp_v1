import pandas as pd

from quantlab.calendar import sessions
from quantlab.data.universe import build_membership, canonical_ticker


def test_canonical():
    assert canonical_ticker("BRK-B") == "BRK.B"
    assert canonical_ticker("brk.b[1]") == "BRK.B"


def test_membership_backward_walk():
    sess = sessions("2024-01-02", "2024-01-31")
    current = pd.DataFrame({"symbol": ["A", "B", "C"]})
    # On Jan 10, C replaced X. On Jan 20, B replaced Y. Before Jan 10: {A, X, Y}.
    changes = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-10", "2024-01-20"]), "added": ["C", "B"], "removed": ["X", "Y"]}
    )
    m = build_membership(sess, current, changes)
    assert set(m.columns) == {"A", "B", "C", "X", "Y"}
    on = lambda d: set(m.columns[m.loc[pd.Timestamp(d)].values])
    assert on("2024-01-31") == {"A", "B", "C"}
    assert on("2024-01-22") == {"A", "B", "C"}
    assert on("2024-01-19") == {"A", "C", "Y"}
    assert on("2024-01-10") == {"A", "C", "Y"}
    assert on("2024-01-09") == {"A", "X", "Y"}
    assert on("2024-01-02") == {"A", "X", "Y"}


def test_membership_one_sided_event():
    sess = sessions("2024-01-02", "2024-01-31")
    current = pd.DataFrame({"symbol": ["A", "B"]})
    changes = pd.DataFrame({"date": pd.to_datetime(["2024-01-15"]), "added": [None], "removed": ["Z"]})
    m = build_membership(sess, current, changes)
    on = lambda d: set(m.columns[m.loc[pd.Timestamp(d)].values])
    assert on("2024-01-16") == {"A", "B"}
    assert on("2024-01-12") == {"A", "B", "Z"}

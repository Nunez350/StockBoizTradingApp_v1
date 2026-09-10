"""NYSE session calendar helpers built on ``exchange_calendars`` (XNYS).

Every panel in the project is indexed by *sessions* (tz-naive midnight Timestamps).
Alpaca returns UTC timestamps; ``to_session`` converts them to the New York session date.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

NY = "America/New_York"


@lru_cache(maxsize=1)
def _xnys():
    import exchange_calendars as xc

    return xc.get_calendar("XNYS", start="1990-01-01", end="2030-12-31")


def sessions(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """All NYSE sessions in [start, end] as tz-naive timestamps."""
    cal = _xnys()
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    idx = cal.sessions_in_range(s, e)
    return pd.DatetimeIndex(idx.tz_localize(None) if idx.tz is not None else idx, name="date")


def is_session(t: str | pd.Timestamp) -> bool:
    return bool(_xnys().is_session(pd.Timestamp(t).normalize()))


def _naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


def next_session(t: str | pd.Timestamp, n: int = 1) -> pd.Timestamp:
    """The n-th session strictly after ``t`` (``t`` need not be a session)."""
    cal = _xnys()
    ts = pd.Timestamp(t).normalize()
    cur = cal.date_to_session(ts, direction="next")  # ts itself if it is a session
    steps = n if cal.is_session(ts) else n - 1
    for _ in range(steps):
        cur = cal.next_session(cur)
    return _naive(cur)


def prev_session(t: str | pd.Timestamp, n: int = 1) -> pd.Timestamp:
    """The n-th session strictly before ``t`` (``t`` need not be a session)."""
    cal = _xnys()
    ts = pd.Timestamp(t).normalize()
    cur = cal.date_to_session(ts, direction="previous")
    steps = n if cal.is_session(ts) else n - 1
    for _ in range(steps):
        cur = cal.previous_session(cur)
    return _naive(cur)


def to_session(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert (possibly tz-aware UTC) timestamps to tz-naive NY session dates."""
    idx = pd.DatetimeIndex(ts)
    if idx.tz is not None:
        idx = idx.tz_convert(NY)
    return pd.DatetimeIndex(idx.tz_localize(None).normalize(), name="date")


def align(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Reindex a date-indexed frame onto sessions; raise if any index value is not a session."""
    idx = pd.DatetimeIndex(df.index)
    bad = [d for d in idx if not is_session(d)]
    if bad:
        raise ValueError(f"{len(bad)} off-calendar dates, e.g. {bad[:3]}")
    s = start or idx.min()
    e = end or idx.max()
    return df.reindex(sessions(s, e))


def rebalance_dates(sess: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """Last session of each period defined by ``rule`` (pandas offset alias, e.g. 'W-FRI', 'ME')."""
    s = pd.Series(sess, index=sess)
    last = s.groupby(s.index.to_period(_period_alias(rule))).last()
    return pd.DatetimeIndex(last.values, name="date")


def _period_alias(rule: str) -> str:
    r = rule.upper()
    if r in {"M", "ME", "BM", "BME"}:
        return "M"
    if r in {"Q", "QE"}:
        return "Q"
    if r.startswith("W"):
        return r if "-" in r else "W-FRI"
    if r == "D":
        return "D"
    raise ValueError(f"unsupported rebalance rule {rule!r}")

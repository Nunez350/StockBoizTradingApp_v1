import pandas as pd

from quantlab.calendar import align, next_session, prev_session, rebalance_dates, sessions, to_session


def test_sessions_basic():
    s = sessions("2024-01-02", "2024-01-10")
    assert len(s) == 7
    assert s[0] == pd.Timestamp("2024-01-02")
    assert s.tz is None


def test_holiday_skipped():
    s = sessions("2024-07-03", "2024-07-05")
    assert list(s.date.astype(str)) == ["2024-07-03", "2024-07-05"]


def test_next_prev():
    assert next_session("2024-07-03") == pd.Timestamp("2024-07-05")
    assert next_session("2024-07-04") == pd.Timestamp("2024-07-05")
    assert prev_session("2024-07-05") == pd.Timestamp("2024-07-03")
    assert next_session("2024-01-02", 2) == pd.Timestamp("2024-01-04")


def test_to_session_from_utc():
    utc = pd.DatetimeIndex(["2024-01-02T05:00:00Z", "2024-01-03T05:00:00Z"])
    assert list(to_session(utc)) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_align_rejects_off_calendar():
    df = pd.DataFrame({"x": [1, 2]}, index=pd.DatetimeIndex(["2024-01-02", "2024-01-06"]))
    try:
        align(df)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_rebalance_dates_weekly_monthly():
    s = sessions("2024-01-01", "2024-03-31")
    w = rebalance_dates(s, "W-FRI")
    assert all(d in s for d in w)
    assert pd.Timestamp("2024-01-05") in w
    m = rebalance_dates(s, "ME")
    assert list(m.date.astype(str)) == ["2024-01-31", "2024-02-29", "2024-03-28"]

"""Cross-validation for overlapping financial labels: purged K-fold, combinatorial purged CV, walk-forward.

Rows carry a label interval ``[t0, t1]`` (decision date, label end date). A train row is
*purged* if its interval overlaps the test interval, and *embargoed* if its t0 falls
within ``embargo`` sessions after the test block ends. Groups are contiguous in time and
all rows sharing a t0 stay together.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def _as_pos(dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> np.ndarray:
    pos = sessions.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("t0/t1 contain dates not in sessions")
    return pos


def purge_and_embargo(train_mask: np.ndarray, t0: np.ndarray, t1: np.ndarray, test_start: int, test_end: int,
                      embargo: int) -> np.ndarray:
    """Return train mask with rows overlapping [test_start, test_end] purged and the embargo applied.
    Positions are integer session indices."""
    overlap = (t0 <= test_end) & (t1 >= test_start)
    emb = (t0 > test_end) & (t0 <= test_end + embargo)
    return train_mask & ~overlap & ~emb


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo: int = 0):
        self.n_splits, self.embargo = n_splits, embargo

    def split(self, t0: pd.DatetimeIndex | np.ndarray, t1: pd.DatetimeIndex | np.ndarray, sessions: pd.DatetimeIndex):
        p0, p1 = _as_pos(pd.DatetimeIndex(t0), sessions), _as_pos(pd.DatetimeIndex(t1), sessions)
        uniq = np.unique(p0)
        folds = np.array_split(uniq, self.n_splits)
        for f in folds:
            test = np.isin(p0, f)
            ts, te = int(p0[test].min()), int(p1[test].max())
            train = purge_and_embargo(~test, p0, p1, ts, te, self.embargo)
            yield np.flatnonzero(train), np.flatnonzero(test)


class CombinatorialPurgedCV:
    """Bailey/Lopez de Prado CPCV: choose ``n_test`` of ``n_groups`` contiguous groups as test.
    ``n_paths = C(n_groups-1, n_test-1)``; each group is tested in exactly that many splits, and
    ``path_of(split_idx, group)`` assigns every tested (split, group) pair to one full OOS path."""

    def __init__(self, n_groups: int = 6, n_test: int = 2, embargo: int = 0):
        self.n_groups, self.n_test, self.embargo = n_groups, n_test, embargo
        self.combos = list(combinations(range(n_groups), n_test))
        self.n_paths = comb(n_groups - 1, n_test - 1)
        self._path = {}
        counter = {g: 0 for g in range(n_groups)}
        for si, c in enumerate(self.combos):
            for g in c:
                self._path[(si, g)] = counter[g]
                counter[g] += 1

    def path_of(self, split_idx: int, group: int) -> int:
        return self._path[(split_idx, group)]

    def groups(self, t0_pos: np.ndarray) -> np.ndarray:
        uniq = np.unique(t0_pos)
        bounds = np.array_split(uniq, self.n_groups)
        g = np.empty(len(t0_pos), dtype=int)
        for gi, b in enumerate(bounds):
            g[np.isin(t0_pos, b)] = gi
        return g

    def split(self, t0, t1, sessions: pd.DatetimeIndex):
        p0, p1 = _as_pos(pd.DatetimeIndex(t0), sessions), _as_pos(pd.DatetimeIndex(t1), sessions)
        grp = self.groups(p0)
        for si, c in enumerate(self.combos):
            test = np.isin(grp, c)
            train = ~test
            for g in c:  # purge around each test block separately
                m = grp == g
                train = purge_and_embargo(train, p0, p1, int(p0[m].min()), int(p1[m].max()), self.embargo)
            yield si, np.flatnonzero(train), np.flatnonzero(test), grp


def walk_forward_splits(t0: pd.DatetimeIndex, t1: pd.DatetimeIndex, sessions: pd.DatetimeIndex,
                        train_min_sessions: int, step_rule: str = "QE", embargo: int = 0):
    """Expanding-window walk-forward: retrain at each period boundary, predict the following period."""
    from quantlab.calendar import rebalance_dates

    p0, p1 = _as_pos(t0, sessions), _as_pos(t1, sessions)
    bounds = rebalance_dates(sessions, step_rule)
    bpos = sessions.get_indexer(bounds)
    bpos = bpos[bpos >= train_min_sessions]
    for a, b in zip(bpos[:-1], bpos[1:]):
        test = (p0 > a) & (p0 <= b)
        if not test.any():
            continue
        train = (p1 <= a - embargo) & (p0 <= a)  # labels fully resolved before the cutoff, minus embargo
        yield np.flatnonzero(train), np.flatnonzero(test), sessions[a]

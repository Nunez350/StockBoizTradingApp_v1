import numpy as np
import pandas as pd

from quantlab.calendar import sessions
from quantlab.stats.cv import CombinatorialPurgedCV, PurgedKFold, walk_forward_splits


def _labels(h=5, n=300):
    sess = sessions("2020-01-01", "2021-12-31")
    t0 = sess[: n]
    pos = np.arange(n) + h
    t1 = sess[pos]
    return sess, pd.DatetimeIndex(t0), pd.DatetimeIndex(t1)


def test_purged_kfold_no_overlap_and_embargo():
    sess, t0, t1 = _labels(h=5)
    p0, p1 = sess.get_indexer(t0), sess.get_indexer(t1)
    cv = PurgedKFold(n_splits=5, embargo=3)
    for tr, te in cv.split(t0, t1, sess):
        ts, te_end = p0[te].min(), p1[te].max()
        assert not ((p0[tr] <= te_end) & (p1[tr] >= ts)).any(), "train label overlaps test window"
        assert not ((p0[tr] > te_end) & (p0[tr] <= te_end + 3)).any(), "embargo violated"
        assert len(set(tr) & set(te)) == 0


def test_cpcv_paths_and_coverage():
    sess, t0, t1 = _labels(h=5)
    cv = CombinatorialPurgedCV(n_groups=6, n_test=2, embargo=2)
    assert cv.n_paths == 5 and len(cv.combos) == 15
    seen = {}
    for si, tr, te, grp in cv.split(t0, t1, sess):
        for g in set(grp[te]):
            path = cv.path_of(si, g)
            seen.setdefault(path, set()).add(g)
        p0, p1 = sess.get_indexer(t0), sess.get_indexer(t1)
        for g in set(grp[te]):
            m = grp == g
            assert not ((p0[tr] <= p1[m].max()) & (p1[tr] >= p0[m].min())).any()
    assert all(seen[p] == set(range(6)) for p in range(5)), "each path must contain every group exactly once"


def test_walk_forward_train_labels_resolved():
    sess, t0, t1 = _labels(h=5, n=400)
    p0, p1 = sess.get_indexer(t0), sess.get_indexer(t1)
    n = 0
    for tr, te, cutoff in walk_forward_splits(t0, t1, sess, train_min_sessions=120, step_rule="ME", embargo=5):
        c = sess.get_loc(cutoff)
        assert p1[tr].max() <= c - 5 and p0[te].min() > c
        n += 1
    assert n > 5

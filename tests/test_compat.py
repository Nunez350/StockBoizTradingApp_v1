from quantlab.compat import smoke_test


def test_fragile_libs_under_pandas3():
    res = smoke_test()
    bad = {k: v for k, v in res.items() if v != "ok"}
    assert not bad, bad

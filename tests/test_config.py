import pytest
import yaml

from quantlab.config import CostConfig, StrategyConfig, load_config


def test_defaults_and_hash_stable():
    a = StrategyConfig(name="x")
    b = StrategyConfig(name="x")
    assert a.hash() == b.hash()
    assert StrategyConfig(name="y").hash() != a.hash()


def test_typo_key_rejected():
    with pytest.raises(Exception):
        StrategyConfig(name="x", portfolio={"gros": 1.0})


def test_load_merges_base(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"costs": {"half_spread_bps": 9.0}, "seed": 7}))
    strat = tmp_path / "s.yaml"
    strat.write_text(yaml.safe_dump({"name": "s", "costs": {"impact_coef": 0.1}}))
    cfg = load_config(strat, base=base)
    assert cfg.costs.half_spread_bps == 9.0 and cfg.costs.impact_coef == 0.1 and cfg.seed == 7


def test_cost_scaling():
    c = CostConfig().scaled(0)
    assert c.half_spread_bps == 0 and c.commission_per_share == 0

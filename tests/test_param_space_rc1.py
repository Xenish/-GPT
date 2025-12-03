import random

import pytest

from finantradealgo.strategies.param_space import (
    ParamSpaceError,
    apply_strategy_params_to_cfg,
    sample_params,
    validate_param_space,
)
from finantradealgo.strategies.rule_param_space import RULE_PARAM_SPACE
from finantradealgo.strategies.trend_param_space import TREND_PARAM_SPACE
from finantradealgo.strategies.sweep_param_space import SWEEP_PARAM_SPACE
from finantradealgo.strategies.volatility_param_space import VOLATILITY_PARAM_SPACE


def test_all_param_spaces_validate():
    for space in [RULE_PARAM_SPACE, TREND_PARAM_SPACE, SWEEP_PARAM_SPACE, VOLATILITY_PARAM_SPACE]:
        validate_param_space(space)


def test_sample_params_returns_values_in_range():
    rng = random.Random(42)
    params = sample_params(RULE_PARAM_SPACE, rng=rng)
    assert params
    assert RULE_PARAM_SPACE["ms_trend_min"].low <= params["ms_trend_min"] <= RULE_PARAM_SPACE["ms_trend_min"].high
    assert RULE_PARAM_SPACE["tp_atr_mult"].low <= params["tp_atr_mult"] <= RULE_PARAM_SPACE["tp_atr_mult"].high


def test_validate_param_space_rejects_bad_spec():
    from finantradealgo.strategies.param_space import ParamSpec

    bad_space = {
        "x": ParamSpec(name="x", type="int", low=5, high=1),
    }
    with pytest.raises(ParamSpaceError):
        validate_param_space(bad_space)


def test_apply_strategy_params_to_cfg_merges_without_mutation():
    cfg = {"strategy": {"rule": {"a": 1}}, "other": 2}
    params = {"b": 3}
    merged = apply_strategy_params_to_cfg(cfg, "rule", params)

    assert merged["strategy"]["rule"]["b"] == 3
    assert cfg["strategy"]["rule"].get("b") is None  # original untouched

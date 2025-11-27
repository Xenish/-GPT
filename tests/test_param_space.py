from __future__ import annotations

import random

import pytest

from finantradealgo.strategies.param_space import (
    ParamSpec,
    ParamSpaceError,
    apply_strategy_params_to_cfg,
    sample_params,
    validate_param_space,
)


def test_validate_param_space_success():
    space = {
        "alpha": ParamSpec(name="alpha", type="float", low=0.1, high=1.0),
        "beta": ParamSpec(name="beta", type="categorical", choices=["a", "b"]),
    }
    validate_param_space(space)


def test_validate_param_space_name_mismatch():
    space = {"alpha": ParamSpec(name="beta", type="float", low=0.1, high=1.0)}
    with pytest.raises(ParamSpaceError):
        validate_param_space(space)


def test_validate_param_space_numeric_range():
    space = {"alpha": ParamSpec(name="alpha", type="float", low=1.0, high=0.5)}
    with pytest.raises(ParamSpaceError):
        validate_param_space(space)


def test_validate_param_space_missing_choices():
    space = {"alpha": ParamSpec(name="alpha", type="categorical")}
    with pytest.raises(ParamSpaceError):
        validate_param_space(space)


def test_sample_params_types_and_ranges():
    space = {
        "n": ParamSpec(name="n", type="int", low=1, high=5),
        "lr": ParamSpec(name="lr", type="float", low=0.001, high=0.01),
        "use_ms": ParamSpec(name="use_ms", type="bool"),
        "side": ParamSpec(name="side", type="categorical", choices=["long", "short"]),
    }
    validate_param_space(space)
    rng = random.Random()
    sample = sample_params(space, rng=rng)
    assert isinstance(sample["n"], int)
    assert 1 <= sample["n"] <= 5
    assert isinstance(sample["lr"], float)
    assert 0.001 <= sample["lr"] <= 0.01
    assert isinstance(sample["use_ms"], bool)
    assert sample["side"] in {"long", "short"}


def test_sample_params_deterministic():
    space = {
        "n": ParamSpec(name="n", type="int", low=1, high=5),
        "lr": ParamSpec(name="lr", type="float", low=0.001, high=0.01),
        "use_ms": ParamSpec(name="use_ms", type="bool"),
        "side": ParamSpec(name="side", type="categorical", choices=["long", "short"]),
    }
    validate_param_space(space)
    rng = random.Random(42)
    first = sample_params(space, rng=rng)
    second = sample_params(space, rng=rng)
    assert first == {
        "n": 1,
        "lr": pytest.approx(0.0012250967970040025),
        "use_ms": False,
        "side": "long",
    }
    assert second == {
        "n": 2,
        "lr": pytest.approx(0.0022558413567262954),
        "use_ms": False,
        "side": "long",
    }


def test_apply_strategy_params_to_cfg():
    cfg = {"strategy": {"rule": {"use_ms_chop_filter": False}}}
    result = apply_strategy_params_to_cfg(cfg, "rule", {"ms_trend_min": -0.5})
    assert result is not cfg
    assert result["strategy"]["rule"]["ms_trend_min"] == -0.5
    assert result["strategy"]["rule"]["use_ms_chop_filter"] is False

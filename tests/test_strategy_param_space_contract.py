from __future__ import annotations

import pytest

from finantradealgo.strategies.param_space import ParamSpace, ParamSpec, ParamSpaceError, validate_param_space
from finantradealgo.strategies.strategy_engine import STRATEGY_REGISTRY


def test_strategy_registry_param_space_contract():
    assert STRATEGY_REGISTRY, "Strategy registry should not be empty"

    for name, meta in STRATEGY_REGISTRY.items():
        # Registry key matches meta name
        assert meta.name == name, f"Meta name mismatch for strategy '{name}'"

        if meta.is_searchable or meta.param_space is not None:
            # Searchable strategies must expose a non-empty param_space
            assert isinstance(meta.param_space, dict) and meta.param_space, f"Strategy '{name}' missing param_space"
            _validate_param_space_contract(name, meta.param_space)
        else:
            # Non-searchable strategies explicitly have no param space
            assert meta.param_space is None or meta.param_space == {}, f"Non-searchable strategy '{name}' should not define param_space"


def _validate_param_space_contract(strategy_name: str, space: ParamSpace) -> None:
    # Reuse existing validator to catch low/high/choices issues
    validate_param_space(space)

    # Extra guardrails: consistent naming and non-empty ranges/choices enforced by validate_param_space
    for key, spec in space.items():
        assert isinstance(spec, ParamSpec), f"Strategy '{strategy_name}' param '{key}' must be ParamSpec"
        assert spec.name == key, f"Strategy '{strategy_name}' param key/name mismatch: {key} vs {spec.name}"


def test_validate_param_space_raises_on_bad_specs():
    bad_space = {
        "foo": ParamSpec(name="foo", type="int", low=10, high=5),  # low >= high
    }
    with pytest.raises(ParamSpaceError):
        validate_param_space(bad_space)

"""
Tests for strategy registry and metadata system.
"""
from __future__ import annotations

import pytest

from finantradealgo.strategies.strategy_engine import (
    STRATEGY_REGISTRY,
    get_searchable_strategies,
    get_strategies_by_family,
    get_strategy_meta,
    list_strategies,
)


def test_all_strategies_have_required_fields():
    """All strategies must have name, description, family, and status."""
    for name, meta in STRATEGY_REGISTRY.items():
        assert meta.name, f"Strategy '{name}' missing name"
        assert meta.description, f"Strategy '{name}' missing description"
        assert meta.family, f"Strategy '{name}' missing family"
        assert meta.status in ("experimental", "candidate", "baseline", "live"), \
            f"Strategy '{name}' has invalid status: {meta.status}"


def test_searchable_strategies_have_param_space():
    """All searchable strategies must have param_space defined."""
    searchable = get_searchable_strategies()
    for name, meta in searchable.items():
        assert meta.param_space is not None, \
            f"Searchable strategy '{name}' missing param_space"
        assert len(meta.param_space) > 0, \
            f"Searchable strategy '{name}' has empty param_space"
        assert meta.is_searchable is True, \
            f"Searchable strategy '{name}' has is_searchable=False"


def test_list_strategies_with_status_filter():
    """list_strategies should filter by status correctly."""
    candidates = list_strategies(status="candidate")
    assert len(candidates) > 0, "Should have at least one candidate strategy"

    for name, meta in candidates.items():
        assert meta.status == "candidate", \
            f"Strategy '{name}' should have status='candidate'"


def test_list_strategies_with_family_filter():
    """list_strategies should filter by family correctly."""
    rule_strategies = list_strategies(family="rule")
    assert "rule" in rule_strategies, "Should include 'rule' strategy"

    for name, meta in rule_strategies.items():
        assert meta.family == "rule", \
            f"Strategy '{name}' should have family='rule'"


def test_list_strategies_with_searchable_only():
    """list_strategies with searchable_only should match get_searchable_strategies."""
    searchable_via_list = list_strategies(searchable_only=True)
    searchable_direct = get_searchable_strategies()

    assert set(searchable_via_list.keys()) == set(searchable_direct.keys()), \
        "searchable_only filter should match get_searchable_strategies"


def test_get_strategy_meta():
    """get_strategy_meta should return correct meta."""
    meta = get_strategy_meta("rule")
    assert meta.name == "rule"
    assert meta.family == "rule"


def test_get_strategy_meta_invalid_name():
    """get_strategy_meta should raise ValueError for invalid name."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy_meta("nonexistent_strategy")


def test_registry_not_empty():
    """Strategy registry should have at least one strategy."""
    assert len(STRATEGY_REGISTRY) > 0, "Registry should not be empty"


def test_no_duplicate_strategy_names():
    """All strategy names should be unique."""
    names = [meta.name for meta in STRATEGY_REGISTRY.values()]
    assert len(names) == len(set(names)), "Duplicate strategy names found"

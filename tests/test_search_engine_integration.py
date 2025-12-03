"""
Integration tests for strategy search engine.
"""
from __future__ import annotations

import pytest
from datetime import datetime, UTC

from finantradealgo.research.strategy_search.search_engine import (
    random_search,
    evaluate_strategy_once,
)
from finantradealgo.research.strategy_search.jobs import StrategySearchJob, create_job_id
from finantradealgo.system.config_loader import load_config


@pytest.fixture
def research_config():
    """Load research config."""
    return load_config("research")


def test_evaluate_strategy_once_with_default_params(research_config):
    """Test single strategy evaluation with default params."""
    result = evaluate_strategy_once(
        strategy_name="rule",
        params=None,
        sys_cfg=research_config,
    )

    assert "params" in result
    assert "cum_return" in result
    assert "sharpe" in result
    assert "max_drawdown" in result
    assert "trade_count" in result


def test_random_search_with_seed(research_config):
    """Test random search with seed for reproducibility."""
    results1 = random_search(
        strategy_name="rule",
        n_samples=5,
        sys_cfg=research_config,
        random_seed=42,
    )

    results2 = random_search(
        strategy_name="rule",
        n_samples=5,
        sys_cfg=research_config,
        random_seed=42,
    )

    # With same seed, params should be identical
    assert len(results1) == 5
    assert len(results2) == 5

    for r1, r2 in zip(results1, results2):
        assert r1["params"] == r2["params"], \
            "Same seed should produce same parameter samples"


def test_random_search_error_handling(research_config):
    """Test that random_search handles errors gracefully."""
    # Use invalid params to trigger errors (this is a simulation)
    # In reality, we'd need a strategy that can fail
    results = random_search(
        strategy_name="rule",
        n_samples=3,
        sys_cfg=research_config,
        random_seed=42,
    )

    # All results should have status field
    for result in results:
        assert "status" in result
        assert result["status"] in ("ok", "error")
        assert "error_message" in result


def test_create_job_id_format():
    """Test job ID creation format."""
    job_id = create_job_id(
        strategy="rule",
        symbol="BTCUSDT",
        timeframe="15m",
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )

    assert "rule" in job_id
    assert "BTCUSDT" in job_id
    assert "15m" in job_id
    assert "20250101" in job_id


def test_strategy_search_job_serialization():
    """Test job serialization/deserialization."""
    job = StrategySearchJob(
        job_id="test_job_123",
        strategy="rule",
        symbol="BTCUSDT",
        timeframe="15m",
        search_type="random",
        n_samples=50,
        config_path="config/system.research.yml",
        created_at=datetime.now(UTC),
        seed=42,
        notes="Test job",
    )

    # Serialize
    job_dict = job.to_dict()
    assert job_dict["job_id"] == "test_job_123"
    assert job_dict["seed"] == 42

    # Deserialize
    job_restored = StrategySearchJob.from_dict(job_dict)
    assert job_restored.job_id == job.job_id
    assert job_restored.seed == job.seed

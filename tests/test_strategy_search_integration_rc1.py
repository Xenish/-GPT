from pathlib import Path

import pandas as pd
import pytest

from finantradealgo.research.strategy_search.search_engine import run_random_search
from finantradealgo.research.strategy_search.jobs import StrategySearchJob
from finantradealgo.strategies.strategy_engine import get_strategy_meta


@pytest.mark.parametrize("strategy_name", ["rule", "trend_continuation", "sweep_reversal", "volatility_breakout"])
def test_strategy_meta_and_searchable(strategy_name):
    meta = get_strategy_meta(strategy_name)
    assert meta.param_space is not None
    assert meta.is_searchable is True


@pytest.mark.parametrize("strategy_name", ["rule", "trend_continuation", "sweep_reversal", "volatility_breakout"])
def test_run_random_search_minimal(strategy_name):
    # Build a minimal job
    job = StrategySearchJob(
        job_id=f"{strategy_name}_test_job",
        strategy=strategy_name,
        symbol="BTCUSDT",
        timeframe="15m",
        search_type="random",
        n_samples=1,
        profile="research",
        created_at=pd.Timestamp("2025-01-01T00:00:00Z"),
    )
    job_dir = run_random_search(job, param_space=None, sys_cfg=None)
    assert job_dir.exists()
    results_path = job_dir / "results.parquet"
    assert results_path.exists()
    df = pd.read_parquet(results_path)
    assert len(df) >= 1

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import finantradealgo.research.strategy_search.search_engine as se
from finantradealgo.research.strategy_search.jobs import StrategySearchJob


def _make_job(search_type: str = "random") -> StrategySearchJob:
    created_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return StrategySearchJob(
        job_id="rule_BTCUSDT_15m_20250101_120000",
        strategy="rule",
        symbol="BTCUSDT",
        timeframe="15m",
        search_type=search_type,
        n_samples=1,
        created_at=created_at,
        profile="research",
        seed=11,
        mode="research",
    )


def test_run_random_search_writes_outputs(monkeypatch, tmp_path: Path):
    job = _make_job()

    fake_results = [
        {
            "params": {"foo": 1},
            "cum_return": 0.1,
            "sharpe": 1.2,
            "max_drawdown": -0.05,
            "win_rate": 0.6,
            "trade_count": 12,
            "status": "ok",
            "error_message": None,
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
            "runtime_sec": 1.5,
        }
    ]

    def fake_random_search(**kwargs):
        return fake_results

    monkeypatch.setattr(se, "random_search", fake_random_search)
    monkeypatch.setattr(se, "BASE_OUTPUT_DIR", tmp_path / "strategy_search")

    job_dir = se.run_random_search(job, param_space=None, sys_cfg=None)

    results_parquet = job_dir / "results.parquet"
    meta_json = job_dir / "meta.json"
    assert results_parquet.exists()
    assert meta_json.exists()

    df = pd.read_parquet(results_parquet)
    assert len(df) >= 1
    for col in se.REQUIRED_RESULT_COLUMNS:
        assert col in df.columns

    loaded_job = StrategySearchJob.load_meta(job_dir)
    assert loaded_job.job_id == job.job_id
    assert loaded_job.strategy == job.strategy
    assert loaded_job.search_type == job.search_type


def test_run_random_search_rejects_grid_job():
    job = _make_job(search_type="grid")
    with pytest.raises(ValueError):
        se.run_random_search(job)

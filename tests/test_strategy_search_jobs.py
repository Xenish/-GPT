from datetime import datetime, timezone
from pathlib import Path

import pytest

from finantradealgo.research.strategy_search.jobs import StrategySearchJob


def _sample_job(created_at: datetime | None = None) -> StrategySearchJob:
    created_at = created_at or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return StrategySearchJob(
        job_id="rule_BTCUSDT_15m_20250101_120000",
        strategy="rule",
        symbol="BTCUSDT",
        timeframe="15m",
        search_type="random",
        n_samples=10,
        created_at=created_at,
        profile="research",
        notes="Round-trip test",
        seed=123,
        mode="research",
        config_snapshot_relpath="cfg/snap.yaml",
    )


def test_strategy_search_job_roundtrip_serialization():
    created_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    job = _sample_job(created_at)

    job_dict = job.to_dict()
    assert job_dict["created_at"] == created_at.isoformat()
    assert job_dict["profile"] == "research"
    assert job_dict["seed"] == 123
    assert job_dict["config_snapshot_relpath"] == "cfg/snap.yaml"

    restored = StrategySearchJob.from_dict(job_dict)
    assert restored == job


def test_save_meta_and_load_meta(tmp_path: Path):
    job = _sample_job()

    meta_path = job.save_meta(tmp_path)
    assert meta_path.exists()

    loaded_job = StrategySearchJob.load_meta(tmp_path)
    assert loaded_job == job

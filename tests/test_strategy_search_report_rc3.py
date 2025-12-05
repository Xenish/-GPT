import json
from pathlib import Path

import pandas as pd

from finantradealgo.research.reporting.base import ReportProfile
from finantradealgo.research.reporting.strategy_search import StrategySearchReportGenerator


def _write_dummy_job(job_dir: Path):
    job_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "params": {"a": 1},
                "param_a": 1,
                "param_b": 2,
                "sharpe": 1.5,
                "cum_return": 0.1,
                "max_drawdown": -0.05,
                "win_rate": 0.6,
                "trade_count": 100,
                "status": "ok",
                "error_message": None,
            },
            {
                "params": {"a": 2},
                "param_a": 2,
                "param_b": 3,
                "sharpe": 0.5,
                "cum_return": 0.02,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
                "trade_count": 60,
                "status": "ok",
                "error_message": None,
            },
            {
                "params": {"a": 3},
                "param_a": 3,
                "param_b": 4,
                "sharpe": None,
                "cum_return": None,
                "max_drawdown": None,
                "win_rate": None,
                "trade_count": None,
                "status": "error",
                "error_message": "fail",
            },
        ]
    )
    df.to_parquet(job_dir / "results.parquet", index=False)

    meta = {
        "job_id": "dummy_job",
        "run_id": "run_dummy",
        "strategy": "rule",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "n_samples": 3,
        "search_type": "random",
        "profile": "research",
        "git_sha": "abc123",
    }
    (job_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_strategy_search_report_generator(tmp_path: Path):
    job_dir = tmp_path / "strategy_search" / "job1"
    _write_dummy_job(job_dir)

    gen = StrategySearchReportGenerator()
    report = gen.generate(job_dir)

    assert report.title.startswith("Strategy Search Report")
    assert report.job_id == "dummy_job"
    assert report.run_id == "run_dummy"
    assert report.profile in (ReportProfile.RESEARCH, "research")
    assert report.strategy_id == "rule"
    assert report.symbol == "BTCUSDT"
    assert report.timeframe == "15m"
    assert len(report.sections) >= 4

    # At least one section should carry data with a DataFrame
    has_df = False
    for section in report.sections:
        if section.data:
            if any(isinstance(v, pd.DataFrame) for v in section.data.values()):
                has_df = True
                break
    assert has_df

    # Artifacts should include heatmap if generated
    assert "heatmap_html" in report.artifacts

    # Smoke render
    report.to_markdown()
    report.to_html()

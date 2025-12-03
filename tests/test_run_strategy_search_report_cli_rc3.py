import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_dummy_job(job_dir: Path):
    job_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"param_a": 1, "param_b": 2, "sharpe": 1.0, "cum_return": 0.1, "max_drawdown": -0.1, "trade_count": 10, "status": "ok", "error_message": None},
            {"param_a": 2, "param_b": 3, "sharpe": 0.5, "cum_return": 0.05, "max_drawdown": -0.2, "trade_count": 8, "status": "ok", "error_message": None},
        ]
    )
    df.to_parquet(job_dir / "results.parquet", index=False)
    meta = {
        "job_id": job_dir.name,
        "strategy": "rule",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "n_samples": 2,
        "search_type": "random",
        "profile": "research",
    }
    (job_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_run_strategy_search_report_cli(tmp_path: Path):
    job_dir = tmp_path / "strategy_search" / "job_cli"
    _write_dummy_job(job_dir)

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_strategy_search_report",
        "--job-dir",
        str(job_dir),
        "--format",
        "markdown",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report_path = job_dir / "report.md"
    assert report_path.exists()
    content = report_path.read_text()
    assert "Strategy Search Report" in content
    assert len(content) > 10

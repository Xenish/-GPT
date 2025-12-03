import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_strategy_search_cli_creates_job_dir(tmp_path: Path):
    output_root = tmp_path / "strategy_search"
    env = os.environ.copy()
    env["STRATEGY_SEARCH_DRYRUN"] = "1"
    env["STRATEGY_SEARCH_OUTPUT_DIR"] = str(output_root)

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_strategy_search",
        "--profile",
        "research",
        "--strategy",
        "rule",
        "--symbol",
        "BTCUSDT",
        "--timeframe",
        "15m",
        "--n-samples",
        "1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr

    assert output_root.exists(), "Output directory should be created"
    job_dirs = [p for p in output_root.iterdir() if p.is_dir()]
    assert job_dirs, "At least one job directory should be created"

    job_dir = job_dirs[0]
    results_parquet = job_dir / "results.parquet"
    meta_json = job_dir / "meta.json"
    assert results_parquet.exists()
    assert meta_json.exists()

    meta = json.loads(meta_json.read_text())
    assert "job_id" in meta
    assert "strategy" in meta

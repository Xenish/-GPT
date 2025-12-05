import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _prepare_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "job_cli"
    job_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "equity_metrics": {"cum_return": 0.1, "sharpe": 1.0, "max_drawdown": -0.05},
        "trade_stats": {"trade_count": 2, "win_rate": 0.5},
    }
    (job_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    pd.DataFrame({"equity": [100, 105, 110]}).to_csv(job_dir / "equity_curve.csv", index=False)
    pd.DataFrame({"pnl": [5, -3], "side": ["long", "short"]}).to_csv(job_dir / "trades.csv", index=False)
    return job_dir


def test_run_backtest_report_cli_generates_reports(tmp_path: Path):
    job_dir = _prepare_job(tmp_path)

    cmd = [
        sys.executable,
        "scripts/run_backtest_report.py",
        "--job-dir",
        str(job_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (job_dir / "report.html").exists()
    assert (job_dir / "report.md").exists()

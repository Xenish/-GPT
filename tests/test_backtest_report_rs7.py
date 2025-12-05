import json
from pathlib import Path

import pandas as pd

from finantradealgo.research.reporting.backtest_report import BacktestReportGenerator
from finantradealgo.research.reporting.base import Report, ReportProfile


def _write_backtest_job(tmp_path: Path):
    job_dir = tmp_path / "job1"
    job_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "job_id": "job1",
        "run_id": "run1",
        "profile": "research",
        "strategy": "rule",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "equity_metrics": {
            "initial_cash": 10000,
            "final_equity": 12000,
            "cum_return": 0.2,
            "max_drawdown": -0.1,
            "sharpe": 1.5,
        },
        "trade_stats": {
            "trade_count": 3,
            "win_rate": 0.66,
            "profit_factor": 1.8,
        },
        "risk_stats": {"blocked_entries": 1, "kill_switch_triggered": False},
    }
    (job_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    equity_df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "equity": [10000, 11000, 12000],
        }
    )
    equity_path = job_dir / "equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)

    trades_df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "pnl": [50, -10, 40],
            "side": ["long", "short", "long"],
        }
    )
    trades_path = job_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    return job_dir, metrics, equity_path, trades_path


def test_backtest_report_generator_builds_sections(tmp_path: Path):
    job_dir, metrics, equity_path, trades_path = _write_backtest_job(tmp_path)

    gen = BacktestReportGenerator()
    report = gen.generate(
        metrics=metrics,
        equity_curve_path=equity_path,
        trades_path=trades_path,
        job_id="job1",
        run_id="run1",
        profile=ReportProfile.RESEARCH,
        strategy_id="rule",
        symbol="BTCUSDT",
        timeframe="15m",
    )

    assert report.job_id == "job1"
    assert report.run_id == "run1"
    assert report.profile == ReportProfile.RESEARCH
    assert report.metrics["sharpe"] == 1.5
    assert report.artifacts["equity_curve_csv"].endswith("equity_curve.csv")
    assert report.artifacts["trades_csv"].endswith("trades.csv")

    titles = [s.title for s in report.sections]
    assert "Overview" in titles
    assert "Equity & Drawdown" in titles
    assert "Trades" in titles
    assert "Risk" in titles

    # Roundtrip serialization with DataFrames
    restored = Report.from_dict(report.to_dict())
    assert restored.metrics["sharpe"] == 1.5
    assert isinstance(restored.sections[1].data["Equity Curve"], pd.DataFrame)

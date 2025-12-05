from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from finantradealgo.research.reporting.base import Report, ReportSection, ReportProfile
from finantradealgo.research.reporting.backtest_report import BacktestReportGenerator
from finantradealgo.research.reporting.strategy_search import StrategySearchReportGenerator
from finantradealgo.research.reporting.live_report import LiveReportGenerator


def test_report_base_serialization_roundtrip():
    section = ReportSection(
        title="Overview",
        content="Test content",
        metrics={"sharpe": 1.2},
        artifacts={"equity_curve": "path/equity.csv"},
        data={"table": pd.DataFrame({"a": [1, 2]})},
    )
    report = Report(
        title="Test Report",
        description="desc",
        job_id="job1",
        run_id="run1",
        profile=ReportProfile.RESEARCH,
        strategy_id="rule",
        symbol="BTCUSDT",
        timeframe="15m",
        metrics={"final_equity": 100.0},
        artifacts={"trades_csv": "path/trades.csv"},
        sections=[section],
        metadata={"report_type": "test"},
    )

    as_dict = report.to_dict()
    restored = Report.from_dict(as_dict)

    assert restored.title == report.title
    assert restored.profile == report.profile
    assert restored.sections and restored.sections[0].title == "Overview"
    md = report.to_markdown()
    html = report.to_html()
    js = report.to_json()
    assert "Test Report" in md
    assert "Test Report" in html
    assert js["title"] == "Test Report"


def _dummy_backtest_inputs(tmp_path: Path):
    equity_path = tmp_path / "equity.csv"
    trades_path = tmp_path / "trades.csv"
    pd.DataFrame({"timestamp": [1], "equity": [100]}).to_csv(equity_path, index=False)
    pd.DataFrame({"id": [1]}).to_csv(trades_path, index=False)
    return equity_path, trades_path


def test_generators_return_report_instances(tmp_path):
    # Backtest
    equity_path, trades_path = _dummy_backtest_inputs(tmp_path)
    bt_report = BacktestReportGenerator().generate(
        metrics={"sharpe": 1.0},
        equity_curve_path=equity_path,
        trades_path=trades_path,
        job_id="job1",
        run_id="run1",
        profile=ReportProfile.RESEARCH,
        strategy_id="rule",
        symbol="BTCUSDT",
        timeframe="15m",
    )
    assert isinstance(bt_report, Report)
    assert bt_report.sections and bt_report.sections[0].title
    assert bt_report.metrics or bt_report.metadata, "Backtest report should expose metrics or metadata"

    # Strategy search
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    # Minimal search results
    results_df = pd.DataFrame(
        {
            "sharpe": [1.0, 0.5],
            "cum_return": [0.1, 0.05],
            "status": ["ok", "ok"],
        }
    )
    results_path = job_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    (job_dir / "meta.json").write_text(json.dumps({"profile": "research", "strategy": "rule", "symbol": "BTCUSDT", "timeframe": "15m"}), encoding="utf-8")

    search_report = StrategySearchReportGenerator().generate(job_dir)
    assert isinstance(search_report, Report)
    assert search_report.sections
    assert search_report.title.lower().startswith("strategy search report")

    # Live report
    live_report = LiveReportGenerator().generate(
        snapshot={"equity": 11000, "symbol": "BTCUSDT", "timeframe": "15m", "run_id": "live1", "mode": "live"},
        trades=pd.DataFrame({"symbol": ["BTCUSDT"], "pnl": [1.0]}),
    )
    assert isinstance(live_report, Report)
    assert live_report.sections
    assert live_report.title

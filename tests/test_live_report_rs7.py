import json
from pathlib import Path

import pandas as pd

from finantradealgo.research.reporting.live_report import LiveReportGenerator
from finantradealgo.research.reporting.base import Report


def test_live_report_generator_builds_sections(tmp_path: Path):
    snapshot = {
        "run_id": "live_run_1",
        "mode": "paper",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "equity_now": 11000.0,
        "equity_start": 10000.0,
        "daily_pnl": 150.0,
        "max_intraday_dd": -0.03,
        "kill_switch_triggered": False,
        "validation_issues": ["latency high"],
        "last_bar_time": "2025-01-01T00:00:00Z",
        "heartbeat_age_sec": 5,
        "strategy": "rule",
    }
    trades_df = pd.DataFrame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "pnl": 10, "side": "long"},
            {"timestamp": "2025-01-01T00:15:00Z", "pnl": -5, "side": "short"},
        ]
    )

    generator = LiveReportGenerator()
    report = generator.generate(snapshot=snapshot, trades=trades_df)

    assert report.profile.value == "live"
    assert report.run_id == "live_run_1"
    assert "Overview" in [s.title for s in report.sections]
    assert "Risk" in [s.title for s in report.sections]
    assert "Recent Trades" in [s.title for s in report.sections]
    assert report.metrics["equity_now"] == 11000.0

    restored = Report.from_dict(report.to_dict())
    assert restored.run_id == "live_run_1"
    assert isinstance(restored.sections[-1].data["trades"], pd.DataFrame)


def test_live_report_generator_reads_files(tmp_path: Path):
    snapshot_path = tmp_path / "live.json"
    trades_path = tmp_path / "trades.csv"
    snapshot_path.write_text(
        json.dumps(
            {
                "run_id": "run_file",
                "symbol": "ETHUSDT",
                "timeframe": "1h",
                "equity": 5000,
                "equity_start": 4500,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"pnl": [1, -1]}).to_csv(trades_path, index=False)

    report = LiveReportGenerator().generate(snapshot_path, trades=trades_path, snapshot_path=snapshot_path)
    assert report.symbol == "ETHUSDT"
    assert report.timeframe == "1h"
    assert report.artifacts["snapshot_path"].endswith("live.json")

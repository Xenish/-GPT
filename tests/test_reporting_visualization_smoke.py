from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from finantradealgo.research.reporting.base import Report, ReportSection, ReportProfile
from finantradealgo.research.visualization.charts import create_chart, save_chart


def test_backtest_report_html_render(tmp_path: Path):
    section = ReportSection(
        title="Equity",
        content="Test content",
        metrics={"sharpe": 1.0},
        data={"equity_curve": pd.DataFrame({"timestamp": [1, 2, 3], "equity": [100, 101, 102]})},
    )
    report = Report(
        title="Backtest Report Smoke",
        description="",
        job_id="job1",
        run_id="run1",
        profile=ReportProfile.RESEARCH,
        strategy_id="rule",
        symbol="BTCUSDT",
        timeframe="15m",
        metrics={"final_equity": 102.0},
        sections=[section],
    )
    html = report.to_html()
    out_path = tmp_path / "backtest_report.html"
    out_path.write_text(html, encoding="utf-8")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert "Backtest Report Smoke" in out_path.read_text(encoding="utf-8")


def test_chart_render_smoke(tmp_path: Path):
    chart = create_chart(
        chart_type="line",
        data=pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}),
        config=None,
        title="Test Chart",
        x="x",
        y="y",
    )
    html_path = tmp_path / "chart.html"
    save_chart(chart, str(html_path), format="html")
    assert html_path.exists()
    assert html_path.stat().st_size > 0

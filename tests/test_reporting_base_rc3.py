from datetime import datetime, timezone

import pandas as pd

from finantradealgo.research.reporting.base import Report, ReportFormat, ReportProfile, ReportSection


def test_report_section_markdown_and_html():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    section = ReportSection(
        title="Section One",
        content="Content here",
        metrics={"rows": 2},
        artifacts={"table_csv": "table.csv"},
        data={"table": df, "stats": {"x": 1}},
    )

    md = section.to_markdown(level=2)
    assert "## Section One" in md
    assert "Content here" in md
    assert "table" in md
    assert "rows: 2" in md

    html = section.to_html(level=2)
    assert "<h2>Section One" in html
    assert "<table" in html
    assert "table.csv" in html


def test_report_to_dict_roundtrip():
    created_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({"a": [1, 2]})
    report = Report(
        title="Test Report",
        description="Desc",
        job_id="job-1",
        run_id="run-1",
        profile=ReportProfile.RESEARCH,
        strategy_id="rule",
        symbol="BTCUSDT",
        timeframe="1h",
        metrics={"sharpe": 1.2},
        artifacts={"equity_csv": "equity.csv"},
        created_at=created_at,
        sections=[
            ReportSection(
                title="Sec",
                content="Body",
                metrics={"rows": 2},
                artifacts={"foo": "bar"},
                data={"numbers": {"a": 1}, "frame": df},
                subsections=[ReportSection(title="Child", content="Child body")],
            )
        ],
        metadata={"k": "v"},
    )

    md = report.to_markdown()
    assert "# Test Report" in md
    assert "Metadata" in md
    assert "Context" in md

    d = report.to_dict()
    restored = Report.from_dict(d)
    assert restored.title == report.title
    assert restored.description == report.description
    assert restored.job_id == "job-1"
    assert restored.run_id == "run-1"
    assert restored.profile == ReportProfile.RESEARCH
    assert restored.strategy_id == "rule"
    assert restored.symbol == "BTCUSDT"
    assert restored.timeframe == "1h"
    assert restored.metrics["sharpe"] == 1.2
    assert restored.artifacts["equity_csv"] == "equity.csv"
    assert restored.metadata == report.metadata
    assert len(restored.sections) == 1
    assert restored.sections[0].subsections[0].title == "Child"
    # DataFrame roundtrip
    assert isinstance(restored.sections[0].data["frame"], pd.DataFrame)
    assert list(restored.sections[0].data["frame"]["a"]) == [1, 2]

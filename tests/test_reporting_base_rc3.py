from datetime import datetime, timezone

import pandas as pd

from finantradealgo.research.reporting.base import Report, ReportFormat, ReportSection


def test_report_section_markdown_and_html():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    section = ReportSection(
        title="Section One",
        content="Content here",
        data={"table": df, "stats": {"x": 1}},
    )

    md = section.to_markdown(level=2)
    assert "## Section One" in md
    assert "Content here" in md
    assert "table" in md

    html = section.to_html(level=2)
    assert "<h2>Section One" in html
    assert "<table" in html


def test_report_to_dict_roundtrip():
    created_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    report = Report(
        title="Test Report",
        description="Desc",
        created_at=created_at,
        sections=[
            ReportSection(
                title="Sec",
                content="Body",
                data={"numbers": {"a": 1}},
                subsections=[ReportSection(title="Child", content="Child body")],
            )
        ],
        metadata={"k": "v"},
    )

    md = report.to_markdown()
    assert "# Test Report" in md
    assert "Metadata" in md

    d = report.to_dict()
    restored = Report.from_dict(d)
    assert restored.title == report.title
    assert restored.description == report.description
    assert restored.metadata == report.metadata
    assert len(restored.sections) == 1
    assert restored.sections[0].subsections[0].title == "Child"

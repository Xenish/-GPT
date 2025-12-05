"""
Base Report Generation Infrastructure.

Provides core classes for creating research reports in various formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

MetricValue = Union[float, int, str]
Metrics = Dict[str, MetricValue]
Artifacts = Dict[str, str]


class ReportFormat(str, Enum):
    """Report output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class ReportProfile(str, Enum):
    """Report execution context."""

    RESEARCH = "research"
    LIVE = "live"


def _serialize_data_value(value: Any) -> Any:
    """Convert data payloads to JSON-friendly structures."""
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "columns": list(value.columns),
            "data": value.to_dict(orient="records"),
        }
    return value


def _deserialize_data_value(value: Any) -> Any:
    """Restore structured data from serialized payloads."""
    if isinstance(value, dict) and value.get("__type__") == "dataframe":
        return pd.DataFrame(value["data"], columns=value.get("columns"))
    return value


@dataclass
class ReportSection:
    """
    A section within a report.

    Attributes:
        title: Section title
        content: Section content (text/markdown)
        metrics: Key metrics for the section (JSON-friendly)
        artifacts: Links/paths to section-level artifacts (charts, tables, etc.)
        data: Structured data (tables, metrics)
        metadata: Additional metadata
        subsections: Nested subsections
    """

    title: str
    content: Optional[str] = ""
    metrics: Metrics = field(default_factory=dict)
    artifacts: Artifacts = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    subsections: List["ReportSection"] = field(default_factory=list)

    def to_markdown(self, level: int = 1) -> str:
        """
        Convert section to Markdown.

        Args:
            level: Heading level (1-6)

        Returns:
            Markdown-formatted string
        """
        lines = []

        # Title
        heading = "#" * min(level, 6)
        lines.append(f"{heading} {self.title}\n")

        # Content
        if self.content:
            lines.append(self.content)
            lines.append("")

        # Metrics
        if self.metrics:
            lines.append("**Metrics:**")
            for key, value in self.metrics.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        # Artifacts
        if self.artifacts:
            lines.append("**Artifacts:**")
            for key, value in self.artifacts.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        # Data tables
        if self.data:
            for key, value in self.data.items():
                if isinstance(value, pd.DataFrame):
                    lines.append(f"**{key}**:\n")
                    try:
                        lines.append(value.to_markdown())
                    except ImportError:
                        # tabulate may be missing; fall back to plain text
                        lines.append(value.to_string(index=False))
                    lines.append("")
                elif isinstance(value, dict):
                    lines.append(f"**{key}**:\n")
                    for k, v in value.items():
                        lines.append(f"- {k}: {v}")
                    lines.append("")

        # Subsections
        for subsection in self.subsections:
            lines.append(subsection.to_markdown(level + 1))

        return "\n".join(lines)

    def to_html(self, level: int = 1) -> str:
        """
        Convert section to HTML.

        Args:
            level: Heading level (1-6)

        Returns:
            HTML-formatted string
        """
        lines = []

        # Title
        h_level = min(level, 6)
        lines.append(f"<h{h_level}>{self.title}</h{h_level}>")

        # Content
        if self.content:
            content_html = self.content.replace("\n\n", "</p><p>")
            lines.append(f"<p>{content_html}</p>")

        # Metrics
        if self.metrics:
            lines.append(f"<h{h_level + 1}>Metrics</h{h_level + 1}>")
            lines.append("<ul>")
            for key, value in self.metrics.items():
                lines.append(f"<li><strong>{key}</strong>: {value}</li>")
            lines.append("</ul>")

        # Artifacts
        if self.artifacts:
            lines.append(f"<h{h_level + 1}>Artifacts</h{h_level + 1}>")
            lines.append("<ul>")
            for key, value in self.artifacts.items():
                lines.append(f"<li><strong>{key}</strong>: {value}</li>")
            lines.append("</ul>")

        # Data tables
        if self.data:
            for key, value in self.data.items():
                if isinstance(value, pd.DataFrame):
                    lines.append(f"<h{h_level + 1}>{key}</h{h_level + 1}>")
                    lines.append(value.to_html(index=False, border=1, classes="table table-striped"))
                elif isinstance(value, dict):
                    lines.append(f"<h{h_level + 1}>{key}</h{h_level + 1}>")
                    lines.append("<ul>")
                    for k, v in value.items():
                        lines.append(f"<li><strong>{k}</strong>: {v}</li>")
                    lines.append("</ul>")

        # Subsections
        for subsection in self.subsections:
            lines.append(subsection.to_html(level + 1))

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize section to dictionary (JSON-friendly)."""
        serialized_data = None
        if self.data is not None:
            serialized_data = {key: _serialize_data_value(value) for key, value in self.data.items()}
        return {
            "title": self.title,
            "content": self.content,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "data": serialized_data,
            "metadata": self.metadata,
            "subsections": [s.to_dict() for s in self.subsections],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportSection":
        """Deserialize section from dictionary."""
        subsections = [cls.from_dict(s) for s in data.get("subsections", []) or []]
        raw_data = data.get("data")
        parsed_data = None
        if raw_data is not None:
            parsed_data = {key: _deserialize_data_value(value) for key, value in raw_data.items()}
        return cls(
            title=data["title"],
            content=data.get("content"),
            metrics=data.get("metrics") or {},
            artifacts=data.get("artifacts") or {},
            data=parsed_data,
            metadata=data.get("metadata"),
            subsections=subsections,
        )


@dataclass
class Report:
    """
    Research report container.

    Attributes:
        title: Report title
        description: Report description
        job_id: Batch or orchestration job identifier
        run_id: Unique execution identifier (per run / live session)
        profile: Execution profile (research or live)
        strategy_id: Strategy identifier (rule, ml, trend_continuation, etc.)
        symbol: Trading symbol
        timeframe: Bar timeframe (e.g., 15m, 1h)
        metrics: Key metrics for the full report
        artifacts: Links/paths to artifacts (equity_csv, trades_csv, heatmap_html, etc.)
        created_at: Creation timestamp
        sections: Report sections
        metadata: Report metadata
    """

    title: str
    description: str = ""
    job_id: Optional[str] = None
    run_id: Optional[str] = None
    profile: Optional[ReportProfile] = None
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    metrics: Metrics = field(default_factory=dict)
    artifacts: Artifacts = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)

    def to_markdown(self) -> str:
        """
        Generate Markdown report.

        Returns:
            Full Markdown document
        """
        lines = []

        # Header
        lines.append(f"# {self.title}\n")

        if self.description:
            lines.append(f"{self.description}\n")

        context_items = [
            ("Profile", self.profile.value if isinstance(self.profile, ReportProfile) else self.profile),
            ("Job ID", self.job_id),
            ("Run ID", self.run_id),
            ("Strategy", self.strategy_id),
            ("Symbol", self.symbol),
            ("Timeframe", self.timeframe),
        ]
        context_items = [(k, v) for k, v in context_items if v is not None]
        if context_items:
            lines.append("**Context:**")
            for key, value in context_items:
                lines.append(f"- {key}: {value}")
            lines.append("")

        if self.metrics:
            lines.append("**Metrics:**")
            for key, value in self.metrics.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        if self.artifacts:
            lines.append("**Artifacts:**")
            for key, value in self.artifacts.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        lines.append(f"**Generated**: {self.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        lines.append("---\n")

        # Sections
        for section in self.sections:
            lines.append(section.to_markdown(level=2))
            lines.append("")

        # Metadata
        if self.metadata:
            lines.append("## Metadata\n")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def to_html(self, include_css: bool = True) -> str:
        """
        Generate HTML report.

        Args:
            include_css: Include basic CSS styling

        Returns:
            Full HTML document
        """
        lines = []

        # HTML header
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append(f"<title>{self.title}</title>")
        lines.append('<meta charset="UTF-8">')

        # CSS
        if include_css:
            lines.append("<style>")
            lines.append(
                """
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
                th { background-color: #3498db; color: white; font-weight: 600; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .metadata { background-color: #ecf0f1; padding: 15px; border-radius: 4px; margin-top: 20px; }
                .timestamp { color: #7f8c8d; font-size: 0.9em; }
                ul { margin: 10px 0; }
                li { margin: 5px 0; }
            """
            )
            lines.append("</style>")

        lines.append("</head>")
        lines.append("<body>")
        lines.append('<div class="container">')

        # Header
        lines.append(f"<h1>{self.title}</h1>")

        if self.description:
            lines.append(f"<p>{self.description}</p>")

        context_items = [
            ("Profile", self.profile.value if isinstance(self.profile, ReportProfile) else self.profile),
            ("Job ID", self.job_id),
            ("Run ID", self.run_id),
            ("Strategy", self.strategy_id),
            ("Symbol", self.symbol),
            ("Timeframe", self.timeframe),
        ]
        context_items = [(k, v) for k, v in context_items if v is not None]
        if context_items or self.metrics or self.artifacts:
            lines.append('<div class="metadata">')
            if context_items:
                lines.append("<h3>Context</h3>")
                lines.append("<ul>")
                for key, value in context_items:
                    lines.append(f"<li><strong>{key}</strong>: {value}</li>")
                lines.append("</ul>")
            if self.metrics:
                lines.append("<h3>Metrics</h3>")
                lines.append("<ul>")
                for key, value in self.metrics.items():
                    lines.append(f"<li><strong>{key}</strong>: {value}</li>")
                lines.append("</ul>")
            if self.artifacts:
                lines.append("<h3>Artifacts</h3>")
                lines.append("<ul>")
                for key, value in self.artifacts.items():
                    lines.append(f"<li><strong>{key}</strong>: {value}</li>")
                lines.append("</ul>")
            lines.append("</div>")

        lines.append(f'<p class="timestamp"><strong>Generated</strong>: {self.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>')
        lines.append("<hr>")

        # Sections
        for section in self.sections:
            lines.append(section.to_html(level=2))

        # Metadata
        if self.metadata:
            lines.append('<div class="metadata">')
            lines.append("<h3>Metadata</h3>")
            lines.append("<ul>")
            for key, value in self.metadata.items():
                lines.append(f"<li><strong>{key}</strong>: {value}</li>")
            lines.append("</ul>")
            lines.append("</div>")

        # Footer
        lines.append("</div>")
        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """
        Generate JSON representation.

        Returns:
            Dictionary representation of report
        """
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary (JSON-friendly)."""
        return {
            "title": self.title,
            "description": self.description,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "profile": self.profile.value if isinstance(self.profile, ReportProfile) else self.profile,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "created_at": self.created_at.isoformat(),
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Report":
        """Deserialize report from dictionary."""
        created_at_raw = data.get("created_at")
        if isinstance(created_at_raw, str):
            created_at = datetime.fromisoformat(created_at_raw)
        else:
            created_at = created_at_raw or datetime.now(timezone.utc)
        sections = [ReportSection.from_dict(s) for s in data.get("sections", []) or []]
        raw_profile = data.get("profile")
        profile = ReportProfile(raw_profile) if raw_profile in {p.value for p in ReportProfile} else raw_profile
        return cls(
            title=data["title"],
            description=data.get("description", ""),
            job_id=data.get("job_id"),
            run_id=data.get("run_id"),
            profile=profile,
            strategy_id=data.get("strategy_id"),
            symbol=data.get("symbol"),
            timeframe=data.get("timeframe"),
            metrics=data.get("metrics") or {},
            artifacts=data.get("artifacts") or {},
            created_at=created_at,
            sections=sections,
            metadata=data.get("metadata"),
        )

    def save(self, output_path: Path, format: ReportFormat = ReportFormat.HTML) -> None:
        """
        Save report to file.

        Args:
            output_path: Output file path
            format: Report format
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ReportFormat.MARKDOWN:
            content = self.to_markdown()
            suffix = ".md"
        elif format == ReportFormat.HTML:
            content = self.to_html()
            suffix = ".html"
        elif format == ReportFormat.JSON:
            import json

            content = json.dumps(self.to_json(), indent=2)
            suffix = ".json"
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Ensure correct suffix
        if not str(output_path).endswith(suffix):
            output_path = output_path.with_suffix(suffix)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(content)

        print(f"[PASS] Report saved to {output_path}")


class ReportGenerator(ABC):
    """
    Abstract base class for report generators.

    Subclasses implement specific report types (strategy search, ensemble, etc.)
    """

    @abstractmethod
    def generate(self, **kwargs) -> Report:
        """
        Generate report.

        Args:
            **kwargs: Report-specific arguments

        Returns:
            Generated report
        """
        pass

    def generate_and_save(
        self,
        output_path: Path,
        format: ReportFormat = ReportFormat.HTML,
        **kwargs,
    ) -> Report:
        """
        Generate report and save to file.

        Args:
            output_path: Output file path
            format: Report format
            **kwargs: Report-specific arguments

        Returns:
            Generated report
        """
        report = self.generate(**kwargs)
        report.save(output_path, format)
        return report


__all__ = [
    "Report",
    "ReportSection",
    "ReportFormat",
    "ReportProfile",
    "ReportGenerator",
]

"""
Base Report Generation Infrastructure.

Provides core classes for creating research reports in various formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ReportFormat(str, Enum):
    """Report output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


@dataclass
class ReportSection:
    """
    A section within a report.

    Attributes:
        title: Section title
        content: Section content (text/markdown)
        subsections: Nested subsections
        data: Structured data (tables, metrics)
        metadata: Additional metadata
    """

    title: str
    content: str = ""
    subsections: List[ReportSection] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

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

        # Data tables
        if self.data:
            for key, value in self.data.items():
                if isinstance(value, pd.DataFrame):
                    lines.append(f"**{key}**:\n")
                    lines.append(value.to_markdown())
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
            # Simple markdown-to-HTML conversion (basic)
            content_html = self.content.replace("\n\n", "</p><p>")
            lines.append(f"<p>{content_html}</p>")

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


@dataclass
class Report:
    """
    Research report container.

    Attributes:
        title: Report title
        description: Report description
        created_at: Creation timestamp
        sections: Report sections
        metadata: Report metadata
    """

    title: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
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
            lines.append("""
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
            """)
            lines.append("</style>")

        lines.append("</head>")
        lines.append("<body>")
        lines.append('<div class="container">')

        # Header
        lines.append(f"<h1>{self.title}</h1>")

        if self.description:
            lines.append(f"<p>{self.description}</p>")

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
        return {
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "data": section.data,
                    "metadata": section.metadata,
                }
                for section in self.sections
            ],
            "metadata": self.metadata,
        }

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

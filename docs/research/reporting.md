# Research Reporting System

Comprehensive guide to the FinanTradeAlgo research reporting infrastructure.

## Overview

The reporting system provides automated generation of professional HTML, Markdown, and JSON reports for research activities. It supports:

- **Strategy Parameter Search Reports**: Top performers, parameter sensitivity, performance distributions
- **Ensemble Strategy Reports**: Component comparison, weight evolution, bandit statistics
- **Customizable Formats**: HTML (styled), Markdown, JSON
- **API Integration**: Generate reports via REST endpoints

## Architecture

### Core Components

```
finantradealgo/research/reporting/
├── __init__.py                # Package exports
├── base.py                    # Core infrastructure (Report, ReportSection, ReportGenerator)
├── strategy_search.py         # Strategy search report generator
└── ensemble.py                # Ensemble report generator
```

### Report Structure

```python
Report
├── title: str
├── description: str
├── created_at: datetime
├── metadata: Dict[str, Any]
└── sections: List[ReportSection]
    └── ReportSection
        ├── title: str
        ├── content: str (markdown text)
        ├── data: Dict[str, DataFrame | Dict]
        └── subsections: List[ReportSection]
```

## Usage

### 1. Strategy Search Reports

Generate reports for parameter optimization jobs.

#### Via Python SDK

```python
from pathlib import Path
from finantradealgo.research.reporting import (
    StrategySearchReportGenerator,
    ReportFormat,
)

# Initialize generator
generator = StrategySearchReportGenerator()

# Generate report
report = generator.generate(
    job_dir=Path("outputs/strategy_search/job_20241130_123456"),
    job_id="job_20241130_123456",
    top_n=10,  # Highlight top 10 performers
)

# Save as HTML
report.save("reports/strategy_search/my_report.html", format=ReportFormat.HTML)

# Save as Markdown
report.save("reports/strategy_search/my_report.md", format=ReportFormat.MARKDOWN)

# Save as JSON
report.save("reports/strategy_search/my_report.json", format=ReportFormat.JSON)
```

#### Via REST API

```bash
# Generate report for job
curl -X POST http://localhost:8001/api/research/reports/strategy-search \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job_20241130_123456",
    "top_n": 10,
    "format": "html"
  }'

# Response
{
  "success": true,
  "report_id": "job_20241130_123456",
  "file_path": "reports/strategy_search/job_20241130_123456.html",
  "format": "html",
  "message": "Report generated successfully"
}
```

#### Via Python Requests

```python
import requests

response = requests.post(
    "http://localhost:8001/api/research/reports/strategy-search",
    json={
        "job_id": "job_20241130_123456",
        "top_n": 10,
        "format": "html",
    }
)

result = response.json()
print(f"Report saved to: {result['file_path']}")
```

### 2. Ensemble Reports

Generate reports for ensemble backtests.

#### Via Python SDK

```python
from finantradealgo.research.reporting import EnsembleReportGenerator
from finantradealgo.research.ensemble.backtest import run_ensemble_backtest

# Run ensemble backtest (see ensemble playbook)
result = run_ensemble_backtest(ensemble_strategy, df, components, sys_cfg)

# Generate report
generator = EnsembleReportGenerator()
report = generator.generate(
    backtest_result=result,
    ensemble_type="weighted",  # or "bandit"
    symbol="AIAUSDT",
    timeframe="15m",
    component_names=["rule", "trend_continuation", "sweep_reversal"],
)

# Save report
report.save("reports/ensemble/ensemble_AIAUSDT_15m.html")
```

### 3. Custom Reports

Create custom reports using the base infrastructure.

```python
from finantradealgo.research.reporting import Report, ReportSection
import pandas as pd

# Create report
report = Report(
    title="My Custom Analysis",
    description="Custom analysis of strategy XYZ",
    metadata={"strategy": "XYZ", "version": "1.0"},
)

# Add section with text
section1 = ReportSection(
    title="Introduction",
    content="This report analyzes strategy XYZ performance...",
)
report.add_section(section1)

# Add section with data
metrics_df = pd.DataFrame({
    "Metric": ["Sharpe", "Return", "Max DD"],
    "Value": [0.85, 0.42, -0.18],
})

section2 = ReportSection(
    title="Performance Metrics",
    content="Key performance indicators:",
    data={"Metrics Table": metrics_df},
)
report.add_section(section2)

# Save
report.save("reports/custom/my_analysis.html")
```

## Report Formats

### HTML Format

- **Professional styling** with embedded CSS
- **Responsive design** (works on mobile)
- **Styled tables** with alternating row colors
- **Section hierarchy** with proper heading levels
- **Best for**: Sharing with stakeholders, presentations, documentation

**Example Output**:
```html
<!DOCTYPE html>
<html>
<head>
  <title>Strategy Search Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', ...; }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; }
    table { border-collapse: collapse; width: 100%; }
    th { background-color: #3498db; color: white; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Strategy Search Report</h1>
    <!-- Report content -->
  </div>
</body>
</html>
```

### Markdown Format

- **Plain text** with markdown formatting
- **Easy to read** in text editors and GitHub
- **Version control friendly** (diffs work well)
- **Best for**: Documentation, version control, developer notes

**Example Output**:
```markdown
# Strategy Search Report

Generated: 2024-11-30 12:34:56 UTC

---

## Job Overview

This report summarizes the results of a parameter search for the **trend_continuation** strategy.

**Job Configuration**:
- Symbol: AIAUSDT
- Timeframe: 15m
```

### JSON Format

- **Structured data** for programmatic access
- **Machine-readable** format
- **Best for**: API integration, data pipelines, further processing

**Example Output**:
```json
{
  "title": "Strategy Search Report",
  "description": "Parameter search results",
  "created_at": "2024-11-30T12:34:56Z",
  "sections": [
    {
      "title": "Job Overview",
      "content": "This report summarizes...",
      "data": {...}
    }
  ],
  "metadata": {
    "job_id": "job_20241130_123456",
    "strategy": "trend_continuation"
  }
}
```

## API Endpoints

### POST `/api/research/reports/strategy-search`

Generate strategy search report.

**Request Body**:
```json
{
  "job_id": "job_20241130_123456",
  "top_n": 10,
  "format": "html"
}
```

**Response**:
```json
{
  "success": true,
  "report_id": "job_20241130_123456",
  "file_path": "reports/strategy_search/job_20241130_123456.html",
  "format": "html",
  "message": "Report generated successfully"
}
```

### GET `/api/research/reports/list`

List available reports.

**Query Parameters**:
- `report_type` (optional): Filter by type (`strategy_search`, `ensemble`)
- `limit` (optional): Max number of reports (default: 50)

**Response**:
```json
{
  "reports": [
    {
      "report_id": "job_20241130_123456",
      "title": "Job 20241130 123456",
      "created_at": "2024-11-30T12:34:56",
      "file_path": "reports/strategy_search/job_20241130_123456.html",
      "format": "html",
      "size_bytes": 45678
    }
  ],
  "total_count": 1
}
```

### GET `/api/research/reports/view/{report_type}/{report_id}`

View or download a report.

**Path Parameters**:
- `report_type`: Report type (`strategy_search`, `ensemble`)
- `report_id`: Report ID

**Query Parameters**:
- `format` (optional): Report format (`html`, `markdown`, `json`)

**Response**:
- HTML reports: Rendered directly in browser
- Other formats: File download

### DELETE `/api/research/reports/{report_type}/{report_id}`

Delete a report.

**Response**:
```json
{
  "success": true,
  "message": "Deleted 1 file(s)",
  "deleted_files": ["reports/strategy_search/job_20241130_123456.html"]
}
```

## Report Sections

### Strategy Search Report Sections

1. **Job Overview**
   - Job configuration (strategy, symbol, timeframe, search type)
   - Execution summary (total evaluations, success rate)
   - Performance summary statistics (mean, median, std dev of metrics)

2. **Top Performers**
   - Top N parameter sets ranked by Sharpe ratio
   - Key metrics for each (Sharpe, return, drawdown, trade count, win rate)
   - Parameter values for reproducibility

3. **Performance Distribution**
   - Quartile analysis for key metrics
   - Min, 25th, 50th (median), 75th, max percentiles
   - Helps understand full distribution, not just top performers

4. **Parameter Analysis**
   - Correlation between parameters and Sharpe ratio
   - Identifies which parameters have strongest impact
   - Helps guide future optimization

5. **Recommendations**
   - Best parameter set with full configuration
   - Next steps for validation
   - Suggestions for ensemble or further testing

### Ensemble Report Sections

1. **Overview**
   - Ensemble type (weighted or bandit)
   - Methodology explanation
   - List of component strategies

2. **Ensemble Performance**
   - Overall ensemble metrics (Sharpe, return, drawdown, etc.)
   - Key performance indicators

3. **Component Comparison**
   - Individual component performance
   - Comparison to ensemble (delta Sharpe)
   - Identifies which components contribute most

4. **Weight Evolution** (Weighted Ensembles)
   - Component weights over time
   - Final weight allocation
   - Percentage of contribution

5. **Bandit Statistics** (Bandit Ensembles)
   - Arm selection counts
   - Mean rewards per arm
   - Selection percentages
   - Exploration vs exploitation balance

6. **Recommendations**
   - Ensemble vs best component comparison
   - Optimization suggestions (reweighting, parameter tuning)
   - Next steps for validation and deployment

## Best Practices

### 1. Report Organization

```
reports/
├── strategy_search/
│   ├── job_20241130_123456.html
│   ├── job_20241130_123456.md
│   └── job_20241201_143021.html
├── ensemble/
│   ├── ensemble_AIAUSDT_15m_weighted.html
│   └── ensemble_BTCUSDT_1h_bandit.html
└── custom/
    └── my_analysis.html
```

### 2. Naming Conventions

- **Strategy Search**: `{job_id}.{format}`
- **Ensemble**: `ensemble_{symbol}_{timeframe}_{type}.{format}`
- **Custom**: Descriptive name with underscores

### 3. Report Retention

- Keep reports in version control for reproducibility
- Archive old reports periodically (e.g., after 90 days)
- Include git SHA in report metadata for code versioning

### 4. Metadata Tracking

Always include in report metadata:
- `job_id` or unique identifier
- `strategy` name
- `symbol` and `timeframe`
- `created_at` timestamp
- `git_sha` (code version)
- `n_samples` or `n_components`

Example:
```python
report = Report(
    title="My Report",
    metadata={
        "job_id": "job_20241130_123456",
        "strategy": "trend_continuation",
        "symbol": "AIAUSDT",
        "timeframe": "15m",
        "git_sha": "abc123def",
        "n_samples": 100,
    }
)
```

### 5. Sharing Reports

- **HTML**: Best for email, Slack, presentations
- **Markdown**: Best for GitHub, documentation sites
- **JSON**: Best for programmatic processing, dashboards

## Customization

### Custom Report Generator

Create your own report generator by subclassing `ReportGenerator`:

```python
from finantradealgo.research.reporting import ReportGenerator, Report, ReportSection

class MyCustomReportGenerator(ReportGenerator):
    """Generate custom reports."""

    def generate(self, data, **kwargs) -> Report:
        """
        Generate custom report.

        Args:
            data: Input data
            **kwargs: Additional arguments

        Returns:
            Generated report
        """
        # Create report
        report = Report(
            title="My Custom Report",
            description="Custom analysis",
        )

        # Add sections
        section = ReportSection(
            title="Analysis",
            content="...",
            data={"Table": data},
        )
        report.add_section(section)

        return report

# Use it
generator = MyCustomReportGenerator()
report = generator.generate(my_data)
report.save("reports/custom/my_report.html")
```

### Custom Section Formatting

Override `to_markdown()` or `to_html()` for custom formatting:

```python
from finantradealgo.research.reporting import ReportSection

class CustomSection(ReportSection):
    """Custom formatted section."""

    def to_html(self, level: int = 1) -> str:
        """Custom HTML rendering."""
        html = f"<div class='custom-section'>"
        html += f"<h{level}>{self.title}</h{level}>"
        html += f"<p class='custom-content'>{self.content}</p>"
        html += "</div>"
        return html
```

### Custom CSS Styling

Modify CSS in `base.py` `Report.to_html()` method or inject custom CSS:

```python
report.save("my_report.html")

# Post-process to inject custom CSS
with open("my_report.html", "r") as f:
    html = f.read()

custom_css = """
<style>
  .custom-section { background-color: #f0f0f0; padding: 20px; }
</style>
"""

html = html.replace("</head>", f"{custom_css}</head>")

with open("my_report.html", "w") as f:
    f.write(html)
```

## Troubleshooting

### Report Generation Fails

**Problem**: `FileNotFoundError: No results file found in {job_dir}`

**Solution**:
- Verify job ID is correct
- Check that job completed successfully
- Ensure `results.parquet` or `results.csv` exists in job directory

**Problem**: `KeyError: 'sharpe'`

**Solution**:
- Ensure backtest results include required metrics
- Check that results DataFrame has expected columns

### Report Not Found via API

**Problem**: `404: Report not found`

**Solution**:
- Verify report was generated (check `reports/` directory)
- Confirm report type and ID are correct
- Check file extensions match format requested

### HTML Report Styling Issues

**Problem**: Report HTML looks unstyled

**Solution**:
- Ensure `include_css=True` when calling `to_html()`
- Check browser console for errors
- Verify HTML file contains `<style>` tag

## Examples

See [examples.md](examples.md) for comprehensive usage examples.

## Related Documentation

- [Research Playbooks](../../finantradealgo/research/playbooks/README.md)
- [Strategy Parameter Search Playbook](../../finantradealgo/research/playbooks/strategy_param_search.md)
- [Ensemble Development Playbook](../../finantradealgo/research/playbooks/ensemble_development.md)
- [Research Service API Documentation](../api/research_service.md)

## Support

For issues or questions:
- Check troubleshooting section above
- Review playbooks for workflow guidance
- See code examples in `tests/test_reporting.py`
- Open an issue on GitHub

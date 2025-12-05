# Research Reporting System

Comprehensive guide to the FinanTradeAlgo research reporting infrastructure.

## Overview

The reporting system provides automated generation of professional HTML, Markdown, and JSON reports for research, live, ensemble, and backtest workflows. All outputs share the same unified `Report` contract so strategy search, ensemble, backtest, live, and portfolio flows can plug into a single model.

## Architecture

### Core Components

```
finantradealgo/research/reporting/
|-- __init__.py            # Package exports
|-- base.py                # Core infrastructure (Report, ReportSection, ReportGenerator)
|-- strategy_search.py     # Strategy search report generator
`-- ensemble.py            # Ensemble report generator
```

### Report Structure

Report:
- title: str
- description: str
- job_id: str | None
- run_id: str | None
- profile: "research" | "live" | None
- strategy_id: str | None (e.g., "rule", "ml", "trend_continuation")
- symbol: str | None
- timeframe: str | None
- metrics: Dict[str, float | int | str]
- artifacts: Dict[str, str] (equity_csv, trades_csv, heatmap_html, etc.)
- created_at: datetime
- metadata: Dict[str, Any] | None
- sections: List[ReportSection]

ReportSection:
- title: str
- content: str (markdown text)
- metrics: Dict[str, float | int | str]
- artifacts: Dict[str, str]
- data: Dict[str, DataFrame | Dict] | None
- metadata: Dict[str, Any] | None
- subsections: List[ReportSection]

Both `Report.to_dict()` and `ReportSection.to_dict()` emit JSON-friendly structures (datetimes are ISO strings, enums are values, dataframes become lists of records) and `from_dict()` restores them.

## Usage

### 1. Strategy Search Reports

Generate reports for parameter optimization jobs.

#### Via Python SDK

```python
from pathlib import Path
from finantradealgo.research.reporting import (
    StrategySearchReportGenerator,
    ReportFormat,
    ReportProfile,
)

generator = StrategySearchReportGenerator()

report = generator.generate(
    job_dir=Path("outputs/strategy_search/job_20241130_123456"),
    job_id="job_20241130_123456",
    run_id="run_001",
    profile=ReportProfile.RESEARCH,
    strategy_id="trend_continuation",
    symbol="AIAUSDT",
    timeframe="15m",
    top_n=10,
)

report.save("reports/strategy_search/my_report.html", format=ReportFormat.HTML)
report.save("reports/strategy_search/my_report.md", format=ReportFormat.MARKDOWN)
report.save("reports/strategy_search/my_report.json", format=ReportFormat.JSON)
```

#### Via REST API

```bash
curl -X POST http://localhost:8001/api/research/reports/strategy-search \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "job_20241130_123456",
    "run_id": "run_001",
    "profile": "research",
    "strategy_id": "trend_continuation",
    "symbol": "AIAUSDT",
    "timeframe": "15m",
    "top_n": 10,
    "format": "html"
  }'
```

Response (abridged):
```json
{
  "success": true,
  "report_id": "job_20241130_123456",
  "file_path": "reports/strategy_search/job_20241130_123456.html",
  "format": "html",
  "message": "Report generated successfully"
}
```

### 2. Ensemble Reports

Generate reports for ensemble backtests.

```python
from finantradealgo.research.reporting import EnsembleReportGenerator, ReportProfile
from finantradealgo.research.ensemble.backtest import run_ensemble_backtest

result = run_ensemble_backtest(ensemble_strategy, df, components, sys_cfg)

report = EnsembleReportGenerator().generate(
    backtest_result=result,
    ensemble_type="weighted",
    strategy_id="ensemble_v1",
    profile=ReportProfile.RESEARCH,
    symbol="AIAUSDT",
    timeframe="15m",
    run_id="run_ensemble_42",
)

report.save("reports/ensemble/ensemble_AIAUSDT_15m.html")
```

### 3. Custom Reports

Create custom reports using the base infrastructure with the unified fields.

```python
from finantradealgo.research.reporting import Report, ReportSection, ReportProfile
import pandas as pd

report = Report(
    title="My Custom Analysis",
    description="Custom analysis of strategy XYZ",
    job_id="job_20241130_123456",
    run_id="run_custom_01",
    profile=ReportProfile.RESEARCH,
    strategy_id="rule",
    symbol="BTCUSDT",
    timeframe="1h",
    metrics={"sharpe": 1.2, "return": 0.35},
    artifacts={"equity_csv": "reports/custom/equity.csv"},
)

section1 = ReportSection(
    title="Introduction",
    content="This report analyzes strategy XYZ performance...",
)
report.add_section(section1)

metrics_df = pd.DataFrame({
    "Metric": ["Sharpe", "Return", "Max DD"],
    "Value": [0.85, 0.42, -0.18],
})

section2 = ReportSection(
    title="Performance Metrics",
    content="Key performance indicators:",
    metrics={"trades": 124, "win_rate": "54%"},
    data={"Metrics Table": metrics_df},
    artifacts={"heatmap_html": "reports/custom/heatmap.html"},
)
report.add_section(section2)

report.save("reports/custom/my_analysis.html")
```

## Report Formats

### HTML Format

- Professional styling with embedded CSS
- Responsive design (works on mobile)
- Context box for job/run/profile/strategy/symbol/timeframe
- Section-level metrics, artifacts, and tables
- Best for sharing with stakeholders

Example (truncated):
```html
<!DOCTYPE html>
<html>
<head>
  <title>Strategy Search Report</title>
  <style>...</style>
</head>
<body>
  <div class="container">
    <h1>Strategy Search Report</h1>
    <div class="metadata">
      <h3>Context</h3>
      <ul>
        <li><strong>Profile</strong>: research</li>
        <li><strong>Job ID</strong>: job_20241130_123456</li>
        <li><strong>Run ID</strong>: run_001</li>
        <li><strong>Strategy</strong>: trend_continuation</li>
        <li><strong>Symbol</strong>: AIAUSDT</li>
        <li><strong>Timeframe</strong>: 15m</li>
      </ul>
      <h3>Metrics</h3>
      <ul>
        <li><strong>sharpe</strong>: 1.24</li>
        <li><strong>return</strong>: 0.42</li>
      </ul>
    </div>
    <!-- Sections ... -->
  </div>
</body>
</html>
```

### Markdown Format

- Plain text with markdown formatting
- Context, metrics, and artifacts rendered as bullet lists
- Version control friendly

Example:
```markdown
# Strategy Search Report

Parameter search results

**Context:**
- Profile: research
- Job ID: job_20241130_123456
- Run ID: run_001
- Strategy: trend_continuation
- Symbol: AIAUSDT
- Timeframe: 15m

**Metrics:**
- sharpe: 1.24
- return: 0.42

**Artifacts:**
- equity_csv: reports/strategy_search/job_20241130_123456/equity.csv
- trades_csv: reports/strategy_search/job_20241130_123456/trades.csv

**Generated**: 2024-11-30 12:34:56 UTC

---

## Job Overview
This report summarizes the results of a parameter search for the trend_continuation strategy.
```

### JSON Format

- Structured data for programmatic access
- DataFrames serialized to `{"__type__": "dataframe", "columns": [...], "data": [...]}` and restored via `from_dict()`

Example:
```json
{
  "title": "Strategy Search Report",
  "description": "Parameter search results",
  "job_id": "job_20241130_123456",
  "run_id": "run_001",
  "profile": "research",
  "strategy_id": "trend_continuation",
  "symbol": "AIAUSDT",
  "timeframe": "15m",
  "metrics": {
    "sharpe": 1.24,
    "return": 0.42
  },
  "artifacts": {
    "equity_csv": "reports/strategy_search/job_20241130_123456/equity.csv",
    "trades_csv": "reports/strategy_search/job_20241130_123456/trades.csv"
  },
  "created_at": "2024-11-30T12:34:56Z",
  "sections": [
    {
      "title": "Job Overview",
      "content": "This report summarizes...",
      "metrics": {
        "evaluations": 500,
        "success_rate": 0.98
      },
      "artifacts": {},
      "data": null,
      "metadata": null,
      "subsections": []
    }
  ],
  "metadata": {
    "git_sha": "abc123def"
  }
}
```

**Strategy Search JSON Example**
```json
{
  "title": "Strategy Search Report: job_20241130_123456",
  "description": "Parameter search results for trend_continuation strategy on AIAUSDT/15m",
  "job_id": "job_20241130_123456",
  "run_id": "run_001",
  "profile": "research",
  "strategy_id": "trend_continuation",
  "symbol": "AIAUSDT",
  "timeframe": "15m",
  "metrics": {
    "best_sharpe": 1.42,
    "best_cum_return": 0.37,
    "samples_total": 500,
    "samples_ok": 480,
    "samples_failed": 20
  },
  "artifacts": {
    "results_parquet": "outputs/strategy_search/job_20241130_123456/results.parquet",
    "results_csv": "outputs/strategy_search/job_20241130_123456/results.csv",
    "meta_json": "outputs/strategy_search/job_20241130_123456/meta.json",
    "heatmap_html": "outputs/strategy_search/job_20241130_123456/param_heatmap.html"
  },
  "created_at": "2024-11-30T12:34:56Z",
  "sections": [
    {
      "title": "Job Overview",
      "content": "This report summarizes...",
      "metrics": {"evaluations": 500, "success_rate": 96.0},
      "artifacts": {},
      "data": null,
      "metadata": null,
      "subsections": []
    }
  ],
  "metadata": {
    "n_samples": 500,
    "search_type": "random",
    "git_sha": "abc123def"
  }
}
```

**Ensemble JSON Example**
```json
{
  "title": "Ensemble Strategy Report: Weighted",
  "description": "Weighted ensemble backtest for BTCUSDT/15m",
  "job_id": null,
  "run_id": "ensemble_run_01",
  "profile": "research",
  "strategy_id": "ensemble_v1",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "metrics": {
    "sharpe": 1.05,
    "cum_return": 0.22,
    "max_drawdown": -0.09,
    "trade_count": 320,
    "win_rate": 0.53
  },
  "artifacts": {},
  "created_at": "2024-11-30T12:34:56Z",
  "sections": [
    {
      "title": "Overview",
      "content": "This report analyzes a weighted ensemble strategy...",
      "metrics": {},
      "artifacts": {},
      "data": null,
      "metadata": null,
      "subsections": []
    }
  ],
  "metadata": {
    "ensemble_type": "weighted",
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "n_components": 3,
    "component_names": ["rule", "ml", "trend_continuation"]
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
  "run_id": "run_001",
  "profile": "research",
  "strategy_id": "trend_continuation",
  "symbol": "AIAUSDT",
  "timeframe": "15m",
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

1. Job Overview
   - Job configuration (strategy, symbol, timeframe, search type)
   - Execution summary (total evaluations, success rate)
   - Performance summary statistics (mean, median, std dev of metrics)
2. Top Performers
   - Top N parameter sets ranked by Sharpe ratio
   - Key metrics for each (Sharpe, return, drawdown, trade count, win rate)
   - Parameter values for reproducibility
3. Performance Distribution
   - Quartile analysis for key metrics
   - Min, 25th, 50th (median), 75th, max percentiles
4. Parameter Analysis
   - Correlation between parameters and Sharpe ratio
   - Identifies which parameters have strongest impact
5. Recommendations
   - Best parameter set with full configuration
   - Next steps for validation
   - Suggestions for ensemble or further testing

### Ensemble Report Sections

1. Overview
   - Ensemble type (weighted or bandit)
   - Methodology explanation
   - List of component strategies
2. Ensemble Performance
   - Overall ensemble metrics (Sharpe, return, drawdown, etc.)
   - Key performance indicators
3. Component Comparison
   - Individual component performance
   - Comparison to ensemble (delta Sharpe)
4. Weight Evolution (Weighted Ensembles)
   - Component weights over time
   - Final weight allocation
5. Bandit Statistics (Bandit Ensembles)
   - Arm selection counts
   - Mean rewards per arm
   - Selection percentages
6. Recommendations
   - Ensemble vs best component comparison
   - Optimization suggestions (reweighting, parameter tuning)
   - Next steps for validation and deployment

## Best Practices

### 1. Report Organization

```
reports/
|-- strategy_search/
|   |-- job_20241130_123456.html
|   |-- job_20241130_123456.md
|   `-- job_20241201_143021.html
|-- ensemble/
|   |-- ensemble_AIAUSDT_15m_weighted.html
|   `-- ensemble_BTCUSDT_1h_bandit.html
`-- custom/
    `-- my_analysis.html
```

### 2. Naming Conventions

- Strategy Search: `{job_id}.{format}`
- Ensemble: `ensemble_{symbol}_{timeframe}_{type}.{format}`
- Custom: Descriptive name with underscores

### 3. Report Retention

- Keep reports in version control for reproducibility
- Archive old reports periodically (e.g., after 90 days)
- Include git SHA in report metadata for code versioning

### 4. Metadata Tracking

Use the dedicated fields instead of stuffing metadata:
- `job_id` and/or `run_id`
- `profile` ("research" or "live")
- `strategy_id`
- `symbol`, `timeframe`
- `metrics` (key performance metrics)
- `artifacts` (paths or URLs to outputs)

Optional extras can still live under `metadata` (e.g., `git_sha`, `n_samples`).

Example:
```python
report = Report(
    title="My Report",
    job_id="job_20241130_123456",
    run_id="run_001",
    profile=ReportProfile.RESEARCH,
    strategy_id="trend_continuation",
    symbol="AIAUSDT",
    timeframe="15m",
    metrics={"sharpe": 1.1, "return": 0.28},
    artifacts={"equity_csv": "reports/.../equity.csv"},
    metadata={"git_sha": "abc123def", "n_samples": 100},
)
```

### 5. Sharing Reports

- HTML: Best for email, Slack, presentations
- Markdown: Best for GitHub, documentation sites
- JSON: Best for programmatic processing, dashboards

## Customization

### Custom Report Generator

Create your own report generator by subclassing `ReportGenerator`:

```python
from finantradealgo.research.reporting import ReportGenerator, Report, ReportSection, ReportProfile

class MyCustomReportGenerator(ReportGenerator):
    """Generate custom reports."""

    def generate(self, data, **kwargs) -> Report:
        report = Report(
            title="My Custom Report",
            description="Custom analysis",
            profile=ReportProfile.RESEARCH,
            strategy_id="custom_strategy",
        )

        section = ReportSection(
            title="Analysis",
            content="...",
            metrics={"sharpe": 0.9},
            artifacts={"chart_html": "reports/custom/chart.html"},
            data={"table": data},
        )
        report.add_section(section)

        return report

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

## Extended Reports

- **Portfolio API**: Responses now include `metrics` (e.g., `sharpe_ratio`, `volatility`, `max_drawdown`, `risk_parity_weight`) and `sections` with `title="Portfolio Overview"` to align with the unified report contract shape.
- **Monte Carlo API**: `/montecarlo/run` adds `metrics` with `median_return`, `p5_return`, `p95_return`, and `worst_case_dd` alongside the summary/risk payload.
- **Scenarios API**: Scenario results return `scenario_id`, `description`, and a `metrics` dict (`strategy`, `symbol`, `timeframe`, `params`, `cum_return`, `sharpe`, `max_drawdown`, `trade_count`) for a consistent contract-like structure.

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

# Research Reporting Examples

Practical examples for generating and using research reports.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Strategy Search Reports](#strategy-search-reports)
3. [Ensemble Reports](#ensemble-reports)
4. [Custom Reports](#custom-reports)
5. [API Integration](#api-integration)
6. [Advanced Workflows](#advanced-workflows)

---

## Quick Start

### Example 1: Generate Report After Parameter Search

```python
from finantradealgo.research.strategy_search import StrategySearchJob
from finantradealgo.research.reporting import StrategySearchReportGenerator

# Run parameter search
job = StrategySearchJob(
    strategy_name="trend_continuation",
    symbol="AIAUSDT",
    timeframe="15m",
    param_space={
        "trend_min": [0.5, 1.0, 1.5, 2.0],
        "atr_period": [10, 14, 20],
    },
    search_type="grid",
)

result = job.run()
print(f"Job ID: {result.job_id}")

# Generate report
generator = StrategySearchReportGenerator()
report = generator.generate(
    job_dir=result.job_dir,
    job_id=result.job_id,
    top_n=5,
)

# Save as HTML
report.save(f"reports/strategy_search/{result.job_id}.html")

print(f"Report saved! Open: reports/strategy_search/{result.job_id}.html")
```

**Output**:
```
Job ID: job_20241130_123456
[PASS] Report saved to reports/strategy_search/job_20241130_123456.html
Report saved! Open: reports/strategy_search/job_20241130_123456.html
```

---

## Strategy Search Reports

### Example 2: Analyze Top Performers

```python
from pathlib import Path
from finantradealgo.research.reporting import StrategySearchReportGenerator
import pandas as pd

# Generate report
generator = StrategySearchReportGenerator()
job_id = "job_20241130_123456"

report = generator.generate(
    job_dir=Path(f"outputs/strategy_search/{job_id}"),
    job_id=job_id,
    top_n=10,
)

# Save in multiple formats
report.save(f"reports/strategy_search/{job_id}.html")  # For sharing
report.save(f"reports/strategy_search/{job_id}.md")    # For docs
report.save(f"reports/strategy_search/{job_id}.json")  # For processing

# Extract data for further analysis
import json
with open(f"reports/strategy_search/{job_id}.json", "r") as f:
    report_data = json.load(f)

# Access top performers from JSON
for section in report_data["sections"]:
    if section["title"] == "Top Performers":
        print("Top 10 Sharpe Ratios:")
        # Process section data
        break
```

### Example 3: Compare Multiple Search Jobs

```python
from finantradealgo.research.reporting import StrategySearchReportGenerator, Report, ReportSection
import pandas as pd

# Load results from multiple jobs
job_ids = ["job_20241130_123456", "job_20241130_143021", "job_20241130_163045"]

comparison_data = []

for job_id in job_ids:
    # Load results
    results_path = Path(f"outputs/strategy_search/{job_id}/results.parquet")
    df = pd.read_parquet(results_path)

    # Get best result
    best = df.nlargest(1, "sharpe").iloc[0]

    comparison_data.append({
        "job_id": job_id,
        "best_sharpe": best["sharpe"],
        "best_return": best["cum_return"],
        "best_dd": best["max_drawdown"],
        "best_params": {k: best[k] for k in df.columns if k.startswith("param_")},
    })

# Create comparison report
comparison_df = pd.DataFrame(comparison_data)

report = Report(
    title="Multi-Job Comparison",
    description="Comparison of 3 parameter search jobs",
)

section = ReportSection(
    title="Best Performers Across Jobs",
    content="This table shows the best parameter set from each job.",
    data={"Comparison": comparison_df[["job_id", "best_sharpe", "best_return", "best_dd"]]},
)
report.add_section(section)

report.save("reports/custom/job_comparison.html")
```

### Example 4: Parameter Sensitivity Analysis

```python
from finantradealgo.research.reporting import Report, ReportSection
import pandas as pd
import matplotlib.pyplot as plt

# Load results
job_id = "job_20241130_123456"
results_df = pd.read_parquet(f"outputs/strategy_search/{job_id}/results.parquet")

# Calculate parameter correlations
param_cols = [col for col in results_df.columns if col.startswith("param_")]

correlations = []
for param in param_cols:
    param_values = pd.to_numeric(results_df[param], errors='coerce')
    sharpe_values = results_df["sharpe"]

    mask = ~(param_values.isna() | sharpe_values.isna())
    if mask.sum() > 1:
        corr = param_values[mask].corr(sharpe_values[mask])
        correlations.append({
            "Parameter": param.replace("param_", ""),
            "Correlation": corr,
        })

corr_df = pd.DataFrame(correlations).sort_values("Correlation", ascending=False)

# Create visualization
plt.figure(figsize=(10, 6))
plt.barh(corr_df["Parameter"], corr_df["Correlation"])
plt.xlabel("Correlation with Sharpe Ratio")
plt.ylabel("Parameter")
plt.title("Parameter Sensitivity Analysis")
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("param_sensitivity.png", dpi=150)
plt.close()

# Create report
report = Report(
    title="Parameter Sensitivity Report",
    description=f"Analysis of parameter impact on Sharpe ratio for {job_id}",
)

section = ReportSection(
    title="Parameter Correlations",
    content="""
This analysis shows which parameters have the strongest impact on strategy performance.

**Interpretation**:
- Positive correlation: Higher parameter value → Higher Sharpe
- Negative correlation: Lower parameter value → Higher Sharpe
- Near zero: Parameter has little impact

![Parameter Sensitivity](param_sensitivity.png)
""",
    data={"Correlations": corr_df},
)
report.add_section(section)

report.save("reports/custom/param_sensitivity.html")
```

---

## Ensemble Reports

### Example 5: Generate Ensemble Report

```python
from finantradealgo.data.ohlcv_loader import load_ohlcv
from finantradealgo.config import load_config
from finantradealgo.research.ensemble import (
    WeightedEnsembleStrategy,
    WeightedEnsembleConfig,
    ComponentStrategy,
    WeightingMethod,
)
from finantradealgo.research.ensemble.backtest import run_ensemble_backtest
from finantradealgo.research.reporting import EnsembleReportGenerator

# Load data
df = load_ohlcv("AIAUSDT", "15m")
sys_cfg = load_config("config/research_config.yaml")

# Define components
components = [
    ComponentStrategy("rule", label="rule"),
    ComponentStrategy("trend_continuation", params={"trend_min": 1.0}, label="trend"),
    ComponentStrategy("sweep_reversal", label="sweep"),
]

# Create weighted ensemble
config = WeightedEnsembleConfig(
    components=components,
    weighting_method=WeightingMethod.SHARPE,
    reweight_period=50,
    signal_threshold=0.5,
)

ensemble = WeightedEnsembleStrategy(config)

# Run backtest
result = run_ensemble_backtest(
    ensemble_strategy=ensemble,
    df=df,
    components=components,
    sys_cfg=sys_cfg,
)

# Generate report
generator = EnsembleReportGenerator()
report = generator.generate(
    backtest_result=result,
    ensemble_type="weighted",
    symbol="AIAUSDT",
    timeframe="15m",
    component_names=["rule", "trend", "sweep"],
)

# Save report
report.save("reports/ensemble/weighted_AIAUSDT_15m.html")

print(f"Ensemble Sharpe: {result.ensemble_metrics['sharpe']:.4f}")
print(f"Report saved to: reports/ensemble/weighted_AIAUSDT_15m.html")
```

### Example 6: Compare Weighted vs Bandit Ensembles

```python
from finantradealgo.research.ensemble import (
    BanditEnsembleStrategy,
    BanditEnsembleConfig,
    BanditAlgorithm,
)
from finantradealgo.research.reporting import Report, ReportSection
import pandas as pd

# (Assume df, sys_cfg, components already loaded from Example 5)

# Run weighted ensemble
weighted_config = WeightedEnsembleConfig(
    components=components,
    weighting_method=WeightingMethod.EQUAL,
)
weighted_ensemble = WeightedEnsembleStrategy(weighted_config)
weighted_result = run_ensemble_backtest(weighted_ensemble, df, components, sys_cfg)

# Run bandit ensemble
bandit_config = BanditEnsembleConfig(
    components=components,
    bandit_algorithm=BanditAlgorithm.UCB1,
    update_period=20,
)
bandit_ensemble = BanditEnsembleStrategy(bandit_config)
bandit_result = run_ensemble_backtest(bandit_ensemble, df, components, sys_cfg)

# Compare results
comparison_df = pd.DataFrame({
    "Weighted": weighted_result.ensemble_metrics,
    "Bandit (UCB1)": bandit_result.ensemble_metrics,
}).T

# Create comparison report
report = Report(
    title="Ensemble Type Comparison",
    description="Weighted vs Bandit ensemble on AIAUSDT/15m",
)

section = ReportSection(
    title="Performance Comparison",
    content="""
Comparison of two ensemble approaches:

**Weighted Ensemble** (Equal Weights):
- Combines all component signals with equal weighting
- Simple, stable, diversified

**Bandit Ensemble** (UCB1):
- Selects best component dynamically
- Adapts to changing market conditions
- Exploration-exploitation tradeoff

**Result**: The ensemble with higher Sharpe ratio is recommended.
""",
    data={"Ensemble Comparison": comparison_df},
)
report.add_section(section)

report.save("reports/ensemble/comparison_AIAUSDT_15m.html")

print(comparison_df)
```

**Output**:
```
                sharpe  cum_return  max_dd  trade_count  win_rate
Weighted        0.6234      0.3421 -0.1823          145    0.5517
Bandit (UCB1)   0.7891      0.4512 -0.1456          132    0.5985
```

---

## Custom Reports

### Example 7: Multi-Strategy Comparison Report

```python
from finantradealgo.strategies.strategy_engine import create_strategy
from finantradealgo.backtesting.backtest_engine import BacktestEngine
from finantradealgo.research.reporting import Report, ReportSection
import pandas as pd

# (Assume df, sys_cfg already loaded)

strategies = [
    {"name": "rule", "label": "Rule-Based"},
    {"name": "trend_continuation", "label": "Trend Following"},
    {"name": "sweep_reversal", "label": "Sweep Reversal"},
]

results = []

for strat_config in strategies:
    strategy = create_strategy(strat_config["name"], sys_cfg)
    engine = BacktestEngine(strategy, sys_cfg)
    result = engine.run(df.copy())

    results.append({
        "Strategy": strat_config["label"],
        "Sharpe": result.metrics.sharpe,
        "Return": result.metrics.cum_return,
        "Max DD": result.metrics.max_drawdown,
        "Trades": result.metrics.trade_count,
        "Win Rate": result.metrics.win_rate,
    })

results_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)

# Create report
report = Report(
    title="Multi-Strategy Comparison",
    description="Performance comparison of 3 strategies on AIAUSDT/15m",
)

# Section 1: Rankings
rankings_section = ReportSection(
    title="Performance Rankings",
    content="Strategies ranked by Sharpe ratio.",
    data={"Rankings": results_df},
)
report.add_section(rankings_section)

# Section 2: Analysis
best_strategy = results_df.iloc[0]["Strategy"]
best_sharpe = results_df.iloc[0]["Sharpe"]

analysis_content = f"""
**Winner**: {best_strategy} with Sharpe ratio of {best_sharpe:.4f}

**Key Findings**:
- All strategies have positive Sharpe ratios
- {best_strategy} outperforms other strategies in risk-adjusted returns
- Win rates are relatively consistent across strategies (50-60%)

**Recommendation**:
- Deploy {best_strategy} for live trading after robustness testing
- Consider ensemble combining top 2-3 strategies for diversification
"""

analysis_section = ReportSection(
    title="Analysis & Recommendations",
    content=analysis_content.strip(),
)
report.add_section(analysis_section)

report.save("reports/custom/strategy_comparison.html")
```

### Example 8: Regime Analysis Report

```python
from finantradealgo.research.reporting import Report, ReportSection
import pandas as pd
import numpy as np

# (Assume df with regime classification already exists)
# See regime_analysis playbook for regime classification methods

def calculate_regime_performance(trades_df, regime_col="regime"):
    """Calculate performance by regime."""
    regime_stats = []

    for regime in trades_df[regime_col].dropna().unique():
        regime_trades = trades_df[trades_df[regime_col] == regime]

        if len(regime_trades) == 0:
            continue

        sharpe = regime_trades["pnl"].mean() / regime_trades["pnl"].std() if regime_trades["pnl"].std() > 0 else 0

        regime_stats.append({
            "Regime": regime,
            "Trade Count": len(regime_trades),
            "Win Rate": (regime_trades["pnl"] > 0).mean(),
            "Avg PnL": regime_trades["pnl"].mean(),
            "Sharpe": sharpe,
        })

    return pd.DataFrame(regime_stats).sort_values("Sharpe", ascending=False)

# Calculate regime performance
regime_perf = calculate_regime_performance(trades_df)

# Create report
report = Report(
    title="Regime Analysis Report",
    description="Strategy performance across market regimes",
)

# Section 1: Regime Distribution
regime_dist = df["regime"].value_counts().to_frame("Count")
regime_dist["Percentage"] = (regime_dist["Count"] / len(df) * 100).round(2)

dist_section = ReportSection(
    title="Regime Distribution",
    content="Distribution of market regimes in the dataset.",
    data={"Regime Distribution": regime_dist},
)
report.add_section(dist_section)

# Section 2: Performance by Regime
perf_section = ReportSection(
    title="Performance by Regime",
    content="Strategy performance varies significantly across regimes.",
    data={"Regime Performance": regime_perf},
)
report.add_section(perf_section)

# Section 3: Recommendations
best_regime = regime_perf.iloc[0]["Regime"]
worst_regime = regime_perf.iloc[-1]["Regime"]

recommendations = f"""
**Best Regime**: {best_regime} (Sharpe: {regime_perf.iloc[0]['Sharpe']:.4f})
**Worst Regime**: {worst_regime} (Sharpe: {regime_perf.iloc[-1]['Sharpe']:.4f})

**Recommendations**:
1. **Regime Filtering**: Only trade in {best_regime} regime
2. **Regime-Aware Sizing**: Reduce position size in {worst_regime} regime
3. **Regime-Based Ensemble**: Use different strategies for different regimes
4. **Monitor Regime Transitions**: Exit positions when regime changes from favorable to unfavorable

**Expected Impact**:
- Filtering to {best_regime} only may improve Sharpe by 30-50%
- Trade frequency will decrease proportionally
"""

rec_section = ReportSection(
    title="Recommendations",
    content=recommendations.strip(),
)
report.add_section(rec_section)

report.save("reports/custom/regime_analysis.html")
```

---

## API Integration

### Example 9: Generate Report via API

```python
import requests
import time

# Step 1: Run parameter search job via API
search_request = {
    "strategy_name": "trend_continuation",
    "symbol": "AIAUSDT",
    "timeframe": "15m",
    "param_space": {
        "trend_min": [0.5, 1.0, 1.5, 2.0],
        "atr_period": [10, 14, 20],
    },
    "search_type": "grid",
}

response = requests.post(
    "http://localhost:8001/api/research/strategy-search/run",
    json=search_request
)

job_id = response.json()["job_id"]
print(f"Job started: {job_id}")

# Step 2: Wait for job to complete
while True:
    status_response = requests.get(
        f"http://localhost:8001/api/research/jobs/{job_id}/status"
    )
    status = status_response.json()["status"]

    if status == "completed":
        print("Job completed!")
        break
    elif status == "failed":
        print("Job failed!")
        break

    time.sleep(5)

# Step 3: Generate report via API
report_request = {
    "job_id": job_id,
    "top_n": 10,
    "format": "html",
}

report_response = requests.post(
    "http://localhost:8001/api/research/reports/strategy-search",
    json=report_request
)

report_info = report_response.json()
print(f"Report generated: {report_info['file_path']}")

# Step 4: View report
view_url = f"http://localhost:8001/api/research/reports/view/strategy_search/{job_id}"
print(f"View report in browser: {view_url}")
```

### Example 10: List and Download Reports

```python
import requests

# List all strategy search reports
response = requests.get(
    "http://localhost:8001/api/research/reports/list",
    params={"report_type": "strategy_search", "limit": 10}
)

reports = response.json()["reports"]

print(f"Found {len(reports)} reports:")
for report in reports:
    print(f"  - {report['report_id']} ({report['format']}) - {report['created_at']}")

# Download a specific report
if reports:
    latest_report = reports[0]
    download_url = f"http://localhost:8001/api/research/reports/view/strategy_search/{latest_report['report_id']}"

    # For HTML, open in browser
    # For other formats, download
    if latest_report['format'] == 'html':
        import webbrowser
        webbrowser.open(download_url)
    else:
        import urllib.request
        urllib.request.urlretrieve(download_url, f"downloaded_{latest_report['report_id']}.{latest_report['format']}")
```

---

## Advanced Workflows

### Example 11: Automated Daily Report Generation

```python
from datetime import datetime
from pathlib import Path
from finantradealgo.research.reporting import (
    StrategySearchReportGenerator,
    Report,
    ReportSection,
)
import pandas as pd

def generate_daily_summary_report():
    """Generate daily summary of all research activities."""

    # Find all jobs from today
    today = datetime.now().strftime("%Y%m%d")
    search_dir = Path("outputs/strategy_search")

    today_jobs = [
        job_dir for job_dir in search_dir.iterdir()
        if job_dir.is_dir() and today in job_dir.name
    ]

    if not today_jobs:
        print("No jobs found for today.")
        return

    # Create summary report
    report = Report(
        title=f"Daily Research Summary - {today}",
        description=f"Summary of {len(today_jobs)} research jobs completed today",
    )

    # Add overview section
    overview_content = f"""
Today's research activities:

- **Total Jobs**: {len(today_jobs)}
- **Date**: {datetime.now().strftime("%Y-%m-%d")}

Individual job summaries below.
"""

    overview_section = ReportSection(
        title="Overview",
        content=overview_content.strip(),
    )
    report.add_section(overview_section)

    # Add section for each job
    for job_dir in today_jobs:
        try:
            # Load results
            results_path = job_dir / "results.parquet"
            if not results_path.exists():
                continue

            results_df = pd.read_parquet(results_path)

            # Get best result
            best = results_df.nlargest(1, "sharpe").iloc[0]

            job_content = f"""
**Best Sharpe**: {best['sharpe']:.4f}
**Best Return**: {best['cum_return']:.2%}
**Total Evaluations**: {len(results_df)}
"""

            job_section = ReportSection(
                title=f"Job: {job_dir.name}",
                content=job_content.strip(),
            )
            report.add_section(job_section)

        except Exception as e:
            print(f"Error processing {job_dir.name}: {e}")

    # Save daily summary
    summary_path = f"reports/daily_summaries/summary_{today}.html"
    report.save(summary_path)

    print(f"Daily summary saved: {summary_path}")

# Run daily
generate_daily_summary_report()
```

### Example 12: Export Report Data to Dashboard

```python
from finantradealgo.research.reporting import StrategySearchReportGenerator
import json

def export_to_dashboard(job_id):
    """Export report data for dashboard visualization."""

    # Generate report in JSON format
    generator = StrategySearchReportGenerator()
    report = generator.generate(
        job_dir=Path(f"outputs/strategy_search/{job_id}"),
        job_id=job_id,
    )

    # Save as JSON
    json_path = f"reports/strategy_search/{job_id}.json"
    report.save(json_path)

    # Load JSON data
    with open(json_path, "r") as f:
        report_data = json.load(f)

    # Extract key metrics for dashboard
    dashboard_data = {
        "job_id": job_id,
        "created_at": report_data["created_at"],
        "metadata": report_data["metadata"],
        "top_performers": [],
    }

    # Extract top performers
    for section in report_data["sections"]:
        if section["title"] == "Top Performers":
            # Process data (implementation depends on section structure)
            dashboard_data["top_performers"] = section.get("data", {})

    # Save dashboard-specific JSON
    dashboard_path = f"dashboard/data/{job_id}.json"
    Path("dashboard/data").mkdir(parents=True, exist_ok=True)

    with open(dashboard_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"Dashboard data exported: {dashboard_path}")

# Export
export_to_dashboard("job_20241130_123456")
```

---

## Integration with Playbooks

Reports should be generated at the end of each playbook workflow.

### Example 13: Complete Strategy Search Workflow

Following the [Strategy Parameter Search Playbook](../../finantradealgo/research/playbooks/strategy_param_search.md):

```python
from finantradealgo.research.strategy_search import StrategySearchJob
from finantradealgo.research.reporting import StrategySearchReportGenerator

# Step 1-5 from playbook: Define, configure, run, analyze
job = StrategySearchJob(
    strategy_name="trend_continuation",
    symbol="AIAUSDT",
    timeframe="15m",
    param_space={
        "trend_min": [0.5, 1.0, 1.5, 2.0],
        "atr_period": [10, 14, 20],
    },
    search_type="grid",
)

result = job.run()

# Step 6: Generate Report (from playbook)
generator = StrategySearchReportGenerator()
report = generator.generate(
    job_dir=result.job_dir,
    job_id=result.job_id,
    top_n=10,
)

report.save(f"reports/strategy_search/{result.job_id}.html")

print("[COMPLETE] Strategy parameter search workflow finished.")
print(f"Report: reports/strategy_search/{result.job_id}.html")
```

### Example 14: Complete Ensemble Workflow

Following the [Ensemble Development Playbook](../../finantradealgo/research/playbooks/ensemble_development.md):

```python
# (Steps 1-5 from ensemble playbook: select components, configure, run)

# Step 6: Generate Report
from finantradealgo.research.reporting import EnsembleReportGenerator

generator = EnsembleReportGenerator()
report = generator.generate(
    backtest_result=result,
    ensemble_type="weighted",
    symbol="AIAUSDT",
    timeframe="15m",
)

report.save("reports/ensemble/weighted_AIAUSDT_15m.html")

print("[COMPLETE] Ensemble development workflow finished.")
print("Report: reports/ensemble/weighted_AIAUSDT_15m.html")
```

---

## Best Practices Summary

1. **Always generate reports** at the end of research workflows
2. **Use descriptive titles** and include metadata (job_id, strategy, symbol, timeframe)
3. **Save in multiple formats** (HTML for viewing, JSON for processing)
4. **Organize reports** by type (strategy_search, ensemble, custom)
5. **Archive reports** with git for reproducibility
6. **Share HTML reports** with stakeholders via email or Slack
7. **Process JSON reports** programmatically for dashboards
8. **Document decisions** in custom report sections

---

## Next Steps

- Review [reporting.md](reporting.md) for detailed API documentation
- Follow [playbooks](../../finantradealgo/research/playbooks/README.md) for complete workflows
- See tests in `tests/test_reporting.py` for more examples

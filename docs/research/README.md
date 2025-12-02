# Research System Documentation

Welcome to the FinanTradeAlgo Research System documentation.

## Overview

The Research System is a comprehensive framework for developing, testing, and validating trading strategies in an isolated environment. It provides tools for:

- **Strategy Parameter Search**: Systematic optimization of strategy parameters
- **Ensemble Strategies**: Combining multiple strategies via weighted or bandit approaches
- **Scenario Testing**: Analyzing strategy performance across different market conditions
- **Automated Reporting**: Professional HTML/Markdown reports for all research activities
- **Research Playbooks**: Standardized workflows and best practices

## Architecture

```
finantradealgo/research/
├── config/                  # Research configuration
│   ├── research_config.py
│   └── strategy_registry.py
├── strategy_search/         # Parameter optimization
│   ├── job.py
│   ├── grid_search.py
│   └── random_search.py
├── ensemble/                # Ensemble strategies
│   ├── base.py
│   ├── weighted.py
│   ├── bandit.py
│   └── backtest.py
├── reporting/               # Report generation
│   ├── base.py
│   ├── strategy_search.py
│   └── ensemble.py
└── playbooks/               # Research workflows
    ├── README.md
    ├── strategy_param_search.md
    ├── ensemble_development.md
    ├── multi_strategy_comparison.md
    ├── regime_analysis.md
    └── robustness_testing.md
```

## Quick Start

### 1. Run Parameter Search

```python
from finantradealgo.research.strategy_search import StrategySearchJob

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
print(f"Best Sharpe: {result.best_sharpe:.4f}")
```

### 2. Build Ensemble

```python
from finantradealgo.research.ensemble import (
    WeightedEnsembleStrategy,
    WeightedEnsembleConfig,
    ComponentStrategy,
    WeightingMethod,
)

components = [
    ComponentStrategy("rule", label="rule"),
    ComponentStrategy("trend_continuation", label="trend"),
    ComponentStrategy("sweep_reversal", label="sweep"),
]

config = WeightedEnsembleConfig(
    components=components,
    weighting_method=WeightingMethod.SHARPE,
)

ensemble = WeightedEnsembleStrategy(config)
```

### 3. Generate Report

```python
from finantradealgo.research.reporting import StrategySearchReportGenerator

generator = StrategySearchReportGenerator()
report = generator.generate(job_dir=result.job_dir, job_id=result.job_id)
report.save(f"reports/{result.job_id}.html")
```

## Documentation

### Core Components

- **[Reporting System](reporting.md)**: Report generation, formats, customization
- **[Examples](examples.md)**: Practical code examples and workflows

### Playbooks

Playbooks provide step-by-step workflows for common research tasks:

1. **[Strategy Parameter Search](../../finantradealgo/research/playbooks/strategy_param_search.md)**
   - Systematic parameter optimization
   - Grid search and random search
   - Analysis and validation
   - Duration: 30-60 minutes

2. **[Ensemble Development](../../finantradealgo/research/playbooks/ensemble_development.md)**
   - Weighted ensembles (5 methods)
   - Bandit ensembles (3 algorithms)
   - Component selection and validation
   - Duration: 1-3 hours

3. **[Multi-Strategy Comparison](../../finantradealgo/research/playbooks/multi_strategy_comparison.md)**
   - Side-by-side strategy comparison
   - Trade-off analysis
   - Statistical significance testing
   - Duration: 30-60 minutes

4. **[Regime Analysis](../../finantradealgo/research/playbooks/regime_analysis.md)**
   - Market regime classification
   - Regime-specific performance
   - Regime-aware strategy selection
   - Duration: 1-2 hours

5. **[Robustness Testing](../../finantradealgo/research/playbooks/robustness_testing.md)**
   - Out-of-sample validation
   - Walk-forward analysis
   - Cross-symbol/timeframe testing
   - Monte Carlo simulation
   - Duration: 2-4 hours

## Research Workflow

Typical research workflow follows this sequence:

```
1. Strategy Parameter Search
   ↓
2. Multi-Strategy Comparison
   ↓
3. Ensemble Development (optional)
   ↓
4. Regime Analysis
   ↓
5. Robustness Testing
   ↓
6. Paper Trading
   ↓
7. Live Deployment
```

Each step has a corresponding playbook with detailed instructions.

## Research API

The Research Service provides REST endpoints for all research operations.

### Base URL

```
http://localhost:8001/api/research
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/strategy-search/run` | POST | Run parameter search job |
| `/strategy-search/{job_id}/status` | GET | Get job status |
| `/strategy-search/{job_id}/results` | GET | Get job results |
| `/ensemble/run` | POST | Run ensemble backtest |
| `/reports/strategy-search` | POST | Generate strategy search report |
| `/reports/list` | GET | List available reports |
| `/reports/view/{type}/{id}` | GET | View/download report |

See [API Documentation](../api/research_service.md) for full endpoint reference (coming soon).

## Configuration

### Research Config

Located in `config/research_config.yaml`:

```yaml
research_cfg:
  mode: research
  strategy_universe:
    - rule
    - trend_continuation
    - sweep_reversal
    - volatility_breakout
  max_parallel_jobs: 4
  default_search_type: grid
  results_retention_days: 90
```

### Strategy Registry

Strategies available for research are defined in:

```python
from finantradealgo.research.config.strategy_registry import StrategyRegistry

registry = StrategyRegistry()
available_strategies = registry.list_strategies()
```

## Best Practices

### 1. Data Management

- **Use consistent data** for all backtests
- **Validate data quality** before research
- **Document data sources** and date ranges
- **Archive data** used in production decisions

### 2. Methodology

- **Start with simple strategies** before complex ensembles
- **Use out-of-sample testing** to avoid overfitting
- **Test across multiple symbols/timeframes** for robustness
- **Document all assumptions** and parameter choices

### 3. Reporting

- **Generate reports** for all research activities
- **Share HTML reports** with stakeholders
- **Version control** report files for reproducibility
- **Include metadata** (job_id, git SHA, date) in all reports

### 4. Validation

- **Never skip robustness testing** before deployment
- **Require minimum thresholds** (e.g., Sharpe > 0.3)
- **Test with realistic costs** (fees, slippage)
- **Paper trade** before going live

### 5. Risk Management

- **Set maximum drawdown limits** (e.g., -20%)
- **Monitor live performance** vs backtest
- **Have kill switches** for catastrophic scenarios
- **Size positions appropriately**

## Common Pitfalls

### ❌ Overfitting

**Problem**: Strategy optimized perfectly on historical data fails live.

**Solution**:
- Use train/test split (70/30)
- Require positive out-of-sample performance
- Test across multiple time periods and symbols
- Avoid over-optimization (limit parameter count)

### ❌ Lookahead Bias

**Problem**: Using future data to make historical decisions.

**Solution**:
- Review indicator calculations carefully
- Use lagged features where appropriate
- Test with walk-forward analysis
- Validate data alignment in backtests

### ❌ Survivorship Bias

**Problem**: Testing only on successful symbols.

**Solution**:
- Test on diverse symbol universe
- Include delisted or failed assets if possible
- Use multiple asset classes

### ❌ Ignoring Transaction Costs

**Problem**: Strategy profitable in backtest but loses money live due to fees.

**Solution**:
- Always include realistic fees (0.04% for Binance)
- Add slippage (0.05-0.1% for market orders)
- Avoid high-frequency strategies unless costs are negligible

### ❌ Cherry-Picking Metrics

**Problem**: Highlighting Sharpe for one strategy, win rate for another.

**Solution**:
- Use consistent primary metric (usually Sharpe)
- Report all metrics for transparency
- Acknowledge trade-offs explicitly

## Tools and Utilities

### Job Management

```python
from finantradealgo.research.strategy_search import JobManager

manager = JobManager()

# List all jobs
jobs = manager.list_jobs()

# Get job status
status = manager.get_job_status("job_20241130_123456")

# Cancel running job
manager.cancel_job("job_20241130_123456")
```

### Results Analysis

```python
import pandas as pd

# Load results
results_df = pd.read_parquet("outputs/strategy_search/job_20241130_123456/results.parquet")

# Top performers
top_10 = results_df.nlargest(10, "sharpe")

# Parameter correlations
param_cols = [c for c in results_df.columns if c.startswith("param_")]
correlations = results_df[param_cols + ["sharpe"]].corr()["sharpe"]
```

### Report Management

```bash
# List reports
curl http://localhost:8001/api/research/reports/list

# View report
open http://localhost:8001/api/research/reports/view/strategy_search/job_20241130_123456

# Delete old reports
curl -X DELETE http://localhost:8001/api/research/reports/strategy_search/old_job_id
```

## Testing

Run research system tests:

```bash
# All research tests
pytest tests/test_research*.py

# Specific modules
pytest tests/test_strategy_search.py
pytest tests/test_ensemble_strategies.py
pytest tests/test_reporting.py
```

## Support

### Documentation

- [Reporting System](reporting.md)
- [Examples](examples.md)
- [Playbooks](../../finantradealgo/research/playbooks/README.md)

### Code Examples

- See `tests/` directory for comprehensive test examples
- See `examples.md` for practical use cases
- See playbooks for step-by-step workflows

### Issues

For bugs or feature requests, open an issue on GitHub.

## Contributing

When adding new research features:

1. **Write tests** for all new functionality
2. **Document** in appropriate markdown files
3. **Create playbook** if new workflow
4. **Update API** if adding endpoints
5. **Generate examples** for documentation

## Changelog

### v0.82 (Current)

- ✅ Strategy parameter search (grid, random)
- ✅ Ensemble strategies (weighted, bandit)
- ✅ Automated reporting (HTML, Markdown, JSON)
- ✅ Research playbooks (5 workflows)
- ✅ Research API endpoints
- ✅ Comprehensive documentation

### Upcoming (v0.83)

- Walk-forward optimization automation
- Multi-objective optimization (Sharpe + drawdown)
- Regime-based ensemble weighting
- Live performance monitoring dashboard
- Integrated paper trading environment

---

**Last Updated**: 2024-11-30

**Version**: 0.82

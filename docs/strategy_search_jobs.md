# Strategy Search Jobs

## Overview

The Strategy Search Jobs system enables systematic parameter optimization for trading strategies with full reproducibility and persistence. Each search job creates a complete record including:

- **Parameter search results** (results.parquet)
- **Job metadata** (meta.json with git SHA, config path, timestamps)
- **Config snapshot** (exact config used for the search)

This ensures every search is fully reproducible and traceable to a specific code version and configuration.

## Architecture

### Core Components

1. **StrategySearchJob** (`finantradealgo/research/strategy_search/jobs.py`)
   - Dataclass model for job specification
   - Contains all metadata needed for reproducibility
   - Serializable to/from JSON

2. **Search Engine** (`finantradealgo/research/strategy_search/search_engine.py`)
   - `random_search()`: Core random parameter sampling
   - `run_random_search()`: Job-based search with full persistence
   - Result validation and git SHA tracking

3. **Analysis Helpers** (`finantradealgo/research/strategy_search/analysis.py`)
   - `load_results()`: Load and validate search results
   - `filter_by_metrics()`: Filter by performance criteria
   - `top_n_by_metric()`: Get top performers
   - `compare_jobs()`: Compare results across multiple jobs

4. **CLI Script** (`scripts/run_strategy_search.py`)
   - Command-line interface for running searches
   - Mode validation (ensures mode='research')
   - Searchable strategy discovery

### Directory Structure

```
outputs/strategy_search/{job_id}/
- results.parquet       # Search results (params + metrics)
- meta.json            # Job metadata (git_sha, config_path, etc.)
- config_snapshot.yml  # Exact config used for search
```

## Quick Start

### 1. List Searchable Strategies

```bash
python -m scripts.run_strategy_search --list-searchable
```

Output:
```
Searchable Strategies:
============================================================
  * rule                      (family: rule)
    Parameters: ms_trend_min, ms_trend_max, tp_atr_mult, ...
  * trend_continuation        (family: trend)
    Parameters: rsi_trend_min, rsi_trend_max, ...
  * sweep_reversal            (family: microstructure)
    Parameters: max_rsi_for_long, min_pullback_ratio, ...
============================================================
```

### 2. Run a Parameter Search

```bash
python -m scripts.run_strategy_search \
  --strategy rule \
  --symbol BTCUSDT \
  --timeframe 15m \
  --n-samples 100 \
  --seed 42
```

This will:
1. Load `config/system.research.yml`
2. Validate mode='research'
3. Sample 100 random parameter sets from the strategy's ParamSpace
4. Run backtest for each parameter set
5. Save results with full metadata

### 3. Analyze Results

```python
from finantradealgo.research.strategy_search.analysis import (
    load_results,
    filter_by_metrics,
    top_n_by_metric,
)

# Load results
df = load_results("outputs/strategy_search/rule_BTCUSDT_15m_20251130_103904")

# Filter for profitable strategies with good Sharpe
profitable = filter_by_metrics(
    df,
    cum_return_min=0.0,
    sharpe_min=1.0,
    max_drawdown_min=-0.20,
    trade_count_min=50,
)

# Get top 10 by Sharpe ratio
top_10 = top_n_by_metric(df, metric="sharpe", n=10)
print(top_10[['params', 'cum_return', 'sharpe', 'max_drawdown']])
```

## CLI Usage

### Basic Search

```bash
python -m scripts.run_strategy_search \
  --strategy rule \
  --symbol BTCUSDT \
  --timeframe 15m \
  --n-samples 100
```

### Advanced Options

```bash
python -m scripts.run_strategy_search \
  --config config/system.research.yml \
  --strategy trend_continuation \
  --symbol AIAUSDT \
  --timeframe 15m \
  --n-samples 50 \
  --job-id custom_job_20251130 \
  --seed 42 \
  --notes "Testing trend strategy with conservative parameters"
```

### Arguments

- `--config`: Path to system config (default: `config/system.research.yml`)
- `--strategy`: Strategy name (required)
- `--symbol`: Trading symbol (required)
- `--timeframe`: Timeframe (required)
- `--n-samples`: Number of parameter samples (required)
- `--job-id`: Custom job ID (default: auto-generated)
- `--seed`: Random seed for reproducibility (optional)
- `--notes`: Optional notes about the job

## Job Metadata

Each job creates a `meta.json` file with complete metadata:

```json
{
  "job_id": "rule_BTCUSDT_15m_20251130_103904",
  "strategy": "rule",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "search_type": "random",
  "n_samples": 100,
  "config_path": "config/system.research.yml",
  "config_snapshot_relpath": "config_snapshot.yml",
  "created_at": "2025-11-30T10:39:04.784983",
  "seed": 42,
  "mode": "research",
  "notes": "Testing new parameter ranges",
  "git_sha": "3846241a",
  "results_path": "results.parquet",
  "n_results": 100
}
```

## Results Format

### Required Columns

All `results.parquet` files must contain these columns:

- `params`: Dict of parameter values used for this evaluation
- `cum_return`: Cumulative return over backtest period
- `sharpe`: Sharpe ratio (risk-adjusted returns)
- `max_drawdown`: Maximum drawdown (negative value)
- `win_rate`: Percentage of winning trades (0.0 - 1.0)
- `trade_count`: Number of trades executed

### Example Results

```python
import pandas as pd

df = pd.read_parquet("outputs/strategy_search/job_id/results.parquet")
print(df.head())
```

Output:
```
                                              params  cum_return    sharpe  max_drawdown  win_rate  trade_count
0  {'ms_trend_min': -0.3, 'ms_trend_max': 0.8, ...}   0.007616  0.235653     -0.209985      None           82
1  {'ms_trend_min': -0.1, 'ms_trend_max': 0.6, ...}  -0.008371  0.203090     -0.212644      None           61
2  {'ms_trend_min': -0.5, 'ms_trend_max': 1.2, ...}  -0.002271  0.197117     -0.206801      None           56
```

## Analysis Examples

### Filter by Multiple Criteria

```python
from finantradealgo.research.strategy_search.analysis import filter_by_metrics

# Find strategies with:
# - Positive returns
# - Sharpe >= 1.5
# - Max drawdown <= 20%
# - At least 50 trades
filtered = filter_by_metrics(
    df,
    cum_return_min=0.0,
    sharpe_min=1.5,
    max_drawdown_min=-0.20,
    trade_count_min=50,
)

print(f"Found {len(filtered)} strategies matching criteria")
```

### Compare Multiple Jobs

```python
from finantradealgo.research.strategy_search.analysis import compare_jobs

# Compare top 5 from each job
comparison = compare_jobs(
    job_dirs=[
        "outputs/strategy_search/rule_BTCUSDT_15m_20251130_103904",
        "outputs/strategy_search/rule_BTCUSDT_15m_20251130_110521",
        "outputs/strategy_search/rule_BTCUSDT_15m_20251130_112045",
    ],
    metric="sharpe",
    top_n=5,
)

print(comparison[['job_id', 'sharpe', 'cum_return', 'max_drawdown']])
```

### Analyze Parameter Importance

```python
from finantradealgo.research.strategy_search.analysis import get_param_importance

# Find which parameters appear most in top 10% performers
importance = get_param_importance(df, metric="sharpe", top_pct=0.1)
print(importance.head(20))
```

Output:
```
       parameter      value  count  frequency
0   ms_trend_min       -0.3      5       0.50
1   ms_trend_min       -0.4      3       0.30
2   ms_trend_max        0.8      4       0.40
3   tp_atr_mult        2.5      6       0.60
...
```

### Load Results with Metadata

```python
from finantradealgo.research.strategy_search.analysis import load_results

# Load results and metadata together
df, meta = load_results(
    "outputs/strategy_search/rule_BTCUSDT_15m_20251130_103904",
    include_meta=True
)

print(f"Strategy: {meta['strategy']}")
print(f"Symbol: {meta['symbol']}")
print(f"Git SHA: {meta['git_sha']}")
print(f"Results: {len(df)}")
```

## Reproducibility

### Full Reproducibility Chain

Every search job is fully reproducible via:

1. **Git SHA**: Exact code version used for the search
2. **Config Snapshot**: Exact config used (copied to job directory)
3. **Random Seed**: Reproducible parameter sampling (if provided)
4. **Timestamp**: When the search was run
5. **Mode**: Ensures correct execution environment (research vs live)

### Reproducing a Search

```bash
# 1. Check out the code version
git checkout 3846241a

# 2. Use the saved config snapshot
python -m scripts.run_strategy_search \
  --config outputs/strategy_search/job_id/config_snapshot.yml \
  --strategy rule \
  --symbol BTCUSDT \
  --timeframe 15m \
  --n-samples 100 \
  --seed 42
```

## Adding New Searchable Strategies

To make a strategy searchable, define a `ParamSpace` and link it in the strategy registry:

### 1. Define ParamSpace

Create `finantradealgo/strategies/{family}_param_space.py`:

```python
from finantradealgo.strategies.param_space import ParamSpace, ParamSpec

MY_STRATEGY_PARAM_SPACE: ParamSpace = {
    "rsi_threshold": ParamSpec(
        name="rsi_threshold",
        type="float",
        low=40.0,
        high=70.0,
    ),
    "atr_mult": ParamSpec(
        name="atr_mult",
        type="float",
        low=1.0,
        high=3.0,
    ),
    "use_filter": ParamSpec(
        name="use_filter",
        type="bool",
    ),
}
```

### 2. Register in Strategy Engine

Update `finantradealgo/strategies/strategy_engine.py`:

```python
from finantradealgo.strategies.my_param_space import MY_STRATEGY_PARAM_SPACE

STRATEGY_SPECS: Dict[str, StrategySpec] = {
    "my_strategy": StrategySpec(
        name="my_strategy",
        strategy_cls=MyStrategy,
        config_cls=MyStrategyConfig,
        config_extractor=_default_extractor("my_strategy"),
        meta=StrategyMeta(
            name="my_strategy",
            family="trend",  # or "range", "microstructure", "volatility", etc.
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=True,
            param_space=MY_STRATEGY_PARAM_SPACE,  # Link the ParamSpace
        ),
    ),
}
```

### 3. Verify Registration

```bash
python -m scripts.run_strategy_search --list-searchable
```

Your strategy should now appear in the list!

## Best Practices

### Search Design

1. **Start Small**: Begin with 10-20 samples to validate the pipeline
2. **Use Seeds**: Always use `--seed` for reproducibility
3. **Iterative Refinement**: Use results to narrow parameter ranges
4. **Multiple Runs**: Run multiple searches with different symbols/timeframes

### Parameter Space Design

1. **Reasonable Ranges**: Based on domain knowledge and initial testing
2. **Avoid Overfitting**: Use cross-validation across symbols/timeframes
3. **Statistical Significance**: Ensure trade_count >= 50 for meaningful metrics

### Result Analysis

1. **Filter First**: Apply minimum criteria (e.g., sharpe_min=1.0)
2. **Look Beyond Sharpe**: Consider cum_return, max_drawdown, trade_count
3. **Parameter Stability**: Check if top performers cluster around similar values
4. **Out-of-Sample Testing**: Validate on different time periods/symbols

## Mode Validation

The system enforces mode validation to prevent accidents:

```python
# This will FAIL if config has mode='live'
python -m scripts.run_strategy_search \
  --config config/system.live.yml \
  --strategy rule \
  ...
```

Error:
```
RuntimeError: Strategy search must run with mode='research' config.
Got mode='live'. Use config/system.research.yml or ensure mode='research'.
```

**Always use `config/system.research.yml` for parameter searches.**

## Troubleshooting

### "Strategy not searchable"

Error:
```
Error: Strategy 'ml' is not searchable (no ParamSpace defined).
```

Solution: Only strategies with `param_space` defined are searchable. Use `--list-searchable` to see available strategies.

### "Missing required columns"

Error:
```
ValueError: Results missing required columns: {'win_rate'}
```

Solution: Ensure your backtest returns all required columns. Check `REQUIRED_RESULT_COLUMNS` in `search_engine.py`.

### "Mode validation failed"

Error:
```
RuntimeError: Strategy search must run with mode='research' config.
```

Solution: Use `config/system.research.yml` or ensure your config has `mode: "research"`.

## Performance Tips

1. **Parallel Searches**: Run multiple searches in parallel for different symbols
2. **Incremental Analysis**: Analyze results after each batch to refine ranges
3. **Efficient Filtering**: Use `filter_by_metrics()` before expensive operations
4. **Parquet Optimization**: Results stored in Parquet for fast loading

## Related Documentation

- [Config Profiles](core_config_profiles.md) - Research vs Live config separation
- [Strategy Engine](strategy_engine.md) - Strategy registration and discovery
- [ParamSpace](param_space.md) - Parameter space definition

## Example Workflow

Complete example of a typical parameter search workflow:

```bash
# 1. List available strategies
python -m scripts.run_strategy_search --list-searchable

# 2. Run initial broad search
python -m scripts.run_strategy_search \
  --strategy rule \
  --symbol BTCUSDT \
  --timeframe 15m \
  --n-samples 100 \
  --seed 42 \
  --notes "Initial broad search"

# 3. Analyze results
python -c "
from finantradealgo.research.strategy_search.analysis import load_results, top_n_by_metric
df = load_results('outputs/strategy_search/rule_BTCUSDT_15m_20251130_103904')
top = top_n_by_metric(df, 'sharpe', n=10)
print(top[['cum_return', 'sharpe', 'max_drawdown']])
"

# 4. Based on results, run focused search with narrowed ranges
# (Update ParamSpace ranges in strategy_param_space.py)

# 5. Run refined search
python -m scripts.run_strategy_search \
  --strategy rule \
  --symbol BTCUSDT \
  --timeframe 15m \
  --n-samples 200 \
  --seed 43 \
  --notes "Refined search based on initial results"

# 6. Compare both searches
python -c "
from finantradealgo.research.strategy_search.analysis import compare_jobs
comparison = compare_jobs([
    'outputs/strategy_search/rule_BTCUSDT_15m_20251130_103904',
    'outputs/strategy_search/rule_BTCUSDT_15m_20251130_112045',
], metric='sharpe', top_n=5)
print(comparison)
"

# 7. Validate top performers on different symbol/timeframe
python -m scripts.run_strategy_search \
  --strategy rule \
  --symbol AIAUSDT \
  --timeframe 1h \
  --n-samples 20 \
  --seed 44 \
  --notes "Out-of-sample validation"
```

## Summary

The Strategy Search Jobs system provides:

- **Full Reproducibility**: Git SHA, config snapshot, seed tracking
- **Easy Analysis**: Rich helper functions for filtering and comparison
- **Mode Safety**: Enforces research mode to prevent live trading accidents
- **Extensibility**: Easy to add new searchable strategies
- **Performance**: Parquet storage for fast I/O

Start with `--list-searchable` to see available strategies, run your first search with `--n-samples 10` to validate, then scale up!

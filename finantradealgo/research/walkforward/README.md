# Walk-Forward Analysis Framework

Comprehensive out-of-sample validation framework for testing strategy robustness and detecting overfitting.

## Overview

Walk-forward analysis is the gold standard for validating trading strategies. It simulates real-world conditions by:
1. Optimizing parameters on **in-sample** (training) data
2. Testing those parameters on **out-of-sample** (unseen) data
3. Rolling the window forward and repeating
4. Analyzing performance degradation from IS to OOS

This framework provides everything needed for professional walk-forward optimization:

- **Anchored (Expanding) Walk-Forward**: Training window expands from a fixed start
- **Rolling Walk-Forward**: Fixed-size windows that roll forward together
- **Walk-Forward Efficiency (WFE)**: Measures OOS/IS performance ratio
- **Overfitting Detection**: Identifies excessive IS-to-OOS degradation
- **Parameter Stability Analysis**: Tracks how parameters change across windows
- **Comprehensive Validation**: Multi-factor robustness scoring

## Features

### 1. Flexible Window Types

#### Rolling Walk-Forward
- Fixed training window size
- Fixed test window size
- Both windows roll forward together
- Best for strategies that adapt to recent market conditions

#### Anchored/Expanding Walk-Forward
- Training window expands from fixed start
- Fixed test window size
- Uses all historical data for training
- Best for strategies that benefit from more data

### 2. Walk-Forward Efficiency (WFE)

Measures how well strategy translates from IS to OOS:

```
WFE = OOS Performance / IS Performance
```

- **WFE ≈ 1.0**: Excellent robustness, minimal overfitting
- **WFE ≈ 0.5-0.8**: Acceptable, some degradation expected
- **WFE < 0.5**: Potential overfitting concerns

### 3. Overfitting Detection

Multi-factor analysis:
- **Degradation Analysis**: IS-to-OOS performance drop
- **Efficiency Metrics**: Sharpe/return efficiency ratios
- **Parameter Stability**: Consistency of optimal parameters
- **Consistency Score**: OOS win rate and reliability

### 4. Validation & Scoring

Automated validation checks:
- ✓ Sharpe degradation < 50%
- ✓ OOS win rate > 40%
- ✓ Positive OOS Sharpe ratio
- ✓ Stable parameters across windows

Overall scores:
- **Excellent**: 80-100 (production ready)
- **Good**: 60-80 (acceptable robustness)
- **Warning**: 40-60 (needs improvement)
- **Failed**: < 40 (likely overfit)

## Quick Start

### Basic Usage

```python
from finantradealgo.research.walkforward import (
    WalkForwardEngine,
    WalkForwardConfig,
    WindowType,
)

# 1. Configure walk-forward
config = WalkForwardConfig(
    in_sample_periods=12,      # 12 months training
    out_sample_periods=3,       # 3 months testing
    window_type=WindowType.ROLLING,
    period_unit="M",
)

# 2. Create engine
engine = WalkForwardEngine(config)

# 3. Define parameter grid
param_grid = {
    'fast_ma': [10, 20, 30],
    'slow_ma': [50, 100, 200],
}

# 4. Run walk-forward
result = engine.run(
    strategy_id="my_strategy",
    param_grid=param_grid,
    data_df=price_data,
    backtest_function=my_backtest_func,
)

# 5. Get validation report
report = engine.validate(result)
print(f"Status: {report.status}")
print(f"Score: {report.overall_score}/100")

# 6. Detect overfitting
overfitting = engine.detect_overfitting(result)
print(f"Risk: {overfitting['risk_level']}")

# 7. Export results with plots
engine.export_results(result, "results/walkforward", include_plots=True)
```

### Backtest Function Format

Your backtest function must follow this signature:

```python
def my_backtest_func(data_df: pd.DataFrame, params: dict) -> tuple:
    """
    Args:
        data_df: Price data for this window
        params: Dictionary of parameters to test

    Returns:
        Tuple of (trades_df, metrics_dict) where:
        - trades_df: DataFrame with trade results
        - metrics: Dict with required keys:
            - 'sharpe_ratio': float
            - 'total_return': float (percentage)
            - 'max_drawdown': float
            - 'win_rate': float (0-1)
            - 'total_trades': int
    """
    # Your strategy logic here
    ...
    return trades_df, metrics
```

## Architecture

```
walkforward/
├── engine.py           # High-level orchestration API
├── optimizer.py        # Window generation & optimization
├── validator.py        # OOS validation & checks
├── analysis.py         # WFE metrics & analysis
├── visualization.py    # Interactive charts
├── models.py          # Data structures
└── examples/          # Usage examples
    └── basic_example.py
```

## Usage Examples

### Example 1: Rolling Walk-Forward

```python
from finantradealgo.research.walkforward import WalkForwardEngine, WalkForwardConfig

config = WalkForwardConfig(
    in_sample_periods=6,
    out_sample_periods=2,
    window_type="rolling",
    period_unit="M",
)

engine = WalkForwardEngine(config)
result = engine.run_rolling(
    strategy_id="Strategy_A",
    param_grid=params,
    data_df=data,
    backtest_function=backtest_func,
)
```

### Example 2: Anchored Walk-Forward

```python
result = engine.run_anchored(
    strategy_id="Strategy_B",
    param_grid=params,
    data_df=data,
    backtest_function=backtest_func,
    in_sample_periods=12,  # Initial 12 months, then expanding
    out_sample_periods=3,   # Always 3 months OOS
)
```

### Example 3: WFE Analysis

```python
# Calculate Walk-Forward Efficiency
wfe = engine.calculate_wfe(result)

print(f"Sharpe Efficiency: {wfe.sharpe_efficiency:.2f}")
print(f"Return Efficiency: {wfe.return_efficiency:.2f}")
print(f"IS-OOS Correlation: {wfe.sharpe_correlation:.2f}")
```

### Example 4: Strategy Comparison

```python
# Run multiple strategies
results = []
for strategy_name, params in strategies:
    result = engine.run(strategy_name, params, data, backtest_func)
    results.append(result)

# Compare
comparison_df = engine.compare_strategies(results)
print(comparison_df)

# Visualize comparison
engine.plot_comparison(results, "comparison.html")
```

### Example 5: Full Analysis & Export

```python
# Run walk-forward
result = engine.run(strategy_id, params, data, backtest_func)

# Get detailed analysis
analysis = engine.analyze(result)
print(analysis['efficiency_metrics'])
print(analysis['parameter_drift'])
print(analysis['regime_sensitivity'])

# Generate text report
print(engine.generate_report(result))

# Export everything (JSON + HTML plots)
engine.export_results(result, "output/", include_plots=True)
```

## Visualization

The framework generates interactive Plotly charts:

### Performance Dashboard
```python
engine.plot_robustness_dashboard(result, "dashboard.html")
```
Shows:
- IS vs OOS Sharpe
- Consistency by window
- Combined equity curve
- Parameter stability gauge
- Returns distribution
- Summary metrics table

### Individual Charts
```python
# IS vs OOS comparison
engine.plot_performance(result, "performance.html")

# Combined OOS equity
engine.plot_equity_curve(result, "equity.html")

# Parameter evolution
engine.plot_parameter_stability(result, "params.html")

# Degradation distribution
engine.plot_degradation(result, "degradation.html")
```

## Metrics Reference

### Performance Metrics
- **Avg IS Sharpe**: Average in-sample Sharpe ratio
- **Avg OOS Sharpe**: Average out-of-sample Sharpe ratio
- **OOS Win Rate**: Percentage of profitable OOS periods
- **Avg Degradation**: Average IS-to-OOS performance drop

### Efficiency Metrics
- **Sharpe Efficiency**: OOS Sharpe / IS Sharpe
- **Return Efficiency**: OOS Return / IS Return
- **Sharpe Correlation**: Correlation between IS and OOS Sharpe

### Robustness Scores
- **Consistency Score**: 0-100, based on OOS win rate and degradation
- **Parameter Stability**: 0-100, based on parameter coefficient of variation
- **Overall Validation**: 0-100, weighted combination of all checks

## Configuration Options

### WalkForwardConfig

```python
WalkForwardConfig(
    # Window configuration
    in_sample_periods=12,          # Training periods
    out_sample_periods=3,           # Test periods
    window_type="rolling",          # or "anchored"/"expanding"
    period_unit="M",                # D/W/M/Q/Y

    # Optimization
    optimization_metric="sharpe_ratio",  # Metric to optimize

    # Constraints
    min_trades_per_period=10,       # Minimum trades required
    require_profitable_is=False,    # Require IS to be profitable
)
```

### Validator Configuration

```python
engine = WalkForwardEngine(
    config=wf_config,
    validator_config={
        'max_sharpe_degradation': 0.5,   # Max 50% degradation
        'min_oos_win_rate': 0.4,          # Min 40% profitable periods
        'min_oos_sharpe': 0.5,            # Min OOS Sharpe
        'max_param_cv': 0.5,              # Max parameter variation
    }
)
```

## Best Practices

### 1. Choose Appropriate Window Sizes
- **Too small**: Unstable results, parameter noise
- **Too large**: Few windows, limited validation
- **Recommendation**: 6-12 months IS, 1-3 months OOS

### 2. Use Sufficient Data
- Minimum: 2-3 years of data
- Recommended: 5+ years for robust validation
- More windows = better statistical confidence

### 3. Parameter Grid Size
- **Too small**: May miss optimal parameters
- **Too large**: Overfitting risk, long runtime
- **Recommendation**: 3-5 values per parameter, < 50 total combinations

### 4. Interpretation Guidelines
- Focus on **OOS metrics**, not IS metrics
- **Consistency > absolute performance**
- Low degradation is more important than high IS Sharpe
- Stable parameters indicate robustness

### 5. Red Flags
- ⚠️ OOS win rate < 40%
- ⚠️ Sharpe degradation > 50%
- ⚠️ Negative OOS Sharpe
- ⚠️ Highly unstable parameters
- ⚠️ WFE < 0.3

## Complete Example

See [examples/basic_example.py](examples/basic_example.py) for complete runnable examples including:
- Sample data generation
- Simple MA crossover strategy
- Rolling walk-forward
- Anchored walk-forward
- Strategy comparison
- Full analysis and export

Run example:
```bash
python -m finantradealgo.research.walkforward.examples.basic_example
```

## API Reference

### WalkForwardEngine

Main API for walk-forward analysis.

**Methods:**
- `run()` - Run complete walk-forward analysis
- `run_rolling()` - Shortcut for rolling window
- `run_anchored()` - Shortcut for anchored window
- `validate()` - Validate result for robustness
- `analyze()` - Detailed performance analysis
- `calculate_wfe()` - Calculate WFE metrics
- `detect_overfitting()` - Overfitting risk assessment
- `compare_strategies()` - Compare multiple results
- `plot_*()` - Various visualization methods
- `export_results()` - Export everything to disk
- `generate_report()` - Text summary report

### WalkForwardResult

Result object containing all windows and aggregate metrics.

**Attributes:**
- `strategy_id` - Strategy identifier
- `windows` - List of WalkForwardWindow objects
- `total_windows` - Number of windows
- `avg_oos_sharpe` - Average OOS Sharpe
- `avg_oos_return` - Average OOS return
- `oos_win_rate` - OOS win rate
- `consistency_score` - Consistency score 0-100
- `param_stability_score` - Parameter stability 0-100
- `combined_equity_curve` - Combined OOS equity

### ValidationReport

Validation result with status and recommendations.

**Attributes:**
- `status` - ValidationStatus enum (excellent/good/warning/failed)
- `overall_score` - Overall score 0-100
- `passed_checks` - List of passed validation checks
- `failed_checks` - List of failed checks
- `warnings` - List of warnings
- `recommendations` - Actionable recommendations

## Contributing

This framework is part of the finantradealgo research module. For improvements or bug reports, please follow the main project guidelines.

## License

Part of the finantradealgo project.

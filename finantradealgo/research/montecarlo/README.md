# Monte Carlo Simulation Framework

Comprehensive Monte Carlo simulation framework for trading strategy robustness testing, risk analysis, and statistical validation.

## Overview

Monte Carlo simulation is essential for understanding the statistical properties of trading strategies. This framework provides:

1. **Trade Sequence Randomization** - Test if results are due to skill or luck
2. **Parameter Perturbation** - Test parameter sensitivity and robustness
3. **Regime Randomization** - Test performance across market conditions
4. **Risk Distribution Analysis** - VaR, CVaR, drawdown distributions
5. **Stress Testing** - Extreme market scenario testing

## Features

### 1. Trade Sequence Randomization (Luck vs. Skill)

Randomly shuffles or resamples trade order to determine if results are statistically significant:

```python
from finantradealgo.research.montecarlo import MonteCarloSimulator, MonteCarloConfig

config = MonteCarloConfig(n_simulations=1000)
simulator = MonteCarloSimulator(config)

# Test if results are luck or skill
result = simulator.test_trade_sequence(trades_df)

print(f"Mean Return: {result.mean_return:.2f}%")
print(f"95% CI: [{result.return_ci_lower:.2f}, {result.return_ci_upper:.2f}]")
print(f"Prob Profit: {result.prob_profit:.1%}")
```

**Benefits:**
- Determines statistical significance of results
- Identifies if performance is repeatable
- Calculates confidence intervals
- Detects lucky/unlucky sequences

### 2. Parameter Perturbation Testing

Tests sensitivity to parameter changes by adding noise to optimal parameters:

```python
# Test parameter robustness
sensitivity = simulator.test_parameter_perturbation(
    optimal_params={'fast_ma': 20, 'slow_ma': 50},
    data_df=price_data,
    backtest_function=my_backtest,
    noise_level=0.1,  # ±10% noise
    n_perturbations=100,
)

print(f"Robustness Score: {sensitivity['robustness_score']:.1f}/100")
print(f"Return Degradation: {sensitivity['degradation']['return_degradation']:.1%}")
```

**Benefits:**
- Identifies parameter stability
- Estimates confidence intervals for performance
- Detects overfitting to specific parameter values
- Helps set parameter tolerances

### 3. Regime Randomization

Tests strategy performance across different market conditions:

```python
from finantradealgo.research.montecarlo import MonteCarloAnalyzer

analyzer = MonteCarloAnalyzer()

regime_analysis = analyzer.analyze_regime_randomization(
    trades_df=trades_df,
    market_data=market_data,
)

print(f"Bull Market: {regime_analysis.bull_market_return:.2f}%")
print(f"Bear Market: {regime_analysis.bear_market_return:.2f}%")
print(f"Sideways: {regime_analysis.sideways_market_return:.2f}%")
print(f"Consistency: {regime_analysis.regime_consistency_score:.1f}/100")
```

**Benefits:**
- Identifies regime dependency
- Tests robustness across market conditions
- Detects strategies that only work in specific conditions

### 4. Risk Distribution Analysis

Comprehensive risk metrics from Monte Carlo distributions:

```python
# Run Monte Carlo simulation
result = simulator.test_trade_sequence(trades_df)

# Risk metrics automatically calculated
print(f"VaR (95%): {result.value_at_risk:.2f}%")
print(f"CVaR (95%): {result.conditional_var:.2f}%")
print(f"Worst Case (1%): {result.percentile_1:.2f}%")
print(f"Best Case (99%): {result.percentile_99:.2f}%")

# Drawdown distribution
dd_analysis = analyzer.analyze_drawdown_distribution(result)
print(f"Mean Max DD: {dd_analysis['mean_max_dd']:.2f}%")
print(f"95% Max DD: {dd_analysis['percentile_95']:.2f}%")
```

**Metrics Provided:**
- Value at Risk (VaR) at various confidence levels
- Conditional VaR (Expected Shortfall)
- Drawdown distribution (mean, percentiles)
- Return distribution (mean, std, skew, kurtosis)
- Probability metrics (profit, large loss)

### 5. Stress Testing

Test strategy under extreme market conditions:

```python
# Run stress tests
analysis = simulator.run_comprehensive_analysis(
    trades_df=trades_df,
    include_stress_tests=True,
)

stress_results = analysis['stress_tests']
for scenario, results in stress_results.items():
    print(f"{scenario}: {results['mean_return']:.2f}%")
```

**Predefined Scenarios:**
- High Volatility (2x normal volatility)
- Market Crash (3x volatility, -5% drift)
- Low Volatility (0.5x volatility)
- Trending Market (strong upward drift)
- Ranging Market (low volatility, no trend)

## Quick Start

### Basic Usage

```python
from finantradealgo.research.montecarlo import MonteCarloSimulator, MonteCarloConfig

# 1. Configure Monte Carlo
config = MonteCarloConfig(
    n_simulations=1000,       # Number of simulations
    resampling_method="bootstrap",  # or "shuffle", "block_bootstrap", "parametric"
    confidence_level=0.95,    # 95% confidence intervals
)

# 2. Create simulator
simulator = MonteCarloSimulator(config)

# 3. Test trade sequence (requires trades_df with 'pnl' column)
result = simulator.test_trade_sequence(trades_df)

# 4. Analyze results
print(simulator.generate_report(result))
```

### Luck vs. Skill Analysis

```python
from finantradealgo.research.montecarlo import MonteCarloAnalyzer

# Create analyzer
analyzer = MonteCarloAnalyzer()

# Analyze luck vs. skill
luck_analysis = analyzer.analyze_luck_vs_skill(trades_df, mc_result)

print(f"Statistical Significance: {luck_analysis.is_statistically_significant}")
print(f"P-Value: {luck_analysis.p_value:.4f}")
print(f"Interpretation: {luck_analysis.interpretation}")
```

## Architecture

```
montecarlo/
├── simulator.py        # High-level orchestration API
├── analysis.py         # Regime analysis & luck vs. skill
├── resampler.py        # Bootstrap resampling methods
├── models.py           # Data structures
├── risk_metrics.py     # Risk assessment
├── stress_test.py      # Stress testing
├── visualization.py    # Charts and plots
└── examples/
    └── basic_example.py
```

## Resampling Methods

### 1. Bootstrap (Default)
Random sampling with replacement - breaks serial correlation:

```python
config = MonteCarloConfig(resampling_method="bootstrap")
```

**Use when:**
- Trades are independent
- Want to test luck vs. skill
- Standard approach for most strategies

### 2. Shuffle
Random permutation without replacement:

```python
config = MonteCarloConfig(resampling_method="shuffle")
```

**Use when:**
- Preserving exact trade distribution
- Testing order dependency only

### 3. Block Bootstrap
Preserves short-term serial correlation:

```python
config = MonteCarloConfig(
    resampling_method="block_bootstrap",
    block_size=10,  # Keep 10-trade blocks together
)
```

**Use when:**
- Trades have serial correlation
- Mean reversion strategies
- Preserving market regime effects

### 4. Parametric
Assumes normal distribution:

```python
config = MonteCarloConfig(resampling_method="parametric")
```

**Use when:**
- Trades approximately normally distributed
- Want smooth theoretical distribution
- Extrapolating beyond observed data

## Complete Example

```python
from finantradealgo.research.montecarlo import (
    MonteCarloSimulator,
    MonteCarloAnalyzer,
    MonteCarloConfig,
)
import pandas as pd

# Load your trades
trades_df = pd.read_csv("trades.csv")  # Must have 'pnl' column

# Configure Monte Carlo
config = MonteCarloConfig(
    n_simulations=1000,
    resampling_method="bootstrap",
    confidence_level=0.95,
)

# Create simulator and analyzer
simulator = MonteCarloSimulator(config)
analyzer = MonteCarloAnalyzer()

# 1. Test trade sequence randomization
print("1. Testing trade sequence randomization...")
mc_result = simulator.test_trade_sequence(trades_df)

# 2. Analyze luck vs. skill
print("\n2. Analyzing luck vs. skill...")
luck_analysis = analyzer.analyze_luck_vs_skill(trades_df, mc_result)
print(f"   Statistical Significance: {luck_analysis.is_statistically_significant}")
print(f"   P-Value: {luck_analysis.p_value:.4f}")

# 3. Risk metrics
print("\n3. Risk Metrics:")
print(f"   VaR (95%): {mc_result.value_at_risk:.2f}%")
print(f"   CVaR (95%): {mc_result.conditional_var:.2f}%")
print(f"   Prob Profit: {mc_result.prob_profit:.1%}")

# 4. Drawdown distribution
print("\n4. Drawdown Distribution:")
dd_analysis = analyzer.analyze_drawdown_distribution(mc_result)
print(f"   Mean Max DD: {dd_analysis['mean_max_dd']:.2f}%")
print(f"   95% Worst Case: {dd_analysis['percentile_95']:.2f}%")

# 5. Full report
print("\n5. Full Report:")
print(simulator.generate_report(mc_result))

# 6. Export results
simulator.export_results(mc_result, "montecarlo_results/")
```

## API Reference

### MonteCarloSimulator

High-level API for Monte Carlo simulation.

**Methods:**
- `test_trade_sequence()` - Trade sequence randomization
- `test_parameter_perturbation()` - Parameter sensitivity testing
- `test_parameter_sensitivity_grid()` - Grid-based sensitivity
- `run_comprehensive_analysis()` - Complete analysis with all tests
- `compare_strategies()` - Compare multiple strategies
- `generate_report()` - Text summary report
- `export_results()` - Export to disk

### MonteCarloAnalyzer

Advanced analysis methods.

**Methods:**
- `analyze_luck_vs_skill()` - Statistical significance testing
- `analyze_regime_randomization()` - Market regime testing
- `analyze_drawdown_distribution()` - Drawdown analysis
- `analyze_return_distribution()` - Return distribution analysis
- `calculate_sharpe_confidence_interval()` - Sharpe CI
- `estimate_win_probability()` - Prob of target return
- `compare_to_benchmark()` - Benchmark comparison
- `generate_comprehensive_report()` - Full analysis report

### MonteCarloConfig

Configuration object.

**Parameters:**
- `n_simulations` (int): Number of Monte Carlo runs (default: 1000)
- `resampling_method` (str): "bootstrap", "shuffle", "block_bootstrap", "parametric"
- `block_size` (int): Block size for block bootstrap (default: 10)
- `confidence_level` (float): Confidence level for intervals (default: 0.95)
- `min_trades_per_sim` (int): Minimum trades required (default: 10)
- `random_seed` (int): Random seed for reproducibility (default: None)

### MonteCarloResult

Result object containing all simulations and aggregate statistics.

**Attributes:**
- `n_simulations` - Number of simulations
- `mean_return` - Mean return across simulations
- `median_return` - Median return
- `std_return` - Standard deviation
- `return_ci_lower/upper` - Confidence interval bounds
- `value_at_risk` - VaR at confidence level
- `conditional_var` - CVaR (Expected Shortfall)
- `percentile_1/5/25/75/95/99` - Return percentiles
- `prob_profit` - Probability of profit
- `prob_loss_exceeds_10pct/20pct` - Large loss probabilities
- `skewness` - Distribution skewness
- `kurtosis` - Distribution kurtosis

## Interpretation Guide

### Return Confidence Intervals

```python
print(f"95% CI: [{result.return_ci_lower:.2f}, {result.return_ci_upper:.2f}]%")
```

- **Narrow CI**: Consistent, predictable performance
- **Wide CI**: High uncertainty, luck-dependent
- **CI includes 0**: Not consistently profitable

### Luck vs. Skill P-Value

```python
p_value = luck_analysis.p_value
```

- **p < 0.05**: Statistically significant (likely skill)
- **p < 0.01**: Highly significant (strong evidence of skill)
- **p > 0.05**: Not significant (may be luck)

### Probability of Profit

```python
prob = result.prob_profit
```

- **> 0.7**: High consistency
- **0.5 - 0.7**: Moderate consistency
- **< 0.5**: Poor consistency (losing more often than winning)

### Robustness Score

```python
score = sensitivity['robustness_score']
```

- **> 70**: Robust to parameter changes
- **50 - 70**: Moderate sensitivity
- **< 50**: High sensitivity (overfitted parameters)

## Best Practices

### 1. Number of Simulations
- **Minimum**: 500 simulations
- **Recommended**: 1000-5000 simulations
- **High precision**: 10000+ simulations

More simulations = more stable statistics but longer runtime.

### 2. Minimum Trades
- Need at least 30 trades for meaningful Monte Carlo
- Prefer 50+ trades for robust statistics
- 100+ trades ideal

### 3. Interpretation
- Focus on **median** over mean (more robust to outliers)
- Consider **worst case** (5th percentile) for risk management
- Use **p-value < 0.05** threshold for significance

### 4. Parameter Testing
- Test ±10-20% noise around optimal parameters
- If robustness score < 50, parameters may be overfit
- Consider widening parameter ranges if unstable

### 5. Regime Analysis
- Consistency score > 70 indicates regime-independent strategy
- Large differences between regimes suggests regime filtering needed
- Test longer time periods for regime robustness

## Red Flags

- ⚠️ Prob Profit < 50% - Strategy not consistently profitable
- ⚠️ Wide confidence intervals - High luck dependency
- ⚠️ P-value > 0.10 - Results not statistically significant
- ⚠️ Robustness score < 40 - Highly sensitive to parameters
- ⚠️ Negative skewness + high kurtosis - Fat left tail (large loss risk)
- ⚠️ Regime consistency < 50 - Very regime-dependent

## Examples

See [examples/basic_example.py](examples/basic_example.py) for complete runnable examples including:
- Trade sequence randomization
- Parameter perturbation testing
- Comprehensive analysis
- Regime analysis
- Strategy comparison

Run examples:
```bash
python -m finantradealgo.research.montecarlo.examples.basic_example
```

## Integration with Walk-Forward

Monte Carlo complements walk-forward analysis:

```python
from finantradealgo.research.walkforward import WalkForwardEngine
from finantradealgo.research.montecarlo import MonteCarloSimulator

# 1. Run walk-forward optimization
wf_engine = WalkForwardEngine(wf_config)
wf_result = wf_engine.run(strategy_id, params, data, backtest_func)

# 2. Test OOS trades with Monte Carlo
oos_trades = extract_oos_trades(wf_result)  # Get combined OOS trades

mc_simulator = MonteCarloSimulator(mc_config)
mc_result = mc_simulator.test_trade_sequence(oos_trades)

# 3. Analyze
print(f"WF OOS Sharpe: {wf_result.avg_oos_sharpe:.2f}")
print(f"MC Mean Return: {mc_result.mean_return:.2f}%")
print(f"MC Prob Profit: {mc_result.prob_profit:.1%}")
```

## Contributing

This framework is part of the finantradealgo research module. For improvements or bug reports, please follow the main project guidelines.

## License

Part of the finantradealgo project.

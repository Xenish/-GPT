# Playbook: Multi-Strategy Comparison

**Objective**: Compare multiple strategies side-by-side to identify best performers and understand trade-offs.

**Duration**: 30-60 minutes

**Output**: Comparative analysis report, performance rankings, selection recommendation

---

## Prerequisites

### Strategies
- ✅ 2+ strategies to compare (can be different strategy types or same type with different params)
- ✅ Each strategy has been validated individually
- ✅ Strategies tested on same data (apples-to-apples comparison)

### Data & Config
- ✅ OHLCV data for testing (same dataset for all strategies)
- ✅ Research mode config (`mode: research`)
- ✅ System config with consistent risk parameters

### Knowledge
- ✅ Understand key performance metrics (Sharpe, drawdown, win rate)
- ✅ Familiar with strategy characteristics (trend-following, mean-reversion, etc.)

---

## Step 1: Define Comparison Criteria

### Primary Metric
Choose the main metric for ranking strategies:
- [ ] **Sharpe Ratio** (risk-adjusted returns) - Most common
- [ ] **Cumulative Return** (absolute performance)
- [ ] **Max Drawdown** (risk management)
- [ ] **Win Rate** (psychological comfort)
- [ ] **Profit Factor** (gross profit / gross loss)

### Secondary Metrics
Additional metrics to consider:
- [ ] Trade frequency (operational complexity)
- [ ] Average trade duration (capital efficiency)
- [ ] Consistency across time periods
- [ ] Robustness across symbols/timeframes

---

## Step 2: Prepare Strategy Configurations

### Option A: Different Strategy Types

Compare fundamentally different strategies:

```python
from finantradealgo.strategies.strategy_engine import create_strategy

strategies_to_compare = [
    {
        "name": "rule",
        "label": "Rule-Based",
        "params": {},
    },
    {
        "name": "trend_continuation",
        "label": "Trend Following",
        "params": {"trend_min": 1.0},
    },
    {
        "name": "sweep_reversal",
        "label": "Sweep Reversal",
        "params": {},
    },
    {
        "name": "volatility_breakout",
        "label": "Vol Breakout",
        "params": {"atr_period": 14},
    },
]
```

### Option B: Same Strategy, Different Parameters

Compare parameter variations:

```python
strategies_to_compare = [
    {
        "name": "trend_continuation",
        "label": "Trend (Conservative)",
        "params": {"trend_min": 2.0, "atr_period": 20},
    },
    {
        "name": "trend_continuation",
        "label": "Trend (Balanced)",
        "params": {"trend_min": 1.0, "atr_period": 14},
    },
    {
        "name": "trend_continuation",
        "label": "Trend (Aggressive)",
        "params": {"trend_min": 0.5, "atr_period": 10},
    },
]
```

---

## Step 3: Run Backtests

### Via Python SDK

```python
from finantradealgo.data.ohlcv_loader import load_ohlcv
from finantradealgo.backtesting.backtest_engine import BacktestEngine
from finantradealgo.config import load_config

# Load data and config
df = load_ohlcv("AIAUSDT", "15m")
sys_cfg = load_config("config/research_config.yaml")

results = {}

for strat_config in strategies_to_compare:
    # Create strategy
    strategy = create_strategy(strat_config["name"], sys_cfg)

    # Override params if specified
    for key, value in strat_config.get("params", {}).items():
        setattr(strategy, key, value)

    # Run backtest
    engine = BacktestEngine(strategy, sys_cfg)
    result = engine.run(df.copy())

    # Store results
    results[strat_config["label"]] = {
        "sharpe": result.metrics.sharpe,
        "cum_return": result.metrics.cum_return,
        "max_dd": result.metrics.max_drawdown,
        "trade_count": result.metrics.trade_count,
        "win_rate": result.metrics.win_rate,
        "avg_trade_duration": result.metrics.avg_trade_duration_bars,
        "profit_factor": result.metrics.profit_factor,
    }
```

### Via Research API (Batch Comparison)

```python
import requests

payload = {
    "symbol": "AIAUSDT",
    "timeframe": "15m",
    "strategies": [
        {"strategy_name": "rule", "label": "Rule-Based"},
        {"strategy_name": "trend_continuation", "label": "Trend Following"},
        {"strategy_name": "sweep_reversal", "label": "Sweep Reversal"},
    ],
}

response = requests.post(
    "http://localhost:8001/api/research/compare",
    json=payload
)

results = response.json()
```

---

## Step 4: Create Comparison Table

```python
import pandas as pd

# Convert results to DataFrame
comparison_df = pd.DataFrame(results).T

# Sort by primary metric (Sharpe)
comparison_df = comparison_df.sort_values("sharpe", ascending=False)

# Add rank column
comparison_df["rank"] = range(1, len(comparison_df) + 1)

# Format for display
display_df = comparison_df.copy()
for col in ["sharpe", "cum_return", "max_dd", "win_rate", "profit_factor"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

print(display_df)
```

**Example Output**:
```
                      sharpe  cum_return  max_dd  trade_count  win_rate  rank
Trend Following       0.8234      0.4521 -0.1823          145    0.5517     1
Sweep Reversal        0.6891      0.3214 -0.2156           98    0.6122     2
Vol Breakout          0.5623      0.2987 -0.1654           87    0.5747     3
Rule-Based            0.4512      0.2134 -0.2543          234    0.4872     4
```

---

## Step 5: Analyze Trade-offs

### Risk-Return Profile

```python
import matplotlib.pyplot as plt

# Create scatter plot: Return vs Drawdown
plt.figure(figsize=(10, 6))
plt.scatter(
    comparison_df["max_dd"],
    comparison_df["cum_return"],
    s=100,
    alpha=0.6
)

# Add labels
for label, row in comparison_df.iterrows():
    plt.annotate(
        label,
        (row["max_dd"], row["cum_return"]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Max Drawdown")
plt.ylabel("Cumulative Return")
plt.title("Risk-Return Profile")
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.show()
```

### Trade Frequency vs Performance

```python
# Plot: Trade count vs Sharpe
plt.figure(figsize=(10, 6))
plt.scatter(
    comparison_df["trade_count"],
    comparison_df["sharpe"],
    s=100,
    alpha=0.6
)

for label, row in comparison_df.iterrows():
    plt.annotate(label, (row["trade_count"], row["sharpe"]))

plt.xlabel("Trade Count")
plt.ylabel("Sharpe Ratio")
plt.title("Trade Frequency vs Performance")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Step 6: Statistical Significance Testing

Test if performance differences are statistically significant:

```python
from scipy import stats

# Get equity curves for top 2 strategies
strat1_returns = results_dict["strategy1"]["trade_returns"]
strat2_returns = results_dict["strategy2"]["trade_returns"]

# T-test for mean return difference
t_stat, p_value = stats.ttest_ind(strat1_returns, strat2_returns)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Performance difference is statistically significant (p < 0.05)")
else:
    print("⚠️  Performance difference is NOT statistically significant (p >= 0.05)")
```

**Interpretation**:
- **p < 0.05**: Significant difference, top strategy is likely better
- **p >= 0.05**: No significant difference, strategies are statistically similar

---

## Step 7: Cross-Validation (Multi-Symbol/Timeframe)

Validate rankings across different market conditions:

```python
symbols = ["AIAUSDT", "BTCUSDT", "ETHUSDT"]
timeframes = ["15m", "1h"]

cross_val_results = {}

for symbol in symbols:
    for tf in timeframes:
        key = f"{symbol}_{tf}"
        df = load_ohlcv(symbol, tf)

        # Run all strategies
        for strat_config in strategies_to_compare:
            strategy = create_strategy(strat_config["name"], sys_cfg)
            engine = BacktestEngine(strategy, sys_cfg)
            result = engine.run(df.copy())

            if key not in cross_val_results:
                cross_val_results[key] = {}

            cross_val_results[key][strat_config["label"]] = result.metrics.sharpe

# Create cross-validation DataFrame
cross_val_df = pd.DataFrame(cross_val_results).T

# Calculate mean rank across all conditions
cross_val_df["mean_rank"] = cross_val_df.rank(axis=1, ascending=False).mean(axis=1)

print("Mean Rank Across Conditions:")
print(cross_val_df["mean_rank"].sort_values())
```

**Goal**: Identify strategies that consistently rank high across different conditions.

---

## Step 8: Generate Comparison Report

```python
from finantradealgo.research.reporting import Report, ReportSection

# Create report
report = Report(
    title="Multi-Strategy Comparison Report",
    description=f"Comparing {len(strategies_to_compare)} strategies on AIAUSDT/15m",
)

# Section 1: Performance Rankings
rankings_section = ReportSection(
    title="Performance Rankings",
    content="Strategies ranked by Sharpe ratio.",
    data={"Rankings": comparison_df},
)
report.add_section(rankings_section)

# Section 2: Trade-offs Analysis
tradeoffs_content = f"""
**Top Performer**: {comparison_df.index[0]} (Sharpe: {comparison_df.iloc[0]['sharpe']:.4f})

**Best Risk Management**: {comparison_df.nsmallest(1, 'max_dd').index[0]} (Max DD: {comparison_df['max_dd'].min():.4f})

**Highest Win Rate**: {comparison_df.nlargest(1, 'win_rate').index[0]} (Win Rate: {comparison_df['win_rate'].max():.4f})

**Trade-offs**:
- High Sharpe strategies may have fewer trades (lower frequency)
- Low drawdown strategies may sacrifice returns
- High win rate doesn't guarantee high Sharpe
"""

tradeoffs_section = ReportSection(
    title="Trade-offs Analysis",
    content=tradeoffs_content.strip(),
)
report.add_section(tradeoffs_section)

# Section 3: Recommendations
recommendations_section = ReportSection(
    title="Recommendations",
    content=f"""
**Primary Recommendation**: {comparison_df.index[0]}
- Best risk-adjusted returns (Sharpe: {comparison_df.iloc[0]['sharpe']:.4f})
- Suitable for live trading if other criteria met

**Alternative Options**:
- Consider {comparison_df.index[1]} for lower drawdown exposure
- Consider ensemble combining top 2-3 strategies

**Next Steps**:
1. Run robustness tests on top 3 strategies
2. Validate on out-of-sample data
3. Test on multiple symbols/timeframes
4. Paper trade top performer for 1-2 weeks
""",
)
report.add_section(recommendations_section)

# Save report
report.save("reports/strategy_comparison_AIAUSDT_15m.html")
```

---

## Quality Checklist

### Comparison Integrity
- [ ] All strategies tested on identical data
- [ ] Same system config (risk params, fees, slippage) for all
- [ ] No lookahead bias in any strategy
- [ ] Warmup periods consistent

### Analysis Completeness
- [ ] Primary and secondary metrics calculated
- [ ] Trade-offs identified (return vs risk, frequency vs performance)
- [ ] Statistical significance tested (if applicable)
- [ ] Cross-validation performed (multiple symbols/timeframes)

### Documentation
- [ ] Comparison table saved
- [ ] Report generated and saved
- [ ] Strategy configurations documented
- [ ] Rationale for ranking criteria documented

### Validation
- [ ] Top performer validated on out-of-sample data
- [ ] Performance differences are meaningful (not noise)
- [ ] Recommendations are actionable

---

## Common Pitfalls

### ❌ Comparing on Different Data
**Problem**: Strategy A tested on 2023 data, Strategy B on 2024 data.

**Solution**:
- Use identical datasets for all strategies
- Same date range, same symbol, same timeframe
- Ensure data quality is consistent

### ❌ Ignoring Transaction Costs
**Problem**: High-frequency strategy looks best, but costs not accounted for.

**Solution**:
- Include realistic fees and slippage in backtest
- Calculate net Sharpe (after costs)
- Penalize strategies with excessive trading

### ❌ Cherry-Picking Metrics
**Problem**: Highlighting win rate for one strategy, Sharpe for another.

**Solution**:
- Use consistent primary metric for ranking
- Report all metrics for transparency
- Acknowledge trade-offs explicitly

### ❌ Overfitting to Single Market Condition
**Problem**: All strategies tested on 2024 bull market only.

**Solution**:
- Test across multiple time periods (bull, bear, sideways)
- Cross-validate on different symbols
- Use walk-forward analysis if possible

---

## Next Steps

After completing this playbook:

1. **If clear winner**: Proceed to [Robustness Testing](robustness_testing.md)
2. **If no clear winner**: Consider [Ensemble Development](ensemble_development.md)
3. **If performance similar**: Analyze [Regime Analysis](regime_analysis.md) to see if strategies excel in different conditions
4. **If ready for production**: Paper trade top 1-2 strategies

---

**Remember**: The "best" strategy depends on your objectives. A lower Sharpe but more consistent strategy may be preferable to a high-but-volatile one.

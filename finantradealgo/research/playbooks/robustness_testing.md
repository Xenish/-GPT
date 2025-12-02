# Playbook: Robustness Testing

**Objective**: Validate strategy performance before live deployment through comprehensive out-of-sample testing.

**Duration**: 2-4 hours

**Output**: Robustness validation report, deployment readiness checklist, risk assessment

---

## Prerequisites

### Strategy
- ✅ Strategy has positive in-sample performance (Sharpe > 0.5)
- ✅ Strategy parameters finalized (no further optimization)
- ✅ Strategy logic is well-documented

### Data
- ✅ Historical data for multiple symbols and timeframes
- ✅ Sufficient data for train/test split (at least 6+ months)
- ✅ Out-of-sample data available (not used in strategy development)

### Knowledge
- ✅ Understand overfitting risks
- ✅ Familiar with walk-forward analysis
- ✅ Know what "good enough" means for your risk tolerance

---

## What is Robustness Testing?

**Robustness** means the strategy performs consistently across:
- Different time periods (out-of-sample)
- Different symbols (cross-asset validation)
- Different timeframes (scalability)
- Slight parameter variations (stability)
- Different market regimes (adaptability)

**Why It Matters**:
- **Overfitting**: A strategy optimized on historical data may fail in live trading
- **Regime Dependency**: A strategy that works in bull markets may fail in bear markets
- **Parameter Sensitivity**: Fragile strategies break with minor parameter changes
- **Deployment Confidence**: Robustness testing gives confidence before risking real capital

---

## Step 1: Out-of-Sample (OOS) Validation

### Train/Test Split

```python
from finantradealgo.data.ohlcv_loader import load_ohlcv
from finantradealgo.strategies.strategy_engine import create_strategy
from finantradealgo.backtesting.backtest_engine import BacktestEngine
from finantradealgo.config import load_config

# Load data
df = load_ohlcv("AIAUSDT", "15m")

# Split: 70% train, 30% test
split_idx = int(len(df) * 0.7)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

print(f"Train period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
print(f"Test period: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

# Load config and strategy
sys_cfg = load_config("config/research_config.yaml")
strategy = create_strategy("trend_continuation", sys_cfg)

# Backtest on train
engine_train = BacktestEngine(strategy, sys_cfg)
result_train = engine_train.run(df_train)

# Backtest on test (out-of-sample)
engine_test = BacktestEngine(strategy, sys_cfg)
result_test = engine_test.run(df_test)

# Compare
print("\n=== Performance Comparison ===")
print(f"Train Sharpe: {result_train.metrics.sharpe:.4f}")
print(f"Test Sharpe: {result_test.metrics.sharpe:.4f}")
print(f"Degradation: {((result_test.metrics.sharpe - result_train.metrics.sharpe) / result_train.metrics.sharpe * 100):.2f}%")
```

**Success Criteria**:
- [ ] Test Sharpe > 0.3 (still profitable)
- [ ] Test Sharpe >= 50% of Train Sharpe (degradation < 50%)
- [ ] Test win rate >= 40%
- [ ] Test max drawdown < 30%

**Red Flags**:
- ❌ Test Sharpe < 0 (strategy loses money OOS)
- ❌ Test Sharpe < 30% of Train Sharpe (severe overfitting)
- ❌ Test win rate < 30% (poor signal quality)

---

## Step 2: Walk-Forward Analysis

Simulate realistic rolling optimization:

```python
def walk_forward_analysis(
    df: pd.DataFrame,
    strategy_name: str,
    sys_cfg,
    train_size: int = 1000,  # bars
    test_size: int = 300,    # bars
    step_size: int = 300,    # bars
):
    """
    Perform walk-forward analysis.

    Args:
        df: OHLCV DataFrame
        strategy_name: Name of strategy
        sys_cfg: System config
        train_size: Size of training window
        test_size: Size of testing window
        step_size: Step size for rolling window

    Returns:
        List of test results
    """
    results = []
    current_idx = 0

    while current_idx + train_size + test_size <= len(df):
        # Split data
        train_start = current_idx
        train_end = current_idx + train_size
        test_start = train_end
        test_end = test_start + test_size

        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[test_start:test_end].copy()

        # Backtest on test period
        strategy = create_strategy(strategy_name, sys_cfg)
        engine = BacktestEngine(strategy, sys_cfg)
        result = engine.run(df_test)

        results.append({
            "test_start": df_test["timestamp"].iloc[0],
            "test_end": df_test["timestamp"].iloc[-1],
            "sharpe": result.metrics.sharpe,
            "cum_return": result.metrics.cum_return,
            "max_dd": result.metrics.max_drawdown,
            "trade_count": result.metrics.trade_count,
        })

        # Move forward
        current_idx += step_size

    return pd.DataFrame(results)

# Run walk-forward analysis
wf_results = walk_forward_analysis(df, "trend_continuation", sys_cfg)

print("Walk-Forward Results:")
print(wf_results)

# Calculate statistics
print(f"\nMean OOS Sharpe: {wf_results['sharpe'].mean():.4f}")
print(f"Median OOS Sharpe: {wf_results['sharpe'].median():.4f}")
print(f"Std Dev: {wf_results['sharpe'].std():.4f}")
print(f"Min Sharpe: {wf_results['sharpe'].min():.4f}")
print(f"Max Sharpe: {wf_results['sharpe'].max():.4f}")
print(f"% Positive Periods: {(wf_results['sharpe'] > 0).mean() * 100:.1f}%")
```

**Success Criteria**:
- [ ] Mean OOS Sharpe > 0.3
- [ ] Median OOS Sharpe > 0.2
- [ ] At least 60% of periods have positive Sharpe
- [ ] Standard deviation of Sharpe < 0.5

---

## Step 3: Cross-Symbol Validation

Test strategy on different symbols:

```python
symbols = ["AIAUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
cross_symbol_results = []

for symbol in symbols:
    try:
        df = load_ohlcv(symbol, "15m")

        strategy = create_strategy("trend_continuation", sys_cfg)
        engine = BacktestEngine(strategy, sys_cfg)
        result = engine.run(df)

        cross_symbol_results.append({
            "symbol": symbol,
            "sharpe": result.metrics.sharpe,
            "cum_return": result.metrics.cum_return,
            "max_dd": result.metrics.max_drawdown,
            "trade_count": result.metrics.trade_count,
            "win_rate": result.metrics.win_rate,
        })

    except Exception as e:
        print(f"Error with {symbol}: {e}")

cross_symbol_df = pd.DataFrame(cross_symbol_results)
print(cross_symbol_df)

# Summary statistics
print(f"\nMean Sharpe: {cross_symbol_df['sharpe'].mean():.4f}")
print(f"Min Sharpe: {cross_symbol_df['sharpe'].min():.4f}")
print(f"% Positive Sharpe: {(cross_symbol_df['sharpe'] > 0).mean() * 100:.1f}%")
```

**Success Criteria**:
- [ ] Positive Sharpe on at least 60% of symbols
- [ ] Mean Sharpe > 0.3
- [ ] No symbol has Sharpe < -0.5 (catastrophic failure)
- [ ] Strategy is profitable on primary trading symbols

---

## Step 4: Cross-Timeframe Validation

Test strategy on different timeframes:

```python
timeframes = ["15m", "30m", "1h", "4h"]
cross_tf_results = []

for tf in timeframes:
    try:
        df = load_ohlcv("AIAUSDT", tf)

        strategy = create_strategy("trend_continuation", sys_cfg)
        engine = BacktestEngine(strategy, sys_cfg)
        result = engine.run(df)

        cross_tf_results.append({
            "timeframe": tf,
            "sharpe": result.metrics.sharpe,
            "cum_return": result.metrics.cum_return,
            "max_dd": result.metrics.max_drawdown,
            "trade_count": result.metrics.trade_count,
        })

    except Exception as e:
        print(f"Error with {tf}: {e}")

cross_tf_df = pd.DataFrame(cross_tf_results)
print(cross_tf_df)
```

**Success Criteria**:
- [ ] Positive Sharpe on primary timeframe
- [ ] Strategy works on at least 2 timeframes
- [ ] Performance doesn't degrade severely with timeframe changes

**Note**: Not all strategies need to work on all timeframes. A 15m scalping strategy may not work on 4h.

---

## Step 5: Parameter Sensitivity Analysis

Test parameter stability:

```python
def parameter_sensitivity_test(
    df: pd.DataFrame,
    strategy_name: str,
    param_name: str,
    base_value: float,
    variations: list,  # e.g., [-20%, -10%, +10%, +20%]
    sys_cfg,
):
    """
    Test strategy performance with parameter variations.

    Args:
        df: OHLCV DataFrame
        strategy_name: Strategy name
        param_name: Parameter to vary
        base_value: Base parameter value
        variations: List of percentage variations (e.g., [-0.2, -0.1, 0.1, 0.2])
        sys_cfg: System config

    Returns:
        DataFrame with results
    """
    results = []

    for var in variations:
        # Calculate new param value
        new_value = base_value * (1 + var)

        # Create strategy with modified parameter
        strategy = create_strategy(strategy_name, sys_cfg)
        setattr(strategy, param_name, new_value)

        # Backtest
        engine = BacktestEngine(strategy, sys_cfg)
        result = engine.run(df.copy())

        results.append({
            "variation": f"{var * 100:+.0f}%",
            "param_value": new_value,
            "sharpe": result.metrics.sharpe,
            "cum_return": result.metrics.cum_return,
            "max_dd": result.metrics.max_drawdown,
        })

    return pd.DataFrame(results)

# Example: Test trend_min parameter
sensitivity_results = parameter_sensitivity_test(
    df=df,
    strategy_name="trend_continuation",
    param_name="trend_min",
    base_value=1.0,
    variations=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
    sys_cfg=sys_cfg,
)

print("Parameter Sensitivity Results:")
print(sensitivity_results)

# Calculate stability metric
sharpe_std = sensitivity_results["sharpe"].std()
sharpe_range = sensitivity_results["sharpe"].max() - sensitivity_results["sharpe"].min()

print(f"\nSharpe Std Dev: {sharpe_std:.4f}")
print(f"Sharpe Range: {sharpe_range:.4f}")

# Stable if Sharpe doesn't change drastically with ±20% param change
if sharpe_std < 0.3:
    print("✅ Parameter is STABLE")
else:
    print("⚠️  Parameter is SENSITIVE - strategy may be fragile")
```

**Success Criteria**:
- [ ] Sharpe remains positive across ±20% parameter variations
- [ ] Sharpe standard deviation < 0.3
- [ ] No single parameter value causes catastrophic failure

---

## Step 6: Monte Carlo Simulation

Simulate random trade sequences to assess strategy risk:

```python
import numpy as np

def monte_carlo_simulation(
    trades_df: pd.DataFrame,
    n_simulations: int = 1000,
    n_trades: int = None,
):
    """
    Run Monte Carlo simulation by randomly resampling trades.

    Args:
        trades_df: DataFrame with 'pnl' column
        n_simulations: Number of simulations
        n_trades: Number of trades per simulation (default: same as original)

    Returns:
        DataFrame with simulation results
    """
    if n_trades is None:
        n_trades = len(trades_df)

    trade_pnls = trades_df["pnl"].values

    simulation_results = []

    for i in range(n_simulations):
        # Randomly sample trades with replacement
        sampled_pnls = np.random.choice(trade_pnls, size=n_trades, replace=True)

        # Calculate metrics
        cum_pnl = sampled_pnls.sum()
        sharpe = sampled_pnls.mean() / sampled_pnls.std() if sampled_pnls.std() > 0 else 0

        # Calculate drawdown
        cumulative = np.cumsum(sampled_pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        max_dd = drawdown.min()

        simulation_results.append({
            "cum_pnl": cum_pnl,
            "sharpe": sharpe,
            "max_dd": max_dd,
        })

    return pd.DataFrame(simulation_results)

# Run Monte Carlo
mc_results = monte_carlo_simulation(result.trades_df, n_simulations=1000)

# Calculate percentiles
sharpe_5th = mc_results["sharpe"].quantile(0.05)
sharpe_median = mc_results["sharpe"].median()
sharpe_95th = mc_results["sharpe"].quantile(0.95)

print("Monte Carlo Results (1000 simulations):")
print(f"5th Percentile Sharpe: {sharpe_5th:.4f}")
print(f"Median Sharpe: {sharpe_median:.4f}")
print(f"95th Percentile Sharpe: {sharpe_95th:.4f}")
print(f"% Simulations with Sharpe > 0: {(mc_results['sharpe'] > 0).mean() * 100:.1f}%")

# Risk assessment
if sharpe_5th > 0:
    print("✅ ROBUST: 95% confidence of positive Sharpe")
elif sharpe_median > 0:
    print("⚠️  MODERATE: Median Sharpe positive, but some risk of losses")
else:
    print("❌ HIGH RISK: Median Sharpe negative, strategy may not be profitable")
```

**Success Criteria**:
- [ ] 5th percentile Sharpe > 0 (95% confidence of profitability)
- [ ] Median Sharpe > 0.3
- [ ] At least 70% of simulations have positive Sharpe

---

## Step 7: Stress Testing

Test strategy under adverse conditions:

### A. Transaction Cost Sensitivity

```python
# Test with higher transaction costs
original_fee = sys_cfg.fee_rate
stress_fees = [original_fee * 1.5, original_fee * 2.0, original_fee * 3.0]

for fee in stress_fees:
    sys_cfg.fee_rate = fee
    engine = BacktestEngine(strategy, sys_cfg)
    result = engine.run(df.copy())

    print(f"Fee Rate: {fee:.4f} | Sharpe: {result.metrics.sharpe:.4f}")

# Reset
sys_cfg.fee_rate = original_fee
```

### B. Slippage Sensitivity

```python
# Test with higher slippage
slippage_scenarios = [0.0005, 0.001, 0.002, 0.005]  # 0.05% to 0.5%

for slippage in slippage_scenarios:
    sys_cfg.slippage = slippage
    engine = BacktestEngine(strategy, sys_cfg)
    result = engine.run(df.copy())

    print(f"Slippage: {slippage:.4f} | Sharpe: {result.metrics.sharpe:.4f}")
```

**Success Criteria**:
- [ ] Strategy remains profitable with 2x transaction costs
- [ ] Strategy Sharpe > 0 with 0.2% slippage

---

## Step 8: Generate Robustness Report

```python
from finantradealgo.research.reporting import Report, ReportSection

# Create report
report = Report(
    title="Robustness Testing Report",
    description=f"Comprehensive validation for trend_continuation strategy on AIAUSDT/15m",
)

# Section 1: Out-of-Sample Performance
oos_content = f"""
**Train Period**: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}
- Train Sharpe: {result_train.metrics.sharpe:.4f}
- Train Return: {result_train.metrics.cum_return:.2%}

**Test Period** (Out-of-Sample): {df_test['timestamp'].min()} to {df_test['timestamp'].max()}
- Test Sharpe: {result_test.metrics.sharpe:.4f}
- Test Return: {result_test.metrics.cum_return:.2%}

**Degradation**: {((result_test.metrics.sharpe - result_train.metrics.sharpe) / result_train.metrics.sharpe * 100):.2f}%

**Assessment**: {"✅ PASS" if result_test.metrics.sharpe > 0.3 and result_test.metrics.sharpe >= result_train.metrics.sharpe * 0.5 else "❌ FAIL"}
"""

oos_section = ReportSection(
    title="Out-of-Sample Validation",
    content=oos_content.strip(),
)
report.add_section(oos_section)

# Section 2: Walk-Forward Analysis
wf_section = ReportSection(
    title="Walk-Forward Analysis",
    content=f"""
**Rolling Window Configuration**:
- Train Size: 1000 bars
- Test Size: 300 bars
- Step Size: 300 bars
- Number of Periods: {len(wf_results)}

**Out-of-Sample Statistics**:
- Mean Sharpe: {wf_results['sharpe'].mean():.4f}
- Median Sharpe: {wf_results['sharpe'].median():.4f}
- Std Dev: {wf_results['sharpe'].std():.4f}
- % Positive Periods: {(wf_results['sharpe'] > 0).mean() * 100:.1f}%

**Assessment**: {"✅ PASS" if wf_results['sharpe'].mean() > 0.3 and (wf_results['sharpe'] > 0).mean() >= 0.6 else "❌ FAIL"}
""",
    data={"Walk-Forward Results": wf_results},
)
report.add_section(wf_section)

# Section 3: Cross-Validation
cross_val_section = ReportSection(
    title="Cross-Symbol Validation",
    content=f"""
**Symbols Tested**: {len(cross_symbol_df)}

**Summary**:
- Mean Sharpe: {cross_symbol_df['sharpe'].mean():.4f}
- Min Sharpe: {cross_symbol_df['sharpe'].min():.4f}
- % Positive: {(cross_symbol_df['sharpe'] > 0).mean() * 100:.1f}%

**Assessment**: {"✅ PASS" if (cross_symbol_df['sharpe'] > 0).mean() >= 0.6 else "❌ FAIL"}
""",
    data={"Cross-Symbol Results": cross_symbol_df},
)
report.add_section(cross_val_section)

# Section 4: Monte Carlo Risk Assessment
mc_section = ReportSection(
    title="Monte Carlo Simulation",
    content=f"""
**Simulations**: 1000

**Sharpe Percentiles**:
- 5th: {mc_results['sharpe'].quantile(0.05):.4f}
- 50th (Median): {mc_results['sharpe'].median():.4f}
- 95th: {mc_results['sharpe'].quantile(0.95):.4f}

**Risk Level**: {"✅ LOW RISK" if mc_results['sharpe'].quantile(0.05) > 0 else "⚠️  MODERATE RISK" if mc_results['sharpe'].median() > 0 else "❌ HIGH RISK"}

**Assessment**: {"✅ PASS" if mc_results['sharpe'].median() > 0.3 else "❌ FAIL"}
""",
)
report.add_section(mc_section)

# Section 5: Final Recommendation
final_recommendation = """
Based on comprehensive robustness testing:

**Deployment Readiness**:
- Out-of-Sample: ✅ PASS
- Walk-Forward: ✅ PASS
- Cross-Symbol: ✅ PASS
- Monte Carlo: ✅ PASS
- Parameter Sensitivity: ✅ PASS

**Overall Assessment**: ✅ STRATEGY IS READY FOR PAPER TRADING

**Recommended Next Steps**:
1. Deploy to paper trading environment for 1-2 weeks
2. Monitor live performance vs backtest expectations
3. Set stop-loss at -15% drawdown threshold
4. Review performance after 50 live trades

**Risk Warnings**:
- Strategy performance may degrade in live trading due to execution slippage
- Monitor regime changes (strategy optimized for trending markets)
- Set position size limits to manage risk
"""

recommendation_section = ReportSection(
    title="Final Recommendation",
    content=final_recommendation.strip(),
)
report.add_section(recommendation_section)

# Save report
report.save("reports/robustness_testing_trend_continuation.html")
print("[PASS] Robustness report saved.")
```

---

## Quality Checklist

### Out-of-Sample Testing
- [ ] Train/test split performed (70/30 or 80/20)
- [ ] Test Sharpe > 0.3
- [ ] Test degradation < 50%
- [ ] Walk-forward analysis completed
- [ ] At least 60% of WF periods are profitable

### Cross-Validation
- [ ] Tested on 3+ symbols
- [ ] Tested on 2+ timeframes
- [ ] Positive Sharpe on majority of symbols
- [ ] No catastrophic failures

### Sensitivity Analysis
- [ ] Parameter sensitivity tested (±20% variations)
- [ ] Strategy is stable (Sharpe std dev < 0.3)
- [ ] Monte Carlo simulation run (1000+ iterations)
- [ ] 5th percentile Sharpe > 0

### Stress Testing
- [ ] Transaction cost sensitivity tested (2x fees)
- [ ] Slippage sensitivity tested (0.2%+)
- [ ] Strategy remains profitable under stress

### Documentation
- [ ] Robustness report generated and saved
- [ ] All test results documented
- [ ] Deployment readiness decision documented
- [ ] Risk warnings identified

---

## Common Pitfalls

### ❌ Insufficient Out-of-Sample Data
**Problem**: Using only 10% of data for OOS testing.

**Solution**:
- Use at least 20-30% for OOS
- Ensure OOS period includes different market conditions
- Use walk-forward for realistic validation

### ❌ Data Leakage
**Problem**: Using test data to inform strategy decisions.

**Solution**:
- Strictly separate train and test data
- Never re-optimize based on test results
- Use walk-forward to avoid leakage

### ❌ Accepting Marginal Performance
**Problem**: Deploying strategy with OOS Sharpe of 0.1.

**Solution**:
- Set minimum thresholds (Sharpe > 0.3)
- Require robustness across multiple tests
- Don't deploy if uncertain

### ❌ Ignoring Transaction Costs
**Problem**: Strategy profitable in backtest, loses money live due to fees.

**Solution**:
- Always include realistic fees and slippage
- Stress test with 2-3x higher costs
- Avoid high-frequency strategies if costs are high

---

## Deployment Decision Framework

Use this framework to decide if strategy is ready:

| Test | Pass Threshold | Result | Pass? |
|------|---------------|--------|-------|
| OOS Sharpe | > 0.3 | 0.45 | ✅ |
| OOS Degradation | < 50% | 32% | ✅ |
| Walk-Forward Mean Sharpe | > 0.3 | 0.38 | ✅ |
| Walk-Forward % Positive | > 60% | 68% | ✅ |
| Cross-Symbol % Positive | > 60% | 75% | ✅ |
| Monte Carlo 5th %ile | > 0 | 0.12 | ✅ |
| Param Sensitivity Std | < 0.3 | 0.21 | ✅ |
| 2x Fee Sharpe | > 0 | 0.23 | ✅ |

**Overall**: ✅ **PASS** - Strategy is robust and ready for paper trading.

---

## Next Steps

After completing robustness testing:

1. **If PASS**: Deploy to paper trading, monitor for 1-2 weeks
2. **If MARGINAL**: Revisit strategy design, consider ensemble approach
3. **If FAIL**: Return to [Strategy Parameter Search](strategy_param_search.md) or abandon strategy

---

**Remember**: A robust strategy is one that performs consistently across time, symbols, parameters, and market conditions. Don't skip this step.

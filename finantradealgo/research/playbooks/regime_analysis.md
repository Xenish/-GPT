# Playbook: Market Regime Analysis

**Objective**: Analyze strategy performance across different market regimes to understand when strategies work best.

**Duration**: 1-2 hours

**Output**: Regime classification, regime-specific performance metrics, regime-aware recommendations

---

## Prerequisites

### Data
- ✅ Historical OHLCV data (at least 6+ months for meaningful regime analysis)
- ✅ Sufficient data to capture different market conditions (bull, bear, sideways)

### Strategies
- ✅ 1+ strategies to analyze across regimes
- ✅ Strategy has been backtested with baseline metrics

### Knowledge
- ✅ Understand market regimes (trend, mean-reversion, volatility)
- ✅ Familiar with regime detection methods

---

## What Are Market Regimes?

Market regimes are distinct periods characterized by different price dynamics:

**Common Regime Types**:
1. **Trending** (Bull/Bear): Persistent directional movement
2. **Mean-Reverting** (Sideways/Range-Bound): Price oscillates around mean
3. **High Volatility**: Large price swings, uncertainty
4. **Low Volatility**: Quiet, compressed price action

**Why Regime Analysis Matters**:
- Trend-following strategies excel in trending regimes but fail in sideways markets
- Mean-reversion strategies excel in range-bound regimes but suffer in trends
- Understanding regime-dependent performance helps with:
  - Strategy selection (switch strategies based on regime)
  - Ensemble design (weight strategies by regime)
  - Risk management (reduce exposure in unfavorable regimes)

---

## Step 1: Choose Regime Detection Method

### Method A: Trend Strength (ADX-Based)

```python
import pandas as pd
import numpy as np

def classify_regime_adx(df: pd.DataFrame, adx_period: int = 14, threshold: float = 25) -> pd.Series:
    """
    Classify regime based on ADX (Average Directional Index).

    - ADX > threshold: Trending
    - ADX <= threshold: Sideways/Mean-reverting

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close'
        adx_period: ADX calculation period
        threshold: ADX threshold for trending regime

    Returns:
        Series with regime labels: "trending" or "sideways"
    """
    from ta.trend import ADXIndicator

    adx_indicator = ADXIndicator(df["high"], df["low"], df["close"], window=adx_period)
    df["adx"] = adx_indicator.adx()

    regime = np.where(df["adx"] > threshold, "trending", "sideways")
    return pd.Series(regime, index=df.index)
```

### Method B: Volatility-Based

```python
def classify_regime_volatility(df: pd.DataFrame, atr_period: int = 14, vol_percentile: float = 0.7) -> pd.Series:
    """
    Classify regime based on ATR (volatility).

    - ATR > 70th percentile: High volatility
    - ATR <= 70th percentile: Low volatility

    Args:
        df: OHLCV DataFrame
        atr_period: ATR period
        vol_percentile: Percentile threshold

    Returns:
        Series with regime labels: "high_vol" or "low_vol"
    """
    from ta.volatility import AverageTrueRange

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=atr_period)
    df["atr"] = atr.average_true_range()

    threshold = df["atr"].quantile(vol_percentile)
    regime = np.where(df["atr"] > threshold, "high_vol", "low_vol")
    return pd.Series(regime, index=df.index)
```

### Method C: Price Action (Bull/Bear/Sideways)

```python
def classify_regime_price_action(df: pd.DataFrame, ma_period: int = 50, slope_threshold: float = 0.001) -> pd.Series:
    """
    Classify regime based on moving average slope.

    - MA slope > threshold: Bull (uptrend)
    - MA slope < -threshold: Bear (downtrend)
    - Otherwise: Sideways

    Args:
        df: OHLCV DataFrame with 'close'
        ma_period: Moving average period
        slope_threshold: Minimum slope for trend detection

    Returns:
        Series with regime labels: "bull", "bear", or "sideways"
    """
    df["ma"] = df["close"].rolling(ma_period).mean()
    df["ma_slope"] = df["ma"].pct_change(periods=5)  # 5-bar slope

    regime = np.where(
        df["ma_slope"] > slope_threshold, "bull",
        np.where(df["ma_slope"] < -slope_threshold, "bear", "sideways")
    )
    return pd.Series(regime, index=df.index)
```

### Method D: Combined (Recommended)

```python
def classify_regime_combined(df: pd.DataFrame) -> pd.Series:
    """
    Combine multiple methods for robust regime classification.

    Classifies into 4 regimes:
    - trending_bull: ADX > 25 AND price above MA50 AND slope > 0
    - trending_bear: ADX > 25 AND price below MA50 AND slope < 0
    - sideways_high_vol: ADX <= 25 AND ATR > 70th percentile
    - sideways_low_vol: ADX <= 25 AND ATR <= 70th percentile

    Returns:
        Series with regime labels
    """
    from ta.trend import ADXIndicator
    from ta.volatility import AverageTrueRange

    # Calculate indicators
    adx_indicator = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx_indicator.adx()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()

    df["ma50"] = df["close"].rolling(50).mean()
    df["ma_slope"] = df["ma50"].pct_change(5)

    # Classify
    regime = []
    for idx, row in df.iterrows():
        if pd.isna(row["adx"]) or pd.isna(row["atr"]):
            regime.append("unknown")
            continue

        if row["adx"] > 25:  # Trending
            if row["close"] > row["ma50"] and row["ma_slope"] > 0:
                regime.append("trending_bull")
            elif row["close"] < row["ma50"] and row["ma_slope"] < 0:
                regime.append("trending_bear")
            else:
                regime.append("trending_mixed")
        else:  # Sideways
            atr_threshold = df["atr"].quantile(0.7)
            if row["atr"] > atr_threshold:
                regime.append("sideways_high_vol")
            else:
                regime.append("sideways_low_vol")

    return pd.Series(regime, index=df.index)
```

---

## Step 2: Apply Regime Classification

```python
from finantradealgo.data.ohlcv_loader import load_ohlcv

# Load data
df = load_ohlcv("AIAUSDT", "15m")

# Classify regimes (choose one method)
df["regime"] = classify_regime_combined(df)

# Check regime distribution
print("Regime Distribution:")
print(df["regime"].value_counts())
print("\nRegime Percentages:")
print(df["regime"].value_counts(normalize=True) * 100)
```

**Example Output**:
```
Regime Distribution:
sideways_low_vol      3421
trending_bull         2134
sideways_high_vol     1823
trending_bear          987
unknown                235

Regime Percentages:
sideways_low_vol     40.12%
trending_bull        25.03%
sideways_high_vol    21.38%
trending_bear        11.58%
unknown               2.76%
```

---

## Step 3: Run Strategy Backtest with Regime Tracking

```python
from finantradealgo.strategies.strategy_engine import create_strategy
from finantradealgo.backtesting.backtest_engine import BacktestEngine
from finantradealgo.config import load_config

# Load config
sys_cfg = load_config("config/research_config.yaml")

# Create strategy
strategy = create_strategy("trend_continuation", sys_cfg)

# Run backtest
engine = BacktestEngine(strategy, sys_cfg)
result = engine.run(df.copy())

# Add regime to trades
trades_df = result.trades_df.copy()
trades_df["regime"] = trades_df["entry_time"].map(
    df.set_index("timestamp")["regime"]
)

print("Trades by Regime:")
print(trades_df["regime"].value_counts())
```

---

## Step 4: Calculate Regime-Specific Performance

```python
def calculate_regime_performance(trades_df: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
    """
    Calculate performance metrics per regime.

    Args:
        trades_df: DataFrame with trades (must have 'pnl', 'regime' columns)
        regime_col: Name of regime column

    Returns:
        DataFrame with regime-specific metrics
    """
    regime_stats = []

    for regime in trades_df[regime_col].dropna().unique():
        regime_trades = trades_df[trades_df[regime_col] == regime].copy()

        if len(regime_trades) == 0:
            continue

        # Calculate metrics
        total_pnl = regime_trades["pnl"].sum()
        avg_pnl = regime_trades["pnl"].mean()
        win_rate = (regime_trades["pnl"] > 0).sum() / len(regime_trades)
        sharpe = regime_trades["pnl"].mean() / regime_trades["pnl"].std() if regime_trades["pnl"].std() > 0 else 0

        regime_stats.append({
            "regime": regime,
            "trade_count": len(regime_trades),
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "win_rate": win_rate,
            "sharpe": sharpe,
        })

    return pd.DataFrame(regime_stats).sort_values("sharpe", ascending=False)

# Calculate
regime_performance = calculate_regime_performance(trades_df)
print(regime_performance)
```

**Example Output**:
```
              regime  trade_count  total_pnl  avg_pnl  win_rate   sharpe
0      trending_bull           45      0.234   0.0052    0.6222   0.8234
1  sideways_high_vol           23      0.087   0.0038    0.5652   0.4512
2   sideways_low_vol           67     -0.045  -0.0007    0.4925  -0.2134
3      trending_bear           12     -0.123  -0.0103    0.3333  -0.6543
```

**Interpretation**:
- Strategy performs best in **trending_bull** regime (Sharpe: 0.82)
- Loses money in **sideways_low_vol** and **trending_bear** regimes
- This is a trend-following strategy (as expected)

---

## Step 5: Visualize Regime Performance

```python
import matplotlib.pyplot as plt

# Plot regime performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sharpe by Regime
axes[0, 0].bar(regime_performance["regime"], regime_performance["sharpe"], color="steelblue")
axes[0, 0].set_title("Sharpe Ratio by Regime")
axes[0, 0].set_xlabel("Regime")
axes[0, 0].set_ylabel("Sharpe Ratio")
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Trade Count by Regime
axes[0, 1].bar(regime_performance["regime"], regime_performance["trade_count"], color="coral")
axes[0, 1].set_title("Trade Count by Regime")
axes[0, 1].set_xlabel("Regime")
axes[0, 1].set_ylabel("# Trades")
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Win Rate by Regime
axes[1, 0].bar(regime_performance["regime"], regime_performance["win_rate"], color="seagreen")
axes[1, 0].set_title("Win Rate by Regime")
axes[1, 0].set_xlabel("Regime")
axes[1, 0].set_ylabel("Win Rate")
axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Avg PnL by Regime
axes[1, 1].bar(regime_performance["regime"], regime_performance["avg_pnl"], color="purple")
axes[1, 1].set_title("Average PnL by Regime")
axes[1, 1].set_xlabel("Regime")
axes[1, 1].set_ylabel("Avg PnL")
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("regime_analysis.png", dpi=150)
plt.show()
```

---

## Step 6: Regime-Aware Strategy Selection

Based on regime analysis, design regime-aware trading logic:

### Option A: Filter Trades by Favorable Regimes

Only trade when in favorable regimes:

```python
def filter_signals_by_regime(df: pd.DataFrame, favorable_regimes: list) -> pd.DataFrame:
    """
    Filter strategy signals to only favorable regimes.

    Args:
        df: DataFrame with 'long_entry', 'regime' columns
        favorable_regimes: List of regime names to allow trading

    Returns:
        DataFrame with filtered signals
    """
    df_filtered = df.copy()

    # Disable signals in unfavorable regimes
    unfavorable_mask = ~df_filtered["regime"].isin(favorable_regimes)
    df_filtered.loc[unfavorable_mask, "long_entry"] = False
    df_filtered.loc[unfavorable_mask, "short_entry"] = False

    return df_filtered

# Example: Only trade in trending regimes
favorable_regimes = ["trending_bull", "trending_bear"]
df_filtered = filter_signals_by_regime(df, favorable_regimes)

# Re-run backtest
result_filtered = engine.run(df_filtered.copy())
print(f"Original Sharpe: {result.metrics.sharpe:.4f}")
print(f"Regime-Filtered Sharpe: {result_filtered.metrics.sharpe:.4f}")
```

### Option B: Regime-Based Ensemble

Use different strategies for different regimes:

```python
regime_strategy_map = {
    "trending_bull": "trend_continuation",
    "trending_bear": "trend_continuation",
    "sideways_low_vol": "sweep_reversal",
    "sideways_high_vol": "volatility_breakout",
}

# Generate signals based on current regime
df["selected_strategy"] = df["regime"].map(regime_strategy_map)

# This would require custom ensemble logic (future enhancement)
```

---

## Step 7: Generate Regime Analysis Report

```python
from finantradealgo.research.reporting import Report, ReportSection

# Create report
report = Report(
    title="Market Regime Analysis Report",
    description=f"Regime-based performance analysis for trend_continuation strategy on AIAUSDT/15m",
)

# Section 1: Regime Distribution
regime_dist_section = ReportSection(
    title="Regime Distribution",
    content="Distribution of market regimes in the dataset.",
    data={"Regime Counts": df["regime"].value_counts().to_frame("count")},
)
report.add_section(regime_dist_section)

# Section 2: Regime-Specific Performance
regime_perf_section = ReportSection(
    title="Performance by Regime",
    content="Strategy performance across different market regimes.",
    data={"Regime Performance": regime_performance},
)
report.add_section(regime_perf_section)

# Section 3: Recommendations
best_regime = regime_performance.iloc[0]["regime"]
worst_regime = regime_performance.iloc[-1]["regime"]

recommendations_content = f"""
**Best Performing Regime**: {best_regime}
- Sharpe: {regime_performance.iloc[0]['sharpe']:.4f}
- Win Rate: {regime_performance.iloc[0]['win_rate']:.2%}
- Recommendation: Continue trading in this regime

**Worst Performing Regime**: {worst_regime}
- Sharpe: {regime_performance.iloc[-1]['sharpe']:.4f}
- Win Rate: {regime_performance.iloc[-1]['win_rate']:.2%}
- Recommendation: **Avoid trading** in this regime or switch strategy

**Regime-Aware Recommendations**:
1. Implement regime filter to only trade in favorable regimes
2. Consider using different strategies for different regimes (ensemble)
3. Adjust position sizing based on regime (larger in favorable, smaller in unfavorable)
4. Monitor regime transitions for strategy switching signals

**Next Steps**:
- Backtest with regime filter enabled
- Compare original vs regime-filtered performance
- Explore regime-based ensemble strategies
"""

recommendations_section = ReportSection(
    title="Recommendations",
    content=recommendations_content.strip(),
)
report.add_section(recommendations_section)

# Save report
report.save("reports/regime_analysis_trend_continuation.html")
```

---

## Quality Checklist

### Regime Classification
- [ ] Regime detection method chosen and documented
- [ ] Regime distribution is reasonable (no single regime > 80%)
- [ ] Sufficient data in each regime for meaningful analysis
- [ ] Regime labels are interpretable

### Performance Analysis
- [ ] Regime-specific metrics calculated (Sharpe, win rate, trade count)
- [ ] Best and worst regimes identified
- [ ] Performance differences are meaningful (not noise)
- [ ] Visualizations created (charts saved)

### Validation
- [ ] Regime classification validated on different time periods
- [ ] Out-of-sample testing performed
- [ ] Regime-filtered backtest run (if applicable)

### Documentation
- [ ] Regime analysis report generated
- [ ] Regime classification method documented
- [ ] Recommendations are actionable

---

## Common Pitfalls

### ❌ Regime Overfitting
**Problem**: Defining 20 micro-regimes that perfectly match historical performance.

**Solution**:
- Keep regime count reasonable (3-5 regimes max)
- Use simple, interpretable regime definitions
- Validate regimes on out-of-sample data

### ❌ Lookahead Bias in Regime Detection
**Problem**: Using future data to classify current regime (e.g., MA50 uses future bars).

**Solution**:
- Ensure regime indicators are calculated with past data only
- Use lagged regime classification if necessary
- Test regime detection logic carefully

### ❌ Insufficient Data Per Regime
**Problem**: Only 5 trades in "high volatility" regime, not enough for statistics.

**Solution**:
- Ensure at least 20-30 trades per regime for meaningful analysis
- Combine similar regimes if data is sparse
- Use longer historical data

### ❌ Ignoring Regime Transitions
**Problem**: Strategy switches regimes mid-trade, enters in bull but exits in bear.

**Solution**:
- Analyze regime stability (how long regimes last)
- Consider regime transition logic (exit when regime changes)
- Use regime confirmation periods

---

## Next Steps

After completing this playbook:

1. **If regime differences are significant**: Implement regime-aware filtering or ensemble
2. **If no regime sensitivity**: Strategy is robust across regimes, proceed to [Robustness Testing](robustness_testing.md)
3. **If multiple strategies**: Use regime analysis to design [Ensemble Development](ensemble_development.md) with regime-based weighting

---

**Remember**: Market regimes are a model, not reality. Don't over-optimize to historical regime patterns.

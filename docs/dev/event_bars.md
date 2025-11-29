# Event Bars Documentation

## Overview

Event bars are alternative bar aggregation methods that create bars based on market activity rather than fixed time intervals. Unlike traditional time-based bars (1m, 5m, 15m, etc.), event bars adapt to market conditions by aggregating data based on volume, dollar notional, or tick count.

**Why use event bars?**
- **Adaptive sampling**: More bars during active periods, fewer during quiet periods
- **Better signal quality**: Event-based sampling can provide clearer price action signals
- **Normalized data**: Each bar represents similar market activity regardless of time duration

## Source Data Requirements

### Recommended Source Timeframe: 1 Minute

Event bars should be built from **1-minute OHLCV data** for best results.

**Why 1m data?**
- **High resolution**: Captures intra-bar dynamics accurately
- **Precise thresholds**: Volume, dollar, and tick targets are measured precisely
- **No data loss**: Minimal information loss during aggregation
- **Wide availability**: Most exchanges and data providers offer 1m data

**What happens with coarser data?**
```yaml
# BAD: Using 15m source data
bars:
  mode: volume
  target_volume: 5000
  source_timeframe: "15m"  # TOO COARSE!
```

If you aggregate 15m bars into event bars:
- Volume/dollar/tick measurements become chunky and imprecise
- Bar boundaries become less meaningful
- You lose the adaptive sampling benefits

**Configuration validation:**
The system will raise an error if you try to use event bars with source data coarser than 1m:
```python
ValueError: Event bars (mode=volume) should be built from 1m data.
Got source_timeframe=15m. Non-time event bars require high-resolution (1m)
data to accurately capture volume, dollar, or tick thresholds.
```

## Event Bar Modes

### 1. Time Mode (time)

**Description**: Traditional fixed-time bars. This is the default pass-through mode.

**Configuration**:
```yaml
bars:
  mode: time
  # No target needed - just passes through original data
```

**Behavior**: Input DataFrame is returned unchanged.

**Use case**: When you want standard time-based bars (1m, 5m, 15m, etc.)

---

### 2. Volume Mode (volume)

**Description**: Aggregate bars based on cumulative traded volume.

**Configuration**:
```yaml
bars:
  mode: volume
  target_volume: 5000.0  # Close bar when cumulative volume >= 5000
  source_timeframe: "1m"
  keep_partial_last_bar: false
```

**How it works**:
1. Start accumulating volume from 1m bars
2. When cumulative volume reaches or exceeds `target_volume`, close the bar
3. Start a new bar and repeat

**Example**:
```
Source 1m bars:
  09:00 | volume: 1000
  09:01 | volume: 2000  (cumulative: 3000)
  09:02 | volume: 1500  (cumulative: 4500)
  09:03 | volume: 1000  (cumulative: 5500) → CLOSE BAR 1 (total: 5500)
  09:04 | volume: 800   (cumulative: 800)
  09:05 | volume: 3000  (cumulative: 3800)
  09:06 | volume: 1500  (cumulative: 5300) → CLOSE BAR 2 (total: 5300)
```

**Result**: 2 volume bars instead of 7 time bars.

**Use case**:
- Trading volatile instruments where volume spikes indicate significant events
- Normalizing data across different market conditions
- Reducing noise during low-volume periods

---

### 3. Dollar Mode (dollar)

**Description**: Aggregate bars based on cumulative dollar notional (price × volume).

**Configuration**:
```yaml
bars:
  mode: dollar
  target_notional: 100000.0  # Close bar when notional >= $100,000
  source_timeframe: "1m"
  keep_partial_last_bar: false
```

**How it works**:
1. Calculate notional for each 1m bar: `notional = close × volume`
2. Accumulate notional across bars
3. When cumulative notional reaches or exceeds `target_notional`, close the bar

**Example**:
```
Source 1m bars:
  09:00 | close: 100, volume: 500  → notional: 50,000  (cumulative: 50,000)
  09:01 | close: 101, volume: 600  → notional: 60,600  (cumulative: 110,600) → CLOSE BAR 1
  09:02 | close: 102, volume: 300  → notional: 30,600  (cumulative: 30,600)
  09:03 | close: 103, volume: 700  → notional: 72,100  (cumulative: 102,700) → CLOSE BAR 2
```

**Result**: 2 dollar bars.

**Use case**:
- Trading instruments where price changes significantly (dollar normalization)
- Institutional strategies that focus on capital flow
- High-frequency trading where notional matters more than tick count

---

### 4. Tick Mode (tick)

**Description**: Aggregate bars based on the number of source bars (ticks).

**Configuration**:
```yaml
bars:
  mode: tick
  target_ticks: 5  # Close bar after 5 source bars
  source_timeframe: "1m"
  keep_partial_last_bar: false
```

**How it works**:
1. Count source bars (each 1m bar is one "tick")
2. When count reaches `target_ticks`, close the event bar
3. Reset counter and start new bar

**Example**:
```
Source 1m bars:
  09:00, 09:01, 09:02, 09:03, 09:04 → CLOSE BAR 1 (5 ticks)
  09:05, 09:06, 09:07, 09:08, 09:09 → CLOSE BAR 2 (5 ticks)
  09:10, 09:11, 09:12 (only 3 ticks, partial)
```

**Result**:
- With `keep_partial_last_bar=false`: 2 bars
- With `keep_partial_last_bar=true`: 3 bars (last one has only 3 ticks)

**Use case**:
- Simple fixed-ratio resampling
- Testing different sampling rates
- When you want consistent bar counts regardless of market activity

## Time Semantics & Metadata

### Bar Time Columns

Event bars add two important time metadata columns:

#### `bar_start_ts` (column)
- The timestamp when the event bar started accumulating
- Always the timestamp of the first 1m bar included in this event bar
- Type: `pd.Timestamp`

#### `bar_end_ts` (index)
- The timestamp when the event bar closed
- Always the timestamp of the last 1m bar included in this event bar
- Type: `pd.DatetimeIndex`
- This is the DataFrame index

### Time Duration

**Important**: Event bars have **variable duration**.

```python
# Example event bar
bar_start_ts: 2023-01-01 09:00:00
bar_end_ts:   2023-01-01 09:03:00
# This bar spans 3 minutes (4 × 1m bars)

# Next event bar might be different
bar_start_ts: 2023-01-01 09:04:00
bar_end_ts:   2023-01-01 09:11:00
# This bar spans 7 minutes (8 × 1m bars)
```

### Time Boundary Semantics

Event bars follow **inclusive time boundaries**:

```
[bar_start_ts, bar_end_ts]
```

A bar includes ALL data from `bar_start_ts` up to and including `bar_end_ts`.

**Example**:
```
Bar 1: [09:00:00, 09:05:00]  ← Includes 09:00, 09:01, 09:02, 09:03, 09:04, 09:05
Bar 2: [09:05:01, 09:08:00]  ← Starts right after Bar 1 ends
```

**Important for microstructure features**: When calculating features like liquidity sweeps that use trade data, the time boundaries are used to filter trades:

```python
# This uses inclusive boundaries
relevant_trades = trades_df.loc[bar_start_ts:bar_end_ts]
```

### OHLCV Aggregation

When aggregating 1m bars into event bars:

- **Open**: First bar's open
- **High**: Maximum of all bars' highs
- **Low**: Minimum of all bars' lows
- **Close**: Last bar's close
- **Volume**: Sum of all bars' volumes

## Configuration Options

### `keep_partial_last_bar`

Controls what happens to the final incomplete bar in the dataset.

**Default**: `false` (drop partial bars)

#### Behavior with `keep_partial_last_bar = false`

```python
# Source: 15 bars of 1m data
# Target: 5000 volume per bar

Bar 1: volume = 5200 ✓ (complete)
Bar 2: volume = 5100 ✓ (complete)
Bar 3: volume = 5300 ✓ (complete)
Bar 4: volume = 3200 ✗ (partial - DROPPED)

# Result: 3 bars
```

**When to use**:
- Backtesting: You don't want to include incomplete data
- Production: Real-time last bar is always incomplete
- Training ML models: Consistent bar quality

#### Behavior with `keep_partial_last_bar = true`

```python
# Source: 15 bars of 1m data
# Target: 5000 volume per bar

Bar 1: volume = 5200 ✓ (complete)
Bar 2: volume = 5100 ✓ (complete)
Bar 3: volume = 5300 ✓ (complete)
Bar 4: volume = 3200 ✓ (kept despite being partial)

# Result: 4 bars
```

**When to use**:
- Research: You want to analyze all available data
- Visualization: Show the current partial bar
- Live trading: The partial bar represents current market state

### Configuration Validation

The system validates your configuration and provides helpful error messages:

#### Missing Target

```yaml
bars:
  mode: volume
  # MISSING: target_volume
```

**Error**: Returns empty DataFrame

#### Invalid Source Timeframe

```yaml
bars:
  mode: dollar
  target_notional: 50000
  source_timeframe: "15m"  # TOO COARSE!
```

**Error**:
```
ValueError: Event bars (mode=dollar) should be built from 1m data.
Got source_timeframe=15m. Non-time event bars require high-resolution (1m)
data to accurately capture volume, dollar, or tick thresholds.
```

## Integration with Feature Pipeline

Event bars work seamlessly with the feature pipeline:

```python
from finantradealgo.features.feature_pipeline import build_feature_pipeline
from finantradealgo.system.config_loader import DataConfig, EventBarConfig

# Configure event bars
data_cfg = DataConfig(
    bars=EventBarConfig(
        mode="volume",
        target_volume=5000.0,
        source_timeframe="1m",
        keep_partial_last_bar=False
    )
)

# Build features on event bars
result = build_feature_pipeline(
    csv_ohlcv_path="path/to/1m_data.csv",
    pipeline_cfg=pipeline_cfg,
    data_cfg=data_cfg  # Event bars applied here
)

# result.df now contains event bars with all features
```

### Microstructure Features on Event Bars

Microstructure features (like liquidity sweeps) use the `bar_start_ts` and `bar_end_ts` columns to filter trade data:

```python
# In microstructure_engine.py
def sweep_for_bar(bar):
    if 'bar_start_ts' in df.columns:
        # Event bars: use explicit timestamps
        bar_start_ts = bar['bar_start_ts']
        bar_end_ts = bar.name  # Index
    else:
        # Regular time bars: infer from timeframe
        bar_start_ts = bar.name
        bar_end_ts = bar_start_ts + timeframe_delta

    return detect_liquidity_sweep(
        bar.open, bar.close,
        bar_start_ts, bar_end_ts,
        trades_df, cfg.sweep
    )
```

This ensures that:
- Each event bar only uses trades from its time range
- Variable-duration bars are handled correctly
- No cross-contamination between bars

## Best Practices

### 1. Start with 1m Data

Always use 1-minute OHLCV data as your source:

```yaml
# GOOD
bars:
  mode: volume
  target_volume: 5000
  source_timeframe: "1m"  # Correct!

# BAD
bars:
  mode: volume
  target_volume: 5000
  source_timeframe: "15m"  # Will raise error
```

### 2. Choose Appropriate Thresholds

Target values should match your instrument's characteristics:

**High-liquidity instruments (e.g., BTC/USDT)**:
```yaml
bars:
  mode: volume
  target_volume: 50000  # Higher threshold for high-volume instruments
```

**Low-liquidity instruments (e.g., altcoins)**:
```yaml
bars:
  mode: volume
  target_volume: 1000   # Lower threshold for low-volume instruments
```

### 3. Use Dollar Bars for Price-Changing Instruments

If your instrument has significant price trends:

```yaml
# Better: Dollar bars normalize for price changes
bars:
  mode: dollar
  target_notional: 100000

# Worse: Volume bars don't account for price
bars:
  mode: volume
  target_volume: 1000  # 1000 at $10 ≠ 1000 at $100
```

### 4. Drop Partial Bars for Backtesting

```yaml
bars:
  mode: volume
  target_volume: 5000
  keep_partial_last_bar: false  # Ensures consistent bar quality
```

### 5. Keep Partial Bars for Live Trading

```yaml
bars:
  mode: volume
  target_volume: 5000
  keep_partial_last_bar: true  # Shows current market state
```

## Common Patterns

### Pattern 1: Volume-Based Adaptive Sampling

```yaml
# Compress low-activity periods, expand high-activity periods
bars:
  mode: volume
  target_volume: 10000
  source_timeframe: "1m"
  keep_partial_last_bar: false
```

**Result**:
- Quiet periods: Few bars (maybe 1 bar per hour)
- Active periods: Many bars (maybe 10 bars per hour)

### Pattern 2: Fixed Downsampling

```yaml
# Simple 5:1 downsampling (5 × 1m → 1 bar)
bars:
  mode: tick
  target_ticks: 5
  source_timeframe: "1m"
  keep_partial_last_bar: false
```

**Result**: Every 5 minutes compressed into 1 bar

### Pattern 3: Capital-Weighted Bars

```yaml
# Each bar represents ~$100k of traded notional
bars:
  mode: dollar
  target_notional: 100000
  source_timeframe: "1m"
  keep_partial_last_bar: false
```

**Result**: Normalized by capital flow, not time or volume

## Troubleshooting

### Issue: Empty DataFrame returned

**Cause**: Missing target parameter

**Solution**:
```yaml
# Add the appropriate target
bars:
  mode: volume
  target_volume: 5000  # Don't forget this!
```

### Issue: "should be built from 1m data" error

**Cause**: Using coarse source timeframe with event bars

**Solution**:
```yaml
# Change to 1m source data
bars:
  mode: volume
  target_volume: 5000
  source_timeframe: "1m"  # Must be 1m!
```

### Issue: Too many/too few bars

**Cause**: Inappropriate threshold for your instrument

**Solution**: Adjust target to match instrument characteristics:
```yaml
# Too many bars? Increase threshold
bars:
  mode: volume
  target_volume: 50000  # Was: 5000

# Too few bars? Decrease threshold
bars:
  mode: volume
  target_volume: 1000   # Was: 10000
```

### Issue: Microstructure features all NaN

**Cause**: Usually missing trade data or incorrect time alignment

**Solution**:
1. Verify trade data exists for the same time period
2. Check that trades DataFrame has correct datetime index
3. Ensure bar time boundaries are reasonable

## Examples

### Example 1: Basic Volume Bars

```python
from finantradealgo.data_engine.event_bars import build_event_bars
from finantradealgo.core.config import EventBarConfig
import pandas as pd

# Load 1m data
df_1m = pd.read_csv("btc_1m.csv", index_col=0, parse_dates=True)

# Configure volume bars
cfg = EventBarConfig(
    mode="volume",
    target_volume=10000.0,
    source_timeframe="1m",
    keep_partial_last_bar=False
)

# Build event bars
df_volume = build_event_bars(df_1m, cfg)

print(f"Original: {len(df_1m)} bars")
print(f"Volume bars: {len(df_volume)} bars")
print(df_volume.head())
```

### Example 2: Dollar Bars with Pipeline

```python
from finantradealgo.features.feature_pipeline import build_feature_pipeline
from finantradealgo.system.config_loader import DataConfig, EventBarConfig

data_cfg = DataConfig(
    bars=EventBarConfig(
        mode="dollar",
        target_notional=100000.0,
        source_timeframe="1m",
        keep_partial_last_bar=False
    )
)

result = build_feature_pipeline(
    csv_ohlcv_path="eth_1m.csv",
    pipeline_cfg=pipeline_cfg,
    data_cfg=data_cfg
)

# result.df has dollar bars with all features
print(result.df[['open', 'close', 'volume', 'bar_start_ts']].head())
```

### Example 3: Tick Bars for Downsampling

```python
# 10:1 downsampling (10 × 1m → 1 bar)
cfg = EventBarConfig(
    mode="tick",
    target_ticks=10,
    source_timeframe="1m",
    keep_partial_last_bar=True  # Keep last partial bar
)

df_downsampled = build_event_bars(df_1m, cfg)

# Approximately 6 bars per hour (60 / 10)
```

## Summary

| Feature | Description |
|---------|-------------|
| **Source Data** | Always use 1-minute OHLCV data |
| **Modes** | time, volume, dollar, tick |
| **Time Columns** | `bar_start_ts` (column), `bar_end_ts` (index) |
| **Duration** | Variable - each bar can span different time periods |
| **Partial Bar** | Control with `keep_partial_last_bar` |
| **Validation** | System enforces 1m source data for non-time modes |
| **Integration** | Works seamlessly with feature pipeline |
| **Use Cases** | Adaptive sampling, normalizing data, reducing noise |

Event bars provide a powerful way to sample market data based on actual activity rather than arbitrary time intervals. When used correctly with 1m source data, they can significantly improve signal quality and strategy performance.

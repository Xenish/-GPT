# Microstructure Sweep Documentation

## Overview

Liquidity sweeps are rapid, one-sided bursts of trading activity that push price in a specific direction. They represent significant market events where large orders "sweep" through available liquidity levels.

This document explains:
- What liquidity sweeps are and how they're detected
- How sweep detection adapts to event bars vs regular time bars
- Implementation details and design decisions
- Performance considerations and future improvements

## What is a Liquidity Sweep?

A liquidity sweep occurs when:
1. **Large one-sided trading**: Heavy buying OR selling activity (not both)
2. **Price impact**: Price moves significantly in the direction of the trading
3. **Short timeframe**: Activity happens within a single bar or short window

**Example scenarios:**
- Large institutional order executing across multiple price levels
- Stop-loss cascade triggering sequential orders
- High-frequency traders front-running major orders
- Market maker inventory rebalancing

## Detection Algorithm

### Core Logic

```python
def detect_liquidity_sweep(
    bar_open: float,
    bar_close: float,
    bar_start_ts: pd.Timestamp,
    bar_end_ts: pd.Timestamp,
    trades_df: pd.DataFrame,
    cfg: LiquiditySweepConfig,
) -> Tuple[float, float]:
```

**Step 1: Filter trades to time window**
```python
window_start = bar_start_ts - pd.to_timedelta(cfg.lookback_ms, unit="ms")
relevant_trades = trades_df.loc[window_start:bar_end_ts]
```

**Step 2: Calculate buy/sell notional**
```python
notional = relevant_trades["price"] * relevant_trades["size"]
buy_notional = notional[relevant_trades["side"] == "buy"].sum()
sell_notional = notional[relevant_trades["side"] == "sell"].sum()
```

**Step 3: Check for price impact + threshold**
```python
# Upward sweep: price moved up + heavy buying
if bar_close > bar_open and buy_notional > cfg.notional_threshold:
    sweep_up = buy_notional

# Downward sweep: price moved down + heavy selling
if bar_close < bar_open and sell_notional > cfg.notional_threshold:
    sweep_down = sell_notional
```

### Key Design Decisions

1. **Requires price impact**: Trading activity alone isn't enough - price must move
2. **One-sided**: Only buying (sweep_up) OR selling (sweep_down), never both
3. **Notional-based**: Uses dollar value (price × size), not just volume
4. **Configurable threshold**: `notional_threshold` filters noise from real sweeps

## Time Range Semantics

### The Challenge: Variable-Duration Bars

**Problem**: Event bars have variable durations, making fixed timeframe assumptions incorrect.

```python
# Regular time bars: All bars are 1 minute
Bar 1: 09:00:00 - 09:01:00  (1 minute)
Bar 2: 09:01:00 - 09:02:00  (1 minute)
Bar 3: 09:02:00 - 09:03:00  (1 minute)

# Event bars: Bars have different durations
Bar 1: 09:00:00 - 09:03:00  (3 minutes)
Bar 2: 09:04:00 - 09:11:00  (7 minutes)
Bar 3: 09:12:00 - 09:13:00  (1 minute)
```

**The old approach** (hardcoded timeframe):
```python
# WRONG: Assumes all bars are same duration
timeframe_delta = pd.Timedelta(minutes=1)
bar_start = bar_end - timeframe_delta  # Breaks for event bars!
```

### The Solution: Explicit Time Boundaries

**New approach** (event bars compatible):

```python
# Check if DataFrame has explicit bar timestamps (from event bars)
has_bar_start_col = 'bar_start_ts' in df.columns
has_bar_end_col = 'bar_end_ts' in df.columns
has_explicit_bounds = has_bar_start_col

if has_explicit_bounds:
    # Event bars: use explicit timestamps
    bar_start_ts = bar['bar_start_ts']
    bar_end_ts = bar['bar_end_ts'] if has_bar_end_col else bar.name
else:
    # Regular time bars: infer timeframe from index
    timeframe_delta = df.index.to_series().diff().min()
    bar_start_ts = bar.name
    bar_end_ts = bar_start_ts + timeframe_delta
```

**Why this works:**
1. **Event bars**: Explicit `bar_start_ts` and `bar_end_ts` columns provide exact boundaries
2. **Regular bars**: Infer timeframe from index differences (fallback)
3. **No assumptions**: Each bar's duration is determined from actual data

### Time Boundary Behavior

**Important**: Time boundaries are **inclusive on both ends** due to pandas `loc` slicing.

```python
# pandas loc[start:end] includes both start and end
relevant_trades = trades_df.loc[window_start:bar_end_ts]
```

**Example**:
```
Bar: [09:00:00, 09:05:00]

Trades:
  09:00:00  ✓ included
  09:02:30  ✓ included
  09:05:00  ✓ included (boundary is inclusive)
  09:05:01  ✗ not included
```

## Integration with Event Bars

### Event Bar Metadata

Event bars provide explicit time metadata that sweep detection uses:

```python
# Example event bar DataFrame
#                      open  high   low  close  volume  bar_start_ts
# bar_end_ts
# 2023-01-01 09:03:00  100   102    99    102    5200   2023-01-01 09:00:00
# 2023-01-01 09:11:00  102   108   101    107    5100   2023-01-01 09:04:00
```

**Columns used:**
- `bar_start_ts` (column): When this bar started accumulating
- `bar_end_ts` (index): When this bar closed

### Integration Point: `compute_microstructure_df`

Located in `finantradealgo/microstructure/microstructure_engine.py`:

```python
if trades_df is not None and not trades_df.empty:
    # Check if DataFrame has explicit bar timestamps (from event bars)
    has_bar_start_col = 'bar_start_ts' in df.columns
    has_bar_end_col = 'bar_end_ts' in df.columns
    has_explicit_bounds = has_bar_start_col

    if not has_explicit_bounds:
        # Fallback for regular time bars: infer timeframe from index differences
        timeframe_delta = df.index.to_series().diff().min()

    def sweep_for_bar(bar):
        if has_explicit_bounds:
            # Use explicit bar timestamps from event bars
            bar_start_ts = bar['bar_start_ts']
            # bar_end_ts can be either a column or the index
            bar_end_ts = bar['bar_end_ts'] if has_bar_end_col else bar.name
        else:
            # Fallback for regular time bars
            bar_start_ts = bar.name
            bar_end_ts = bar_start_ts + timeframe_delta

        return detect_liquidity_sweep(
            bar.open, bar.close, bar_start_ts, bar_end_ts, trades_df, cfg.sweep
        )

    sweeps = df.apply(sweep_for_bar, axis=1, result_type="expand")
    features_df["ms_sweep_up"] = sweeps[0]
    features_df["ms_sweep_down"] = sweeps[1]
```

### Why This Design?

**Key insight**: The sweep function signature was changed to accept explicit timestamps instead of relying on implicit timeframe assumptions.

**Before (broken for event bars)**:
```python
def detect_liquidity_sweep(
    bar_open, bar_close,
    bar_timestamp,  # Just one timestamp
    trades_df,
    timeframe_delta,  # Hardcoded assumption!
):
    bar_start = bar_timestamp - timeframe_delta  # WRONG for event bars
    bar_end = bar_timestamp
```

**After (works with event bars)**:
```python
def detect_liquidity_sweep(
    bar_open, bar_close,
    bar_start_ts,  # Explicit start
    bar_end_ts,    # Explicit end
    trades_df,
    cfg,
):
    # No assumptions - uses exact boundaries
```

**Benefits:**
1. ✓ Works with variable-duration event bars
2. ✓ Works with regular time bars (backward compatible)
3. ✓ Explicit and clear - no hidden assumptions
4. ✓ Testable - exact time boundaries are verifiable

## Configuration

### LiquiditySweepConfig

```python
@dataclass
class LiquiditySweepConfig:
    lookback_ms: int = 0              # Lookback window in milliseconds
    notional_threshold: float = 0.0   # Minimum notional to qualify as sweep
```

### Parameters Explained

#### `lookback_ms`

Extends the time window before the bar starts.

```python
window_start = bar_start_ts - pd.to_timedelta(lookback_ms, unit="ms")
```

**Example with `lookback_ms = 5000` (5 seconds)**:
```
Bar: [09:01:00, 09:02:00]
Lookback: 5000ms = 5 seconds

Window: [09:00:55, 09:02:00]  ← Includes 5s before bar start
```

**Use cases:**
- `lookback_ms = 0`: Only trades within the bar (strict)
- `lookback_ms = 5000`: Include lead-in trades (detect early sweeps)
- `lookback_ms = 60000`: Include 1 minute before bar (macro context)

#### `notional_threshold`

Minimum dollar value to qualify as a sweep.

**Example**:
```python
# Threshold: $50,000
cfg = LiquiditySweepConfig(
    lookback_ms=0,
    notional_threshold=50000.0
)

# Scenario 1: buy_notional = $75,000
# Result: sweep_up = 75000 ✓ (exceeds threshold)

# Scenario 2: buy_notional = $30,000
# Result: sweep_up = 0 ✗ (below threshold)
```

**Choosing threshold:**
- **High-liquidity instruments** (BTC/USDT): $100,000+
- **Medium-liquidity**: $10,000 - $50,000
- **Low-liquidity/testing**: $0 - $5,000

## Output Format

### Return Values

```python
(sweep_up, sweep_down) -> Tuple[float, float]
```

**Rules:**
1. Only ONE can be non-zero (never both)
2. Values represent total notional in dollars
3. Zero means no sweep detected

**Examples**:
```python
# Upward sweep detected
(75000.0, 0.0)  # $75k buy notional, no sell sweep

# Downward sweep detected
(0.0, 120000.0)  # No buy sweep, $120k sell notional

# No sweep (price didn't move enough or below threshold)
(0.0, 0.0)

# Invalid (never happens in correct implementation)
(50000.0, 50000.0)  # Can't have both!
```

### DataFrame Columns

Added to microstructure features DataFrame:

```python
features_df["ms_sweep_up"]    # Float: upward sweep notional (0 or positive)
features_df["ms_sweep_down"]  # Float: downward sweep notional (0 or positive)
```

**Example DataFrame**:
```
                     ms_sweep_up  ms_sweep_down
bar_end_ts
2023-01-01 09:03:00      75000.0           0.0  ← Upward sweep
2023-01-01 09:11:00          0.0           0.0  ← No sweep
2023-01-01 09:15:00          0.0      125000.0  ← Downward sweep
```

## Performance Considerations

### Current Implementation: Bar-by-Bar Loop

**Current approach** (in `microstructure_engine.py`):
```python
def sweep_for_bar(bar):
    # ... determine bar_start_ts and bar_end_ts ...
    return detect_liquidity_sweep(
        bar.open, bar.close, bar_start_ts, bar_end_ts, trades_df, cfg.sweep
    )

# Apply function to each bar (NOT vectorized)
sweeps = df.apply(sweep_for_bar, axis=1, result_type="expand")
```

**Performance characteristics:**
- ✗ **O(n × m)** worst case: n bars × m trades
- ✗ Not fully vectorized
- ✓ Simple and readable
- ✓ Easy to debug
- ✓ Works correctly with event bars

### Why Not Vectorized?

**Challenge 1: Variable time windows**

Each bar has different start/end times (especially event bars):
```python
Bar 1: [09:00, 09:03]  # 3 minutes
Bar 2: [09:04, 09:11]  # 7 minutes
Bar 3: [09:12, 09:13]  # 1 minute
```

Standard pandas vectorization assumes fixed windows.

**Challenge 2: Trade filtering**

Each bar needs different trade subsets:
```python
# Bar 1 needs trades from [09:00, 09:03]
bar1_trades = trades_df.loc['09:00':'09:03']

# Bar 2 needs trades from [09:04, 09:11]
bar2_trades = trades_df.loc['09:04':'09:11']
```

This per-bar filtering is inherently iterative.

### Performance Profile

**Typical performance**:
```
100 bars × 1000 trades = ~0.5 seconds
1000 bars × 10000 trades = ~5 seconds
```

**When it becomes slow**:
- Very large datasets (10,000+ bars with 100,000+ trades)
- High-frequency data (tick-by-tick trades)

### Future Optimization Plan

**Option A: Vectorized time-based groupby** (complex)
```python
# Pseudo-code for future optimization
# 1. Create interval index from bar boundaries
bar_intervals = pd.IntervalIndex.from_arrays(df['bar_start_ts'], df['bar_end_ts'])

# 2. Map each trade to its bar
trades_df['bar_idx'] = trades_df.index.map(lambda ts: bar_intervals.get_loc(ts))

# 3. Groupby and aggregate
sweep_stats = trades_df.groupby('bar_idx').apply(vectorized_sweep_calc)
```

**Challenges**:
- Complex implementation
- Edge cases with boundary overlaps
- Requires thorough testing
- May not be much faster for typical datasets

**Option B: Cython/Numba acceleration** (simpler)
```python
@numba.jit(nopython=True)
def fast_sweep_detection(bars, trades, thresholds):
    # Compile to machine code
    # Same logic, 10-100x faster
    pass
```

**Recommendation**: Don't optimize unless profiling shows it's a bottleneck.

> "Premature optimization is the root of all evil" - Donald Knuth

The current implementation is:
- ✓ Correct
- ✓ Clear
- ✓ Fast enough for most use cases
- ✓ Easy to maintain

Optimize only when:
1. Profiling shows this is the bottleneck
2. Performance is unacceptable for your use case
3. You have comprehensive tests to prevent regressions

## Trade Data Requirements

### Required Format

Trades DataFrame must have:
```python
# Index: datetime (pd.DatetimeIndex)
# Columns:
#   - side: str ("buy" or "sell")
#   - price: float
#   - size: float
```

**Example**:
```python
                     side  price    size
timestamp
2023-01-01 09:00:15  buy   100.5   10.0
2023-01-01 09:00:30  buy   100.7   25.0
2023-01-01 09:00:45  sell  100.3   15.0
```

### Trade Sources

**Backtest/research**:
- Historical trades from exchange API
- Simulated trades from order book reconstruction

**Live trading**:
- Real-time trade stream from exchange WebSocket
- Accumulated trades from REST API polling

### Common Issues

#### Missing trades data

```python
# If trades_df is None or empty
sweep_up, sweep_down = detect_liquidity_sweep(...)
# Result: (0.0, 0.0) - no sweep detected
```

#### Wrong datetime index

```python
# BAD: string index instead of datetime
trades_df.index = ['2023-01-01 09:00:00', ...]  # str

# GOOD: datetime index
trades_df.index = pd.to_datetime(['2023-01-01 09:00:00', ...])
```

The function automatically converts if needed:
```python
if not isinstance(trades_df.index, pd.DatetimeIndex):
    trades_df = trades_df.set_index("timestamp")
```

#### Time zone mismatches

All timestamps should use the same timezone (preferably UTC):
```python
# Ensure consistent timezone
df.index = df.index.tz_localize('UTC')
trades_df.index = trades_df.index.tz_localize('UTC')
```

## Testing

### Test Coverage

Located in `tests/test_microstructure_sweep_time.py`:

**Regular time bars**:
- `test_sweep_regular_time_bars_upward`: Fixed 1m bars with upward sweep
- `test_sweep_regular_time_bars_downward`: Fixed 1m bars with downward sweep

**Event bars**:
- `test_sweep_event_bars_variable_duration`: Variable-duration event bars
- `test_sweep_event_bars_precise_time_boundaries`: Exact boundary handling

**Edge cases**:
- `test_sweep_no_price_impact_no_sweep`: Heavy trading without price move
- `test_sweep_with_lookback_window`: Lookback parameter behavior

### Key Test Scenarios

#### Test 1: Variable Duration Event Bars

```python
# Bar durations: 2min, 5min, 3min, 1min
Bar 1: [09:00, 09:02]  # 2 minutes
Bar 2: [09:02, 09:07]  # 5 minutes ← Heavy buying here
Bar 3: [09:07, 09:10]  # 3 minutes ← Heavy selling here
Bar 4: [09:10, 09:11]  # 1 minute

# Verify each bar only captures its trades
assert bar2_sweep_up > 50000  # Bar 2 has upward sweep
assert bar3_sweep_down > 50000  # Bar 3 has downward sweep
assert bar1_sweep_up == 0  # Bar 1 has no sweep
```

#### Test 2: No Price Impact

```python
# Heavy buying but price doesn't move (close == open)
Bar: open=100, close=100, buy_notional=100000

# No sweep because price didn't move
assert sweep_up == 0
```

This prevents false positives from large trades that don't impact price.

#### Test 3: Lookback Window

```python
# Bar: [09:01:00, 09:02:00]
# Lookback: 60 seconds

# Without lookback:
#   Only captures trades from [09:01:00, 09:02:00]
#   Result: 10,300 notional

# With 60s lookback:
#   Captures trades from [09:00:00, 09:02:00]
#   Result: 40,600 notional (4x more)
```

## Integration Example

### Complete Pipeline

```python
from finantradealgo.features.feature_pipeline import build_feature_pipeline
from finantradealgo.system.config_loader import DataConfig, EventBarConfig
from finantradealgo.microstructure.config import MicrostructureConfig, LiquiditySweepConfig

# Configure event bars
data_cfg = DataConfig(
    bars=EventBarConfig(
        mode="volume",
        target_volume=5000.0,
        source_timeframe="1m",
        keep_partial_last_bar=False
    )
)

# Configure sweep detection
pipeline_cfg = FeaturePipelineConfig(
    use_microstructure=True,
    microstructure=MicrostructureConfig(
        sweep=LiquiditySweepConfig(
            lookback_ms=5000,      # 5 second lookback
            notional_threshold=50000.0  # $50k threshold
        )
    )
)

# Build features
result = build_feature_pipeline(
    csv_ohlcv_path="data/btc_1m.csv",
    pipeline_cfg=pipeline_cfg,
    data_cfg=data_cfg,
    # trades_df parameter would be passed here in real usage
)

# Access sweep features
sweeps = result.df[['ms_sweep_up', 'ms_sweep_down']]
print(sweeps[sweeps['ms_sweep_up'] > 0])  # Show upward sweeps
```

## Summary

### Key Takeaways

1. **Time semantics**: Sweep detection uses explicit `bar_start_ts` and `bar_end_ts` for event bars, falls back to timeframe inference for regular bars

2. **Event bar compatible**: The refactored implementation works correctly with variable-duration bars

3. **One-sided detection**: Sweeps are either up OR down, never both

4. **Notional-based**: Uses dollar value (price × size) to detect significant activity

5. **Configurable**: Threshold and lookback window are adjustable

6. **Performance**: Current implementation is O(n×m) but fast enough for typical use cases

7. **Future work**: Vectorization is possible but not currently necessary

### Design Philosophy

The implementation prioritizes:
- **Correctness** over performance
- **Clarity** over cleverness
- **Testability** over optimization

This makes the code:
- Easy to understand 6 months later ✓
- Easy to debug when issues arise ✓
- Easy to extend with new features ✓
- Safe to refactor with confidence ✓

### References

- Event bars documentation: `docs/dev/event_bars.md`
- Implementation: `finantradealgo/microstructure/liquidity_sweep.py`
- Integration: `finantradealgo/microstructure/microstructure_engine.py`
- Tests: `tests/test_microstructure_sweep_time.py`

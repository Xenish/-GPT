# Developer Guide: trades_df Data Contract

## 1. Purpose

This document defines the standardized format for `trades_df` DataFrames used throughout the microstructure analysis pipeline. The contract ensures consistent handling of tick-level trade data across all modules.

## 2. Contract Specification

### Index
- **Type**: `pd.DatetimeIndex`
- **Timezone**: UTC
- **Sorting**: Ascending (monotonic increasing)
- **Name**: `timestamp`

### Columns

| Column | Type    | Description                                      |
|--------|---------|--------------------------------------------------|
| `side` | `str`   | Trade direction: `"buy"` or `"sell"`            |
| `price`| `float` | Execution price                                  |
| `size` | `float` | Trade size (volume in base asset)               |

### Example

```python
import pandas as pd

# Valid trades_df structure:
trades_df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=5, freq="1s", tz="UTC"),
    "side": ["buy", "sell", "buy", "buy", "sell"],
    "price": [40000.0, 39995.0, 40005.0, 40010.0, 40000.0],
    "size": [0.5, 0.3, 0.8, 0.2, 0.6]
}).set_index("timestamp")

# Verify contract
assert isinstance(trades_df.index, pd.DatetimeIndex)
assert trades_df.index.is_monotonic_increasing
assert set(trades_df.columns) == {"side", "price", "size"}
```

## 3. Contract Enforcement

### 3.1 Data Loading

The contract is **enforced at load time** by `load_trades()`:

**File**: [finantradealgo/data_engine/orderbook_loader.py](../../finantradealgo/data_engine/orderbook_loader.py)

```python
from finantradealgo.data_engine.orderbook_loader import load_trades

# Automatically returns DatetimeIndex format
trades_df = load_trades("BTCUSDT", "1m")

# trades_df.index is already pd.DatetimeIndex (UTC, sorted)
# No need to call set_index manually
```

**Implementation**:
```python
def load_trades(symbol: str, timeframe: str, trades_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load trades data with DatetimeIndex contract.

    Returns:
        DataFrame with:
        - Index: DatetimeIndex (UTC, sorted ascending)
        - Columns: side, price, size
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df
```

### 3.2 Runtime Validation

The contract is **validated at use time** by microstructure functions:

**File**: [finantradealgo/microstructure/liquidity_sweep.py](../../finantradealgo/microstructure/liquidity_sweep.py)

```python
from finantradealgo.microstructure.liquidity_sweep import detect_liquidity_sweep

# Function validates DatetimeIndex contract
sweep_up, sweep_down = detect_liquidity_sweep(
    bar_open=40000,
    bar_close=40050,
    bar_start_ts=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
    bar_end_ts=pd.Timestamp("2024-01-01 00:01:00", tz="UTC"),
    trades_df=trades_df,  # Must have DatetimeIndex
    cfg=config
)
```

**Validation logic**:
```python
# Contract: trades_df must have DatetimeIndex
if not isinstance(trades_df.index, pd.DatetimeIndex):
    raise ValueError(
        "trades_df must have a DatetimeIndex. "
        "Use load_trades() or ensure trades_df.set_index('timestamp') was called."
    )
```

## 4. Usage Guidelines

### ✓ Correct Usage

```python
# 1. Load trades using official loader
trades_df = load_trades("BTCUSDT", "1m")

# 2. Pass directly to microstructure functions
sweep_up, sweep_down = detect_liquidity_sweep(
    bar_open=40000,
    bar_close=40050,
    bar_start_ts=start_ts,
    bar_end_ts=end_ts,
    trades_df=trades_df,  # Already has DatetimeIndex
    cfg=config
)

# 3. Filter using DatetimeIndex slicing
recent_trades = trades_df.loc[start_ts:end_ts]
```

### ✗ Incorrect Usage

```python
# DON'T: Load CSV without setting index
trades_df = pd.read_csv("trades.csv")  # timestamp is a column, not index

# This will raise ValueError:
detect_liquidity_sweep(..., trades_df=trades_df, ...)
# ValueError: trades_df must have a DatetimeIndex

# DON'T: Reset index after loading
trades_df = load_trades("BTCUSDT", "1m")
trades_df = trades_df.reset_index()  # Breaks contract!

# This will also raise ValueError:
detect_liquidity_sweep(..., trades_df=trades_df, ...)
```

## 5. Migration Guide

If you have legacy code that handles trades data differently, migrate as follows:

### Before (Legacy Code)
```python
# Old: Manual index handling
trades_df = pd.read_csv("trades.csv", parse_dates=["timestamp"])

# Conditional set_index hack
if "timestamp" in trades_df.columns:
    trades_df = trades_df.set_index("timestamp")

# Function accepts both formats (error-prone)
result = some_function(trades_df)
```

### After (Contract-Compliant Code)
```python
# New: Use official loader
trades_df = load_trades("BTCUSDT", "1m")

# Contract is guaranteed - no conditional logic needed
result = some_function(trades_df)
```

## 6. Testing

Contract validation is tested in [tests/test_trades_df_contract.py](../../tests/test_trades_df_contract.py):

```python
def test_load_trades_returns_datetime_index():
    """Test that load_trades() returns DataFrame with DatetimeIndex."""
    trades_df = load_trades("BTCUSDT", "1m", trades_dir=sample_dir)

    # Verify contract
    assert isinstance(trades_df.index, pd.DatetimeIndex)
    assert trades_df.index.is_monotonic_increasing
    assert set(trades_df.columns) == {"side", "price", "size"}
    assert "timestamp" not in trades_df.columns  # It's the index!

def test_detect_sweep_raises_on_invalid_index():
    """Test that functions raise ValueError on contract violations."""
    invalid_df = pd.DataFrame({
        "timestamp": [...],  # Column, not index
        "side": [...],
        "price": [...],
        "size": [...]
    })

    # Should raise clear error
    with pytest.raises(ValueError, match="trades_df must have a DatetimeIndex"):
        detect_liquidity_sweep(..., trades_df=invalid_df, ...)
```

## 7. Benefits

1. **Type Safety**: Contract violations caught early with clear error messages
2. **Consistency**: All modules expect the same format
3. **Simplicity**: No conditional `set_index` checks in business logic
4. **Performance**: Index operations on DatetimeIndex are optimized
5. **Maintainability**: Single source of truth for trades data format

## 8. Related Files

- **Loader**: [finantradealgo/data_engine/orderbook_loader.py](../../finantradealgo/data_engine/orderbook_loader.py) - `load_trades()`
- **Validator**: [finantradealgo/microstructure/liquidity_sweep.py](../../finantradealgo/microstructure/liquidity_sweep.py) - `detect_liquidity_sweep()`
- **Tests**: [tests/test_trades_df_contract.py](../../tests/test_trades_df_contract.py)
- **Config**: [finantradealgo/microstructure/config.py](../../finantradealgo/microstructure/config.py) - `LiquiditySweepConfig`

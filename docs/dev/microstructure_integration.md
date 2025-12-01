# Microstructure Integration Guide

**Task S2.E4: Documentation for Microstructure Module Integration**

This document describes the microstructure module architecture, data requirements, and integration patterns for strategies and risk management.

---

## Overview

The microstructure module provides fine-grained market signals derived from:
- **OHLCV data** (volatility regime, chop, momentum bursts, exhaustion, parabolic trends)
- **Trade data** (liquidity sweeps)
- **Order book data** (bid/ask imbalance)

All features follow a standardized `ms_*` naming convention for clarity and consistency.

---

## Architecture

### Core Modules

1. **`finantradealgo/microstructure/config.py`**
   - Configuration dataclasses for all microstructure signals
   - `MicrostructureConfig`: Main config with nested configs for each signal type
   - YAML-compatible via `from_dict()` methods

2. **`finantradealgo/microstructure/microstructure_engine.py`**
   - **SINGLE ENTRY-POINT**: `compute_microstructure_df()` / `compute_microstructure_features()`
   - Enforces input contracts and output guarantees
   - Handles optional trades_df and book_df for advanced features
   - Task S2.E2: Implements `max_lookback_seconds` truncation for live/paper trading latency control

3. **`finantradealgo/microstructure/types.py`**
   - `MicrostructureSignals`: Defines standardized output columns with `ms_*` prefix

4. **Individual detectors** (internal use only):
   - `chop_detector.py`, `volatility_regime.py`, `burst_detector.py`, etc.
   - **NOT for direct use** - always go through `compute_microstructure_df()`

---

## Input Specification

### Required: OHLCV DataFrame

```python
df = pd.DataFrame({
    "open": [...],
    "high": [...],
    "low": [...],
    "close": [...],
    "volume": [...],
}, index=pd.DatetimeIndex([...]))
```

**Contract:**
- Must have all 5 columns: open, high, low, close, volume
- Must have DatetimeIndex
- Cannot be empty

### Optional: Trades DataFrame

```python
trades_df = pd.DataFrame({
    "side": ["buy", "sell", ...],  # Trade direction
    "price": [100.0, 100.1, ...],  # Execution price
    "size": [10, 20, ...],         # Trade size
}, index=pd.DatetimeIndex([...]))  # Trade timestamps
```

**Used for:**
- `ms_sweep_up` / `ms_sweep_down`: Liquidity sweep detection

**Contract:**
- Required columns: side, price, size
- Must have DatetimeIndex
- Can be None (features default to 0.0)

### Optional: Order Book DataFrame

```python
book_df = pd.DataFrame({
    "bid_price_0": [...],  # Best bid price
    "bid_size_0": [...],   # Best bid size
    "ask_price_0": [...],  # Best ask price
    "ask_size_0": [...],   # Best ask size
    # ... up to depth N
    "bid_price_N": [...],
    "bid_size_N": [...],
    "ask_price_N": [...],
    "ask_size_N": [...],
}, index=pd.DatetimeIndex([...]))
```

**Used for:**
- `ms_imbalance`: Order book bid/ask imbalance

**Contract:**
- Must have DatetimeIndex
- Required columns depend on `depth` config parameter
- Can be None (feature defaults to 0.0)

**See:** `finantradealgo/microstructure/input_spec.py` for detailed validation specifications

---

## Output Specification

### Guaranteed Columns (all prefixed with `ms_*`)

From `MicrostructureSignals.columns()`:

```python
[
    "ms_vol_regime",       # Volatility regime: -2 (low), 0 (normal), 2 (high)
    "ms_chop",             # Chop index: 0.0-1.0 (higher = more choppy)
    "ms_burst_up",         # Upward momentum burst: 0 or 1
    "ms_burst_down",       # Downward momentum burst: 0 or 1
    "ms_exhaustion_up",    # Upward exhaustion: 0 or 1
    "ms_exhaustion_down",  # Downward exhaustion: 0 or 1
    "ms_parabolic_trend",  # Parabolic trend: 0 or 1
    "ms_imbalance",        # Order book imbalance: -1.0 to 1.0 (negative = bid heavy)
    "ms_sweep_up",         # Upward liquidity sweep: 0 or 1
    "ms_sweep_down",       # Downward liquidity sweep: 0 or 1
]
```

**Contract Guarantees:**
- All columns exist even if inputs are incomplete
- Default value: 0.0 for unavailable features
- Index matches input OHLCV DataFrame
- No NaN values (initialized to 0.0)
- Column order is consistent

---

## Configuration

### In `config/system.yml`

```yaml
microstructure:
  enabled: true
  max_lookback_seconds: 3600  # S2.E2: Max lookback for trades/book (1 hour)

  imbalance:
    depth: 5                   # Order book depth
    threshold: 2.0             # Imbalance threshold

  sweep:
    lookback_ms: 5000          # Lookback window for sweeps (ms)
    notional_threshold: 50000.0  # Min notional for sweep detection

  chop:
    lookback_period: 14        # ADX-based chop calculation period

  vol_regime:
    period: 20                 # Volatility calculation window
    z_score_window: 100        # Z-score normalization window
    low_z_threshold: -1.5      # Low volatility threshold
    high_z_threshold: 1.5      # High volatility threshold

  burst:
    return_window: 5           # Window for return calculation
    z_score_window: 100        # Z-score normalization window
    z_up_threshold: 2.0        # Upward burst threshold
    z_down_threshold: 2.0      # Downward burst threshold

  exhaustion:
    min_consecutive_bars: 5    # Min consecutive bars for exhaustion
    volume_z_score_window: 50  # Volume z-score window
    volume_z_threshold: -0.5   # Low volume threshold

  parabolic:
    rolling_std_window: 20     # Rolling std window
    curvature_threshold: 1.5   # Parabolic curvature threshold
```

### In Python

```python
from finantradealgo.microstructure.config import MicrostructureConfig

# Load from YAML
cfg = MicrostructureConfig.from_dict(yaml_data["microstructure"])

# Or create programmatically
cfg = MicrostructureConfig(
    enabled=True,
    max_lookback_seconds=1800,  # 30 minutes
)
```

---

## Usage Patterns

### 1. Direct Computation (Offline/Batch)

```python
from finantradealgo.microstructure import (
    compute_microstructure_df,
    MicrostructureConfig,
)

# Minimal usage (OHLCV only)
df = pd.read_csv("ohlcv.csv", index_col=0, parse_dates=True)
cfg = MicrostructureConfig()
features = compute_microstructure_df(df, cfg)

# With trades data
trades_df = pd.read_csv("trades.csv", index_col=0, parse_dates=True)
features = compute_microstructure_df(df, cfg, trades_df=trades_df)

# With order book data
book_df = pd.read_csv("book.csv", index_col=0, parse_dates=True)
features = compute_microstructure_df(df, cfg, book_df=book_df)

# With both
features = compute_microstructure_df(df, cfg, trades_df=trades_df, book_df=book_df)
```

### 2. Feature Pipeline Integration

The microstructure module integrates with the main feature pipeline via:

```python
# finantradealgo/features/microstructure_features.py
def add_microstructure_features(
    df: pd.DataFrame,
    cfg: Optional[MicrostructureConfig] = None
) -> pd.DataFrame:
    """
    Feature pipeline integration point.
    Called by build_feature_pipeline() when use_microstructure=True.
    """
    ms_df = compute_microstructure_df(df, cfg)
    return pd.concat([df, ms_df], axis=1)
```

**In pipeline config:**
```python
from finantradealgo.features.feature_pipeline import build_feature_pipeline

pipeline_cfg = FeaturePipelineConfig(
    use_microstructure=True,  # Enable microstructure features
    # ... other configs
)

result = build_feature_pipeline(df_ohlcv=df, pipeline_cfg=pipeline_cfg)
# result.df now contains ms_* columns
```

### 3. Strategy Usage

**IMPORTANT:** Strategies should ONLY use the standardized `ms_*` columns, NOT direct calls to microstructure detectors.

```python
# ✅ CORRECT
class MyStrategy(BaseStrategy):
    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        chop = row.get("ms_chop", 0.0)
        burst_up = row.get("ms_burst_up", 0)
        vol_regime = row.get("ms_vol_regime", 0)

        if chop < 0.4 and burst_up == 1 and vol_regime == 0:
            return "LONG"
        return None

# ❌ INCORRECT - Do NOT call detectors directly
from finantradealgo.microstructure.chop_detector import compute_chop
class BadStrategy(BaseStrategy):
    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        # BAD: Direct detector call
        chop = compute_chop(df["close"], cfg)
        ...
```

**Example strategies using microstructure:**
- `volatility_breakout.py`: Uses `ms_chop` for chop filtering
- Future strategies can reference `ms_sweep_*`, `ms_exhaustion_*`, etc.

### 4. Risk Engine Integration

**Current Status:** RiskEngine does NOT directly reference microstructure features.

Risk calculations are based on:
- Historical volatility (`hv_20`)
- ATR-based position sizing
- Daily loss limits

If future risk logic needs microstructure signals, it should:
1. Accept them via `row` parameter (already available)
2. Use standardized `ms_*` column names
3. Document the dependency

**Example (hypothetical):**
```python
class RiskEngine:
    def can_open_new_trade(self, row: pd.Series, ...) -> bool:
        # Check if market is too choppy
        chop = row.get("ms_chop", 0.0)
        if chop > 0.7:  # High chop
            return False
        # ... rest of risk checks
```

---

## Live/Paper Trading Considerations

### Latency Control (Task S2.E2)

**Problem:** In live environments, accumulating trades/book data can cause computation latency.

**Solution:** `max_lookback_seconds` config parameter

```yaml
microstructure:
  max_lookback_seconds: 3600  # Only use last 1 hour of trades/book data
```

**How it works:**
1. Engine truncates `trades_df` and `book_df` before computation
2. Cutoff time = `df.index[-1] - max_lookback_seconds`
3. Only data after cutoff is used for feature calculation
4. Set to `0` to disable truncation (use all data)

**Recommended values:**
- **Live trading:** 1800-3600 seconds (30 min - 1 hour)
- **Paper trading:** 3600 seconds (1 hour)
- **Backtesting:** 0 (use all data)

---

## Testing and Health Checks

### Unit Tests

```bash
pytest tests/test_microstructure_engine.py -v
pytest tests/test_microstructure_config.py -v
```

**Coverage:**
- Input validation (OHLCV, trades_df, book_df contracts)
- Feature computation correctness
- Output contract enforcement
- Lookback window truncation (S2.E2)

### Debug Helpers

```python
from finantradealgo.microstructure.microstructure_debug import (
    summarize_microstructure_features,
    get_microstructure_health_metrics,
)

# Text summary
summary = summarize_microstructure_features(features_df, df)
print(summary)

# Metrics dictionary
metrics = get_microstructure_health_metrics(features_df)
print(f"Total bars: {metrics['total_bars']}")
print(f"Burst events: {metrics['burst_events']}")
print(f"Missing columns: {metrics['missing_cols']}")
```

### Integration Tests

```bash
pytest tests/test_feature_pipeline_ms_micro.py -v
```

**Validates:**
- Market structure + Microstructure coexistence
- Pipeline metadata
- Column prefix consistency

---

## Migration Notes

### For Existing Code

If you have old code referencing microstructure features without `ms_` prefix:

```python
# Old (deprecated)
chop = df["chop"]
burst = df["burst_up"]

# New (correct)
chop = df["ms_chop"]
burst = df["ms_burst_up"]
```

**Where to check:**
- Strategy implementations (`.strategies/`)
- Risk engine (`risk/risk_engine.py`)
- Custom analysis scripts

**Status:** Current audit shows strategies already use `ms_*` prefixes. No legacy code found.

---

## Future Work

### Planned Enhancements

1. **Order flow imbalance** (beyond simple bid/ask)
2. **VWAP-based microstructure signals**
3. **Market maker detection**
4. **Smart order routing signals**

### Performance Optimization

Current implementation uses:
- Row-by-row iteration for liquidity sweeps (vectorization planned)
- Simple reindex/ffill for book data alignment

Future optimization:
- Vectorized sweep detection
- Efficient time-series join strategies
- Cython/Numba acceleration for hot paths

---

## References

- **Config:** `finantradealgo/microstructure/config.py`
- **Engine:** `finantradealgo/microstructure/microstructure_engine.py`
- **Types:** `finantradealgo/microstructure/types.py`
- **Input Spec:** `finantradealgo/microstructure/input_spec.py`
- **Debug Tools:** `finantradealgo/microstructure/microstructure_debug.py`
- **Tests:** `tests/test_microstructure_engine.py`, `tests/test_feature_pipeline_ms_micro.py`
- **Pipeline Integration:** `finantradealgo/features/microstructure_features.py`

---

## Support

For questions or issues:
1. Check unit tests for usage examples
2. Review this documentation
3. Consult `finantradealgo/microstructure/__init__.py` for public API

---

**Last Updated:** 2025-12-01
**Sprint:** Core-S2 (Microstructure Engine Hardening)
**Tasks:** S2.1-S2.6, S2.E1-S2.E4

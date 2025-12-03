# Market Structure Integration Guide

**Version:** 1.0
**Last Updated:** 2025-01-01
**Task Reference:** Core-S1

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Feature Pipeline Integration](#feature-pipeline-integration)
6. [Using Market Structure in Strategies](#using-market-structure-in-strategies)
7. [Debug and Visualization](#debug-and-visualization)
8. [Column Reference](#column-reference)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The **Market Structure Engine** identifies key market patterns and structures from price action, including:

- **Swing Points**: Higher highs, higher lows, lower highs, lower lows
- **Trend Regime**: Current market trend direction (uptrend, downtrend, neutral)
- **Chop Regime**: Measure of market choppiness vs trending behavior (0-1 scale)
- **Fair Value Gaps (FVG)**: Price inefficiencies and imbalances
- **Supply/Demand Zones**: Areas of strong institutional interest
- **Break of Structure (BoS)**: Trend continuation signals
- **Change of Character (ChoCh)**: Potential trend reversal signals

### Key Features

- **Smoothing Pipeline**: Noise reduction for more robust structure identification
- **Standardized Output**: All features use consistent `ms_*` prefix
- **Configuration-Driven**: Full control via `system.yml`
- **Dependency Validation**: Automatic checking of required features for strategies
- **Debug Tools**: Visualization and summary functions for analysis

---

## Quick Start

### 1. Enable Market Structure Features

Edit your `config/system.research.yml` (or `config/system.live.yml` for live):

```yaml
features:
  use_market_structure: true  # Enable market structure features

market_structure:
  smoothing:
    enabled: true
    price_ma_window: 3          # Smoothing window for price
    swing_min_distance: 3       # Minimum bars between swing points
    swing_min_zscore: 0.5       # Minimum swing significance
  chop:
    lookback_period: 14         # Chop calculation lookback
```

### 2. Use in Feature Pipeline

```python
from finantradealgo.features.market_structure_features import add_market_structure_features

# Add market structure features to your DataFrame
df_with_ms = add_market_structure_features(df_ohlcv)

# Check available columns
print([col for col in df_with_ms.columns if col.startswith('ms_')])
```

### 3. Access in Strategy

```python
from finantradealgo.core.strategy import BaseStrategy, StrategyContext, SignalType

class MyStrategy(BaseStrategy):
    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        # Access market structure features
        trend = row['ms_trend_regime']  # 1=up, -1=down, 0=neutral
        chop = row['ms_chop_regime']    # 0=trending, 1=choppy

        swing_high = row['ms_swing_high']  # 1 if swing high, else 0
        fvg_up = row['ms_fvg_up']          # 1 if bullish FVG, else 0

        # Example logic
        if trend > 0 and chop < 0.4 and fvg_up == 1:
            return "LONG"

        return None
```

---

## Architecture

### Component Hierarchy

```
MarketStructureEngine
├── Smoothing Pipeline (smoothing.py)
│   ├── smooth_price()           # Price smoothing
│   └── filter_swing_points()    # Swing filtering
├── Swing Detection (swings.py)
│   └── detect_swings()          # Identify swing highs/lows
├── Trend Regime (regime.py)
│   └── infer_trend_regime()     # Trend classification
├── Chop Detection (chop_detector.py)
│   └── compute_chop()           # Chop score calculation
├── FVG Detection (fvg.py)
│   └── detect_fvg_series()      # Fair value gap identification
├── Zone Building (zones.py)
│   └── build_zones()            # Supply/demand zone detection
└── Structure Breaks (breaks.py)
    └── detect_bos_choch()       # BoS/ChoCh identification
```

### Data Flow

```
OHLCV DataFrame
    ↓
smooth_price() → price_smooth column
    ↓
detect_swings() → ms_swing_high, ms_swing_low
    ↓
filter_swing_points() → Filtered swings
    ↓
infer_trend_regime() → ms_trend_regime
compute_chop() → ms_chop_regime
detect_fvg_series() → ms_fvg_up, ms_fvg_down
build_zones() → ms_zone_demand, ms_zone_supply
detect_bos_choch() → ms_bos_up, ms_bos_down, ms_choch
    ↓
MarketStructureResult
├── features: DataFrame with all ms_* columns
└── zones: List[Zone] objects
```

### Entry Points

**Single Entry Point (Recommended):**
```python
from finantradealgo.features.market_structure_features import add_market_structure_features

df_with_ms = add_market_structure_features(df_ohlcv, cfg)
```

**Advanced Usage (with zones):**
```python
from finantradealgo.features.market_structure_features import compute_market_structure_with_zones

result = compute_market_structure_with_zones(df_ohlcv, cfg)
df_features = result.features
zones = result.zones  # List of Zone objects with detailed info
```

---

## Configuration

### SmoothingConfig

Controls price smoothing and swing filtering to reduce noise.

```yaml
market_structure:
  smoothing:
    enabled: true              # Enable/disable smoothing
    price_ma_window: 3         # Moving average window for price (1-10)
    swing_min_distance: 3      # Min bars between swings (1-10)
    swing_min_zscore: 0.5      # Min z-score for swing significance (0-2)
```

**Parameters:**
- `enabled`: Master switch for smoothing pipeline
- `price_ma_window`: Larger values = smoother price, fewer whipsaws
- `swing_min_distance`: Prevents micro-swings, larger = fewer swings
- `swing_min_zscore`: Filters out insignificant swings based on range

### ChopConfig

Controls chop regime calculation.

```yaml
market_structure:
  chop:
    lookback_period: 14        # Bars to look back for chop calculation (5-50)
```

**Parameters:**
- `lookback_period`: Larger values = smoother chop indicator, slower response

### SwingConfig

Controls swing point detection sensitivity.

```yaml
market_structure:
  swing:
    lookback: 2                # Bars to left/right for swing validation (1-5)
    min_swing_size_pct: 0.003  # Min swing size as % of price (0-0.01)
```

### Full Configuration Example

```yaml
market_structure:
  smoothing:
    enabled: true
    price_ma_window: 3
    swing_min_distance: 3
    swing_min_zscore: 0.5
  chop:
    lookback_period: 14
  swing:
    lookback: 2
    min_swing_size_pct: 0.003
  trend:
    min_swings: 4              # Min swings to determine trend
  fvg:
    min_gap_pct: 0.001         # Min gap size for FVG (0.1%)
    max_bars_ahead: 50         # Max bars to track FVG fill
  zone:
    price_proximity_pct: 0.003 # Price proximity for zone touches
    min_touches: 2             # Min touches to form a zone
    window_bars: 500           # Lookback window for zone detection
  breaks:
    swing_break_buffer_pct: 0.0005  # Tolerance for swing breaks (0.05%)
```

---

## Feature Pipeline Integration

### Automatic Integration

The feature pipeline automatically adds market structure features when enabled:

```python
# In your backtest or live trading setup
from finantradealgo.system.config_loader import load_config

cfg = load_config("research")  # or "live"

# Feature pipeline will check cfg.features.use_market_structure
# and automatically call add_market_structure_features() if True
```

### Manual Integration

For custom pipelines:

```python
from finantradealgo.features.market_structure_features import add_market_structure_features
from finantradealgo.market_structure.config import MarketStructureConfig

# Load config
ms_cfg = MarketStructureConfig.from_dict(system_config.get("market_structure", {}))

# Add features
df_enhanced = add_market_structure_features(df_ohlcv, ms_cfg)
```

### Dependency Validation

Check if a strategy's dependencies are met:

```python
from finantradealgo.features.dependency_check import validate_strategy_dependencies
from finantradealgo.strategies.strategy_engine import STRATEGY_SPECS

strategy_spec = STRATEGY_SPECS["rule"]
meta = strategy_spec.meta

# Validate dependencies
is_valid = validate_strategy_dependencies(
    df=feature_df,
    strategy_name=meta.name,
    uses_market_structure=meta.uses_market_structure,
    uses_microstructure=meta.uses_microstructure,
    strict=False,  # If True, raises ValueError on missing features
)

if not is_valid:
    print("Warning: Strategy dependencies not satisfied!")
```

---

## Using Market Structure in Strategies

### Strategy Registration

When creating a strategy, declare your dependencies in the `StrategyMeta`:

```python
from finantradealgo.strategies.strategy_engine import StrategyMeta, StrategySpec

STRATEGY_SPECS["my_strategy"] = StrategySpec(
    name="my_strategy",
    strategy_cls=MyStrategy,
    config_cls=MyStrategyConfig,
    config_extractor=_default_extractor("my_strategy"),
    meta=StrategyMeta(
        name="my_custom_strategy",
        family="trend",
        uses_ml=False,
        uses_microstructure=False,
        uses_market_structure=True,  # ← Declare dependency
        default_feature_preset="extended",
    ),
)
```

### Example Strategies

#### Trend Following with Market Structure

```python
class TrendMSStrategy(BaseStrategy):
    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        # Check trend and chop
        trend = row['ms_trend_regime']
        chop = row['ms_chop_regime']

        # Only trade in strong trends (low chop)
        if chop > 0.6:
            return None  # Too choppy, stay out

        # Enter on trend confirmation
        if trend > 0 and row['ms_swing_low'] == 1:
            return "LONG"  # Swing low in uptrend
        elif trend < 0 and row['ms_swing_high'] == 1:
            return "SHORT"  # Swing high in downtrend

        return None
```

#### FVG-Based Reversal

```python
class FVGReversalStrategy(BaseStrategy):
    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        # Look for FVG signals
        fvg_up = row['ms_fvg_up']
        fvg_down = row['ms_fvg_down']

        # Check zone proximity
        in_demand_zone = row['ms_zone_demand'] > 0
        in_supply_zone = row['ms_zone_supply'] > 0

        # Long at demand zone with bullish FVG
        if in_demand_zone and fvg_up == 1:
            return "LONG"

        # Short at supply zone with bearish FVG
        if in_supply_zone and fvg_down == 1:
            return "SHORT"

        return None
```

---

## Debug and Visualization

### Text Summary

Get a quick overview of market structure:

```python
from finantradealgo.market_structure.debug_plot import summarize_market_structure
from finantradealgo.features.market_structure_features import compute_market_structure_with_zones

result = compute_market_structure_with_zones(df)
summary = summarize_market_structure(result, df)
print(summary)
```

**Output:**
```
============================================================
MARKET STRUCTURE SUMMARY
============================================================

Total bars: 1000
Swing Highs: 45
Swing Lows: 48

Current Trend: UPTREND (1)
Current Chop: TRENDING (0.312)

FVG Up: 12
FVG Down: 8

BoS Up: 5
BoS Down: 2
ChoCh: 3

Supply Zones: 8
Demand Zones: 10
  Avg Supply Strength: 3.25
  Avg Demand Strength: 2.90

Price Context:
  Current: 42350.50
  Change: +1250.00 (+3.04%)
============================================================
```

### Zone Details

Print detailed zone information:

```python
from finantradealgo.market_structure.debug_plot import print_zone_details

result = compute_market_structure_with_zones(df)
print_zone_details(result.zones)
```

**Output:**
```
================================================================================
ZONE DETAILS
================================================================================
Type     Start        End          Price Low    Price High   Strength
--------------------------------------------------------------------------------
DEMAND   450          475          41250.00     41350.00     3
SUPPLY   520          545          42500.00     42650.00     4
DEMAND   600          625          41800.00     41900.00     2
================================================================================
```

### Visual Plotting

Plot comprehensive market structure analysis (requires `matplotlib`):

```python
from finantradealgo.market_structure.debug_plot import plot_market_structure

result = compute_market_structure_with_zones(df)

plot_market_structure(
    df=df,
    result=result,
    title="BTC/USDT 15m Market Structure",
    show_price_smooth=True,
    show_swings=True,
    show_fvg=True,
    show_zones=True,
    show_bos_choch=True,
    show_regime=True,
    figsize=(16, 10),
)
```

---

## Column Reference

### Standard Columns

All market structure features use the `ms_` prefix for consistency.

| Column Name | Type | Description | Range/Values |
|-------------|------|-------------|--------------|
| `price_smooth` | float | Smoothed price (moving average) | Price value |
| `ms_swing_high` | int | Swing high marker | 0 or 1 |
| `ms_swing_low` | int | Swing low marker | 0 or 1 |
| `ms_trend_regime` | int | Trend direction | -1 (down), 0 (neutral), 1 (up) |
| `ms_chop_regime` | float | Chop score | 0.0 (trending) to 1.0 (choppy) |
| `ms_fvg_up` | int | Bullish FVG marker | 0 or 1 |
| `ms_fvg_down` | int | Bearish FVG marker | 0 or 1 |
| `ms_zone_demand` | float | Demand zone strength | 0.0 (no zone) to N (touches) |
| `ms_zone_supply` | float | Supply zone strength | 0.0 (no zone) to N (touches) |
| `ms_bos_up` | int | Bullish break of structure | 0 or 1 |
| `ms_bos_down` | int | Bearish break of structure | 0 or 1 |
| `ms_choch` | int | Change of character | -1 (bearish), 0 (none), 1 (bullish) |

### Column Contract

The `MarketStructureColumns` dataclass ensures consistent naming:

```python
from finantradealgo.market_structure.types import MarketStructureColumns

cols = MarketStructureColumns()
print(cols.swing_high)  # "ms_swing_high"
print(cols.trend_regime)  # "ms_trend_regime"
```

---

## Best Practices

### 1. Configuration Tuning

- **Start with defaults**: Test before tweaking parameters
- **Timeframe-specific**: Higher timeframes may need larger `swing_min_distance`
- **Market-specific**: Crypto vs. stocks may need different `min_swing_size_pct`

### 2. Strategy Development

- **Validate dependencies**: Use `validate_strategy_dependencies()` in tests
- **Check chop regime**: Avoid trading in choppy markets (`ms_chop_regime > 0.6`)
- **Combine signals**: Use multiple market structure features for confirmation

### 3. Debugging

- **Use visualization**: `plot_market_structure()` for visual inspection
- **Print summaries**: `summarize_market_structure()` for quick stats
- **Check raw data**: Use `compute_market_structure_with_zones()` for Zone objects

### 4. Performance

- **Cache results**: Market structure is expensive, compute once per bar
- **Disable if unused**: Set `use_market_structure: false` to save compute
- **Monitor memory**: Large DataFrames with zones can consume significant memory

### 5. Testing

- **Unit tests**: Test individual functions with small DataFrames
- **Integration tests**: Test full pipeline with realistic data
- **Backtest validation**: Compare manual structure identification with automated

---

## Troubleshooting

### Missing Columns Error

**Error:** `KeyError: 'ms_swing_high'`

**Cause:** Market structure features not enabled or not computed.

**Solution:**
```yaml
# In config/system.research.yml
features:
  use_market_structure: true
```

### Dependency Validation Failed

**Warning:** `Strategy 'rule_signals' requires market structure features, but columns are missing`

**Cause:** Strategy requires market structure but features are disabled.

**Solution:**
1. Enable features in config (see above)
2. Or set strategy `uses_market_structure: false` if not needed

### Too Many/Too Few Swings

**Issue:** Swing detection too sensitive or insensitive.

**Solution:**
Adjust smoothing parameters:
```yaml
market_structure:
  smoothing:
    swing_min_distance: 5      # Increase for fewer swings
    swing_min_zscore: 1.0      # Increase for more significant swings only
```

### Chop Regime Always High

**Issue:** `ms_chop_regime` consistently > 0.8 even in trends.

**Solution:**
- Check if market is actually choppy
- Increase `lookback_period` for smoother response:
```yaml
market_structure:
  chop:
    lookback_period: 20        # Larger = smoother
```

### Visualization Not Working

**Error:** `ImportError: matplotlib is required for plotting`

**Solution:**
```bash
pip install matplotlib
```

### Performance Issues

**Issue:** Market structure computation is slow.

**Solutions:**
1. Reduce DataFrame size (process in chunks)
2. Disable unused features
3. Profile code to identify bottlenecks:
   ```python
   import cProfile
   cProfile.run('add_market_structure_features(df)')
   ```

---

## Further Reading

- **Market Structure Theory**: [docs/trading_concepts.md](trading_concepts.md)
- **Strategy Development**: [docs/strategy_guide.md](strategy_guide.md)
- **Feature Pipeline**: [docs/feature_pipeline.md](feature_pipeline.md)
- **API Reference**: [finantradealgo/market_structure/README.md](../finantradealgo/market_structure/README.md)

---

**Maintained by:** FinanTradeAlgo Team
**Questions:** See [docs/FAQ.md](FAQ.md) or open an issue on GitHub

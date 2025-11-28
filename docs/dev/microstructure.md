# Developer Documentation: Microstructure Module

## 1. Purpose

The Microstructure module is designed to compute high-frequency trading signals that go beyond simple OHLCV-based technical analysis. The goal is to capture market dynamics related to volatility, trend state, momentum acceleration, and order flow pressure. These signals can be used as features in ML models or as direct inputs for rule-based strategies.

## 2. Signal Definitions

The module produces a set of signals, each prefixed with `ms_`.

### 2.1. Volatility Regime (`ms_vol_regime`)
- **Calculation:**
  1.  `log_returns = log(close / close.shift(1))`
  2.  `rolling_vol = rolling_std(log_returns, period)`
  3.  `z_score = (rolling_vol - rolling_mean(rolling_vol)) / rolling_std(rolling_vol)`
- **Output:** An integer representing the regime:
    - `1`: High Volatility (`z_score >= high_z_threshold`)
    - `0`: Normal Volatility
    - `-1`: Low Volatility (`z_score <= low_z_threshold`)

### 2.2. Chop Score (`ms_chop`)
- **Calculation:**
  1.  `range_sum = rolling_sum(|close.diff()|, lookback)`
  2.  `net_move = |close - close.shift(lookback)|`
  3.  `chop = 1 - net_move / range_sum`
- **Output:** A score from `0.0` to `1.0`.
    - `~0.0`: Strong directional trend.
    - `~1.0`: Choppy, sideways market.

### 2.3. Price Bursts (`ms_burst_up`, `ms_burst_down`)
- **Calculation:**
  1.  `returns = pct_change(close, return_window)`
  2.  `z_score = z_score(returns, z_score_window)`
  3.  `burst_up = max(0, z_score - z_up_threshold)`
  4.  `burst_down = max(0, -z_score - z_down_threshold)`
- **Output:** The magnitude of the z-score's excursion beyond the threshold, indicating a significant momentum event.

### 2.4. Trend Exhaustion (`ms_exhaustion_up`, `ms_exhaustion_down`)
- **Calculation:** An exhaustion signal is triggered when a sustained trend occurs on low volume.
  1.  Count consecutive bars in the same direction (`consecutive_up`, `consecutive_down`).
  2.  Calculate the rolling z-score of `volume`.
  3.  `exhaustion_up = 1` if `consecutive_up >= min_consecutive_bars` AND `volume_z_score <= volume_z_threshold`.
- **Output:** A boolean flag (`1` or `0`) indicating trend exhaustion.

### 2.5. Parabolic Trend (`ms_parabolic_trend`)
- **Calculation:** Measures the curvature of the price series.
  1.  `second_diff = close - 2*close.shift(1) + close.shift(2)`
  2.  `curvature = second_diff / rolling_std(close, window)`
- **Output:** An integer representing the trend's shape:
    - `1`: Parabolic (convex) upward trend.
    - `-1`: Parabolic (concave) downward trend.
    - `0`: Linear or no significant trend.

### 2.6. Order Book Imbalance (`ms_imbalance`)
- **Calculation:**
  1.  `total_bid = sum(size)` for bids up to `depth_levels`.
  2.  `total_ask = sum(size)` for asks up to `depth_levels`.
  3.  `imbalance = (total_bid - total_ask) / (total_bid + total_ask)`
- **Output:** A score from `-1.0` (ask-heavy) to `1.0` (bid-heavy).

### 2.7. Liquidity Sweep (`ms_sweep_up`, `ms_sweep_down`)
- **Calculation:** Detects large, one-sided notional trades within a bar's time window that result in price impact.
  1.  Sum the notional (`price * size`) of buy and sell trades in a window.
  2.  If `buy_notional > threshold` and `close > open`, a `sweep_up` is detected.
  3.  If `sell_notional > threshold` and `close < open`, a `sweep_down` is detected.
- **Output:** The total notional value of the sweep event.

## 3. Output Columns

The module produces the following columns, which are accessible via `MicrostructureSignals.columns()`:

| Column Name            | Type  | Description                                 |
| ---------------------- | ----- | ------------------------------------------- |
| `ms_imbalance`         | float | (Optional) Order book bid/ask imbalance.    |
| `ms_sweep_up`          | float | (Optional) Notional of an upward sweep.     |
| `ms_sweep_down`        | float | (Optional) Notional of a downward sweep.    |
| `ms_chop`              | float | Chop vs. Trend score (0=Trend, 1=Chop).     |
| `ms_burst_up`          | float | Magnitude of an upward momentum burst.      |
| `ms_burst_down`        | float | Magnitude of a downward momentum burst.     |
| `ms_vol_regime`        | int   | Volatility regime (-1=Low, 0=Normal, 1=High).|
| `ms_exhaustion_up`     | int   | Flag for an exhausting uptrend.             |
| `ms_exhaustion_down`   | int   | Flag for an exhausting downtrend.           |
| `ms_parabolic_trend`   | int   | Parabolic trend flag (-1=Down, 1=Up).       |

## 4. Configuration Parameters

All parameters are configured via their respective dataclasses within the main `MicrostructureConfig`.

- **`VolatilityRegimeConfig`**
  - `period`: Lookback for calculating rolling volatility of returns.
  - `z_score_window`: Lookback for calculating the z-score of the volatility.
  - `low_z_threshold`/`high_z_threshold`: Z-score thresholds to define low/high regimes.
- **`ChopConfig`**
  - `lookback_period`: The window size for comparing net movement vs. total movement.
- **`BurstConfig`**
  - `return_window`: The window for calculating `pct_change`.
  - `z_score_window`: Lookback for calculating the z-score of the returns.
  - `z_up_threshold`/`z_down_threshold`: Z-score thresholds to define a burst.
- **`ExhaustionConfig`**
  - `min_consecutive_bars`: Minimum number of same-direction bars to constitute a trend.
  - `volume_z_score_window`: Lookback for calculating the volume z-score.
  - `volume_z_threshold`: Z-score threshold below which volume is considered "low".
- **`ParabolicConfig`**
  - `rolling_std_window`: Lookback for calculating the rolling standard deviation to normalize curvature.
  - `curvature_threshold`: Threshold for the normalized second derivative to be considered parabolic.
- **`ImbalanceConfig`**
  - `depth`: Number of order book levels to include in the calculation.
- **`LiquiditySweepConfig`**
  - `lookback_ms`: How many milliseconds to look back from the start of a bar for relevant trades.
  - `notional_threshold`: The minimum notional value to trigger a sweep event.

## 5. Known Limitations

- **Order Book / Trade Features:** The `imbalance` and `sweep` features are considered **optional**. They are only computed if `trades_df` and `book_df` DataFrames are passed to `compute_microstructure_df`. If not provided, their corresponding columns will be filled with `0.0`.
- **`detect_liquidity_sweep` Performance:** The current integration for liquidity sweeps iterates over each OHLCV bar (`df.apply`). While acceptable for moderate datasets, this can be a bottleneck for very large backtests. A future optimization ("Option B") would involve a more efficient, vectorized mapping of trades to bars.

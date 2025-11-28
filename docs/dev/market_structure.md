# Developer Guide: Market Structure Module

## 1. Purpose

The `finantradealgo.market_structure` module is designed to interpret and quantify the narrative of price action. It is not a traditional indicator that produces a simple oscillating value. Instead, it functions as a **state machine** for the market, identifying key structural events and states like trends, breaks of structure, and liquidity zones.

The primary goal is to provide a rich, contextual layer that strategy logic can use to understand *what* the market is doing, not just where the price is.

## 2. Feature Columns (`ms_`)

The `MarketStructureEngine` outputs the following columns, which are prefixed with `ms_`.

| Column Name          | Type  | Value Range              | Meaning                                                                                              |
| -------------------- | ----- | ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `ms_swing_high`      | `int` | `{0, 1}`                 | A `1` marks the confirmation of a swing high point.                                                  |
| `ms_swing_low`       | `int` | `{0, 1}`                 | A `1` marks the confirmation of a swing low point.                                                   |
| `ms_trend_regime`    | `int` | `{-1, 0, 1}`             | The current market trend: `1` for Uptrend, `-1` for Downtrend, `0` for Ranging/Undetermined.       |
| `ms_bos_up`          | `int` | `{0, 1}`                 | A `1` signals a **Break of Structure** above a prior swing high in an uptrend.                       |
| `ms_bos_down`        | `int` | `{0, 1}`                 | A `1` signals a **Break of Structure** below a prior swing low in a downtrend.                       |
| `ms_choch`           | `int` | `{0, 1}`                 | A `1` signals a **Change of Character**, flagged when the `ms_trend_regime` flips sign.            |
| `ms_fvg_up`          | `int` | `{0, 1}`                 | A `1` marks a bullish **Fair Value Gap** (imbalance) formed on the preceding bar.                  |
| `ms_fvg_down`        | `int` | `{0, 1}`                 | A `1` marks a bearish **Fair Value Gap** (imbalance) formed on the preceding bar.                  |
| `ms_zone_demand`     | `float` | `>= 0.0`                 | If `> 0`, price is in a demand zone. The value is the zone's strength (touches + volume).        |
| `ms_zone_supply`     | `float` | `>= 0.0`                 | If `> 0`, price is in a supply zone. The value is the zone's strength (touches + volume).          |

## 3. Configuration Parameters

The behavior of the engine is controlled by the `MarketStructureConfig` dataclass, which contains sub-configurations for each component.

-   **`swing.lookback`**: The number of bars to the left and right of a candle to confirm it as a swing point. Higher values result in fewer, more significant swings.
-   **`swing.min_swing_size_pct`**: The minimum required price movement (as a percentage) from a previous swing for a new one to be considered valid. Filters out minor "wiggles."
-   **`trend.min_swings`**: The minimum number of swings required before a trend regime (`1` or `-1`) can be established.
-   **`zone.price_proximity_pct`**: The percentage of price difference allowed for two swing points to be clustered into the same supply/demand zone.
-   **`zone.min_touches`**: The minimum number of swing points required to form a valid zone.
-   **`zone.window_bars`**: The lookback window (in bars) used to find recent swing points for zone creation.
-   **`fvg.min_gap_pct`**: The minimum size of an imbalance, as a percentage of price, to be considered a valid Fair Value Gap.
-   **`breaks.swing_break_buffer_pct`**: A small tolerance buffer (in percent) to confirm a Break of Structure, preventing false signals from breaks that are too marginal.

## 4. Performance Notes

The engine is designed to run on a full DataFrame in a batch process. The performance characteristics of its components are as follows:

-   **`detect_swings`, `detect_fvg_series`, `infer_trend_regime`**: These are vectorized or have linear complexity **O(N)**, where N is the number of bars. They are very fast.
-   **`build_zones`**: This function involves a nested loop for clustering (`for swing in sorted_swings: for cluster in clusters:`). Its complexity is roughly **O(S * C)**, where S is the number of swings in the window and C is the number of zones found. In highly choppy markets with many swings, this could become a minor bottleneck.
-   **`detect_bos_choch`**: This function contains a Python loop over all detected swings. The original code contains a `TODO` to vectorize this calculation if performance becomes an issue. For DataFrames with a very large number of swings, this is the **most likely bottleneck** in the engine.

## 5. Known Limitations

This implementation is a robust foundation but has several areas for future improvement:

-   **Volume Profile is Basic**: The volume profile used for zone strength is a simple price histogram. It does not identify distinct High Volume Nodes (HVNs) or Low Volume Nodes (LVNs) in detail.
-   **No FVG Fill Tracking**: The engine flags the creation of FVGs but does not track their state (e.g., if/when they are filled or partially filled). This logic would need to be handled by the strategy itself.
-   **Simplified ChoCh**: The Change of Character (`ms_choch`) detection is currently tied to the `ms_trend_regime` flipping. A more traditional definition involves price breaking the last minor swing structure against the primary trend, which is a more nuanced calculation not yet implemented.
-   **Static Zone Analysis**: Zones are built based on a fixed lookback window. The engine does not currently have a concept of a zone's "freshness" or whether it has been mitigated.

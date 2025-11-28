import numpy as np
import pandas as pd
import pytest

from finantradealgo.features.market_structure_features import (
    add_market_structure_features,
)
from finantradealgo.market_structure.config import MarketStructureConfig


def test_add_market_structure_features_integration():
    """
    Tests that the main wrapper function correctly adds all expected columns.
    """
    np.random.seed(42)
    num_bars = 50
    data = {
        "open": 100 + np.random.randn(num_bars).cumsum(),
        "high": 100 + np.random.randn(num_bars).cumsum() + 1,
        "low": 100 + np.random.randn(num_bars).cumsum() - 1,
        "close": 100 + np.random.randn(num_bars).cumsum(),
        "volume": np.random.randint(100, 1000, num_bars),
    }
    df_ohlcv = pd.DataFrame(data, index=pd.to_datetime(pd.date_range("2023-01-01", periods=num_bars, freq="15min")))

    cfg = MarketStructureConfig()
    df_out, zones = add_market_structure_features(df_ohlcv, cfg)

    expected_cols = {
        "ms_swing_high", "ms_swing_low", "ms_trend_regime",
        "ms_fvg_up", "ms_fvg_down", "ms_zone_demand", "ms_zone_supply",
    }
    assert expected_cols.issubset(df_out.columns)
    assert "close" in df_out.columns
    assert len(df_out) == len(df_ohlcv)
    # Check that columns that can be non-zero are numeric
    assert pd.api.types.is_numeric_dtype(df_out["ms_zone_demand"])
    assert pd.api.types.is_numeric_dtype(df_out["ms_zone_supply"])


def test_market_structure_zones_are_identified():
    """
    Uses crafted data to test if a demand zone is correctly identified
    and flagged in the output DataFrame.
    """
    # 1. Craft data to create a clear demand zone around 100
    # timestamps are just indices for simplicity in this test
    prices = [
        # 1. First swing low (part of our future zone)
        105, 103, 101, 100, 102, 
        # 2. Rally up
        105, 110, 115, 120, 118,
        # 3. Second swing low, reinforcing the zone at 100
        104, 102, 100.5, 103,
        # 4. Another rally
        108, 112, 118, 125, 122,
        # 5. Price comes back to test the demand zone
        115, 110, 105, 101.5, # <-- This bar should be in the zone
        # 6. Price moves away again
        104, 108, 111
    ]
    num_bars = len(prices)
    df = pd.DataFrame({
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": [100] * num_bars  # Constant volume for simplicity
    })

    # 2. Configure and run the engine
    # Use a small lookback to form swings easily
    cfg = MarketStructureConfig()
    cfg.swing.lookback = 2
    cfg.swing.min_swing_size_pct = 0.01 # 1%
    cfg.zone.min_touches = 2
    cfg.zone.price_proximity_pct = 0.02 # 2% proximity to cluster 100 and 100.5
    
    df_out, zones = add_market_structure_features(df, cfg)

    # 3. Assertions
    # A demand zone should have formed around price 100-102 from the first two lows.
    assert zones, "The engine should have returned at least one zone"
    assert zones[0].type == "demand"
    
    # The bar at index 22 has close=101.5, which should be inside this zone.
    in_zone_bar_index = 22
    out_of_zone_bar_index = 17 # High up at 125

    # Check the bar that should be IN the demand zone
    demand_strength_in_zone = df_out.loc[in_zone_bar_index, "ms_zone_demand"]
    assert demand_strength_in_zone > 0, "A demand zone should be active at this bar"

    # Check a bar that should be OUT of the zone
    demand_strength_out_of_zone = df_out.loc[out_of_zone_bar_index, "ms_zone_demand"]
    assert demand_strength_out_of_zone == 0.0, "No demand zone should be active high up"
    
    # No significant swing highs were made to form a supply zone
    assert (df_out["ms_zone_supply"] == 0.0).all(), "No supply zones should be formed"


def test_market_structure_golden_fixture():
    """
    Performs a regression test against a golden fixture file.
    This test ensures that the output of the market structure engine remains
    consistent over time. To update the golden file, run the
    `scripts/generate_ms_fixture.py` script.
    """
    # 1. Define paths
    fixture_path = "tests/data/market_structure_fixture.csv"
    golden_path = "tests/golden/market_structure_fixture_out.parquet"

    # 2. Load input data and the golden output
    try:
        df_input = pd.read_csv(fixture_path, parse_dates=["timestamp"])
        df_golden = pd.read_parquet(golden_path)
    except FileNotFoundError as e:
        pytest.fail(
            f"{e}. Please generate the fixture and golden files by running "
            "`python scripts/generate_ms_fixture.py`"
        )

    # 3. Run the engine on the input data
    cfg = MarketStructureConfig()  # Use default config
    df_with_features, _ = add_market_structure_features(df_input, cfg)
    
    # 4. Isolate the market structure columns from the new output
    ms_cols = [col for col in df_with_features.columns if col.startswith("ms_")]
    df_new_output = df_with_features[ms_cols]
    
    # Reset index to ensure alignment, as the golden file has no index
    df_new_output = df_new_output.reset_index(drop=True)
    df_golden = df_golden.reset_index(drop=True)


    # 5. Compare the new output with the golden file
    pd.testing.assert_frame_equal(df_new_output, df_golden, check_dtype=False)

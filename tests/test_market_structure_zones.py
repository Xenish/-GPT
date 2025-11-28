"""
Tests for the market structure zone clustering logic.
"""
import numpy as np
import pandas as pd
import pytest

from finantradealgo.market_structure.config import ZoneConfig
from finantradealgo.market_structure.types import SwingPoint
from finantradealgo.market_structure.zones import build_zones


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Create a sample DataFrame of bars for testing."""
    # Index up to 101 to make sure `last_ts=100` is a valid loc index
    index = np.arange(101)
    # Create somewhat realistic price data that covers the swing prices
    close_prices = np.linspace(90, 125, 101)
    
    # Make volume higher around the price level of the demand zone (100)
    # and lower around the supply zone (110)
    volumes = np.full(101, 100.0)  # Base volume
    # Find indices where price is near 100
    demand_price_indices = np.where((close_prices >= 99) & (close_prices <= 101))
    volumes[demand_price_indices] *= 5  # 5x volume in demand area

    return pd.DataFrame({"close": close_prices, "volume": volumes}, index=index)


def test_build_zones_clustering_with_volume(sample_bars):
    """
    Tests the core logic of clustering and volume-weighting.
    """
    swings = [
        SwingPoint(ts=10, price=100.0, kind="low"),
        SwingPoint(ts=20, price=100.1, kind="low"),
        SwingPoint(ts=30, price=99.9, kind="low"),
        SwingPoint(ts=40, price=95.0, kind="low"), # Ignored (1 touch)
        SwingPoint(ts=50, price=110.0, kind="high"),
        SwingPoint(ts=60, price=110.2, kind="high"),
        SwingPoint(ts=70, price=120.0, kind="high"), # Ignored (1 touch)
        SwingPoint(ts=80, price=100.3, kind="low"),
    ]
    cfg = ZoneConfig(price_proximity_pct=0.01, min_touches=2, window_bars=100)

    zones = build_zones(swings, cfg, sample_bars, last_ts=100)

    assert len(zones) == 2
    zones.sort(key=lambda z: z.type)
    demand_zone, supply_zone = zones[0], zones[1]

    # --- Acceptance Criteria for Demand Zone ---
    assert demand_zone.type == "demand"
    assert demand_zone.strength > 4.0  # Base strength is 4 touches
    assert demand_zone.low == 99.9
    assert demand_zone.high == 100.3

    # --- Acceptance Criteria for Supply Zone ---
    assert supply_zone.type == "supply"
    # Its volume was lower, so its strength should be less than the demand zone's
    assert supply_zone.strength > 2.0  # Base strength is 2 touches
    assert supply_zone.strength < demand_zone.strength
    assert supply_zone.low == 110.0
    assert supply_zone.high == 110.2
    
    # Check that strength is touches + volume factor
    # A rough check is enough, as exact volume factor is implementation-dependent
    assert demand_zone.strength > 4 and demand_zone.strength < 5
    assert supply_zone.strength > 2 and supply_zone.strength < 3


def test_build_zones_empty_and_window(sample_bars):
    """
    Tests edge cases: no swings, and windowing logic.
    """
    cfg = ZoneConfig(price_proximity_pct=0.01, min_touches=2, window_bars=50)

    # Test with no swings
    assert build_zones([], cfg, sample_bars, last_ts=100) == []

    swings = [
        SwingPoint(ts=10, price=100.0, kind="low"), # Outside window
        SwingPoint(ts=20, price=100.1, kind="low"), # Outside window
        SwingPoint(ts=80, price=110.0, kind="high"), # Inside window
        SwingPoint(ts=90, price=110.1, kind="high"), # Inside window
    ]

    # last_ts=100, window_bars=50 -> window starts at ts 50.
    zones = build_zones(swings, cfg, sample_bars, last_ts=100)

    assert len(zones) == 1
    assert zones[0].type == "supply"
    assert zones[0].strength > 2.0 # Base strength is 2 touches
    assert zones[0].first_ts == 80

"""
Tests for the market structure swing detection and trend regime logic.
"""
import numpy as np
import pandas as pd
import pytest

from finantradealgo.market_structure.config import SwingConfig, TrendRegimeConfig
from finantradealgo.market_structure.swings import detect_swings
from finantradealgo.market_structure.regime import infer_trend_regime


@pytest.fixture
def wave_data() -> pd.DataFrame:
    """
    Generates a DataFrame with a clear wave pattern for testing swings.
    Peaks are at indices 5, 25. Troughs are at 15, 35.
    """
    # Sine wave for prices: period=20, amplitude=10
    indices = np.arange(40)
    prices = 100 + 10 * np.sin(2 * np.pi * (indices - 5) / 20)
    df = pd.DataFrame({
        "high": prices,
        "low": prices,
    })
    return df


def test_detect_swings(wave_data):
    """
    Tests that swing points are correctly identified on a simple sine wave.
    """
    cfg = SwingConfig(lookback=4, min_swing_size_pct=0.01)
    
    swings = detect_swings(wave_data["high"], wave_data["low"], cfg)

    swing_highs = [s for s in swings if s.kind == "high"]
    swing_lows = [s for s in swings if s.kind == "low"]

    assert len(swing_highs) == 2, "Should detect two swing highs"
    assert len(swing_lows) == 2, "Should detect two swing lows"

    # Assert that the swings are found at the correct timestamps (indices)
    assert swing_highs[0].ts == 5
    assert swing_highs[1].ts == 25
    assert swing_lows[0].ts == 15
    assert swing_lows[1].ts == 35

    # Assert prices are correct
    assert swing_highs[0].price == pytest.approx(110.0)
    assert swing_lows[0].price == pytest.approx(90.0)


def test_infer_trend_regime():
    """
    Tests the trend regime logic based on a manually crafted sequence of swings.
    """
    from finantradealgo.market_structure.types import SwingPoint

    cfg = TrendRegimeConfig(min_swings=4)

    # 1. Uptrend: Higher Highs (HH), Higher Lows (HL)
    uptrend_swings = [
        SwingPoint(ts=10, price=100, kind="low"),   # L
        SwingPoint(ts=20, price=110, kind="high"),  # H
        SwingPoint(ts=30, price=105, kind="low"),   # HL
        SwingPoint(ts=40, price=115, kind="high"),  # HH
        SwingPoint(ts=50, price=110, kind="low"),   # HL
        SwingPoint(ts=60, price=120, kind="high"),  # HH
    ]
    trend = infer_trend_regime(uptrend_swings, cfg.min_swings)
    assert trend == 1, "Should be in an uptrend (1)"

    # 2. Downtrend: Lower Highs (LH), Lower Lows (LL)
    downtrend_swings = [
        SwingPoint(ts=10, price=120, kind="high"),  # H
        SwingPoint(ts=20, price=110, kind="low"),   # L
        SwingPoint(ts=30, price=115, kind="high"),  # LH
        SwingPoint(ts=40, price=105, kind="low"),   # LL
        SwingPoint(ts=50, price=110, kind="high"),  # LH
        SwingPoint(ts=60, price=100, kind="low"),   # LL
    ]
    trend = infer_trend_regime(downtrend_swings, cfg.min_swings)
    assert trend == -1, "Should be in a downtrend (-1)"

    # 3. Ranging market / not enough data
    ranging_swings = [
        SwingPoint(ts=10, price=100, kind="low"),
        SwingPoint(ts=20, price=110, kind="high"),
        SwingPoint(ts=30, price=100, kind="low"),
        SwingPoint(ts=40, price=110, kind="high"),
    ]
    trend = infer_trend_regime(ranging_swings, cfg.min_swings)
    assert trend == 0, "Should be in a ranging market (0)"

    # Not enough swings
    not_enough_swings = uptrend_swings[:3]
    trend = infer_trend_regime(not_enough_swings, cfg.min_swings)
    assert trend == 0, "Should be ranging (0) with too few swings"

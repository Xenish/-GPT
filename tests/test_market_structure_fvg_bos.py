import numpy as np
import pandas as pd
import pytest

from finantradealgo.market_structure.breaks import detect_bos_choch
from finantradealgo.market_structure.config import BreakConfig, FVGConfig
from finantradealgo.market_structure.fvg import detect_fvg_series
from finantradealgo.market_structure.regime import infer_trend_regime
from finantradealgo.market_structure.swings import detect_swings
from finantradealgo.market_structure.types import SwingPoint

# --- Fixtures ---


@pytest.fixture
def fvg_cfg():
    return FVGConfig(min_gap_pct=0.001)


@pytest.fixture
def break_cfg():
    return BreakConfig(swing_break_buffer_pct=0.0)


# --- FVG Tests ---


def test_detect_bullish_fvg(fvg_cfg):
    """Tests that a bullish FVG is correctly identified."""
    data = {
        "high": [10, 11, 10, 13],
        "low": [9, 10, 9, 12],
    }
    df = pd.DataFrame(data)
    # Create a gap between bar 1 (high=10) and bar 3 (low=12)
    # The FVG is on bar 2
    fvg_up, fvg_down = detect_fvg_series(df, fvg_cfg)

    assert fvg_up.iloc[1] > 0
    assert fvg_down.sum() == 0


def test_detect_bearish_fvg(fvg_cfg):
    """Tests that a bearish FVG is correctly identified."""
    data = {
        "high": [10, 11, 10, 13],
        "low": [9, 10, 9, 8],
    }
    df = pd.DataFrame(data)
    # Create a gap between bar 1 (low=10) and bar 3 (high=10 -> changed to 8)
    df.loc[3, "high"] = 8
    fvg_up, fvg_down = detect_fvg_series(df, fvg_cfg)

    assert fvg_down.iloc[1] > 0
    assert fvg_up.sum() == 0


def test_no_fvg(fvg_cfg):
    """Tests that no FVG is detected when there are no gaps."""
    data = {"high": [10, 11, 10.5], "low": [9, 10, 9.5]}
    df = pd.DataFrame(data)
    fvg_up, fvg_down = detect_fvg_series(df, fvg_cfg)
    assert fvg_up.sum() == 0
    assert fvg_down.sum() == 0


# --- BoS / ChoCh Tests ---


def test_detect_bos_up(break_cfg):
    """Tests a clear Break of Structure in an uptrend."""
    # Manually create swings for a clear uptrend (HH, HL)
    swings = [
        SwingPoint(ts=1, price=100, kind="low"),
        SwingPoint(ts=3, price=110, kind="high"),
        SwingPoint(ts=5, price=105, kind="low"),  # HL
        SwingPoint(ts=7, price=115, kind="high"),  # HH
    ]
    # Trend regime is 1 (uptrend)
    trend_regime = pd.Series([1] * 10)
    # Create a DF where the price breaks the last swing high (115) at index 9
    df = pd.DataFrame({"close": [110] * 9 + [116]})

    bos_up, bos_down, _ = detect_bos_choch(df, swings, trend_regime, break_cfg)

    assert bos_up.iloc[9] == 1
    assert bos_down.sum() == 0


def test_detect_bos_down(break_cfg):
    """Tests a clear Break of Structure in a downtrend."""
    swings = [
        SwingPoint(ts=1, price=115, kind="high"),
        SwingPoint(ts=3, price=105, kind="low"),
        SwingPoint(ts=5, price=110, kind="high"),  # LH
        SwingPoint(ts=7, price=100, kind="low"),   # LL
    ]
    trend_regime = pd.Series([-1] * 10) # Downtrend
    # Price breaks the last swing low (100) at index 9
    df = pd.DataFrame({"close": [105] * 9 + [99]})

    bos_up, bos_down, _ = detect_bos_choch(df, swings, trend_regime, break_cfg)

    assert bos_down.iloc[9] == 1
    assert bos_up.sum() == 0


def test_detect_choch(break_cfg):
    """
    Tests a Change of Character when the trend flips.
    Note: The current implementation simply flags when the trend_regime column
    changes, which is a simplification of a true price-based ChoCh. This
    test validates that simplified logic.
    """
    # Swings and DF are not as important here as the trend_regime series
    swings = []
    df = pd.DataFrame({"close": [100] * 10})
    # Create a trend regime that flips from up (1) to down (-1)
    trend_regime = pd.Series([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

    _, _, choch = detect_bos_choch(df, swings, trend_regime, break_cfg)
    
    # ChoCh should be 1 at index 5 where the regime flips from 1 to -1
    assert choch.iloc[5] == 1
    assert choch.sum() == 1  # Only one change of character

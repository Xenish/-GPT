"""
Parameter space definition for Volatility Breakout strategy.

Defines searchable parameter ranges for strategy_search optimization.
"""
from __future__ import annotations

from finantradealgo.strategies.param_space import ParamSpace, ParamSpec

VOLATILITY_PARAM_SPACE: ParamSpace = {
    # Range detection lookback
    "range_lookback": ParamSpec(
        name="range_lookback",
        type="int",
        low=10,
        high=50,
        log=False,
    ),
    # Breakout multiplier (how far from range to trigger)
    "breakout_mult": ParamSpec(
        name="breakout_mult",
        type="float",
        low=0.5,
        high=2.5,
        log=False,
    ),
    # Volatility regime filter
    "vol_regime_filter": ParamSpec(
        name="vol_regime_filter",
        type="bool",
    ),
    # Volume confirmation threshold
    "volume_confirm_mult": ParamSpec(
        name="volume_confirm_mult",
        type="float",
        low=1.0,
        high=3.0,
        log=False,
    ),
    # ATR-based stop loss multiplier
    "atr_sl_mult": ParamSpec(
        name="atr_sl_mult",
        type="float",
        low=0.5,
        high=2.0,
        log=False,
    ),
    # ATR-based take profit multiplier
    "atr_tp_mult": ParamSpec(
        name="atr_tp_mult",
        type="float",
        low=1.0,
        high=4.0,
        log=False,
    ),
}

__all__ = ["VOLATILITY_PARAM_SPACE"]

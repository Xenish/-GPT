"""
ParamSpace definition for Trend family strategies.

This module defines the searchable parameter space for trend-following strategies
like TrendContinuationStrategy.
"""

from __future__ import annotations

from finantradealgo.strategies.param_space import ParamSpace, ParamSpec


TREND_PARAM_SPACE: ParamSpace = {
    # RSI trend filter (for trend confirmation)
    "rsi_trend_min": ParamSpec(
        name="rsi_trend_min",
        type="float",
        low=40.0,
        high=55.0,
    ),
    "rsi_trend_max": ParamSpec(
        name="rsi_trend_max",
        type="float",
        low=60.0,
        high=80.0,
    ),
    # Trend score filter
    "min_trend_score": ParamSpec(
        name="min_trend_score",
        type="float",
        low=-0.2,
        high=0.3,
    ),
    # Microstructure trend filter
    "use_ms_trend_filter": ParamSpec(
        name="use_ms_trend_filter",
        type="bool",
    ),
    "ms_trend_min": ParamSpec(
        name="ms_trend_min",
        type="float",
        low=-0.5,
        high=0.1,
    ),
    "ms_trend_max": ParamSpec(
        name="ms_trend_max",
        type="float",
        low=0.5,
        high=1.5,
    ),
}


__all__ = ["TREND_PARAM_SPACE"]

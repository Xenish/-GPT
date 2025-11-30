"""
ParamSpace definition for Microstructure family strategies.

This module defines the searchable parameter space for microstructure-based strategies
like SweepReversalStrategy.
"""

from __future__ import annotations

from finantradealgo.strategies.param_space import ParamSpace, ParamSpec


SWEEP_PARAM_SPACE: ParamSpace = {
    # RSI filter for sweep reversal
    "max_rsi_for_long": ParamSpec(
        name="max_rsi_for_long",
        type="float",
        low=45.0,
        high=65.0,
    ),
    # Minimum pullback ratio to qualify as sweep
    "min_pullback_ratio": ParamSpec(
        name="min_pullback_ratio",
        type="float",
        low=0.001,
        high=0.01,
    ),
    # FVG filter toggle
    "use_fvg_filter": ParamSpec(
        name="use_fvg_filter",
        type="bool",
    ),
}


__all__ = ["SWEEP_PARAM_SPACE"]

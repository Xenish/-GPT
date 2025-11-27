from __future__ import annotations

from finantradealgo.strategies.param_space import (
    ParamSpec,
    ParamSpace,
    validate_param_space,
)

RULE_PARAM_SPACE: ParamSpace = {
    "ms_trend_min": ParamSpec(
        name="ms_trend_min",
        type="float",
        low=-1.0,
        high=0.0,
    ),
    "ms_trend_max": ParamSpec(
        name="ms_trend_max",
        type="float",
        low=0.0,
        high=1.0,
    ),
    "tp_atr_mult": ParamSpec(
        name="tp_atr_mult",
        type="float",
        low=0.5,
        high=4.0,
    ),
    "sl_atr_mult": ParamSpec(
        name="sl_atr_mult",
        type="float",
        low=0.5,
        high=4.0,
    ),
    "use_ms_chop_filter": ParamSpec(
        name="use_ms_chop_filter",
        type="bool",
    ),
}

validate_param_space(RULE_PARAM_SPACE)

__all__ = ["RULE_PARAM_SPACE"]

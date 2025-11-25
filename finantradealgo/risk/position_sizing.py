from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSizingInput:
    equity: float
    price: float
    atr: Optional[float] = None
    capital_risk_pct_per_trade: float = 0.01
    max_notional_per_symbol: Optional[float] = None


def _apply_notional_limit(size: float, price: float, limit: Optional[float]) -> float:
    if limit is not None and limit > 0 and price > 0:
        size = min(size, limit / price)
    return max(size, 0.0)


def calc_size_fixed_risk_pct(inp: PositionSizingInput) -> float:
    if inp.equity <= 0 or inp.price <= 0:
        return 0.0

    risk_capital = inp.equity * max(inp.capital_risk_pct_per_trade, 0.0)
    if risk_capital <= 0:
        return 0.0

    stop_distance = inp.price * 0.01
    if stop_distance <= 0:
        return 0.0

    size = risk_capital / stop_distance
    return _apply_notional_limit(size, inp.price, inp.max_notional_per_symbol)


def calc_size_atr_stop(
    inp: PositionSizingInput,
    atr_mult_for_stop: float = 2.0,
) -> float:
    if inp.equity <= 0 or inp.price <= 0:
        return 0.0

    risk_capital = inp.equity * max(inp.capital_risk_pct_per_trade, 0.0)
    if risk_capital <= 0:
        return 0.0

    atr_value = inp.atr if inp.atr and inp.atr > 0 else None
    stop_distance = atr_mult_for_stop * atr_value if atr_value else inp.price * 0.01
    if stop_distance <= 0:
        return 0.0

    size = risk_capital / stop_distance
    return _apply_notional_limit(size, inp.price, inp.max_notional_per_symbol)

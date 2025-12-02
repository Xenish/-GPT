from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import math

from finantradealgo.execution import (
    ExecutionContext,
    ExecutionSimulationConfig,
    OrderSide,
    OrderType,
    SimulatedFill,
)

"""
Slippage simulator for paper/live execution modeling.

Assumptions:
- Uses a simplified mapping from order size vs. available volume and spread to
  estimate execution price.
- Models partial fills by capping filled quantity to a configurable multiple of
  visible volume; the remainder is left unfilled for the caller to handle.
- Aggregates slippage into a single fill (instead of multiple price levels) to
  keep math simple; slippage is recorded as execution_price - reference_price
  (positive is worse for buys, better for sells).
"""


@dataclass
class SlippageModelConfig:
    base_spread_bps: float = 1.0
    max_spread_bps: float = 20.0
    volume_impact_coef: float = 0.1
    volatility_impact_coef: float = 0.1
    partial_fill_threshold: float = 1.0
    allow_price_improvement: bool = False
    metadata: dict[str, Any] | None = None


class SlippageSimulator:
    def __init__(
        self,
        model_config: SlippageModelConfig | None = None,
    ) -> None:
        self.model_config = model_config or SlippageModelConfig()

    def simulate_fill(self, ctx: ExecutionContext, *, simulation_config: ExecutionSimulationConfig | None = None) -> Sequence[SimulatedFill]:
        """
        Generate one or more simulated fills for the provided execution context.

        - Reference price: best quote if available; otherwise mid +/- spread/2.
        - Spread: starts at base_spread_bps and widens for low liquidity / high vol.
        - Volume impact: scales with order size relative to available volume.
        - Partial fills: if enabled and available volume is limited, we cap the
          filled quantity; the remainder is left to the caller.
        """
        model = self.model_config
        sim_cfg = simulation_config or ExecutionSimulationConfig()

        reference_price = self._reference_price(ctx)
        if reference_price is None:
            return []

        spread_bps = self._compute_effective_spread_bps(ctx, model)

        available_vol = self._side_available_volume(ctx.available_volume, ctx.side)
        filled_qty = ctx.order_qty

        # Partial fill logic based on available volume.
        if sim_cfg.enable_partial_fills and available_vol is not None:
            max_fillable = available_vol * model.partial_fill_threshold
            filled_qty = min(ctx.order_qty, max_fillable)
            if filled_qty <= 0:
                return []

        volume_impact_bps = self._compute_volume_impact(
            filled_qty=filled_qty,
            available_vol=available_vol,
            coef=model.volume_impact_coef,
        )

        # Volatility widens slippage linearly relative to base.
        volatility_bps = 0.0
        if ctx.volatility is not None:
            volatility_bps = math.fabs(ctx.volatility) * 10_000 * model.volatility_impact_coef

        total_bps = spread_bps / 2 + volume_impact_bps + volatility_bps
        if not model.allow_price_improvement:
            total_bps = max(total_bps, spread_bps / 2)

        direction = 1.0 if ctx.side == OrderSide.BUY else -1.0
        execution_price = reference_price * (1 + direction * (total_bps / 10_000))

        # Respect limit prices: if limit would not execute, return no fills.
        if ctx.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and ctx.limit_price is not None:
            if ctx.side == OrderSide.BUY and execution_price > ctx.limit_price:
                return []
            if ctx.side == OrderSide.SELL and execution_price < ctx.limit_price:
                return []
            # Clamp execution to limit if price improvement is not allowed.
            if not model.allow_price_improvement:
                if ctx.side == OrderSide.BUY:
                    execution_price = min(execution_price, ctx.limit_price)
                else:
                    execution_price = max(execution_price, ctx.limit_price)

        slippage = execution_price - reference_price
        fill = SimulatedFill(
            price=execution_price,
            qty=filled_qty,
            timestamp=ctx.timestamp,
            liquidity_taken=self._compute_liquidity_taken(filled_qty, available_vol),
            slippage=slippage,
        )
        return [fill]

    def _reference_price(self, ctx: ExecutionContext) -> float | None:
        spread = ctx.spread
        if spread is None and ctx.best_bid is not None and ctx.best_ask is not None:
            spread = ctx.best_ask - ctx.best_bid

        if ctx.side == OrderSide.BUY:
            if ctx.order_type == OrderType.MARKET:
                if ctx.best_ask is not None:
                    return ctx.best_ask
            elif ctx.limit_price is not None:
                return ctx.limit_price
        elif ctx.side == OrderSide.SELL:
            if ctx.order_type == OrderType.MARKET:
                if ctx.best_bid is not None:
                    return ctx.best_bid
            elif ctx.limit_price is not None:
                return ctx.limit_price

        if ctx.mid_price is not None and spread is not None:
            half_spread = spread / 2
            if ctx.side == OrderSide.BUY:
                return ctx.mid_price + half_spread
            return ctx.mid_price - half_spread
        return ctx.mid_price

    def _compute_effective_spread_bps(self, ctx: ExecutionContext, model: SlippageModelConfig) -> float:
        spread_bps = model.base_spread_bps
        if ctx.liquidity_regime in {"low_liquidity", "high_volatility"}:
            spread_bps = min(model.max_spread_bps, spread_bps * 4)
        elif ctx.liquidity_regime == "normal":
            spread_bps = model.base_spread_bps

        if ctx.volatility is not None:
            spread_bps += math.fabs(ctx.volatility) * 10_000 * model.volatility_impact_coef
        return min(spread_bps, model.max_spread_bps)

    def _compute_volume_impact(self, *, filled_qty: float, available_vol: float | None, coef: float) -> float:
        if available_vol is None or available_vol <= 0 or filled_qty <= 0:
            return 0.0
        ratio = filled_qty / max(available_vol, 1e-9)
        return coef * ratio * 10_000  # convert to bps

    def _compute_liquidity_taken(self, filled_qty: float, available_vol: float | None) -> float | None:
        if available_vol is None or available_vol <= 0:
            return None
        return min(1.0, filled_qty / available_vol)

    def _side_available_volume(self, available_volume: dict[str, float] | None, side: OrderSide) -> float | None:
        if not available_volume:
            return None
        key = "ask" if side == OrderSide.BUY else "bid"
        return available_volume.get(key)


__all__ = [
    "SlippageModelConfig",
    "SlippageSimulator",
]

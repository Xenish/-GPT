from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Protocol

import time

from finantradealgo.execution.order_types import (
    OrderSpec,
    OrderSlice,
    AlgoOrderSpec,
    AlgoOrderKind,
)
from finantradealgo.execution.smart_routing import (
    SmartOrderRouter,
    SmartRoutePlan,
)


class ExecutionClient(Protocol):
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Place a single child order on a specific exchange.

        In a real implementation:
        - `symbol` is the exchange-specific symbol.
        - `side` is "buy"/"sell" or similar.
        - `order_type` is "market"/"limit"/etc.
        - This method would return an order ID or execution report.
        """
        ...


@dataclass
class AlgoExecutionContext:
    algo_spec: AlgoOrderSpec
    created_at: float = field(default_factory=lambda: time.time())
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAlgoExecutor:
    """
    Base class for algorithmic order executors (TWAP, VWAP, etc.).
    """

    def __init__(
        self,
        router: SmartOrderRouter,
    ) -> None:
        self.router = router

    def build_slices(self, ctx: AlgoExecutionContext) -> list[OrderSlice]:
        """
        Plan the list of OrderSlice objects to execute for this algo order.

        Subclasses override this.
        """
        raise NotImplementedError

    def execute_slices(
        self,
        ctx: AlgoExecutionContext,
        execution_client_resolver: Any,
    ) -> list[Any]:
        """
        Execute the planned slices by:
        - Using SmartOrderRouter to get routing plans per slice.
        - Resolving ExecutionClient per exchange via `execution_client_resolver`.
        - Submitting orders.

        `execution_client_resolver` is a callable or mapping that, given an
        exchange ID, returns an ExecutionClient.
        """
        results: list[Any] = []
        slices = self.build_slices(ctx)

        for sl in slices:
            # Plan route for this slice (usually 1 venue per slice)
            base_spec = OrderSpec(
                internal_symbol=sl.internal_symbol,
                side=sl.side,
                qty=sl.qty,
                order_type=(
                    # derive from algo_spec or default to MARKET
                    ctx.algo_spec.base.order_type
                ),
                limit_price=sl.limit_price or ctx.algo_spec.base.limit_price,
                time_in_force=sl.time_in_force,
            )
            route_plan: SmartRoutePlan = self.router.plan_route(base_spec)
            route_slices = self.router.to_slices(route_plan)

            for rs in route_slices:
                # Resolve execution client and exchange symbol.
                # For now we assume the resolver returns:
                #   (execution_client, exchange_symbol)
                exec_client, exchange_symbol = execution_client_resolver(rs)
                result = exec_client.place_order(
                    symbol=exchange_symbol,
                    side=rs.side.name.lower(),
                    qty=rs.qty,
                    order_type=ctx.algo_spec.base.order_type.name.lower(),
                    limit_price=rs.limit_price,
                )
                results.append(result)

        return results


@dataclass
class TWAPConfig:
    """
    Configuration for TWAP scheduling.

    - n_slices:
        Number of slices between start_ts and end_ts.
    - randomize:
        If True, introduce small random jitter in slice times.
    """

    n_slices: int = 10
    randomize: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TWAPExecutor(BaseAlgoExecutor):
    def __init__(
        self,
        router: SmartOrderRouter,
        config: TWAPConfig | None = None,
    ) -> None:
        super().__init__(router)
        self.config = config or TWAPConfig()

    def build_slices(self, ctx: AlgoExecutionContext) -> list[OrderSlice]:
        spec = ctx.algo_spec.base
        start_ts = ctx.algo_spec.start_ts or ctx.created_at
        end_ts = ctx.algo_spec.end_ts or (start_ts + 60.0)  # default 1 min
        n = self.config.n_slices

        if n <= 0:
            raise ValueError("TWAPConfig.n_slices must be > 0")

        dt = (end_ts - start_ts) / n
        qty_per_slice = spec.qty / n

        slices: list[OrderSlice] = []
        for i in range(n):
            planned_ts = start_ts + i * dt
            # Optional: jitter if randomize=True (TODO)
            sl = OrderSlice(
                internal_symbol=spec.internal_symbol,
                side=spec.side,
                qty=qty_per_slice,
                limit_price=spec.limit_price,
                planned_ts=planned_ts,
                time_in_force=spec.time_in_force,
                metadata={"algo": "twap"},
            )
            slices.append(sl)
        return slices


@dataclass
class VWAPConfig:
    """
    Basic VWAP execution config.

    - volume_profile:
        Optional intraday volume profile (list of weights summing to 1.0).
        If None, slices fallback to uniform, like TWAP.
    """

    volume_profile: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VWAPExecutor(BaseAlgoExecutor):
    def __init__(
        self,
        router: SmartOrderRouter,
        config: VWAPConfig | None = None,
    ) -> None:
        super().__init__(router)
        self.config = config or VWAPConfig()

    def build_slices(self, ctx: AlgoExecutionContext) -> list[OrderSlice]:
        spec = ctx.algo_spec.base
        start_ts = ctx.algo_spec.start_ts or ctx.created_at
        end_ts = ctx.algo_spec.end_ts or (start_ts + 60.0)
        n = ctx.algo_spec.target_slices or (
            len(self.config.volume_profile) if self.config.volume_profile else 10
        )

        if n <= 0:
            raise ValueError("VWAPExecutor: number of slices must be > 0")

        dt = (end_ts - start_ts) / n

        # Build weights
        if self.config.volume_profile and len(self.config.volume_profile) == n:
            weights = self.config.volume_profile
        elif self.config.volume_profile:
            # TODO: resample profile to n slices
            weights = [1.0 / n] * n
        else:
            weights = [1.0 / n] * n

        total_qty = spec.qty
        slices: list[OrderSlice] = []
        for i in range(n):
            planned_ts = start_ts + i * dt
            qty_i = total_qty * weights[i]
            sl = OrderSlice(
                internal_symbol=spec.internal_symbol,
                side=spec.side,
                qty=qty_i,
                limit_price=spec.limit_price,
                planned_ts=planned_ts,
                time_in_force=spec.time_in_force,
                metadata={"algo": "vwap"},
            )
            slices.append(sl)
        return slices


@dataclass
class ImplementationShortfallConfig:
    """
    Placeholder config for implementation shortfall execution.

    Future extensions:
    - benchmark_price (arrival price)
    - risk_aversion
    - participation limits
    """

    metadata: dict[str, Any] = field(default_factory=dict)


class ImplementationShortfallExecutor(BaseAlgoExecutor):
    def __init__(
        self,
        router: SmartOrderRouter,
        config: ImplementationShortfallConfig | None = None,
    ) -> None:
        super().__init__(router)
        self.config = config or ImplementationShortfallConfig()

    def build_slices(self, ctx: AlgoExecutionContext) -> list[OrderSlice]:
        """
        TODO: implement implementation shortfall scheduling:
        - Minimize slippage vs benchmark price under risk constraints.
        - For now, just fallback to a simple TWAP-like schedule.
        """
        # Simple fallback
        twap = TWAPExecutor(router=self.router)
        return twap.build_slices(ctx)


@dataclass
class AdaptiveExecutionConfig:
    """
    Placeholder config for adaptive execution.

    Future extensions:
    - dynamic participation rate
    - volatility/volume-driven adaptation
    - feedback from realized slippage
    """

    metadata: dict[str, Any] = field(default_factory=dict)


class AdaptiveExecutionExecutor(BaseAlgoExecutor):
    def __init__(
        self,
        router: SmartOrderRouter,
        config: AdaptiveExecutionConfig | None = None,
    ) -> None:
        super().__init__(router)
        self.config = config or AdaptiveExecutionConfig()

    def build_slices(self, ctx: AlgoExecutionContext) -> list[OrderSlice]:
        """
        TODO: implement adaptive scheduling based on real-time market conditions.

        For now, fallback to TWAP, but in the future this method should:
        - Observe market volatility/liquidity.
        - Speed up/slow down execution accordingly.
        """
        twap = TWAPExecutor(router=self.router)
        return twap.build_slices(ctx)

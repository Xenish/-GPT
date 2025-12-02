from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


class AttributionLevel(Enum):
    """
    Granularity of performance attribution.
    """

    TRADE = auto()
    DAY = auto()
    WEEK = auto()
    MONTH = auto()
    REGIME = auto()
    FACTOR = auto()
    STRATEGY_COMPONENT = auto()


@dataclass
class PerformanceAttributionConfig:
    """
    Configuration for performance attribution.

    - level:
        Primary aggregation level (trade, day, regime, etc.).
    - include_factors:
        Whether to join factor exposures (trend/vol/market-structure/time-of-day).
    - include_components:
        Whether to attempt signal/filter/sizing/risk decomposition.
    """

    level: AttributionLevel = AttributionLevel.TRADE
    include_factors: bool = True
    include_components: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentContribution:
    """
    Decomposition of PnL into strategy components.

    All values are expressed in PnL units (e.g. USD) or basis points,
    depending on how the caller constructs them.
    """

    signal: float | None = None  # raw signal quality
    filters: float | None = None  # filters improving/worsening trades
    sizing: float | None = None  # position sizing impact
    risk_management: float | None = None  # stop-loss, take-profit, caps
    slippage: float | None = None  # execution quality impact
    fees: float | None = None  # commissions + funding
    residual: float | None = None  # leftover after decomposition
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeAttributionRow:
    """
    Trade-level performance attribution.

    Assumes we have trade-level PnL and various tags (regime, factors, metadata).
    """

    trade_id: str
    internal_symbol: str
    strategy_id: str | None

    # PnL and basic stats
    gross_pnl: float
    net_pnl: float
    fees: float
    slippage: float | None
    return_pct: float | None  # pct on risk or equity
    holding_period_seconds: float | None

    # Regime / grouping fields
    regime: str | None = None
    entry_tag: str | None = None
    exit_tag: str | None = None

    # Factor exposures (to be filled from factor_analysis)
    trend_exposure: float | None = None
    volatility_exposure: float | None = None
    market_structure_exposure: float | None = None
    time_of_day_bucket: str | None = None

    # Component-level decomposition
    components: ComponentContribution = field(default_factory=ComponentContribution)

    # Raw metadata (execution venue, order ids, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionSummary:
    """
    Aggregated view of performance attribution.

    `group_key` is a dictionary of grouping dimensions:
    - e.g. {"regime": "trend", "time_of_day": "US_open"}
    """

    group_key: dict[str, Any]
    n_trades: int

    gross_pnl: float
    net_pnl: float
    avg_return_pct: float | None
    win_rate: float | None

    # Factor-level aggregates (optional)
    avg_trend_exposure: float | None = None
    avg_volatility_exposure: float | None = None
    avg_market_structure_exposure: float | None = None

    # Component-level aggregates (optional)
    component_totals: ComponentContribution = field(default_factory=ComponentContribution)

    metadata: dict[str, Any] = field(default_factory=dict)


def build_trade_attribution_row(trade: Mapping[str, Any]) -> TradeAttributionRow:
    """
    Construct a TradeAttributionRow from a trade dict/record.

    Expected keys (adapt as needed to your real trade schema):

    - id / trade_id
    - internal_symbol
    - strategy_id
    - gross_pnl
    - net_pnl
    - fees
    - slippage
    - return_pct
    - holding_period_seconds
    - regime
    - entry_tag
    - exit_tag

    This function does not handle factor exposures or component contributions;
    those can be joined later via separate modules (e.g. factor_analysis).
    """
    return TradeAttributionRow(
        trade_id=str(trade.get("id") or trade.get("trade_id")),
        internal_symbol=trade.get("internal_symbol") or trade.get("symbol"),
        strategy_id=trade.get("strategy_id"),
        gross_pnl=float(trade.get("gross_pnl", 0.0)),
        net_pnl=float(trade.get("net_pnl", trade.get("pnl", 0.0))),
        fees=float(trade.get("fees", 0.0)),
        slippage=float(trade.get("slippage", 0.0)),
        return_pct=trade.get("return_pct"),
        holding_period_seconds=trade.get("holding_period_seconds"),
        regime=trade.get("regime"),
        entry_tag=trade.get("entry_tag"),
        exit_tag=trade.get("exit_tag"),
        metadata=dict(trade),
    )


def build_trade_attribution_rows(trades: Iterable[Mapping[str, Any]]) -> list[TradeAttributionRow]:
    return [build_trade_attribution_row(t) for t in trades]


class PerformanceAttributionEngine:
    """
    High-level orchestrator for performance attribution.

    Responsibilities:
    - Build TradeAttributionRow objects from raw trades / backtest results.
    - Join factor exposures (via factor_analysis module).
    - Aggregate by chosen dimensions (regime, time-of-day, factor buckets, etc.).
    - Provide summaries usable for reporting and dashboards.
    """

    def __init__(
        self,
        config: PerformanceAttributionConfig | None = None,
    ) -> None:
        self.config = config or PerformanceAttributionConfig()

    def from_trades(
        self,
        trades: Iterable[Mapping[str, Any]],
    ) -> list[TradeAttributionRow]:
        """
        Build raw TradeAttributionRow objects from a list of trade dicts.

        This is the entry point when you already have trade history as dicts.
        Factor exposures and component contributions can be filled in later.
        """
        return build_trade_attribution_rows(trades)

    def from_backtest_result(self, result: Any) -> list[TradeAttributionRow]:
        """
        Extract trade-level attribution rows from a backtest result object.

        This method assumes that `result` exposes either:
        - result.trades: Sequence[dict[str, Any]]
        - or a method like result.to_trades_dataframe()

        For now we implement a conservative, duck-typed approach and document it.
        """
        trades: Iterable[Mapping[str, Any]]

        if hasattr(result, "trades"):
            trades = getattr(result, "trades")
        elif hasattr(result, "to_trades_dataframe"):
            df = result.to_trades_dataframe()
            trades = df.to_dict(orient="records")
        else:
            raise TypeError("Unsupported backtest result type for attribution")

        return self.from_trades(trades)

    def attach_factor_exposures(
        self,
        rows: list[TradeAttributionRow],
        factor_df: pd.DataFrame,
        *,
        key: str = "trade_id",
    ) -> list[TradeAttributionRow]:
        """
        Attach factor exposures to existing TradeAttributionRow list.

        factor_df:
            DataFrame with one row per trade, containing columns:
            - trade_id
            - trend_exposure
            - volatility_exposure
            - market_structure_exposure
            - time_of_day_bucket

        This method mutates the given rows in-place and also returns them.
        """
        factor_df = factor_df.set_index(key)
        for row in rows:
            if row.trade_id not in factor_df.index:
                continue
            f = factor_df.loc[row.trade_id]
            row.trend_exposure = (
                float(f.get("trend_exposure"))
                if "trend_exposure" in f
                else row.trend_exposure
            )
            row.volatility_exposure = (
                float(f.get("volatility_exposure"))
                if "volatility_exposure" in f
                else row.volatility_exposure
            )
            row.market_structure_exposure = (
                float(f.get("market_structure_exposure"))
                if "market_structure_exposure" in f
                else row.market_structure_exposure
            )
            row.time_of_day_bucket = (
                str(f.get("time_of_day_bucket"))
                if "time_of_day_bucket" in f
                else row.time_of_day_bucket
            )
        return rows

    def aggregate(
        self,
        rows: Sequence[TradeAttributionRow],
        group_by: Sequence[str],
    ) -> list[AttributionSummary]:
        """
        Aggregate trade-level attribution rows by given fields.

        group_by:
            Sequence of attribute names on TradeAttributionRow or keys inside
            row.metadata.

        Example:
            group_by=["regime", "time_of_day_bucket"]
        """
        if not rows:
            return []

        # Build a DataFrame to leverage pandas aggregation.
        records: list[dict[str, Any]] = []
        for r in rows:
            rec = {
                "trade_id": r.trade_id,
                "internal_symbol": r.internal_symbol,
                "strategy_id": r.strategy_id,
                "gross_pnl": r.gross_pnl,
                "net_pnl": r.net_pnl,
                "fees": r.fees,
                "return_pct": r.return_pct,
                "holding_period_seconds": r.holding_period_seconds,
                "regime": r.regime,
                "time_of_day_bucket": r.time_of_day_bucket,
                "trend_exposure": r.trend_exposure,
                "volatility_exposure": r.volatility_exposure,
                "market_structure_exposure": r.market_structure_exposure,
            }
            # You could also flatten components here if needed.
            records.append(rec)

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return []

        grouped = df.groupby(list(group_by), dropna=False)

        summaries: list[AttributionSummary] = []
        for group_values, g in grouped:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group_key = {name: value for name, value in zip(group_by, group_values)}

            n_trades = int(len(g))
            gross_pnl = float(g["gross_pnl"].sum())
            net_pnl = float(g["net_pnl"].sum())
            avg_return_pct = (
                float(g["return_pct"].mean())
                if "return_pct" in g and g["return_pct"].notna().any()
                else None
            )
            wins = (g["net_pnl"] > 0).sum()
            win_rate = float(wins / n_trades) if n_trades > 0 else None

            avg_trend = (
                float(g["trend_exposure"].mean())
                if "trend_exposure" in g and g["trend_exposure"].notna().any()
                else None
            )
            avg_vol = (
                float(g["volatility_exposure"].mean())
                if "volatility_exposure" in g and g["volatility_exposure"].notna().any()
                else None
            )
            avg_ms = (
                float(g["market_structure_exposure"].mean())
                if "market_structure_exposure" in g
                and g["market_structure_exposure"].notna().any()
                else None
            )

            summary = AttributionSummary(
                group_key=group_key,
                n_trades=n_trades,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                avg_return_pct=avg_return_pct,
                win_rate=win_rate,
                avg_trend_exposure=avg_trend,
                avg_volatility_exposure=avg_vol,
                avg_market_structure_exposure=avg_ms,
                # Component aggregation can be added later via separate helper.
            )
            summaries.append(summary)

        return summaries


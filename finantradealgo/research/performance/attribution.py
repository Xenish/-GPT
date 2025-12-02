"""
Performance Attribution Analysis.

Analyzes sources of PnL and identifies contributing factors to performance.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from finantradealgo.research.performance.models import PerformanceAttribution


class AttributionAnalyzer:
    """
    Analyze performance attribution.

    Breaks down PnL by various factors (regime, symbol, timeframe, etc.).
    """

    def __init__(self, strategy_id: str):
        """
        Initialize attribution analyzer.

        Args:
            strategy_id: Strategy identifier
        """
        self.strategy_id = strategy_id

    def analyze_trades(
        self,
        trades_df: pd.DataFrame,
        group_by: List[str] = None,
    ) -> PerformanceAttribution:
        """
        Analyze trade attribution.

        Args:
            trades_df: DataFrame with trades (must have 'pnl' column)
            group_by: Columns to group by (e.g., ['regime', 'symbol'])

        Returns:
            Performance attribution
        """
        if trades_df.empty:
            return PerformanceAttribution(
                strategy_id=self.strategy_id,
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow(),
            )

        # Determine period
        if "entry_time" in trades_df.columns:
            period_start = pd.to_datetime(trades_df["entry_time"].min())
            period_end = pd.to_datetime(trades_df["entry_time"].max())
        else:
            period_start = datetime.utcnow() - timedelta(days=30)
            period_end = datetime.utcnow()

        # Total PnL
        total_pnl = trades_df["pnl"].sum()

        # Initialize attribution
        attribution = PerformanceAttribution(
            strategy_id=self.strategy_id,
            period_start=period_start,
            period_end=period_end,
            total_pnl=total_pnl,
        )

        # Group by regime if available
        if "regime" in trades_df.columns:
            attribution.pnl_by_regime = self._group_pnl(trades_df, "regime")

        # Group by symbol if available
        if "symbol" in trades_df.columns:
            attribution.pnl_by_symbol = self._group_pnl(trades_df, "symbol")

        # Group by timeframe if available
        if "timeframe" in trades_df.columns:
            attribution.pnl_by_timeframe = self._group_pnl(trades_df, "timeframe")

        # Group by component if available (for ensembles)
        if "component" in trades_df.columns:
            attribution.pnl_by_component = self._group_pnl(trades_df, "component")

        # Find top and worst trades
        attribution.top_trades = self._get_top_trades(trades_df, n=10, best=True)
        attribution.worst_trades = self._get_top_trades(trades_df, n=10, best=False)

        return attribution

    def _group_pnl(self, trades_df: pd.DataFrame, column: str) -> Dict[str, float]:
        """
        Group PnL by column.

        Args:
            trades_df: Trades DataFrame
            column: Column to group by

        Returns:
            Dictionary of {group_value: total_pnl}
        """
        if column not in trades_df.columns:
            return {}

        grouped = trades_df.groupby(column)["pnl"].sum()

        return grouped.to_dict()

    def _get_top_trades(
        self,
        trades_df: pd.DataFrame,
        n: int = 10,
        best: bool = True,
    ) -> List[Dict]:
        """
        Get top N trades.

        Args:
            trades_df: Trades DataFrame
            n: Number of trades to return
            best: If True, get best trades; if False, get worst

        Returns:
            List of trade dictionaries
        """
        # Sort by PnL
        sorted_df = trades_df.sort_values("pnl", ascending=not best)

        # Get top N
        top_n = sorted_df.head(n)

        # Convert to list of dicts
        trades = []

        for _, row in top_n.iterrows():
            trade = {
                "pnl": float(row["pnl"]),
            }

            # Add optional fields if available
            for field in ["entry_time", "exit_time", "symbol", "regime", "component"]:
                if field in row:
                    value = row[field]
                    if pd.notna(value):
                        if isinstance(value, (pd.Timestamp, datetime)):
                            trade[field] = value.isoformat()
                        else:
                            trade[field] = str(value)

            trades.append(trade)

        return trades

    def analyze_by_period(
        self,
        trades_df: pd.DataFrame,
        period: str = "D",  # D, W, M
    ) -> pd.DataFrame:
        """
        Analyze PnL by time period.

        Args:
            trades_df: Trades DataFrame
            period: Pandas period code ('D', 'W', 'M')

        Returns:
            DataFrame with period-based analysis
        """
        if trades_df.empty or "entry_time" not in trades_df.columns:
            return pd.DataFrame()

        # Convert to datetime
        trades_df = trades_df.copy()
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])

        # Set as index for resampling
        trades_df = trades_df.set_index("entry_time")

        # Resample by period
        period_analysis = trades_df.resample(period).agg({
            "pnl": ["sum", "mean", "count"],
        })

        period_analysis.columns = ["total_pnl", "avg_pnl", "trade_count"]

        # Calculate cumulative PnL
        period_analysis["cumulative_pnl"] = period_analysis["total_pnl"].cumsum()

        # Calculate win rate
        wins_by_period = trades_df[trades_df["pnl"] > 0].resample(period).size()
        period_analysis["win_rate"] = wins_by_period / period_analysis["trade_count"]
        period_analysis["win_rate"] = period_analysis["win_rate"].fillna(0)

        return period_analysis.reset_index()

    def calculate_regime_contribution(
        self,
        trades_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate contribution of each regime to overall performance.

        Args:
            trades_df: Trades DataFrame with 'regime' column

        Returns:
            DataFrame with regime contributions
        """
        if "regime" not in trades_df.columns:
            return pd.DataFrame()

        # Group by regime
        regime_stats = trades_df.groupby("regime").agg({
            "pnl": ["sum", "mean", "count"],
        })

        regime_stats.columns = ["total_pnl", "avg_pnl", "trade_count"]

        # Calculate contribution percentage
        total_pnl = trades_df["pnl"].sum()
        if total_pnl != 0:
            regime_stats["contribution_pct"] = (regime_stats["total_pnl"] / total_pnl) * 100
        else:
            regime_stats["contribution_pct"] = 0

        # Calculate win rate per regime
        for regime in regime_stats.index:
            regime_trades = trades_df[trades_df["regime"] == regime]
            wins = (regime_trades["pnl"] > 0).sum()
            regime_stats.loc[regime, "win_rate"] = wins / len(regime_trades) if len(regime_trades) > 0 else 0

        # Sort by contribution
        regime_stats = regime_stats.sort_values("total_pnl", ascending=False)

        return regime_stats.reset_index()

    def calculate_component_contribution(
        self,
        trades_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate contribution of each component (for ensembles).

        Args:
            trades_df: Trades DataFrame with 'component' column

        Returns:
            DataFrame with component contributions
        """
        if "component" not in trades_df.columns:
            return pd.DataFrame()

        # Group by component
        component_stats = trades_df.groupby("component").agg({
            "pnl": ["sum", "mean", "count"],
        })

        component_stats.columns = ["total_pnl", "avg_pnl", "trade_count"]

        # Calculate contribution percentage
        total_pnl = trades_df["pnl"].sum()
        if total_pnl != 0:
            component_stats["contribution_pct"] = (component_stats["total_pnl"] / total_pnl) * 100
        else:
            component_stats["contribution_pct"] = 0

        # Calculate Sharpe per component
        for component in component_stats.index:
            component_trades = trades_df[trades_df["component"] == component]
            pnls = component_trades["pnl"]

            if len(pnls) > 1 and pnls.std() > 0:
                sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)
            else:
                sharpe = 0.0

            component_stats.loc[component, "sharpe"] = sharpe

        # Sort by contribution
        component_stats = component_stats.sort_values("total_pnl", ascending=False)

        return component_stats.reset_index()

    def identify_underperforming_segments(
        self,
        trades_df: pd.DataFrame,
        segment_column: str,
        threshold_sharpe: float = 0.0,
    ) -> List[str]:
        """
        Identify underperforming segments (e.g., regimes, symbols).

        Args:
            trades_df: Trades DataFrame
            segment_column: Column to segment by
            threshold_sharpe: Sharpe threshold for underperformance

        Returns:
            List of underperforming segment names
        """
        if segment_column not in trades_df.columns:
            return []

        underperforming = []

        for segment in trades_df[segment_column].unique():
            segment_trades = trades_df[trades_df[segment_column] == segment]
            pnls = segment_trades["pnl"]

            if len(pnls) > 1 and pnls.std() > 0:
                sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)

                if sharpe < threshold_sharpe:
                    underperforming.append(str(segment))

        return underperforming

    def generate_attribution_report(
        self,
        attribution: PerformanceAttribution,
    ) -> str:
        """
        Generate human-readable attribution report.

        Args:
            attribution: Performance attribution

        Returns:
            Formatted report string
        """
        lines = []

        lines.append("=" * 70)
        lines.append(f"PERFORMANCE ATTRIBUTION REPORT: {attribution.strategy_id}")
        lines.append("=" * 70)
        lines.append(f"Period: {attribution.period_start.strftime('%Y-%m-%d')} to {attribution.period_end.strftime('%Y-%m-%d')}")
        lines.append(f"Total PnL: {attribution.total_pnl:+.2f}")
        lines.append("")

        # Regime attribution
        if attribution.pnl_by_regime:
            lines.append("PnL BY REGIME:")
            lines.append("-" * 70)

            sorted_regimes = sorted(
                attribution.pnl_by_regime.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for regime, pnl in sorted_regimes:
                contribution_pct = (pnl / attribution.total_pnl * 100) if attribution.total_pnl != 0 else 0
                lines.append(f"  {regime:<20} {pnl:+10.2f}  ({contribution_pct:+.1f}%)")

            lines.append("")

        # Symbol attribution
        if attribution.pnl_by_symbol:
            lines.append("PnL BY SYMBOL:")
            lines.append("-" * 70)

            sorted_symbols = sorted(
                attribution.pnl_by_symbol.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for symbol, pnl in sorted_symbols[:10]:  # Top 10
                contribution_pct = (pnl / attribution.total_pnl * 100) if attribution.total_pnl != 0 else 0
                lines.append(f"  {symbol:<20} {pnl:+10.2f}  ({contribution_pct:+.1f}%)")

            lines.append("")

        # Component attribution
        if attribution.pnl_by_component:
            lines.append("PnL BY COMPONENT:")
            lines.append("-" * 70)

            sorted_components = sorted(
                attribution.pnl_by_component.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for component, pnl in sorted_components:
                contribution_pct = (pnl / attribution.total_pnl * 100) if attribution.total_pnl != 0 else 0
                lines.append(f"  {component:<20} {pnl:+10.2f}  ({contribution_pct:+.1f}%)")

            lines.append("")

        # Top trades
        if attribution.top_trades:
            lines.append("TOP 10 TRADES:")
            lines.append("-" * 70)

            for i, trade in enumerate(attribution.top_trades, 1):
                pnl = trade["pnl"]
                entry_time = trade.get("entry_time", "N/A")
                symbol = trade.get("symbol", "")
                regime = trade.get("regime", "")

                info = f"{symbol} {regime}".strip()
                lines.append(f"  {i:2d}. {pnl:+10.2f}  {entry_time[:10] if len(entry_time) > 10 else entry_time}  {info}")

            lines.append("")

        # Worst trades
        if attribution.worst_trades:
            lines.append("WORST 10 TRADES:")
            lines.append("-" * 70)

            for i, trade in enumerate(attribution.worst_trades, 1):
                pnl = trade["pnl"]
                entry_time = trade.get("entry_time", "N/A")
                symbol = trade.get("symbol", "")
                regime = trade.get("regime", "")

                info = f"{symbol} {regime}".strip()
                lines.append(f"  {i:2d}. {pnl:+10.2f}  {entry_time[:10] if len(entry_time) > 10 else entry_time}  {info}")

            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

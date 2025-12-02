"""
Portfolio Rebalancing Engine.

Handles periodic rebalancing of portfolio weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from finantradealgo.research.ensemble.portfolio import (
    Portfolio,
    PortfolioConfig,
    RebalanceFrequency,
)


@dataclass
class RebalanceEvent:
    """Record of a rebalancing event."""

    timestamp: datetime
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    drift: float  # Total weight drift that triggered rebalance
    turnover: float  # Sum of absolute weight changes / 2
    reason: str  # Why rebalance happened


class PortfolioRebalancer:
    """Manage portfolio rebalancing."""

    def __init__(self, portfolio: Portfolio):
        """
        Initialize rebalancer.

        Args:
            portfolio: Portfolio to rebalance
        """
        self.portfolio = portfolio
        self.config = portfolio.config
        self.rebalance_history: List[RebalanceEvent] = []

    def check_rebalance_needed(
        self,
        current_weights: Dict[str, float],
        current_time: datetime,
    ) -> tuple[bool, str]:
        """
        Check if rebalancing is needed.

        Args:
            current_weights: Current portfolio weights
            current_time: Current timestamp

        Returns:
            (needs_rebalance, reason)
        """
        # Check drift-based rebalancing
        if self._check_drift_threshold(current_weights):
            return True, "drift_threshold"

        # Check calendar-based rebalancing
        if self._check_calendar_rebalance(current_time):
            return True, "calendar"

        return False, ""

    def _check_drift_threshold(self, current_weights: Dict[str, float]) -> bool:
        """Check if weight drift exceeds threshold."""
        target_weights = self.portfolio.get_weights_dict()

        max_drift = 0.0
        for strategy_id, target_weight in target_weights.items():
            current_weight = current_weights.get(strategy_id, 0.0)
            drift = abs(current_weight - target_weight)
            max_drift = max(max_drift, drift)

        return max_drift > self.config.rebalance_threshold

    def _check_calendar_rebalance(self, current_time: datetime) -> bool:
        """Check if calendar-based rebalance is due."""
        if self.config.rebalance_frequency == RebalanceFrequency.NEVER:
            return False

        if self.portfolio.last_rebalance is None:
            return True

        time_since_rebalance = current_time - self.portfolio.last_rebalance

        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            return time_since_rebalance >= timedelta(days=1)
        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            return time_since_rebalance >= timedelta(days=7)
        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            return time_since_rebalance >= timedelta(days=30)
        elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            return time_since_rebalance >= timedelta(days=90)

        return False

    def rebalance(
        self,
        current_weights: Dict[str, float],
        current_time: datetime,
        reason: str,
    ) -> RebalanceEvent:
        """
        Execute rebalancing.

        Args:
            current_weights: Current portfolio weights
            current_time: Current timestamp
            reason: Reason for rebalancing

        Returns:
            Rebalance event record
        """
        target_weights = self.portfolio.get_weights_dict()

        # Calculate drift and turnover
        total_drift = 0.0
        turnover = 0.0

        for strategy_id in target_weights.keys():
            current = current_weights.get(strategy_id, 0.0)
            target = target_weights[strategy_id]
            drift = abs(current - target)
            total_drift += drift
            turnover += drift

        turnover = turnover / 2.0  # Divide by 2 since buys = sells

        # Create rebalance event
        event = RebalanceEvent(
            timestamp=current_time,
            old_weights=current_weights.copy(),
            new_weights=target_weights.copy(),
            drift=total_drift,
            turnover=turnover,
            reason=reason,
        )

        # Update portfolio
        self.portfolio.last_rebalance = current_time
        self.portfolio.rebalance_count += 1

        # Record event
        self.rebalance_history.append(event)

        return event

    def get_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance.

        Args:
            current_weights: Current portfolio weights
            portfolio_value: Total portfolio value

        Returns:
            Dictionary of strategy_id -> dollar amount to trade (positive = buy, negative = sell)
        """
        target_weights = self.portfolio.get_weights_dict()
        trades = {}

        for strategy_id, target_weight in target_weights.items():
            current_weight = current_weights.get(strategy_id, 0.0)
            weight_change = target_weight - current_weight
            dollar_change = weight_change * portfolio_value
            trades[strategy_id] = dollar_change

        return trades

    def calculate_rebalancing_cost(
        self,
        current_weights: Dict[str, float],
        portfolio_value: float,
        commission_rate: float = 0.001,
    ) -> float:
        """
        Estimate cost of rebalancing.

        Args:
            current_weights: Current portfolio weights
            portfolio_value: Total portfolio value
            commission_rate: Commission rate (default 0.1%)

        Returns:
            Estimated rebalancing cost in dollars
        """
        trades = self.get_rebalance_trades(current_weights, portfolio_value)

        # Calculate total traded value
        total_traded = sum(abs(amount) for amount in trades.values())

        # Cost is commission on total traded value
        cost = total_traded * commission_rate

        return cost

    def get_rebalance_summary(self) -> pd.DataFrame:
        """
        Get summary of rebalancing history.

        Returns:
            DataFrame with rebalancing events
        """
        if not self.rebalance_history:
            return pd.DataFrame()

        records = []
        for event in self.rebalance_history:
            records.append({
                "timestamp": event.timestamp,
                "reason": event.reason,
                "drift": round(event.drift, 4),
                "turnover": round(event.turnover, 4),
            })

        return pd.DataFrame(records)

    def simulate_rebalancing(
        self,
        strategy_returns: Dict[str, pd.Series],
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """
        Simulate rebalancing over historical returns.

        Args:
            strategy_returns: Dictionary of strategy returns
            initial_weights: Initial weights (uses portfolio weights if None)

        Returns:
            Dictionary with simulation results
        """
        # Align all returns series
        returns_df = pd.DataFrame(strategy_returns)
        dates = returns_df.index

        # Initialize weights
        if initial_weights is None:
            current_weights = self.portfolio.get_weights_dict()
        else:
            current_weights = initial_weights.copy()

        # Initialize values
        strategy_ids = list(returns_df.columns)
        values = {sid: 1.0 for sid in strategy_ids}
        total_value = sum(values.values())

        # Track portfolio over time
        portfolio_values = [total_value]
        weight_history = [current_weights.copy()]
        rebalance_dates = []

        # Simulate
        for i, date in enumerate(dates):
            # Update values based on returns
            for sid in strategy_ids:
                ret = returns_df.loc[date, sid]
                values[sid] *= (1 + ret)

            # Calculate new total value and weights
            total_value = sum(values.values())
            new_weights = {sid: values[sid] / total_value for sid in strategy_ids}

            # Check if rebalancing needed
            needs_rebalance, reason = self.check_rebalance_needed(new_weights, date)

            if needs_rebalance:
                # Execute rebalance
                target_weights = self.portfolio.get_weights_dict()

                # Update values to match target weights
                for sid in strategy_ids:
                    values[sid] = total_value * target_weights[sid]

                current_weights = target_weights.copy()
                rebalance_dates.append(date)

                # Record event
                self.rebalance(new_weights, date, reason)
            else:
                current_weights = new_weights

            # Record
            portfolio_values.append(total_value)
            weight_history.append(current_weights.copy())

        # Calculate metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        volatility = portfolio_returns.std() * np.sqrt(252) * 100

        # Max drawdown
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.cummax()
        drawdown = (portfolio_series - running_max) / running_max
        max_dd = drawdown.min() * 100

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "volatility": float(volatility),
            "max_drawdown": float(max_dd),
            "rebalance_count": len(rebalance_dates),
            "avg_turnover": float(np.mean([e.turnover for e in self.rebalance_history])) if self.rebalance_history else 0,
            "portfolio_values": portfolio_values,
            "weight_history": weight_history,
            "rebalance_dates": rebalance_dates,
        }


class AdaptiveRebalancer(PortfolioRebalancer):
    """Rebalancer that adapts thresholds based on market conditions."""

    def __init__(
        self,
        portfolio: Portfolio,
        volatility_window: int = 20,
        base_threshold: float = 0.05,
    ):
        """
        Initialize adaptive rebalancer.

        Args:
            portfolio: Portfolio to rebalance
            volatility_window: Window for volatility calculation
            base_threshold: Base drift threshold
        """
        super().__init__(portfolio)
        self.volatility_window = volatility_window
        self.base_threshold = base_threshold

    def calculate_adaptive_threshold(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> float:
        """
        Calculate adaptive threshold based on market volatility.

        Higher volatility -> higher threshold (less frequent rebalancing)
        Lower volatility -> lower threshold (more frequent rebalancing)

        Args:
            strategy_returns: Dictionary of strategy returns

        Returns:
            Adaptive drift threshold
        """
        returns_df = pd.DataFrame(strategy_returns)

        # Calculate portfolio volatility (recent)
        recent_returns = returns_df.tail(self.volatility_window)
        portfolio_returns = recent_returns.mean(axis=1)  # Equal weighted for simplicity
        current_vol = portfolio_returns.std()

        # Calculate long-term volatility
        long_term_vol = returns_df.mean(axis=1).std()

        if long_term_vol == 0:
            return self.base_threshold

        # Adjust threshold based on relative volatility
        vol_ratio = current_vol / long_term_vol

        # Higher volatility -> higher threshold (scale linearly)
        adaptive_threshold = self.base_threshold * vol_ratio

        # Clamp to reasonable range
        adaptive_threshold = np.clip(adaptive_threshold, 0.02, 0.15)

        return float(adaptive_threshold)

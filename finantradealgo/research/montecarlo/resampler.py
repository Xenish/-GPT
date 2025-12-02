"""
Bootstrap Resampling Engine.

Resampling methods for Monte Carlo simulation.
"""

from __future__ import annotations

from typing import List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from finantradealgo.research.montecarlo.models import (
    MonteCarloConfig,
    SimulationResult,
    MonteCarloResult,
    ResamplingMethod,
)


class BootstrapResampler:
    """
    Bootstrap resampling engine for Monte Carlo simulation.

    Generates resampled trade sequences using various methods.
    """

    def __init__(self, config: MonteCarloConfig):
        """
        Initialize resampler.

        Args:
            config: Monte Carlo configuration
        """
        self.config = config

        # Set random seed if specified
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

    def run_monte_carlo(
        self,
        strategy_id: str,
        trades_df: pd.DataFrame,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on trades.

        Args:
            strategy_id: Strategy identifier
            trades_df: DataFrame with trade results (must have 'pnl' column)

        Returns:
            MonteCarloResult with all simulations
        """
        start_time = datetime.now()

        # Validate input
        if 'pnl' not in trades_df.columns:
            raise ValueError("trades_df must have 'pnl' column")

        if len(trades_df) < self.config.min_trades_per_sim:
            raise ValueError(
                f"Need at least {self.config.min_trades_per_sim} trades, "
                f"got {len(trades_df)}"
            )

        # Run simulations
        simulations = []
        for i in range(self.config.n_simulations):
            sim_result = self._run_single_simulation(i, trades_df)
            simulations.append(sim_result)

        # Create result
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = MonteCarloResult(
            strategy_id=strategy_id,
            config=self.config,
            simulations=simulations,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=duration,
        )

        # Calculate statistics
        result.calculate_statistics()

        return result

    def _run_single_simulation(
        self,
        simulation_id: int,
        trades_df: pd.DataFrame,
    ) -> SimulationResult:
        """
        Run single Monte Carlo simulation.

        Args:
            simulation_id: Simulation identifier
            trades_df: Original trades

        Returns:
            SimulationResult
        """
        # Resample trades
        if self.config.resampling_method == ResamplingMethod.BOOTSTRAP:
            resampled_pnl = self._bootstrap_resample(trades_df['pnl'])
        elif self.config.resampling_method == ResamplingMethod.BLOCK_BOOTSTRAP:
            resampled_pnl = self._block_bootstrap_resample(trades_df['pnl'])
        elif self.config.resampling_method == ResamplingMethod.SHUFFLE:
            resampled_pnl = self._shuffle_resample(trades_df['pnl'])
        elif self.config.resampling_method == ResamplingMethod.PARAMETRIC:
            resampled_pnl = self._parametric_resample(trades_df['pnl'])
        else:
            raise ValueError(f"Unknown resampling method: {self.config.resampling_method}")

        # Calculate metrics
        metrics = self._calculate_metrics(resampled_pnl)

        # Create equity curve
        equity_curve = resampled_pnl.cumsum()

        return SimulationResult(
            simulation_id=simulation_id,
            trades_sampled=len(resampled_pnl),
            total_pnl=metrics['total_pnl'],
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            sortino_ratio=metrics['sortino_ratio'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            max_drawdown_duration=metrics['max_dd_duration'],
            equity_curve=equity_curve,
        )

    def _bootstrap_resample(self, pnl_series: pd.Series) -> pd.Series:
        """
        Standard bootstrap resampling with replacement.

        Args:
            pnl_series: Original PnL series

        Returns:
            Resampled PnL series
        """
        n = len(pnl_series)
        indices = np.random.choice(n, size=n, replace=True)
        resampled = pnl_series.iloc[indices].reset_index(drop=True)

        return resampled

    def _block_bootstrap_resample(self, pnl_series: pd.Series) -> pd.Series:
        """
        Block bootstrap for dependent/serially correlated data.

        Args:
            pnl_series: Original PnL series

        Returns:
            Resampled PnL series
        """
        n = len(pnl_series)
        block_size = self.config.block_size

        # Number of blocks needed
        n_blocks = int(np.ceil(n / block_size))

        # Sample blocks
        resampled_values = []
        for _ in range(n_blocks):
            # Random starting point
            start_idx = np.random.randint(0, max(1, n - block_size + 1))
            end_idx = min(start_idx + block_size, n)

            # Extract block
            block = pnl_series.iloc[start_idx:end_idx].values
            resampled_values.extend(block)

        # Trim to original length
        resampled_values = resampled_values[:n]

        return pd.Series(resampled_values)

    def _shuffle_resample(self, pnl_series: pd.Series) -> pd.Series:
        """
        Random shuffle (permutation) of original data.

        Args:
            pnl_series: Original PnL series

        Returns:
            Shuffled PnL series
        """
        resampled = pnl_series.copy()
        resampled = resampled.sample(frac=1).reset_index(drop=True)

        return resampled

    def _parametric_resample(self, pnl_series: pd.Series) -> pd.Series:
        """
        Parametric resampling using fitted normal distribution.

        Args:
            pnl_series: Original PnL series

        Returns:
            Resampled PnL series from fitted distribution
        """
        n = len(pnl_series)

        # Fit normal distribution
        mean = pnl_series.mean()
        std = pnl_series.std()

        # Generate from normal distribution
        resampled_values = np.random.normal(mean, std, size=n)

        return pd.Series(resampled_values)

    def _calculate_metrics(self, pnl_series: pd.Series) -> dict:
        """
        Calculate performance metrics from PnL series.

        Args:
            pnl_series: PnL series

        Returns:
            Dictionary with metrics
        """
        # Basic metrics
        total_pnl = pnl_series.sum()
        starting_capital = 10000.0  # Assume $10k starting capital
        total_return = (total_pnl / starting_capital) * 100

        # Win rate
        wins = (pnl_series > 0).sum()
        total_trades = len(pnl_series)
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Sharpe ratio
        if pnl_series.std() > 0:
            sharpe_ratio = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = pnl_series[pnl_series < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (pnl_series.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio

        # Max drawdown
        equity = starting_capital + pnl_series.cumsum()
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100

        max_drawdown = drawdown.min()

        # Max drawdown duration
        in_drawdown = drawdown < -0.01  # Consider >0.01% as drawdown
        dd_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0

        return {
            'total_pnl': total_pnl,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_dd_duration': dd_duration,
        }

    def generate_confidence_intervals(
        self,
        result: MonteCarloResult,
        metric: str = "total_return",
    ) -> tuple[float, float, float]:
        """
        Generate confidence interval for a metric.

        Args:
            result: Monte Carlo result
            metric: Metric name

        Returns:
            Tuple of (lower_bound, median, upper_bound)
        """
        values = np.array([getattr(sim, metric) for sim in result.simulations])

        alpha = 1 - result.config.confidence_level
        lower = np.percentile(values, alpha / 2 * 100)
        median = np.median(values)
        upper = np.percentile(values, (1 - alpha / 2) * 100)

        return float(lower), float(median), float(upper)

    def estimate_probability(
        self,
        result: MonteCarloResult,
        metric: str,
        threshold: float,
        above: bool = True,
    ) -> float:
        """
        Estimate probability of metric exceeding threshold.

        Args:
            result: Monte Carlo result
            metric: Metric name
            threshold: Threshold value
            above: If True, P(metric > threshold), else P(metric < threshold)

        Returns:
            Probability
        """
        values = np.array([getattr(sim, metric) for sim in result.simulations])

        if above:
            prob = np.sum(values > threshold) / len(values)
        else:
            prob = np.sum(values < threshold) / len(values)

        return float(prob)

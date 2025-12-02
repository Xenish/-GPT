"""
Walk-Forward Optimizer.

Handles rolling/anchored window optimization and validation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import json

import pandas as pd
import numpy as np

from finantradealgo.research.walkforward.models import (
    WalkForwardConfig,
    WalkForwardWindow,
    WalkForwardResult,
    WindowType,
    OptimizationMetric,
)

# Optional imports - these may not be implemented yet
try:
    from finantradealgo.research.strategy_search.param_search import ParameterSearchEngine
    from finantradealgo.research.strategy_search.models import SearchJob, SearchConfig
except ImportError:
    ParameterSearchEngine = None  # type: ignore
    SearchJob = None  # type: ignore
    SearchConfig = None  # type: ignore


class WalkForwardOptimizer:
    """
    Walk-forward optimization engine.

    Performs rolling or anchored window optimization with out-of-sample testing.
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        param_search_engine: Optional[ParameterSearchEngine] = None,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            config: Walk-forward configuration
            param_search_engine: Parameter search engine for optimization
        """
        self.config = config
        if param_search_engine is None and ParameterSearchEngine is not None:
            self.param_search_engine = ParameterSearchEngine()
        else:
            self.param_search_engine = param_search_engine

    def generate_windows(
        self,
        data_start: datetime,
        data_end: datetime,
    ) -> List[tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows.

        Args:
            data_start: Start of available data
            data_end: End of available data

        Returns:
            List of (is_start, is_end, oos_start, oos_end) tuples
        """
        windows = []

        # Calculate period duration based on unit
        if self.config.period_unit == "D":
            is_delta = timedelta(days=self.config.in_sample_periods)
            oos_delta = timedelta(days=self.config.out_sample_periods)
        elif self.config.period_unit == "W":
            is_delta = timedelta(weeks=self.config.in_sample_periods)
            oos_delta = timedelta(weeks=self.config.out_sample_periods)
        elif self.config.period_unit == "M":
            is_delta = timedelta(days=30 * self.config.in_sample_periods)
            oos_delta = timedelta(days=30 * self.config.out_sample_periods)
        elif self.config.period_unit == "Q":
            is_delta = timedelta(days=90 * self.config.in_sample_periods)
            oos_delta = timedelta(days=90 * self.config.out_sample_periods)
        elif self.config.period_unit == "Y":
            is_delta = timedelta(days=365 * self.config.in_sample_periods)
            oos_delta = timedelta(days=365 * self.config.out_sample_periods)
        else:
            raise ValueError(f"Invalid period_unit: {self.config.period_unit}")

        current_start = data_start

        while True:
            # In-sample period
            is_start = current_start
            is_end = is_start + is_delta

            # Out-of-sample period
            oos_start = is_end
            oos_end = oos_start + oos_delta

            # Check if we have enough data
            if oos_end > data_end:
                break

            windows.append((is_start, is_end, oos_start, oos_end))

            # Move to next window
            if self.config.window_type == WindowType.ROLLING:
                # Roll forward by OOS period
                current_start = oos_start
            elif self.config.window_type in [WindowType.ANCHORED, WindowType.EXPANDING]:
                # Keep start fixed, expand window
                current_start = data_start
                # For next iteration, IS period extends to end of current OOS
                # This is handled by updating is_delta
                is_delta = oos_end - data_start
            else:
                raise ValueError(f"Unknown window type: {self.config.window_type}")

        return windows

    def optimize_window(
        self,
        strategy_id: str,
        param_grid: Dict[str, List[Any]],
        data_df: pd.DataFrame,
        is_start: datetime,
        is_end: datetime,
        backtest_function: Callable,
    ) -> tuple[Dict[str, Any], float, Dict[str, float]]:
        """
        Optimize parameters on in-sample window.

        Args:
            strategy_id: Strategy identifier
            param_grid: Parameter grid to search
            data_df: Price data
            is_start: In-sample start
            is_end: In-sample end
            backtest_function: Function to run backtest with parameters

        Returns:
            Tuple of (best_params, best_score, best_metrics)
        """
        # Filter data to IS window
        is_data = data_df[(data_df.index >= is_start) & (data_df.index < is_end)]

        # Run parameter search
        results = []
        for params in self._generate_param_combinations(param_grid):
            # Run backtest with these parameters
            trades_df, metrics = backtest_function(is_data, params)

            # Extract optimization metric
            score = self._extract_metric(metrics, self.config.optimization_metric)

            results.append({
                "params": params,
                "score": score,
                "metrics": metrics,
                "trades": len(trades_df) if trades_df is not None else 0,
            })

        # Filter by minimum trades
        valid_results = [
            r for r in results
            if r["trades"] >= self.config.min_trades_per_period
        ]

        if not valid_results:
            # No valid results, return empty
            return {}, 0.0, {}

        # Filter by profitability if required
        if self.config.require_profitable_is:
            valid_results = [
                r for r in valid_results
                if r["metrics"].get("total_return", 0) > 0
            ]

        if not valid_results:
            return {}, 0.0, {}

        # Find best result
        best_result = max(valid_results, key=lambda x: x["score"])

        return best_result["params"], best_result["score"], best_result["metrics"]

    def test_window(
        self,
        data_df: pd.DataFrame,
        oos_start: datetime,
        oos_end: datetime,
        params: Dict[str, Any],
        backtest_function: Callable,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        """
        Test parameters on out-of-sample window.

        Args:
            data_df: Price data
            oos_start: Out-of-sample start
            oos_end: Out-of-sample end
            params: Parameters to test
            backtest_function: Function to run backtest

        Returns:
            Tuple of (trades_df, metrics)
        """
        # Filter data to OOS window
        oos_data = data_df[(data_df.index >= oos_start) & (data_df.index < oos_end)]

        # Run backtest with parameters
        trades_df, metrics = backtest_function(oos_data, params)

        return trades_df, metrics

    def run_walk_forward(
        self,
        strategy_id: str,
        param_grid: Dict[str, List[Any]],
        data_df: pd.DataFrame,
        backtest_function: Callable,
    ) -> WalkForwardResult:
        """
        Run complete walk-forward optimization.

        Args:
            strategy_id: Strategy identifier
            param_grid: Parameter grid to search
            data_df: Price data with datetime index
            backtest_function: Function(data_df, params) -> (trades_df, metrics)

        Returns:
            WalkForwardResult with all windows and aggregate metrics
        """
        start_time = datetime.now()

        # Generate windows
        data_start = data_df.index.min()
        data_end = data_df.index.max()
        window_specs = self.generate_windows(data_start, data_end)

        # Process each window
        windows = []
        all_oos_trades = []

        for window_id, (is_start, is_end, oos_start, oos_end) in enumerate(window_specs, 1):
            print(f"Processing window {window_id}/{len(window_specs)}...")

            # Optimize on in-sample
            best_params, best_score, is_metrics = self.optimize_window(
                strategy_id=strategy_id,
                param_grid=param_grid,
                data_df=data_df,
                is_start=is_start,
                is_end=is_end,
                backtest_function=backtest_function,
            )

            # Test on out-of-sample
            oos_trades, oos_metrics = self.test_window(
                data_df=data_df,
                oos_start=oos_start,
                oos_end=oos_end,
                params=best_params,
                backtest_function=backtest_function,
            )

            # Create window result
            window = WalkForwardWindow(
                window_id=window_id,
                in_sample_start=is_start,
                in_sample_end=is_end,
                out_sample_start=oos_start,
                out_sample_end=oos_end,
                best_params=best_params,
                best_is_score=best_score,
                # In-sample metrics
                is_total_trades=is_metrics.get("total_trades", 0),
                is_sharpe=is_metrics.get("sharpe_ratio", 0.0),
                is_total_return=is_metrics.get("total_return", 0.0),
                is_max_drawdown=is_metrics.get("max_drawdown", 0.0),
                is_win_rate=is_metrics.get("win_rate", 0.0),
                # Out-of-sample metrics
                oos_total_trades=oos_metrics.get("total_trades", 0),
                oos_sharpe=oos_metrics.get("sharpe_ratio", 0.0),
                oos_total_return=oos_metrics.get("total_return", 0.0),
                oos_max_drawdown=oos_metrics.get("max_drawdown", 0.0),
                oos_win_rate=oos_metrics.get("win_rate", 0.0),
            )

            # Calculate degradation
            window.calculate_degradation()

            windows.append(window)

            # Collect OOS trades for combined equity curve
            if oos_trades is not None and not oos_trades.empty:
                all_oos_trades.append(oos_trades)

        # Combine OOS equity curve
        combined_equity = None
        if all_oos_trades:
            combined_trades = pd.concat(all_oos_trades, ignore_index=True)
            if "pnl" in combined_trades.columns:
                combined_equity = combined_trades["pnl"].cumsum()

        # Create result
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = WalkForwardResult(
            strategy_id=strategy_id,
            config=self.config,
            windows=windows,
            combined_equity_curve=combined_equity,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=duration,
        )

        # Calculate aggregate metrics
        result.calculate_aggregate_metrics()

        return result

    def _generate_param_combinations(
        self, param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        import itertools

        keys = param_grid.keys()
        values = param_grid.values()

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _extract_metric(
        self, metrics: Dict[str, float], metric_type: OptimizationMetric
    ) -> float:
        """Extract specific metric from metrics dict."""
        if metric_type == OptimizationMetric.SHARPE_RATIO:
            return metrics.get("sharpe_ratio", 0.0)
        elif metric_type == OptimizationMetric.TOTAL_RETURN:
            return metrics.get("total_return", 0.0)
        elif metric_type == OptimizationMetric.PROFIT_FACTOR:
            return metrics.get("profit_factor", 0.0)
        elif metric_type == OptimizationMetric.WIN_RATE:
            return metrics.get("win_rate", 0.0)
        elif metric_type == OptimizationMetric.CALMAR_RATIO:
            return metrics.get("calmar_ratio", 0.0)
        elif metric_type == OptimizationMetric.SORTINO_RATIO:
            return metrics.get("sortino_ratio", 0.0)
        else:
            return 0.0

    def save_result(self, result: WalkForwardResult, output_dir: Path):
        """
        Save walk-forward result to disk.

        Args:
            result: Walk-forward result
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_summary_dict(), f, indent=2)

        # Save windows
        windows_data = []
        for window in result.windows:
            windows_data.append({
                "window_id": window.window_id,
                "in_sample_start": window.in_sample_start.isoformat(),
                "in_sample_end": window.in_sample_end.isoformat(),
                "out_sample_start": window.out_sample_start.isoformat(),
                "out_sample_end": window.out_sample_end.isoformat(),
                "best_params": window.best_params,
                "best_is_score": window.best_is_score,
                "is_sharpe": window.is_sharpe,
                "is_return": window.is_total_return,
                "oos_sharpe": window.oos_sharpe,
                "oos_return": window.oos_total_return,
                "sharpe_degradation": window.sharpe_degradation,
            })

        windows_path = output_dir / "windows.json"
        with open(windows_path, "w") as f:
            json.dump(windows_data, f, indent=2)

        # Save combined equity curve if available
        if result.combined_equity_curve is not None:
            equity_path = output_dir / "equity_curve.csv"
            result.combined_equity_curve.to_csv(equity_path)

        print(f"Walk-forward result saved to {output_dir}")

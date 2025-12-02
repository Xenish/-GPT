"""
Portfolio Backtesting Utilities.

Tools for backtesting portfolio construction strategies.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from finantradealgo.research.ensemble.portfolio import (
    PortfolioConfig,
    PortfolioBacktestResult,
    PortfolioComparisonResult,
    PortfolioWeightingMethod,
)
from finantradealgo.research.ensemble.optimizer import PortfolioOptimizer
from finantradealgo.research.ensemble.rebalancer import PortfolioRebalancer
from finantradealgo.research.ensemble.correlation import CorrelationAnalyzer


class PortfolioBacktester:
    """Backtest portfolio construction strategies."""

    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio backtester.

        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.optimizer = PortfolioOptimizer(config)
        self.corr_analyzer = CorrelationAnalyzer()

    def backtest_portfolio(
        self,
        strategy_returns: Dict[str, pd.Series],
        method: Optional[PortfolioWeightingMethod] = None,
    ) -> PortfolioBacktestResult:
        """
        Backtest a portfolio with given weighting method.

        Args:
            strategy_returns: Dictionary of strategy_id -> returns series
            method: Weighting method (uses config if None)

        Returns:
            Portfolio backtest result
        """
        # Optimize weights
        portfolio = self.optimizer.optimize_weights(strategy_returns, method)

        # Setup rebalancer
        rebalancer = PortfolioRebalancer(portfolio)

        # Run simulation with rebalancing
        sim_results = rebalancer.simulate_rebalancing(strategy_returns)

        # Calculate correlation analysis
        corr_matrix = self.corr_analyzer.calculate_correlation_matrix(strategy_returns)

        # Build result
        portfolio_values = pd.Series(sim_results["portfolio_values"])
        portfolio_returns = portfolio_values.pct_change().dropna()

        # Performance metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        annual_return = (portfolio_returns.mean() * 252) * 100
        volatility = portfolio_returns.std() * np.sqrt(252) * 100

        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std()
        sortino = portfolio_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Calmar ratio
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values - running_max) / running_max
        max_dd = abs(drawdown.min()) * 100

        calmar = abs(annual_return / max_dd) if max_dd > 0 else 0

        # VaR and CVaR at 95%
        var_95 = np.percentile(portfolio_returns, 5) * 100
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100

        # Trade count (number of rebalances)
        total_trades = sim_results["rebalance_count"]

        # Turnover
        avg_turnover = sim_results["avg_turnover"] * 100

        # Best individual strategy Sharpe
        best_individual_sharpe = 0.0
        for sid, returns in strategy_returns.items():
            strat_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            best_individual_sharpe = max(best_individual_sharpe, strat_sharpe)

        improvement = ((sharpe - best_individual_sharpe) / abs(best_individual_sharpe) * 100) if best_individual_sharpe != 0 else 0

        return PortfolioBacktestResult(
            portfolio_id=self.config.portfolio_id,
            config=self.config,
            total_return=float(total_return),
            annual_return=float(annual_return),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            volatility=float(volatility),
            max_drawdown=float(max_dd),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            diversification_ratio=corr_matrix.diversification_ratio,
            effective_n=corr_matrix.effective_n,
            avg_correlation=corr_matrix.avg_correlation,
            total_trades=total_trades,
            rebalance_count=sim_results["rebalance_count"],
            turnover=float(avg_turnover),
            equity_curve=portfolio_values,
            weights_over_time=pd.DataFrame(sim_results["weight_history"]),
            returns=portfolio_returns,
            drawdown_series=drawdown,
            component_weights=portfolio.weights,
            best_individual_sharpe=float(best_individual_sharpe),
            portfolio_vs_best_improvement=float(improvement),
            backtest_start=portfolio_returns.index[0] if len(portfolio_returns) > 0 else None,
            backtest_end=portfolio_returns.index[-1] if len(portfolio_returns) > 0 else None,
        )

    def compare_weighting_methods(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> PortfolioComparisonResult:
        """
        Compare all weighting methods.

        Args:
            strategy_returns: Dictionary of strategy returns

        Returns:
            Comparison result
        """
        methods = [
            PortfolioWeightingMethod.EQUAL,
            PortfolioWeightingMethod.PERFORMANCE,
            PortfolioWeightingMethod.SHARPE,
            PortfolioWeightingMethod.RISK_PARITY,
            PortfolioWeightingMethod.MINIMUM_VARIANCE,
            PortfolioWeightingMethod.MAXIMUM_SHARPE,
            PortfolioWeightingMethod.HIERARCHICAL_RISK_PARITY,
        ]

        results = []
        for method in methods:
            try:
                # Update config for this method
                self.config.weighting_method = method

                # Backtest
                result = self.backtest_portfolio(strategy_returns, method)
                results.append(result)
            except Exception as e:
                print(f"[WARN] {method.value} backtest failed: {e}")
                continue

        # Find best performers
        best_sharpe_method = ""
        best_sharpe = -float('inf')

        best_return_method = ""
        best_return = -float('inf')

        best_risk_adjusted_method = ""
        best_calmar = -float('inf')

        for result in results:
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_sharpe_method = result.config.weighting_method.value

            if result.total_return > best_return:
                best_return = result.total_return
                best_return_method = result.config.weighting_method.value

            if result.calmar_ratio > best_calmar:
                best_calmar = result.calmar_ratio
                best_risk_adjusted_method = result.config.weighting_method.value

        # Create performance table
        records = []
        for result in results:
            records.append({
                "method": result.config.weighting_method.value,
                "total_return": result.total_return,
                "annual_return": result.annual_return,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "calmar_ratio": result.calmar_ratio,
                "volatility": result.volatility,
                "max_drawdown": result.max_drawdown,
                "diversification_ratio": result.diversification_ratio,
                "effective_n": result.effective_n,
                "rebalance_count": result.rebalance_count,
                "turnover": result.turnover,
            })

        performance_table = pd.DataFrame(records)

        return PortfolioComparisonResult(
            portfolio_results=results,
            best_sharpe_method=best_sharpe_method,
            best_return_method=best_return_method,
            best_risk_adjusted_method=best_risk_adjusted_method,
            performance_table=performance_table,
        )

    def walk_forward_backtest(
        self,
        strategy_returns: Dict[str, pd.Series],
        train_periods: int = 252,
        test_periods: int = 63,
        method: Optional[PortfolioWeightingMethod] = None,
    ) -> List[PortfolioBacktestResult]:
        """
        Walk-forward backtest with rolling optimization.

        Args:
            strategy_returns: Dictionary of strategy returns
            train_periods: Training period length
            test_periods: Test period length
            method: Weighting method

        Returns:
            List of backtest results for each window
        """
        returns_df = pd.DataFrame(strategy_returns)
        n_periods = len(returns_df)

        results = []
        current_pos = 0

        while current_pos + train_periods + test_periods <= n_periods:
            # Split data
            train_end = current_pos + train_periods
            test_end = train_end + test_periods

            train_data = {
                sid: returns_df[sid].iloc[current_pos:train_end]
                for sid in returns_df.columns
            }
            test_data = {
                sid: returns_df[sid].iloc[train_end:test_end]
                for sid in returns_df.columns
            }

            # Optimize on training data
            portfolio = self.optimizer.optimize_weights(train_data, method)

            # Test on out-of-sample data
            # Create temporary config for testing
            test_config = PortfolioConfig(
                portfolio_id=f"{self.config.portfolio_id}_wf_{current_pos}",
                strategy_ids=self.config.strategy_ids,
                weighting_method=portfolio.config.weighting_method,
                rebalance_frequency=self.config.rebalance_frequency,
            )

            test_backtester = PortfolioBacktester(test_config)

            # Backtest on test period
            result = test_backtester.backtest_portfolio(test_data, method)
            results.append(result)

            # Move window
            current_pos += test_periods

        return results


def load_strategy_returns_from_files(
    strategy_files: Dict[str, str],
    return_column: str = "returns",
) -> Dict[str, pd.Series]:
    """
    Load strategy returns from CSV files.

    Args:
        strategy_files: Dictionary of strategy_id -> file path
        return_column: Name of returns column

    Returns:
        Dictionary of strategy_id -> returns series
    """
    strategy_returns = {}

    for strategy_id, file_path in strategy_files.items():
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            if return_column in df.columns:
                strategy_returns[strategy_id] = df[return_column]
            else:
                # Try to calculate returns from equity
                if "equity" in df.columns:
                    strategy_returns[strategy_id] = df["equity"].pct_change().dropna()
                else:
                    print(f"[WARN] Could not find returns for {strategy_id}")

        except Exception as e:
            print(f"[ERROR] Failed to load {strategy_id}: {e}")

    return strategy_returns


def calculate_contribution_attribution(
    portfolio_result: PortfolioBacktestResult,
    strategy_returns: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Calculate contribution attribution for portfolio components.

    Args:
        portfolio_result: Portfolio backtest result
        strategy_returns: Dictionary of strategy returns

    Returns:
        DataFrame with contribution metrics
    """
    records = []

    for component in portfolio_result.component_weights:
        strategy_id = component.strategy_id
        weight = component.weight

        if strategy_id in strategy_returns:
            returns = strategy_returns[strategy_id]

            # Contribution to return
            contribution_return = returns.mean() * weight * 252 * 100

            # Contribution to risk
            contribution_risk = returns.std() * weight * np.sqrt(252) * 100

            records.append({
                "strategy_id": strategy_id,
                "weight": round(weight, 4),
                "contribution_return": round(contribution_return, 2),
                "contribution_risk": round(contribution_risk, 2),
                "sharpe_ratio": round(component.sharpe_ratio, 3),
                "annual_return": round(component.annual_return, 2),
                "volatility": round(component.volatility, 2),
                "max_drawdown": round(component.max_drawdown, 2),
            })

    return pd.DataFrame(records)

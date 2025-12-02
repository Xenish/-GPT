"""
Portfolio Optimization Engine.

Implements various portfolio optimization methods for strategy weighting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from finantradealgo.research.ensemble.portfolio import (
    PortfolioConfig,
    Portfolio,
    PortfolioWeight,
    PortfolioWeightingMethod,
)


class PortfolioOptimizer:
    """Optimize portfolio weights using various methods."""

    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio optimizer.

        Args:
            config: Portfolio configuration
        """
        self.config = config

    def optimize_weights(
        self,
        strategy_returns: Dict[str, pd.Series],
        method: Optional[PortfolioWeightingMethod] = None,
    ) -> Portfolio:
        """
        Optimize portfolio weights.

        Args:
            strategy_returns: Dictionary of strategy_id -> returns series
            method: Weighting method (uses config.weighting_method if None)

        Returns:
            Optimized portfolio
        """
        if method is None:
            method = self.config.weighting_method

        # Calculate weights based on method
        if method == PortfolioWeightingMethod.EQUAL:
            weights_dict = self._equal_weights(strategy_returns)
        elif method == PortfolioWeightingMethod.PERFORMANCE:
            weights_dict = self._performance_weights(strategy_returns)
        elif method == PortfolioWeightingMethod.SHARPE:
            weights_dict = self._sharpe_weights(strategy_returns)
        elif method == PortfolioWeightingMethod.RISK_PARITY:
            weights_dict = self._risk_parity_weights(strategy_returns)
        elif method == PortfolioWeightingMethod.MINIMUM_VARIANCE:
            weights_dict = self._minimum_variance_weights(strategy_returns)
        elif method == PortfolioWeightingMethod.MAXIMUM_SHARPE:
            weights_dict = self._maximum_sharpe_weights(strategy_returns)
        elif method == PortfolioWeightingMethod.HIERARCHICAL_RISK_PARITY:
            weights_dict = self._hrp_weights(strategy_returns)
        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Create PortfolioWeight objects
        weights = []
        for strategy_id, weight in weights_dict.items():
            returns = strategy_returns[strategy_id]
            sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            annual_return = returns.mean() * 252  # Assuming daily returns
            volatility = returns.std() * np.sqrt(252)

            # Calculate max drawdown
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()

            weights.append(
                PortfolioWeight(
                    strategy_id=strategy_id,
                    weight=weight,
                    sharpe_ratio=float(sharpe),
                    annual_return=float(annual_return * 100),
                    volatility=float(volatility * 100),
                    max_drawdown=float(max_dd * 100),
                )
            )

        # Calculate portfolio metrics
        portfolio_returns = self._calculate_portfolio_returns(strategy_returns, weights_dict)
        portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
        portfolio_annual_return = portfolio_returns.mean() * 252
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        portfolio_max_dd = drawdown.min()

        return Portfolio(
            portfolio_id=self.config.portfolio_id,
            config=self.config,
            weights=weights,
            total_return=float((cum_returns.iloc[-1] - 1) * 100),
            sharpe_ratio=float(portfolio_sharpe),
            volatility=float(portfolio_volatility * 100),
            max_drawdown=float(portfolio_max_dd * 100),
        )

    def _equal_weights(self, strategy_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """Equal weight allocation."""
        n = len(strategy_returns)
        return {sid: 1.0 / n for sid in strategy_returns.keys()}

    def _performance_weights(self, strategy_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """Weight by historical returns."""
        returns = {sid: rets.mean() for sid, rets in strategy_returns.items()}

        # Only use positive returns
        positive_returns = {sid: max(ret, 0) for sid, ret in returns.items()}
        total = sum(positive_returns.values())

        if total == 0:
            # Fallback to equal weights
            return self._equal_weights(strategy_returns)

        return {sid: ret / total for sid, ret in positive_returns.items()}

    def _sharpe_weights(self, strategy_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """Weight by Sharpe ratio."""
        sharpe_ratios = {}
        for sid, rets in strategy_returns.items():
            sharpe = rets.mean() / rets.std() if rets.std() > 0 else 0
            sharpe_ratios[sid] = max(sharpe, 0)  # Only positive Sharpe

        total = sum(sharpe_ratios.values())

        if total == 0:
            # Fallback to equal weights
            return self._equal_weights(strategy_returns)

        return {sid: sharpe / total for sid, sharpe in sharpe_ratios.items()}

    def _risk_parity_weights(self, strategy_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """Risk parity (inverse volatility) weights."""
        volatilities = {sid: rets.std() for sid, rets in strategy_returns.items()}

        # Inverse volatility
        inv_vols = {sid: 1.0 / vol if vol > 0 else 0 for sid, vol in volatilities.items()}
        total = sum(inv_vols.values())

        if total == 0:
            # Fallback to equal weights
            return self._equal_weights(strategy_returns)

        return {sid: inv_vol / total for sid, inv_vol in inv_vols.items()}

    def _minimum_variance_weights(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """
        Minimum variance portfolio optimization.

        Minimizes portfolio variance subject to:
        - Weights sum to 1
        - Weights >= min_weight
        - Weights <= max_weight
        """
        returns_df = pd.DataFrame(strategy_returns)
        strategy_ids = list(returns_df.columns)
        n = len(strategy_ids)

        # Covariance matrix
        cov_matrix = returns_df.cov().values

        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights.T

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]

        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]

        # Initial guess (equal weights)
        w0 = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000},
        )

        if not result.success:
            # Fallback to equal weights
            return self._equal_weights(strategy_returns)

        return {sid: float(w) for sid, w in zip(strategy_ids, result.x)}

    def _maximum_sharpe_weights(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """
        Maximum Sharpe ratio portfolio optimization.

        Maximizes Sharpe ratio subject to:
        - Weights sum to 1
        - Weights >= min_weight
        - Weights <= max_weight
        """
        returns_df = pd.DataFrame(strategy_returns)
        strategy_ids = list(returns_df.columns)
        n = len(strategy_ids)

        # Expected returns and covariance
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values

        # Objective function: negative Sharpe ratio
        def negative_sharpe(weights):
            portfolio_return = weights @ mean_returns
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights.T)

            if portfolio_vol == 0:
                return 1e10

            # Subtract risk-free rate (annualized)
            rf_daily = self.config.risk_free_rate / 252
            sharpe = (portfolio_return - rf_daily) / portfolio_vol

            return -sharpe  # Negative for minimization

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]

        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]

        # Initial guess (equal weights)
        w0 = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(
            negative_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000},
        )

        if not result.success:
            # Fallback to Sharpe weights
            return self._sharpe_weights(strategy_returns)

        return {sid: float(w) for sid, w in zip(strategy_ids, result.x)}

    def _hrp_weights(self, strategy_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (HRP) weights.

        Based on López de Prado's HRP algorithm:
        1. Cluster strategies by correlation
        2. Allocate weights hierarchically based on variance
        """
        returns_df = pd.DataFrame(strategy_returns)
        strategy_ids = list(returns_df.columns)

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Convert correlation to distance matrix
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # Hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix.values), method='single')

        # Get quasi-diagonalization order
        sort_ix = self._get_quasi_diag(linkage_matrix)

        # Allocate weights recursively
        weights = self._recursive_bisection(returns_df, sort_ix)

        return {sid: float(weights[i]) for i, sid in enumerate(strategy_ids)}

    def _get_quasi_diag(self, linkage_matrix: np.ndarray) -> List[int]:
        """Get quasi-diagonal order from linkage matrix."""
        n = linkage_matrix.shape[0] + 1
        sort_ix = []

        def _append_leaves(node_id):
            if node_id < n:
                # Leaf node
                sort_ix.append(node_id)
            else:
                # Internal node
                left_child = int(linkage_matrix[node_id - n, 0])
                right_child = int(linkage_matrix[node_id - n, 1])
                _append_leaves(left_child)
                _append_leaves(right_child)

        # Start from root
        _append_leaves(2 * n - 2)

        return sort_ix

    def _recursive_bisection(
        self,
        returns_df: pd.DataFrame,
        sort_ix: List[int],
    ) -> np.ndarray:
        """Recursive bisection to allocate weights."""
        n = len(sort_ix)
        weights = np.ones(n)

        # Reorder data by quasi-diagonal order
        reordered_returns = returns_df.iloc[:, sort_ix]

        # Cluster variance
        cov_matrix = reordered_returns.cov()

        def _recursive_split(indices):
            if len(indices) == 1:
                return

            # Split into two clusters
            mid = len(indices) // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]

            # Calculate cluster variances
            left_cov = cov_matrix.iloc[left_indices, left_indices]
            right_cov = cov_matrix.iloc[right_indices, right_indices]

            left_var = np.trace(left_cov.values) / len(left_indices)
            right_var = np.trace(right_cov.values) / len(right_indices)

            # Allocate weights inversely proportional to variance
            total_var = left_var + right_var
            if total_var > 0:
                left_weight = 1.0 - left_var / total_var
                right_weight = 1.0 - right_var / total_var

                # Normalize
                total_weight = left_weight + right_weight
                left_weight /= total_weight
                right_weight /= total_weight
            else:
                left_weight = 0.5
                right_weight = 0.5

            # Apply weights to clusters
            for i in left_indices:
                weights[i] *= left_weight
            for i in right_indices:
                weights[i] *= right_weight

            # Recurse
            _recursive_split(left_indices)
            _recursive_split(right_indices)

        _recursive_split(list(range(n)))

        # Reorder weights back to original order
        original_weights = np.zeros(n)
        for i, orig_i in enumerate(sort_ix):
            original_weights[orig_i] = weights[i]

        return original_weights

    def _calculate_portfolio_returns(
        self,
        strategy_returns: Dict[str, pd.Series],
        weights: Dict[str, float],
    ) -> pd.Series:
        """Calculate portfolio returns given weights."""
        returns_df = pd.DataFrame(strategy_returns)

        # Align weights
        weight_array = np.array([weights.get(sid, 0.0) for sid in returns_df.columns])

        # Portfolio returns
        portfolio_returns = (returns_df.values @ weight_array).squeeze()

        return pd.Series(portfolio_returns, index=returns_df.index)

    def compare_methods(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> Dict[str, Portfolio]:
        """
        Compare all weighting methods.

        Args:
            strategy_returns: Dictionary of strategy returns

        Returns:
            Dictionary of method -> Portfolio
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

        results = {}
        for method in methods:
            try:
                portfolio = self.optimize_weights(strategy_returns, method)
                results[method.value] = portfolio
            except Exception as e:
                print(f"[WARN] {method.value} optimization failed: {e}")
                continue

        return results

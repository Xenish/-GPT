"""
Correlation and Diversification Analysis.

Tools for analyzing strategy correlations and portfolio diversification.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from finantradealgo.research.ensemble.portfolio import CorrelationMatrix


class CorrelationAnalyzer:
    """Analyze correlations between strategies."""

    def __init__(self):
        """Initialize correlation analyzer."""
        pass

    def calculate_correlation_matrix(
        self,
        strategy_returns: Dict[str, pd.Series],
        method: str = "pearson",
    ) -> CorrelationMatrix:
        """
        Calculate correlation matrix between strategies.

        Args:
            strategy_returns: Dictionary of strategy_id -> returns series
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            CorrelationMatrix with analysis results
        """
        # Convert to DataFrame
        returns_df = pd.DataFrame(strategy_returns)

        # Calculate correlation matrix
        corr_matrix = returns_df.corr(method=method)

        # Calculate summary statistics
        # Get upper triangle (excluding diagonal)
        n = len(corr_matrix)
        corr_values = []
        for i in range(n):
            for j in range(i + 1, n):
                corr_values.append(corr_matrix.iloc[i, j])

        avg_corr = float(np.mean(corr_values)) if corr_values else 0.0
        min_corr = float(np.min(corr_values)) if corr_values else 0.0
        max_corr = float(np.max(corr_values)) if corr_values else 0.0

        # Calculate diversification metrics
        strategy_ids = list(strategy_returns.keys())

        # Assume equal weights for diversification ratio calculation
        weights = np.array([1.0 / len(strategy_ids)] * len(strategy_ids))

        # Calculate individual volatilities
        vols = returns_df.std().values

        # Weighted average volatility
        weighted_avg_vol = np.sum(weights * vols)

        # Portfolio volatility (with correlations)
        cov_matrix = returns_df.cov().values
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights.T)

        # Diversification ratio = weighted avg vol / portfolio vol
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        # Effective N (Effective number of independent bets)
        # Formula: (sum of weights)^2 / sum of (weights^2 * (1 + sum of correlations with other strategies))
        effective_n = self._calculate_effective_n(corr_matrix.values, weights)

        return CorrelationMatrix(
            strategy_ids=strategy_ids,
            correlation_matrix=corr_matrix,
            avg_correlation=avg_corr,
            min_correlation=min_corr,
            max_correlation=max_corr,
            diversification_ratio=div_ratio,
            effective_n=effective_n,
        )

    def _calculate_effective_n(
        self,
        corr_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Calculate effective number of independent strategies.

        Uses formula from Meucci (2009):
        Effective N = (sum of weights)^2 / sum of squared weights * (1 + avg correlation)

        Args:
            corr_matrix: Correlation matrix
            weights: Strategy weights

        Returns:
            Effective N
        """
        n = len(weights)

        # Simple approximation using average pairwise correlation
        total_corr = 0.0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total_corr += corr_matrix[i, j]
                    count += 1

        avg_corr = total_corr / count if count > 0 else 0.0

        # Effective N formula
        sum_weights = np.sum(weights)
        sum_sq_weights = np.sum(weights ** 2)

        effective_n = (sum_weights ** 2) / (sum_sq_weights * (1 + avg_corr * (n - 1)))

        return float(effective_n)

    def find_diversifying_strategies(
        self,
        strategy_returns: Dict[str, pd.Series],
        target_strategy: str,
        max_correlation: float = 0.5,
        top_n: int = 5,
    ) -> List[tuple[str, float]]:
        """
        Find strategies that diversify well with target strategy.

        Args:
            strategy_returns: Dictionary of strategy returns
            target_strategy: Strategy to find diversifiers for
            max_correlation: Maximum correlation threshold
            top_n: Number of top diversifiers to return

        Returns:
            List of (strategy_id, correlation) tuples, sorted by correlation
        """
        if target_strategy not in strategy_returns:
            raise ValueError(f"Target strategy {target_strategy} not found")

        target_returns = strategy_returns[target_strategy]
        diversifiers = []

        for strategy_id, returns in strategy_returns.items():
            if strategy_id == target_strategy:
                continue

            # Calculate correlation
            corr = target_returns.corr(returns)

            if abs(corr) <= max_correlation:
                diversifiers.append((strategy_id, float(corr)))

        # Sort by absolute correlation (ascending)
        diversifiers.sort(key=lambda x: abs(x[1]))

        return diversifiers[:top_n]

    def calculate_rolling_correlation(
        self,
        strategy_returns: Dict[str, pd.Series],
        window: int = 60,
    ) -> Dict[tuple[str, str], pd.Series]:
        """
        Calculate rolling correlation between strategy pairs.

        Args:
            strategy_returns: Dictionary of strategy returns
            window: Rolling window size

        Returns:
            Dictionary of (strategy1, strategy2) -> rolling correlation series
        """
        returns_df = pd.DataFrame(strategy_returns)
        strategy_ids = list(strategy_returns.keys())

        rolling_corr = {}

        for i, sid1 in enumerate(strategy_ids):
            for j, sid2 in enumerate(strategy_ids):
                if i < j:  # Only calculate upper triangle
                    corr_series = returns_df[sid1].rolling(window).corr(returns_df[sid2])
                    rolling_corr[(sid1, sid2)] = corr_series

        return rolling_corr

    def identify_correlation_clusters(
        self,
        corr_matrix: CorrelationMatrix,
        threshold: float = 0.7,
    ) -> List[List[str]]:
        """
        Identify clusters of highly correlated strategies.

        Uses simple threshold-based clustering.

        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold for clustering

        Returns:
            List of strategy clusters
        """
        strategy_ids = corr_matrix.strategy_ids
        matrix = corr_matrix.correlation_matrix.values
        n = len(strategy_ids)

        # Track which strategies are already in clusters
        assigned = [False] * n
        clusters = []

        for i in range(n):
            if assigned[i]:
                continue

            # Start new cluster with strategy i
            cluster = [strategy_ids[i]]
            assigned[i] = True

            # Find all strategies correlated with i
            for j in range(i + 1, n):
                if assigned[j]:
                    continue

                if abs(matrix[i, j]) >= threshold:
                    cluster.append(strategy_ids[j])
                    assigned[j] = True

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters


class DiversificationOptimizer:
    """Optimize portfolio for maximum diversification."""

    def __init__(self):
        """Initialize diversification optimizer."""
        pass

    def select_diversified_subset(
        self,
        strategy_returns: Dict[str, pd.Series],
        target_n: int = 5,
        method: str = "greedy",
    ) -> List[str]:
        """
        Select a diversified subset of strategies.

        Args:
            strategy_returns: Dictionary of strategy returns
            target_n: Target number of strategies
            method: Selection method ('greedy' or 'random')

        Returns:
            List of selected strategy IDs
        """
        if len(strategy_returns) <= target_n:
            return list(strategy_returns.keys())

        if method == "greedy":
            return self._greedy_selection(strategy_returns, target_n)
        elif method == "random":
            return self._random_selection(strategy_returns, target_n)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _greedy_selection(
        self,
        strategy_returns: Dict[str, pd.Series],
        target_n: int,
    ) -> List[str]:
        """
        Greedily select strategies to maximize diversification.

        Starts with best Sharpe ratio, then adds strategies with lowest
        average correlation to already selected strategies.

        Args:
            strategy_returns: Dictionary of strategy returns
            target_n: Target number of strategies

        Returns:
            List of selected strategy IDs
        """
        strategy_ids = list(strategy_returns.keys())
        returns_df = pd.DataFrame(strategy_returns)

        # Calculate Sharpe ratios
        sharpe_ratios = {}
        for sid in strategy_ids:
            returns = returns_df[sid]
            sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            sharpe_ratios[sid] = sharpe

        # Start with best Sharpe ratio
        selected = [max(sharpe_ratios, key=sharpe_ratios.get)]

        # Greedily add strategies
        while len(selected) < target_n and len(selected) < len(strategy_ids):
            best_candidate = None
            best_avg_corr = float('inf')

            for candidate in strategy_ids:
                if candidate in selected:
                    continue

                # Calculate average correlation with selected strategies
                correlations = []
                for selected_sid in selected:
                    corr = returns_df[candidate].corr(returns_df[selected_sid])
                    correlations.append(abs(corr))

                avg_corr = np.mean(correlations)

                if avg_corr < best_avg_corr:
                    best_avg_corr = avg_corr
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
            else:
                break

        return selected

    def _random_selection(
        self,
        strategy_returns: Dict[str, pd.Series],
        target_n: int,
    ) -> List[str]:
        """
        Randomly select strategies.

        Args:
            strategy_returns: Dictionary of strategy returns
            target_n: Target number of strategies

        Returns:
            List of selected strategy IDs
        """
        strategy_ids = list(strategy_returns.keys())
        return list(np.random.choice(strategy_ids, size=min(target_n, len(strategy_ids)), replace=False))

    def calculate_portfolio_diversification(
        self,
        strategy_returns: Dict[str, pd.Series],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate diversification metrics for a weighted portfolio.

        Args:
            strategy_returns: Dictionary of strategy returns
            weights: Dictionary of strategy weights

        Returns:
            Dictionary of diversification metrics
        """
        returns_df = pd.DataFrame(strategy_returns)

        # Align weights with returns
        strategy_ids = list(returns_df.columns)
        weight_array = np.array([weights.get(sid, 0.0) for sid in strategy_ids])

        # Normalize weights
        weight_array = weight_array / weight_array.sum()

        # Calculate metrics
        cov_matrix = returns_df.cov().values
        vols = returns_df.std().values

        # Portfolio volatility
        portfolio_vol = np.sqrt(weight_array @ cov_matrix @ weight_array.T)

        # Weighted average volatility
        weighted_avg_vol = np.sum(weight_array * vols)

        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        # Correlation matrix
        corr_matrix = returns_df.corr().values

        # Effective N
        effective_n = self._calculate_effective_n(corr_matrix, weight_array)

        # Average pairwise correlation
        n = len(strategy_ids)
        total_corr = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_corr += corr_matrix[i, j]
                count += 1
        avg_corr = total_corr / count if count > 0 else 0.0

        return {
            "portfolio_volatility": float(portfolio_vol),
            "weighted_avg_volatility": float(weighted_avg_vol),
            "diversification_ratio": float(div_ratio),
            "effective_n": float(effective_n),
            "avg_correlation": float(avg_corr),
        }

    def _calculate_effective_n(
        self,
        corr_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Calculate effective number of independent strategies."""
        n = len(weights)

        # Calculate average pairwise correlation
        total_corr = 0.0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total_corr += corr_matrix[i, j]
                    count += 1

        avg_corr = total_corr / count if count > 0 else 0.0

        # Effective N formula
        sum_weights = np.sum(weights)
        sum_sq_weights = np.sum(weights ** 2)

        effective_n = (sum_weights ** 2) / (sum_sq_weights * (1 + avg_corr * (n - 1)))

        return float(effective_n)

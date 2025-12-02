"""
Monte Carlo Analysis & Regime Randomization.

Advanced analysis methods for Monte Carlo results including regime randomization,
luck vs. skill testing, and comprehensive statistical analysis.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats

from finantradealgo.research.montecarlo.models import (
    MonteCarloResult,
    SimulationResult,
)


@dataclass
class LuckVsSkillAnalysis:
    """
    Luck vs. Skill analysis result.

    Determines if results are statistically significant or due to chance.
    """

    # Original performance
    original_return: float
    original_sharpe: float

    # Monte Carlo distribution
    mc_mean_return: float
    mc_std_return: float
    mc_mean_sharpe: float

    # Statistical tests
    percentile_rank: float  # Where original falls in MC distribution
    z_score: float  # Standard deviations from mean
    p_value: float  # Probability of getting this result by chance

    # Verdict
    is_statistically_significant: bool  # p < 0.05
    confidence_level: float  # 0-100

    # Interpretation
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_performance": {
                "return": round(self.original_return, 2),
                "sharpe": round(self.original_sharpe, 3),
            },
            "monte_carlo_distribution": {
                "mean_return": round(self.mc_mean_return, 2),
                "std_return": round(self.mc_std_return, 2),
                "mean_sharpe": round(self.mc_mean_sharpe, 3),
            },
            "statistical_tests": {
                "percentile_rank": round(self.percentile_rank, 1),
                "z_score": round(self.z_score, 3),
                "p_value": round(self.p_value, 4),
            },
            "verdict": {
                "is_significant": self.is_statistically_significant,
                "confidence_level": round(self.confidence_level, 1),
                "interpretation": self.interpretation,
            },
        }


@dataclass
class RegimeAnalysis:
    """
    Market regime analysis result.

    Tests strategy performance across different market conditions.
    """

    # Regime performance
    bull_market_return: float
    bear_market_return: float
    sideways_market_return: float

    # Regime consistency
    regime_consistency_score: float  # 0-100, higher = more consistent

    # Risk by regime
    bull_max_dd: float
    bear_max_dd: float
    sideways_max_dd: float

    # Statistical significance
    regime_impact_significant: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime_returns": {
                "bull": round(self.bull_market_return, 2),
                "bear": round(self.bear_market_return, 2),
                "sideways": round(self.sideways_market_return, 2),
            },
            "regime_risk": {
                "bull_max_dd": round(self.bull_max_dd, 2),
                "bear_max_dd": round(self.bear_max_dd, 2),
                "sideways_max_dd": round(self.sideways_max_dd, 2),
            },
            "consistency_score": round(self.regime_consistency_score, 1),
            "regime_impact_significant": self.regime_impact_significant,
        }


class MonteCarloAnalyzer:
    """
    Advanced Monte Carlo analysis.

    Provides:
    1. Luck vs. Skill testing - Statistical significance
    2. Regime Randomization - Performance across market conditions
    3. Drawdown distribution analysis
    4. Comprehensive statistical tests
    """

    def __init__(self):
        """Initialize analyzer."""
        pass

    def analyze_luck_vs_skill(
        self,
        original_trades_df: pd.DataFrame,
        mc_result: MonteCarloResult,
    ) -> LuckVsSkillAnalysis:
        """
        Determine if results are due to luck or skill.

        Compares original performance to Monte Carlo distribution
        to assess statistical significance.

        Args:
            original_trades_df: Original trade results
            mc_result: Monte Carlo simulation result

        Returns:
            LuckVsSkillAnalysis with statistical tests
        """
        # Calculate original performance
        original_pnl = original_trades_df['pnl'].sum()
        original_return = (original_pnl / 10000) * 100  # Assume $10k capital

        # Calculate original Sharpe
        if original_trades_df['pnl'].std() > 0:
            original_sharpe = (original_trades_df['pnl'].mean() /
                              original_trades_df['pnl'].std()) * np.sqrt(252)
        else:
            original_sharpe = 0.0

        # Extract MC distribution
        mc_returns = np.array([s.total_return for s in mc_result.simulations])
        mc_mean = np.mean(mc_returns)
        mc_std = np.std(mc_returns)

        # Calculate percentile rank
        percentile = (np.sum(mc_returns < original_return) / len(mc_returns)) * 100

        # Calculate z-score
        if mc_std > 0:
            z_score = (original_return - mc_mean) / mc_std
        else:
            z_score = 0.0

        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Determine significance
        is_significant = p_value < 0.05

        # Calculate confidence level
        confidence = (1 - p_value) * 100 if p_value < 1 else 0

        # Generate interpretation
        interpretation = self._interpret_luck_vs_skill(
            percentile, z_score, p_value, is_significant
        )

        return LuckVsSkillAnalysis(
            original_return=original_return,
            original_sharpe=original_sharpe,
            mc_mean_return=mc_mean,
            mc_std_return=mc_std,
            mc_mean_sharpe=mc_result.mean_return,
            percentile_rank=percentile,
            z_score=z_score,
            p_value=p_value,
            is_statistically_significant=is_significant,
            confidence_level=confidence,
            interpretation=interpretation,
        )

    def analyze_regime_randomization(
        self,
        trades_df: pd.DataFrame,
        market_data: pd.DataFrame,
        n_randomizations: int = 500,
    ) -> RegimeAnalysis:
        """
        Test strategy across randomized market regimes.

        Shuffles market conditions to see if strategy is regime-dependent.

        Args:
            trades_df: Trade results with datetime index
            market_data: Market data with datetime index and 'returns' column
            n_randomizations: Number of regime shuffles

        Returns:
            RegimeAnalysis with regime performance
        """
        # Identify market regimes in original data
        regimes = self._identify_market_regimes(market_data)

        # Split trades by regime
        bull_trades = []
        bear_trades = []
        sideways_trades = []

        for idx, row in trades_df.iterrows():
            trade_date = idx if isinstance(idx, pd.Timestamp) else row.get('exit_date', idx)

            if trade_date in regimes.index:
                regime = regimes.loc[trade_date]

                if regime == 'bull':
                    bull_trades.append(row['pnl'])
                elif regime == 'bear':
                    bear_trades.append(row['pnl'])
                else:
                    sideways_trades.append(row['pnl'])

        # Calculate regime performance
        bull_return = sum(bull_trades) / 10000 * 100 if bull_trades else 0
        bear_return = sum(bear_trades) / 10000 * 100 if bear_trades else 0
        sideways_return = sum(sideways_trades) / 10000 * 100 if sideways_trades else 0

        # Calculate regime drawdowns
        bull_dd = self._calculate_max_drawdown(bull_trades) if bull_trades else 0
        bear_dd = self._calculate_max_drawdown(bear_trades) if bear_trades else 0
        sideways_dd = self._calculate_max_drawdown(sideways_trades) if sideways_trades else 0

        # Consistency score (lower variance = more consistent)
        regime_returns = [bull_return, bear_return, sideways_return]
        regime_std = np.std(regime_returns)
        regime_mean = np.mean([abs(r) for r in regime_returns])

        if regime_mean > 0:
            consistency_score = max(0, 100 * (1 - min(regime_std / regime_mean, 1)))
        else:
            consistency_score = 0

        # Test if regime differences are significant
        # Use ANOVA or Kruskal-Wallis test
        try:
            if len(bull_trades) > 0 and len(bear_trades) > 0 and len(sideways_trades) > 0:
                statistic, p_value = stats.kruskal(bull_trades, bear_trades, sideways_trades)
                regime_impact_significant = p_value < 0.05
            else:
                regime_impact_significant = False
        except:
            regime_impact_significant = False

        return RegimeAnalysis(
            bull_market_return=bull_return,
            bear_market_return=bear_return,
            sideways_market_return=sideways_return,
            regime_consistency_score=consistency_score,
            bull_max_dd=bull_dd,
            bear_max_dd=bear_dd,
            sideways_max_dd=sideways_dd,
            regime_impact_significant=regime_impact_significant,
        )

    def analyze_drawdown_distribution(
        self,
        mc_result: MonteCarloResult,
    ) -> Dict[str, Any]:
        """
        Analyze drawdown distribution from Monte Carlo simulations.

        Args:
            mc_result: Monte Carlo result

        Returns:
            Dictionary with drawdown analysis
        """
        drawdowns = np.array([s.max_drawdown for s in mc_result.simulations])

        return {
            "mean_max_dd": float(np.mean(drawdowns)),
            "median_max_dd": float(np.median(drawdowns)),
            "std_max_dd": float(np.std(drawdowns)),
            "percentile_5": float(np.percentile(drawdowns, 5)),
            "percentile_25": float(np.percentile(drawdowns, 25)),
            "percentile_75": float(np.percentile(drawdowns, 75)),
            "percentile_95": float(np.percentile(drawdowns, 95)),
            "worst_case": float(np.min(drawdowns)),
            "best_case": float(np.max(drawdowns)),
        }

    def calculate_sharpe_confidence_interval(
        self,
        mc_result: MonteCarloResult,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for Sharpe ratio.

        Args:
            mc_result: Monte Carlo result
            confidence_level: Confidence level (default 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        sharpes = np.array([s.sharpe_ratio for s in mc_result.simulations])

        alpha = 1 - confidence_level
        lower = np.percentile(sharpes, alpha / 2 * 100)
        upper = np.percentile(sharpes, (1 - alpha / 2) * 100)

        return float(lower), float(upper)

    def estimate_win_probability(
        self,
        mc_result: MonteCarloResult,
        target_return: float,
    ) -> float:
        """
        Estimate probability of achieving target return.

        Args:
            mc_result: Monte Carlo result
            target_return: Target return percentage

        Returns:
            Probability (0-1)
        """
        returns = np.array([s.total_return for s in mc_result.simulations])
        prob = np.sum(returns >= target_return) / len(returns)

        return float(prob)

    def analyze_return_distribution(
        self,
        mc_result: MonteCarloResult,
    ) -> Dict[str, Any]:
        """
        Comprehensive return distribution analysis.

        Args:
            mc_result: Monte Carlo result

        Returns:
            Dictionary with distribution analysis
        """
        returns = np.array([s.total_return for s in mc_result.simulations])

        # Test for normality
        _, normality_p_value = stats.normaltest(returns)
        is_normal = normality_p_value > 0.05

        # Jarque-Bera test
        _, jb_p_value = stats.jarque_bera(returns)

        return {
            "distribution_stats": {
                "mean": float(np.mean(returns)),
                "median": float(np.median(returns)),
                "mode": float(stats.mode(returns, keepdims=True)[0][0]),
                "std": float(np.std(returns)),
                "variance": float(np.var(returns)),
                "range": float(np.max(returns) - np.min(returns)),
                "iqr": float(np.percentile(returns, 75) - np.percentile(returns, 25)),
            },
            "shape": {
                "skewness": mc_result.skewness,
                "kurtosis": mc_result.kurtosis,
                "is_normal": is_normal,
                "normality_p_value": float(normality_p_value),
                "jarque_bera_p_value": float(jb_p_value),
            },
            "percentiles": {
                "1st": mc_result.percentile_1,
                "5th": mc_result.percentile_5,
                "25th": mc_result.percentile_25,
                "50th": mc_result.median_return,
                "75th": mc_result.percentile_75,
                "95th": mc_result.percentile_95,
                "99th": mc_result.percentile_99,
            },
        }

    def compare_to_benchmark(
        self,
        strategy_result: MonteCarloResult,
        benchmark_result: MonteCarloResult,
    ) -> Dict[str, Any]:
        """
        Compare strategy to benchmark using Monte Carlo results.

        Args:
            strategy_result: Strategy Monte Carlo result
            benchmark_result: Benchmark Monte Carlo result

        Returns:
            Comparison dictionary
        """
        strategy_returns = np.array([s.total_return for s in strategy_result.simulations])
        benchmark_returns = np.array([s.total_return for s in benchmark_result.simulations])

        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(strategy_returns, benchmark_returns)

        # Win probability against benchmark
        win_prob = self._estimate_win_vs_benchmark(strategy_returns, benchmark_returns)

        return {
            "mean_difference": {
                "strategy": strategy_result.mean_return,
                "benchmark": benchmark_result.mean_return,
                "difference": strategy_result.mean_return - benchmark_result.mean_return,
            },
            "risk_difference": {
                "strategy_var": strategy_result.value_at_risk,
                "benchmark_var": benchmark_result.value_at_risk,
                "difference": strategy_result.value_at_risk - benchmark_result.value_at_risk,
            },
            "statistical_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significantly_different": p_value < 0.05,
            },
            "win_probability": win_prob,
            "sharpe_comparison": {
                "strategy": strategy_result.mean_return / strategy_result.std_return
                if strategy_result.std_return > 0 else 0,
                "benchmark": benchmark_result.mean_return / benchmark_result.std_return
                if benchmark_result.std_return > 0 else 0,
            },
        }

    def generate_comprehensive_report(
        self,
        original_trades_df: pd.DataFrame,
        mc_result: MonteCarloResult,
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive Monte Carlo analysis report.

        Args:
            original_trades_df: Original trade results
            mc_result: Monte Carlo result
            market_data: Optional market data for regime analysis

        Returns:
            Complete analysis dictionary
        """
        report = {
            "monte_carlo_summary": mc_result.to_summary_dict(),
        }

        # Luck vs. Skill
        luck_analysis = self.analyze_luck_vs_skill(original_trades_df, mc_result)
        report["luck_vs_skill"] = luck_analysis.to_dict()

        # Regime analysis (if market data provided)
        if market_data is not None and 'returns' in market_data.columns:
            regime_analysis = self.analyze_regime_randomization(
                original_trades_df, market_data
            )
            report["regime_analysis"] = regime_analysis.to_dict()

        # Drawdown distribution
        report["drawdown_distribution"] = self.analyze_drawdown_distribution(mc_result)

        # Return distribution
        report["return_distribution"] = self.analyze_return_distribution(mc_result)

        # Sharpe CI
        sharpe_ci = self.calculate_sharpe_confidence_interval(mc_result)
        report["sharpe_confidence_interval"] = {
            "lower": round(sharpe_ci[0], 3),
            "upper": round(sharpe_ci[1], 3),
        }

        return report

    # Helper methods

    def _identify_market_regimes(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Identify market regimes (bull, bear, sideways).

        Args:
            market_data: Market data with 'returns' column

        Returns:
            Series with regime labels
        """
        # Calculate rolling metrics
        window = 20  # 20-day window
        returns = market_data['returns']

        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        # Define regimes
        regimes = pd.Series(index=market_data.index, data='sideways')

        # Bull: positive trend + low volatility
        bull_mask = (rolling_mean > 0.001) & (rolling_std < rolling_std.median())
        regimes[bull_mask] = 'bull'

        # Bear: negative trend
        bear_mask = rolling_mean < -0.001
        regimes[bear_mask] = 'bear'

        return regimes

    def _calculate_max_drawdown(self, pnl_list: List[float]) -> float:
        """Calculate max drawdown from PnL list."""
        if not pnl_list:
            return 0.0

        equity = 10000 + np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100

        return float(np.min(drawdown))

    def _interpret_luck_vs_skill(
        self,
        percentile: float,
        z_score: float,
        p_value: float,
        is_significant: bool,
    ) -> str:
        """Generate interpretation of luck vs. skill analysis."""
        if is_significant and percentile > 95:
            return (
                f"Results are statistically significant (p={p_value:.4f}). "
                f"Performance at {percentile:.1f}th percentile suggests skill, not luck."
            )
        elif is_significant and percentile < 5:
            return (
                f"Results are significantly poor (p={p_value:.4f}). "
                f"Performance at {percentile:.1f}th percentile is worse than random."
            )
        elif percentile > 75:
            return (
                f"Results are above average ({percentile:.1f}th percentile) "
                f"but not statistically significant (p={p_value:.4f})."
            )
        elif percentile < 25:
            return (
                f"Results are below average ({percentile:.1f}th percentile) "
                f"and may be due to bad luck or poor strategy."
            )
        else:
            return (
                f"Results are near median ({percentile:.1f}th percentile). "
                f"Cannot distinguish from random outcomes (p={p_value:.4f})."
            )

    def _estimate_win_vs_benchmark(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """Estimate probability of beating benchmark."""
        # Run paired comparison
        n_simulations = min(len(strategy_returns), len(benchmark_returns))

        wins = 0
        for i in range(n_simulations):
            if strategy_returns[i] > benchmark_returns[i]:
                wins += 1

        return wins / n_simulations

"""
Monte Carlo Simulator.

High-level API for Monte Carlo simulation including trade sequence randomization,
parameter perturbation, and robustness testing.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np

from finantradealgo.research.montecarlo.models import (
    MonteCarloConfig,
    MonteCarloResult,
    ResamplingMethod,
)
from finantradealgo.research.montecarlo.resampler import BootstrapResampler
from finantradealgo.research.montecarlo.risk_metrics import RiskMetricsCalculator, RiskAssessment
from finantradealgo.research.montecarlo.stress_test import StressTestEngine


class MonteCarloSimulator:
    """
    Complete Monte Carlo simulation engine.

    Provides high-level API for:
    1. Trade Sequence Randomization - Test if results are due to luck or skill
    2. Parameter Perturbation - Test parameter sensitivity
    3. Robustness Testing - Stress tests and risk analysis

    Example:
        >>> from finantradealgo.research.montecarlo import MonteCarloSimulator, MonteCarloConfig
        >>>
        >>> # Configure Monte Carlo
        >>> config = MonteCarloConfig(
        ...     n_simulations=1000,
        ...     resampling_method="bootstrap",
        ...     confidence_level=0.95
        ... )
        >>>
        >>> # Create simulator
        >>> simulator = MonteCarloSimulator(config)
        >>>
        >>> # Test trade sequence randomization
        >>> result = simulator.test_trade_sequence(trades_df)
        >>> print(f"Mean Return: {result.mean_return:.2f}%")
        >>> print(f"95% CI: [{result.return_ci_lower:.2f}, {result.return_ci_upper:.2f}]")
        >>>
        >>> # Test parameter sensitivity
        >>> sensitivity = simulator.test_parameter_sensitivity(
        ...     optimal_params={'fast_ma': 20, 'slow_ma': 50},
        ...     param_ranges={'fast_ma': [15, 25], 'slow_ma': [40, 60]},
        ...     backtest_function=my_backtest
        ... )
    """

    def __init__(self, config: MonteCarloConfig):
        """
        Initialize Monte Carlo simulator.

        Args:
            config: Monte Carlo configuration
        """
        self.config = config
        self.resampler = BootstrapResampler(config)
        self.risk_calculator = RiskMetricsCalculator()
        self.stress_tester = StressTestEngine()

    def test_trade_sequence(
        self,
        trades_df: pd.DataFrame,
        strategy_id: str = "strategy",
    ) -> MonteCarloResult:
        """
        Test trade sequence randomization (Luck vs. Skill).

        Randomly shuffles/resamples trade order to determine if results
        are statistically significant or just lucky ordering.

        Args:
            trades_df: DataFrame with trade results (must have 'pnl' column)
            strategy_id: Strategy identifier

        Returns:
            MonteCarloResult with distribution of possible outcomes
        """
        print(f"Running trade sequence randomization ({self.config.n_simulations} simulations)...")

        result = self.resampler.run_monte_carlo(strategy_id, trades_df)

        # Calculate original performance
        original_return = (trades_df['pnl'].sum() / 10000) * 100

        # Compare to Monte Carlo distribution
        percentile = self._calculate_percentile(result, original_return)

        print(f"\nOriginal Return: {original_return:.2f}%")
        print(f"Mean MC Return: {result.mean_return:.2f}%")
        print(f"95% CI: [{result.return_ci_lower:.2f}, {result.return_ci_upper:.2f}]")
        print(f"Original is at {percentile:.1f}th percentile of MC distribution")

        if percentile > 95:
            print("✓ Results appear statistically significant (>95th percentile)")
        elif percentile > 50:
            print("✓ Results are above average but not exceptional")
        else:
            print("⚠ Results may be due to luck (below median)")

        return result

    def test_parameter_perturbation(
        self,
        optimal_params: Dict[str, float],
        data_df: pd.DataFrame,
        backtest_function: Callable,
        noise_level: float = 0.1,
        n_perturbations: int = 100,
    ) -> Dict[str, Any]:
        """
        Test parameter sensitivity through perturbation.

        Adds noise to optimal parameters to test robustness:
        - How sensitive is performance to parameter changes?
        - What's the confidence interval for performance?
        - Are parameters stable?

        Args:
            optimal_params: Optimal parameters found through optimization
            data_df: Price data
            backtest_function: Function(data_df, params) -> (trades_df, metrics)
            noise_level: Noise level as fraction (0.1 = ±10%)
            n_perturbations: Number of perturbations to test

        Returns:
            Dictionary with perturbation analysis
        """
        print(f"\nTesting parameter sensitivity (noise level: {noise_level*100:.0f}%)...")

        results = []
        failed_runs = 0

        for i in range(n_perturbations):
            # Perturb parameters
            perturbed_params = self._perturb_parameters(optimal_params, noise_level)

            try:
                # Run backtest with perturbed parameters
                trades_df, metrics = backtest_function(data_df, perturbed_params)

                if trades_df is not None and len(trades_df) > 0:
                    results.append({
                        'params': perturbed_params,
                        'total_return': metrics.get('total_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'total_trades': len(trades_df),
                    })
            except Exception as e:
                failed_runs += 1
                continue

        if not results:
            raise ValueError("All perturbation runs failed")

        # Run backtest with optimal parameters
        optimal_trades, optimal_metrics = backtest_function(data_df, optimal_params)
        optimal_return = optimal_metrics.get('total_return', 0)
        optimal_sharpe = optimal_metrics.get('sharpe_ratio', 0)

        # Analyze results
        returns = np.array([r['total_return'] for r in results])
        sharpes = np.array([r['sharpe_ratio'] for r in results])

        analysis = {
            'optimal_params': optimal_params,
            'optimal_return': optimal_return,
            'optimal_sharpe': optimal_sharpe,
            'noise_level': noise_level,
            'n_perturbations': len(results),
            'failed_runs': failed_runs,
            'return_stats': {
                'mean': float(np.mean(returns)),
                'median': float(np.median(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns)),
                'percentile_5': float(np.percentile(returns, 5)),
                'percentile_95': float(np.percentile(returns, 95)),
            },
            'sharpe_stats': {
                'mean': float(np.mean(sharpes)),
                'median': float(np.median(sharpes)),
                'std': float(np.std(sharpes)),
                'percentile_5': float(np.percentile(sharpes, 5)),
                'percentile_95': float(np.percentile(sharpes, 95)),
            },
            'degradation': {
                'return_degradation': float((optimal_return - np.mean(returns)) / abs(optimal_return))
                    if optimal_return != 0 else 0,
                'sharpe_degradation': float((optimal_sharpe - np.mean(sharpes)) / abs(optimal_sharpe))
                    if optimal_sharpe != 0 else 0,
            },
            'robustness_score': self._calculate_robustness_score(optimal_return, returns, optimal_sharpe, sharpes),
        }

        # Print summary
        print(f"\nParameter Perturbation Results:")
        print(f"  Successful Runs: {len(results)}/{n_perturbations}")
        print(f"  Optimal Return: {optimal_return:.2f}%")
        print(f"  Mean Perturbed: {analysis['return_stats']['mean']:.2f}%")
        print(f"  5-95% Range: [{analysis['return_stats']['percentile_5']:.2f}, "
              f"{analysis['return_stats']['percentile_95']:.2f}]%")
        print(f"  Robustness Score: {analysis['robustness_score']:.1f}/100")

        if analysis['robustness_score'] > 70:
            print("  ✓ Parameters appear robust")
        elif analysis['robustness_score'] > 50:
            print("  ⚠ Moderate parameter sensitivity")
        else:
            print("  ⚠ High parameter sensitivity - results may be unstable")

        return analysis

    def test_parameter_sensitivity_grid(
        self,
        optimal_params: Dict[str, float],
        param_ranges: Dict[str, tuple[float, float]],
        data_df: pd.DataFrame,
        backtest_function: Callable,
        n_samples: int = 50,
    ) -> Dict[str, Any]:
        """
        Test parameter sensitivity using grid sampling around optimal.

        Args:
            optimal_params: Optimal parameters
            param_ranges: Dict of param_name -> (min, max) ranges to test
            data_df: Price data
            backtest_function: Backtest function
            n_samples: Number of samples per parameter

        Returns:
            Sensitivity analysis dictionary
        """
        print(f"\nTesting parameter sensitivity grid...")

        sensitivity_results = {}

        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name not in optimal_params:
                continue

            print(f"  Testing {param_name}: [{min_val}, {max_val}]")

            param_values = np.linspace(min_val, max_val, n_samples)
            returns = []
            sharpes = []

            for value in param_values:
                # Create params with this value
                test_params = optimal_params.copy()
                test_params[param_name] = value

                try:
                    trades_df, metrics = backtest_function(data_df, test_params)
                    returns.append(metrics.get('total_return', 0))
                    sharpes.append(metrics.get('sharpe_ratio', 0))
                except:
                    returns.append(0)
                    sharpes.append(0)

            sensitivity_results[param_name] = {
                'values': param_values.tolist(),
                'returns': returns,
                'sharpes': sharpes,
                'optimal_value': optimal_params[param_name],
                'sensitivity_score': self._calculate_param_sensitivity(returns),
            }

        return {
            'optimal_params': optimal_params,
            'sensitivity_by_param': sensitivity_results,
            'overall_sensitivity': np.mean([v['sensitivity_score']
                                           for v in sensitivity_results.values()]),
        }

    def run_comprehensive_analysis(
        self,
        trades_df: pd.DataFrame,
        strategy_id: str = "strategy",
        include_stress_tests: bool = True,
    ) -> Dict[str, Any]:
        """
        Run comprehensive Monte Carlo analysis.

        Includes:
        - Trade sequence randomization
        - Risk metrics (VaR, CVaR)
        - Drawdown distribution
        - Stress tests (optional)

        Args:
            trades_df: Trade results DataFrame
            strategy_id: Strategy identifier
            include_stress_tests: Whether to run stress tests

        Returns:
            Complete analysis dictionary
        """
        print(f"\nRunning comprehensive Monte Carlo analysis...")

        # 1. Trade sequence randomization
        mc_result = self.test_trade_sequence(trades_df, strategy_id)

        # 2. Calculate risk metrics
        print("\nCalculating risk metrics...")
        risk_assessment = self.risk_calculator.calculate_comprehensive_risk(mc_result)

        # 3. Stress tests (optional)
        stress_results = None
        if include_stress_tests:
            print("\nRunning stress tests...")
            stress_results = self.stress_tester.run_stress_test(
                strategy_id=strategy_id,
                trades_df=trades_df,
                n_simulations=self.config.n_simulations // 2,  # Use fewer for speed
            )

        # Compile comprehensive report
        analysis = {
            'strategy_id': strategy_id,
            'monte_carlo': mc_result.to_summary_dict(),
            'risk_assessment': risk_assessment.to_dict() if risk_assessment else {},
            'stress_tests': stress_results,
            'recommendations': self._generate_recommendations(mc_result, risk_assessment),
        }

        return analysis

    def compare_strategies(
        self,
        strategy_trades: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compare multiple strategies using Monte Carlo.

        Args:
            strategy_trades: Dict of strategy_id -> trades_df

        Returns:
            Comparison DataFrame
        """
        print(f"\nComparing {len(strategy_trades)} strategies via Monte Carlo...")

        comparison_data = []

        for strategy_id, trades_df in strategy_trades.items():
            print(f"  Analyzing {strategy_id}...")

            result = self.resampler.run_monte_carlo(strategy_id, trades_df)

            comparison_data.append({
                'strategy_id': strategy_id,
                'mean_return': result.mean_return,
                'median_return': result.median_return,
                'std_return': result.std_return,
                'var_95': result.value_at_risk,
                'cvar_95': result.conditional_var,
                'prob_profit': result.prob_profit,
                'best_case': result.percentile_95,
                'worst_case': result.percentile_5,
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('mean_return', ascending=False)

        print("\nComparison Results:")
        print(df.to_string(index=False))

        return df

    # Helper methods

    def _perturb_parameters(
        self,
        params: Dict[str, float],
        noise_level: float,
    ) -> Dict[str, float]:
        """
        Add random noise to parameters.

        Args:
            params: Original parameters
            noise_level: Noise level as fraction

        Returns:
            Perturbed parameters
        """
        perturbed = {}

        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Add uniform noise
                noise = np.random.uniform(-noise_level, noise_level)
                new_value = value * (1 + noise)

                # Ensure positive for most trading parameters
                new_value = max(new_value, 1.0)

                # Keep integer types as integers
                if isinstance(value, int):
                    new_value = int(round(new_value))

                perturbed[key] = new_value
            else:
                perturbed[key] = value

        return perturbed

    def _calculate_percentile(
        self,
        result: MonteCarloResult,
        value: float,
    ) -> float:
        """Calculate percentile of value in MC distribution."""
        returns = np.array([s.total_return for s in result.simulations])
        percentile = (np.sum(returns < value) / len(returns)) * 100
        return float(percentile)

    def _calculate_robustness_score(
        self,
        optimal_return: float,
        perturbed_returns: np.ndarray,
        optimal_sharpe: float,
        perturbed_sharpes: np.ndarray,
    ) -> float:
        """
        Calculate robustness score (0-100).

        Higher score = more robust to parameter changes.
        """
        # Return stability (lower std = higher score)
        return_cv = np.std(perturbed_returns) / abs(np.mean(perturbed_returns)) \
            if np.mean(perturbed_returns) != 0 else 1.0
        return_stability = max(0, 100 * (1 - min(return_cv, 1)))

        # Percentage of runs that are positive
        positive_rate = np.sum(perturbed_returns > 0) / len(perturbed_returns)
        positive_score = positive_rate * 100

        # How close to optimal
        degradation = abs(optimal_return - np.mean(perturbed_returns)) / abs(optimal_return) \
            if optimal_return != 0 else 1.0
        degradation_score = max(0, 100 * (1 - min(degradation, 1)))

        # Weighted combination
        robustness = (
            return_stability * 0.4 +
            positive_score * 0.3 +
            degradation_score * 0.3
        )

        return float(robustness)

    def _calculate_param_sensitivity(self, returns: List[float]) -> float:
        """
        Calculate sensitivity score for a parameter.

        Lower sensitivity = more stable.
        """
        returns_array = np.array(returns)

        if len(returns_array) < 2:
            return 0.0

        # Coefficient of variation
        cv = np.std(returns_array) / abs(np.mean(returns_array)) \
            if np.mean(returns_array) != 0 else 1.0

        # Range relative to mean
        range_ratio = (np.max(returns_array) - np.min(returns_array)) / abs(np.mean(returns_array)) \
            if np.mean(returns_array) != 0 else 1.0

        # Combined sensitivity (lower is better)
        sensitivity = (cv + range_ratio) / 2

        return float(sensitivity)

    def _generate_recommendations(
        self,
        mc_result: MonteCarloResult,
        risk_assessment: Optional[RiskAssessment],
    ) -> List[str]:
        """Generate recommendations based on MC results."""
        recommendations = []

        # Check probability of profit
        if mc_result.prob_profit < 0.6:
            recommendations.append(
                f"Low probability of profit ({mc_result.prob_profit:.1%}). "
                "Consider: 1) Improving entry/exit logic, 2) Better filters, 3) Risk management"
            )

        # Check tail risk
        if mc_result.value_at_risk < -20:
            recommendations.append(
                f"High tail risk (VaR: {mc_result.value_at_risk:.1f}%). "
                "Consider: 1) Position sizing, 2) Stop losses, 3) Maximum drawdown limits"
            )

        # Check distribution shape
        if mc_result.skewness < -0.5:
            recommendations.append(
                "Negatively skewed distribution (more frequent small wins, rare large losses). "
                "Ensure proper risk controls are in place."
            )

        # Positive signals
        if mc_result.prob_profit > 0.7 and mc_result.value_at_risk > -10:
            recommendations.append(
                "Strategy shows good robustness in Monte Carlo testing. "
                "Consider: 1) Position sizing optimization, 2) Live testing, 3) Risk capital allocation"
            )

        if not recommendations:
            recommendations.append("Monte Carlo results appear reasonable. Proceed with caution and proper risk management.")

        return recommendations

    def export_results(
        self,
        result: MonteCarloResult,
        output_dir: str,
    ) -> None:
        """
        Export Monte Carlo results to disk.

        Args:
            result: Monte Carlo result
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_path / "monte_carlo_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_summary_dict(), f, indent=2)

        # Save simulation details
        simulations_data = []
        for sim in result.simulations:
            simulations_data.append({
                'simulation_id': sim.simulation_id,
                'total_return': sim.total_return,
                'sharpe_ratio': sim.sharpe_ratio,
                'max_drawdown': sim.max_drawdown,
                'win_rate': sim.win_rate,
            })

        sims_df = pd.DataFrame(simulations_data)
        sims_df.to_csv(output_path / "simulations.csv", index=False)

        print(f"Monte Carlo results exported to {output_dir}")

    def generate_report(self, result: MonteCarloResult) -> str:
        """
        Generate text report.

        Args:
            result: Monte Carlo result

        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 80)
        report.append(f"MONTE CARLO SIMULATION REPORT: {result.strategy_id}")
        report.append("=" * 80)
        report.append("")

        # Configuration
        report.append("Configuration:")
        report.append(f"  Simulations: {result.n_simulations}")
        report.append(f"  Method: {result.config.resampling_method.value}")
        report.append(f"  Confidence Level: {result.config.confidence_level:.0%}")
        report.append("")

        # Returns
        report.append("Return Distribution:")
        report.append(f"  Mean: {result.mean_return:.2f}%")
        report.append(f"  Median: {result.median_return:.2f}%")
        report.append(f"  Std Dev: {result.std_return:.2f}%")
        report.append(f"  95% CI: [{result.return_ci_lower:.2f}, {result.return_ci_upper:.2f}]%")
        report.append("")

        # Risk metrics
        report.append("Risk Metrics:")
        report.append(f"  VaR (95%): {result.value_at_risk:.2f}%")
        report.append(f"  CVaR (95%): {result.conditional_var:.2f}%")
        report.append(f"  Worst Case (1%): {result.percentile_1:.2f}%")
        report.append(f"  Best Case (99%): {result.percentile_99:.2f}%")
        report.append("")

        # Probabilities
        report.append("Probabilities:")
        report.append(f"  Profit: {result.prob_profit:.1%}")
        report.append(f"  Loss > 10%: {result.prob_loss_exceeds_10pct:.1%}")
        report.append(f"  Loss > 20%: {result.prob_loss_exceeds_20pct:.1%}")
        report.append("")

        # Distribution shape
        report.append("Distribution Shape:")
        report.append(f"  Skewness: {result.skewness:.3f}")
        report.append(f"  Kurtosis: {result.kurtosis:.3f}")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

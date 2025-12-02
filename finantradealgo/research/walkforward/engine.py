"""
Walk-Forward Analysis Engine.

High-level API for running complete walk-forward analysis including
optimization, validation, analysis, and visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

import pandas as pd

from finantradealgo.research.walkforward.models import (
    WalkForwardConfig,
    WalkForwardResult,
    WindowType,
    OptimizationMetric,
)
from finantradealgo.research.walkforward.optimizer import WalkForwardOptimizer
from finantradealgo.research.walkforward.validator import OutOfSampleValidator, ValidationReport
from finantradealgo.research.walkforward.analysis import WalkForwardAnalyzer, EfficiencyMetrics
from finantradealgo.research.walkforward.visualization import WalkForwardVisualizer


class WalkForwardEngine:
    """
    Complete walk-forward analysis engine.

    Orchestrates all components: optimization, validation, analysis, and visualization.
    Provides simple API for running anchored or rolling walk-forward tests.

    Example:
        >>> from finantradealgo.research.walkforward import WalkForwardEngine, WalkForwardConfig
        >>>
        >>> # Configure walk-forward
        >>> config = WalkForwardConfig(
        ...     in_sample_periods=12,  # 12 months
        ...     out_sample_periods=3,   # 3 months
        ...     window_type="rolling",
        ...     period_unit="M"
        ... )
        >>>
        >>> # Create engine
        >>> engine = WalkForwardEngine(config)
        >>>
        >>> # Run analysis
        >>> result = engine.run(
        ...     strategy_id="my_strategy",
        ...     param_grid={"fast_ma": [10, 20], "slow_ma": [50, 100]},
        ...     data_df=price_data,
        ...     backtest_function=my_backtest_func
        ... )
        >>>
        >>> # Get validation report
        >>> report = engine.validate(result)
        >>> print(f"Status: {report.status}, Score: {report.overall_score:.1f}")
        >>>
        >>> # Generate visualizations
        >>> engine.plot_performance(result, save_path="results/wf_performance.html")
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        validator_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize walk-forward engine.

        Args:
            config: Walk-forward configuration
            validator_config: Optional validator configuration overrides
        """
        self.config = config
        self.optimizer = WalkForwardOptimizer(config)

        # Initialize validator with custom config if provided
        if validator_config:
            self.validator = OutOfSampleValidator(**validator_config)
        else:
            self.validator = OutOfSampleValidator()

        self.analyzer = WalkForwardAnalyzer()
        self.visualizer = WalkForwardVisualizer()

    def run(
        self,
        strategy_id: str,
        param_grid: Dict[str, List[Any]],
        data_df: pd.DataFrame,
        backtest_function: Callable,
        auto_validate: bool = True,
    ) -> WalkForwardResult:
        """
        Run complete walk-forward analysis.

        Args:
            strategy_id: Strategy identifier
            param_grid: Parameter grid for optimization
                Example: {"fast_ma": [10, 20, 30], "slow_ma": [50, 100, 200]}
            data_df: Price data with datetime index
            backtest_function: Function(data_df, params) -> (trades_df, metrics_dict)
                Must return:
                - trades_df: DataFrame with trade results
                - metrics: Dict with keys like "sharpe_ratio", "total_return", etc.
            auto_validate: Automatically run validation after optimization

        Returns:
            WalkForwardResult with all windows and metrics
        """
        print(f"Starting walk-forward analysis for {strategy_id}...")
        print(f"Config: {self.config.window_type.value} window, "
              f"{self.config.in_sample_periods}{self.config.period_unit} IS, "
              f"{self.config.out_sample_periods}{self.config.period_unit} OOS")

        # Run optimization
        result = self.optimizer.run_walk_forward(
            strategy_id=strategy_id,
            param_grid=param_grid,
            data_df=data_df,
            backtest_function=backtest_function,
        )

        print(f"\nWalk-forward completed: {result.total_windows} windows processed")
        print(f"Duration: {result.total_duration_seconds:.1f} seconds")
        print(f"Avg OOS Sharpe: {result.avg_oos_sharpe:.2f}")
        print(f"Consistency Score: {result.consistency_score:.1f}/100")

        # Auto-validate if requested
        if auto_validate:
            validation = self.validate(result)
            print(f"\nValidation Status: {validation.status.value.upper()}")
            print(f"Overall Score: {validation.overall_score:.1f}/100")

        return result

    def run_anchored(
        self,
        strategy_id: str,
        param_grid: Dict[str, List[Any]],
        data_df: pd.DataFrame,
        backtest_function: Callable,
        in_sample_periods: int = 12,
        out_sample_periods: int = 3,
        period_unit: str = "M",
    ) -> WalkForwardResult:
        """
        Run anchored (expanding) walk-forward analysis.

        Training window expands from a fixed start point.
        Test window is fixed size and rolls forward.

        Args:
            strategy_id: Strategy identifier
            param_grid: Parameter grid for optimization
            data_df: Price data
            backtest_function: Backtest function
            in_sample_periods: Initial in-sample periods
            out_sample_periods: Out-of-sample periods (fixed)
            period_unit: Period unit (D/W/M/Q/Y)

        Returns:
            WalkForwardResult
        """
        # Create anchored config
        config = WalkForwardConfig(
            in_sample_periods=in_sample_periods,
            out_sample_periods=out_sample_periods,
            window_type=WindowType.ANCHORED,
            period_unit=period_unit,
            optimization_metric=self.config.optimization_metric,
            min_trades_per_period=self.config.min_trades_per_period,
            require_profitable_is=self.config.require_profitable_is,
        )

        # Create temporary engine with anchored config
        engine = WalkForwardEngine(config)
        return engine.run(strategy_id, param_grid, data_df, backtest_function)

    def run_rolling(
        self,
        strategy_id: str,
        param_grid: Dict[str, List[Any]],
        data_df: pd.DataFrame,
        backtest_function: Callable,
        in_sample_periods: int = 12,
        out_sample_periods: int = 3,
        period_unit: str = "M",
    ) -> WalkForwardResult:
        """
        Run rolling walk-forward analysis.

        Both training and test windows are fixed size and roll forward together.

        Args:
            strategy_id: Strategy identifier
            param_grid: Parameter grid for optimization
            data_df: Price data
            backtest_function: Backtest function
            in_sample_periods: In-sample periods (fixed)
            out_sample_periods: Out-of-sample periods (fixed)
            period_unit: Period unit (D/W/M/Q/Y)

        Returns:
            WalkForwardResult
        """
        # Create rolling config
        config = WalkForwardConfig(
            in_sample_periods=in_sample_periods,
            out_sample_periods=out_sample_periods,
            window_type=WindowType.ROLLING,
            period_unit=period_unit,
            optimization_metric=self.config.optimization_metric,
            min_trades_per_period=self.config.min_trades_per_period,
            require_profitable_is=self.config.require_profitable_is,
        )

        # Create temporary engine with rolling config
        engine = WalkForwardEngine(config)
        return engine.run(strategy_id, param_grid, data_df, backtest_function)

    def validate(self, result: WalkForwardResult) -> ValidationReport:
        """
        Validate walk-forward result.

        Checks for:
        - Excessive IS-to-OOS degradation (overfitting)
        - Poor OOS consistency
        - Low absolute OOS performance
        - Parameter instability

        Args:
            result: Walk-forward result to validate

        Returns:
            ValidationReport with status and recommendations
        """
        return self.validator.validate(result)

    def analyze(self, result: WalkForwardResult) -> Dict[str, Any]:
        """
        Perform detailed analysis of walk-forward result.

        Calculates:
        - Walk-forward efficiency (WFE) metrics
        - Regime sensitivity
        - Parameter drift
        - Window patterns

        Args:
            result: Walk-forward result

        Returns:
            Dictionary with comprehensive analysis
        """
        return self.analyzer.generate_summary_report(result)

    def calculate_wfe(self, result: WalkForwardResult) -> EfficiencyMetrics:
        """
        Calculate Walk-Forward Efficiency (WFE) metrics.

        WFE measures how well the strategy translates from IS to OOS:
        - WFE = OOS Performance / IS Performance
        - Values close to 1.0 indicate robust performance
        - Values << 1.0 suggest overfitting

        Args:
            result: Walk-forward result

        Returns:
            EfficiencyMetrics with detailed efficiency analysis
        """
        return self.analyzer.calculate_efficiency_metrics(result)

    def detect_overfitting(self, result: WalkForwardResult) -> Dict[str, Any]:
        """
        Detect signs of overfitting in walk-forward result.

        Analyzes:
        - IS-to-OOS degradation patterns
        - Parameter stability
        - Performance consistency
        - Regime sensitivity

        Args:
            result: Walk-forward result

        Returns:
            Dictionary with overfitting analysis and risk assessment
        """
        efficiency = self.analyzer.calculate_efficiency_metrics(result)
        validation = self.validator.validate(result)
        param_drift = self.analyzer.analyze_parameter_drift(result)
        regime = self.analyzer.identify_regime_sensitivity(result)

        # Calculate overfitting risk score (0-100, higher = more risk)
        degradation_risk = abs(result.avg_sharpe_degradation) * 100
        efficiency_risk = max(0, (1.0 - efficiency.sharpe_efficiency) * 100)
        stability_risk = 100 - result.param_stability_score
        consistency_risk = 100 - result.consistency_score

        overall_risk = (
            degradation_risk * 0.3 +
            efficiency_risk * 0.25 +
            stability_risk * 0.25 +
            consistency_risk * 0.2
        )

        return {
            "overfitting_risk_score": round(overall_risk, 1),
            "risk_level": self._classify_risk(overall_risk),
            "components": {
                "degradation_risk": round(degradation_risk, 1),
                "efficiency_risk": round(efficiency_risk, 1),
                "stability_risk": round(stability_risk, 1),
                "consistency_risk": round(consistency_risk, 1),
            },
            "efficiency_metrics": efficiency.to_dict(),
            "validation_status": validation.status.value,
            "parameter_drift": param_drift,
            "regime_sensitivity": regime,
            "recommendations": validation.recommendations,
        }

    def _classify_risk(self, risk_score: float) -> str:
        """Classify overfitting risk level."""
        if risk_score < 25:
            return "low"
        elif risk_score < 50:
            return "moderate"
        elif risk_score < 75:
            return "high"
        else:
            return "severe"

    def compare_strategies(
        self,
        results: List[WalkForwardResult],
        metric: str = "consistency_score",
    ) -> pd.DataFrame:
        """
        Compare multiple walk-forward results.

        Args:
            results: List of walk-forward results to compare
            metric: Primary comparison metric

        Returns:
            DataFrame with comparison metrics, sorted by specified metric
        """
        return self.analyzer.compare_strategies(results)

    # Visualization methods

    def plot_performance(
        self,
        result: WalkForwardResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot IS vs OOS performance comparison.

        Shows:
        - Sharpe ratio: IS vs OOS
        - Total return: IS vs OOS
        - Degradation over time
        - Win rate comparison

        Args:
            result: Walk-forward result
            save_path: Optional path to save chart (HTML format)
        """
        fig = self.visualizer.plot_performance_comparison(result)

        if save_path:
            self.visualizer.save(fig, save_path)
        else:
            fig.show()

    def plot_equity_curve(
        self,
        result: WalkForwardResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot combined OOS equity curve.

        Args:
            result: Walk-forward result
            save_path: Optional path to save chart
        """
        fig = self.visualizer.plot_equity_curve(result)

        if save_path:
            self.visualizer.save(fig, save_path)
        else:
            fig.show()

    def plot_parameter_stability(
        self,
        result: WalkForwardResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot parameter evolution across windows.

        Args:
            result: Walk-forward result
            save_path: Optional path to save chart
        """
        fig = self.visualizer.plot_parameter_stability(result)

        if save_path:
            self.visualizer.save(fig, save_path)
        else:
            fig.show()

    def plot_robustness_dashboard(
        self,
        result: WalkForwardResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive robustness dashboard.

        Shows multiple panels with key metrics:
        - IS vs OOS performance
        - Consistency by window
        - Combined equity
        - Parameter stability
        - Returns distribution
        - Efficiency metrics

        Args:
            result: Walk-forward result
            save_path: Optional path to save chart
        """
        fig = self.visualizer.plot_robustness_dashboard(result)

        if save_path:
            self.visualizer.save(fig, save_path)
        else:
            fig.show()

    def plot_degradation(
        self,
        result: WalkForwardResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot IS-to-OOS degradation distribution.

        Args:
            result: Walk-forward result
            save_path: Optional path to save chart
        """
        fig = self.visualizer.plot_degradation_distribution(result)

        if save_path:
            self.visualizer.save(fig, save_path)
        else:
            fig.show()

    def plot_comparison(
        self,
        results: List[WalkForwardResult],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare multiple strategies visually.

        Args:
            results: List of walk-forward results
            save_path: Optional path to save chart
        """
        fig = self.visualizer.plot_comparison(results)

        if save_path:
            self.visualizer.save(fig, save_path)
        else:
            fig.show()

    # Export and reporting

    def export_results(
        self,
        result: WalkForwardResult,
        output_dir: str,
        include_plots: bool = True,
    ) -> None:
        """
        Export complete walk-forward results.

        Creates:
        - summary.json: Overall metrics
        - windows.json: Window-by-window results
        - equity_curve.csv: Combined OOS equity
        - analysis.json: Detailed analysis
        - validation.json: Validation report
        - [Optional] Various plots (HTML)

        Args:
            result: Walk-forward result
            output_dir: Output directory
            include_plots: Whether to generate and save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting results to {output_dir}...")

        # Save basic results
        self.optimizer.save_result(result, output_path)

        # Save analysis
        analysis = self.analyze(result)
        import json
        with open(output_path / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Save validation
        validation = self.validate(result)
        with open(output_path / "validation.json", "w") as f:
            json.dump(validation.to_dict(), f, indent=2)

        # Save overfitting analysis
        overfitting = self.detect_overfitting(result)
        with open(output_path / "overfitting_analysis.json", "w") as f:
            json.dump(overfitting, f, indent=2, default=str)

        # Generate plots if requested
        if include_plots:
            print("Generating visualizations...")
            plots_dir = output_path / "plots"
            plots_dir.mkdir(exist_ok=True)

            self.plot_performance(result, str(plots_dir / "performance.html"))
            self.plot_equity_curve(result, str(plots_dir / "equity_curve.html"))
            self.plot_parameter_stability(result, str(plots_dir / "parameter_stability.html"))
            self.plot_degradation(result, str(plots_dir / "degradation.html"))
            self.plot_robustness_dashboard(result, str(plots_dir / "dashboard.html"))

        print(f"Export completed: {output_dir}")

    def generate_report(self, result: WalkForwardResult) -> str:
        """
        Generate text report summary.

        Args:
            result: Walk-forward result

        Returns:
            Formatted text report
        """
        validation = self.validate(result)
        efficiency = self.calculate_wfe(result)
        overfitting = self.detect_overfitting(result)

        report = []
        report.append("=" * 80)
        report.append(f"WALK-FORWARD ANALYSIS REPORT: {result.strategy_id}")
        report.append("=" * 80)
        report.append("")

        # Configuration
        report.append("Configuration:")
        report.append(f"  Window Type: {result.config.window_type.value}")
        report.append(f"  In-Sample: {result.config.in_sample_periods}{result.config.period_unit}")
        report.append(f"  Out-of-Sample: {result.config.out_sample_periods}{result.config.period_unit}")
        report.append(f"  Total Windows: {result.total_windows}")
        report.append("")

        # Performance
        report.append("Performance Metrics:")
        report.append(f"  Avg IS Sharpe: {result.avg_is_sharpe:.2f}")
        report.append(f"  Avg OOS Sharpe: {result.avg_oos_sharpe:.2f}")
        report.append(f"  Avg OOS Return: {result.avg_oos_return:.2f}%")
        report.append(f"  OOS Win Rate: {result.oos_win_rate:.1%}")
        report.append("")

        # Efficiency
        report.append("Walk-Forward Efficiency:")
        report.append(f"  Sharpe Efficiency: {efficiency.sharpe_efficiency:.2f}")
        report.append(f"  Return Efficiency: {efficiency.return_efficiency:.2f}")
        report.append(f"  Sharpe Correlation: {efficiency.sharpe_correlation:.2f}")
        report.append("")

        # Degradation
        report.append("Degradation Analysis:")
        report.append(f"  Avg Sharpe Degradation: {result.avg_sharpe_degradation:.1%}")
        report.append(f"  Avg Return Degradation: {result.avg_return_degradation:.1%}")
        report.append("")

        # Stability
        report.append("Stability Metrics:")
        report.append(f"  Parameter Stability: {result.param_stability_score:.1f}/100")
        report.append(f"  Consistency Score: {result.consistency_score:.1f}/100")
        report.append("")

        # Overfitting
        report.append("Overfitting Risk:")
        report.append(f"  Risk Score: {overfitting['overfitting_risk_score']:.1f}/100")
        report.append(f"  Risk Level: {overfitting['risk_level'].upper()}")
        report.append("")

        # Validation
        report.append("Validation:")
        report.append(f"  Status: {validation.status.value.upper()}")
        report.append(f"  Overall Score: {validation.overall_score:.1f}/100")
        report.append(f"  Passed Checks: {len(validation.passed_checks)}")
        report.append(f"  Failed Checks: {len(validation.failed_checks)}")
        report.append(f"  Warnings: {len(validation.warnings)}")
        report.append("")

        # Recommendations
        if validation.recommendations:
            report.append("Recommendations:")
            for i, rec in enumerate(validation.recommendations, 1):
                report.append(f"  {i}. {rec}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

"""
Ensemble Strategy Report Generator.

Creates comprehensive reports for ensemble strategy backtests.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from finantradealgo.research.ensemble.backtest import EnsembleBacktestResult
from finantradealgo.research.reporting.base import (
    Report,
    ReportGenerator,
    ReportSection,
)


class EnsembleReportGenerator(ReportGenerator):
    """
    Generate reports for ensemble strategy backtests.

    Creates a comprehensive report including:
    - Ensemble configuration and methodology
    - Overall ensemble performance
    - Component performance comparison
    - Weight evolution (for weighted ensembles)
    - Bandit statistics (for bandit ensembles)
    - Recommendations for optimization
    """

    def generate(
        self,
        backtest_result: EnsembleBacktestResult,
        ensemble_type: str,
        symbol: str,
        timeframe: str,
        component_names: Optional[list[str]] = None,
    ) -> Report:
        """
        Generate ensemble backtest report.

        Args:
            backtest_result: Ensemble backtest results
            ensemble_type: Type of ensemble ("weighted" or "bandit")
            symbol: Trading symbol
            timeframe: Trading timeframe
            component_names: Optional list of component strategy names

        Returns:
            Generated report
        """
        # Create report
        report = Report(
            title=f"Ensemble Strategy Report: {ensemble_type.title()}",
            description=f"{ensemble_type.title()} ensemble backtest for {symbol}/{timeframe}",
            metadata={
                "ensemble_type": ensemble_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "n_components": len(backtest_result.component_metrics),
            },
        )

        # Section 1: Overview
        report.add_section(self._create_overview_section(
            backtest_result,
            ensemble_type,
            symbol,
            timeframe,
        ))

        # Section 2: Ensemble Performance
        report.add_section(self._create_ensemble_performance_section(backtest_result))

        # Section 3: Component Comparison
        report.add_section(self._create_component_comparison_section(backtest_result))

        # Section 4: Ensemble-specific analysis
        if ensemble_type == "weighted" and backtest_result.weight_history is not None:
            report.add_section(self._create_weights_section(backtest_result))
        elif ensemble_type == "bandit" and backtest_result.bandit_stats is not None:
            report.add_section(self._create_bandit_stats_section(backtest_result))

        # Section 5: Recommendations
        report.add_section(self._create_ensemble_recommendations_section(
            backtest_result,
            ensemble_type,
        ))

        return report

    def _create_overview_section(
        self,
        result: EnsembleBacktestResult,
        ensemble_type: str,
        symbol: str,
        timeframe: str,
    ) -> ReportSection:
        """Create overview section."""
        n_components = len(result.component_metrics)

        if ensemble_type == "weighted":
            methodology = """
**Weighted Ensemble Methodology**:
- Combines signals from all component strategies
- Uses weighted aggregation (weighted sum)
- Weights can be static or dynamically recomputed
- Trading decision based on weighted signal threshold
"""
        else:  # bandit
            methodology = """
**Multi-Armed Bandit Methodology**:
- Selects ONE component strategy at a time
- Balances exploration (trying new strategies) vs exploitation (using best)
- Adapts based on realized performance
- Updates arm statistics periodically
"""

        content = f"""
This report analyzes a **{ensemble_type}** ensemble strategy combining **{n_components}** component strategies on **{symbol}/{timeframe}**.

{methodology.strip()}

**Component Strategies**:
"""
        for i, row in result.component_metrics.iterrows():
            content += f"\n- {row['component']}"

        section = ReportSection(
            title="Overview",
            content=content.strip(),
        )

        return section

    def _create_ensemble_performance_section(self, result: EnsembleBacktestResult) -> ReportSection:
        """Create ensemble performance section."""
        metrics = result.ensemble_metrics

        content = f"""
Overall performance of the ensemble strategy.

**Key Metrics**:
- Cumulative Return: {metrics.get('cum_return', 0.0):.4f} ({metrics.get('cum_return', 0.0) * 100:.2f}%)
- Sharpe Ratio: {metrics.get('sharpe', 0.0):.4f}
- Max Drawdown: {metrics.get('max_dd', 0.0):.4f} ({metrics.get('max_dd', 0.0) * 100:.2f}%)
- Trade Count: {metrics.get('trade_count', 0)}
- Win Rate: {metrics.get('win_rate', 0.0):.4f} ({metrics.get('win_rate', 0.0) * 100:.2f}%)
"""

        # Create metrics table
        metrics_df = pd.DataFrame([metrics])
        metrics_df = metrics_df.T
        metrics_df.columns = ["Value"]
        metrics_df.index.name = "Metric"
        metrics_df = metrics_df.reset_index()

        section = ReportSection(
            title="Ensemble Performance",
            content=content.strip(),
            data={"Ensemble Metrics": metrics_df},
        )

        return section

    def _create_component_comparison_section(self, result: EnsembleBacktestResult) -> ReportSection:
        """Create component comparison section."""
        content = "Performance comparison of individual component strategies."

        # Format component metrics
        comp_df = result.component_metrics.copy()

        # Round numeric columns
        for col in ["cum_return", "sharpe", "max_dd", "win_rate"]:
            if col in comp_df.columns:
                comp_df[col] = comp_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

        # Add vs Ensemble column
        ensemble_sharpe = result.ensemble_metrics.get("sharpe", 0.0)
        if "sharpe" in result.component_metrics.columns:
            comp_df["vs_Ensemble (Sharpe)"] = result.component_metrics["sharpe"].apply(
                lambda x: f"+{(x - ensemble_sharpe):.4f}" if x > ensemble_sharpe else f"{(x - ensemble_sharpe):.4f}"
            )

        section = ReportSection(
            title="Component Performance Comparison",
            content=content,
            data={"Component Metrics": comp_df},
        )

        return section

    def _create_weights_section(self, result: EnsembleBacktestResult) -> ReportSection:
        """Create weights section (for weighted ensembles)."""
        content = "Component weights in the weighted ensemble."

        weights_df = result.weight_history.copy() if result.weight_history is not None else None

        if weights_df is not None and not weights_df.empty:
            # Format weights as percentages
            if "weight" in weights_df.columns:
                weights_df["weight_pct"] = weights_df["weight"].apply(lambda x: f"{x * 100:.2f}%")

        section = ReportSection(
            title="Component Weights",
            content=content,
            data={"Final Weights": weights_df} if weights_df is not None else None,
        )

        return section

    def _create_bandit_stats_section(self, result: EnsembleBacktestResult) -> ReportSection:
        """Create bandit statistics section (for bandit ensembles)."""
        content = "Multi-armed bandit arm statistics and selection history."

        bandit_df = result.bandit_stats.copy() if result.bandit_stats is not None else None

        if bandit_df is not None and not bandit_df.empty:
            # Calculate selection percentage
            total_pulls = bandit_df["n_pulls"].sum()
            if total_pulls > 0:
                bandit_df["selection_pct"] = bandit_df["n_pulls"].apply(
                    lambda x: f"{x / total_pulls * 100:.2f}%"
                )

            # Format mean reward
            if "mean_reward" in bandit_df.columns:
                bandit_df["mean_reward_fmt"] = bandit_df["mean_reward"].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                )

        section = ReportSection(
            title="Bandit Statistics",
            content=content,
            data={"Arm Statistics": bandit_df} if bandit_df is not None else None,
        )

        return section

    def _create_ensemble_recommendations_section(
        self,
        result: EnsembleBacktestResult,
        ensemble_type: str,
    ) -> ReportSection:
        """Create recommendations section."""
        ensemble_sharpe = result.ensemble_metrics.get("sharpe", 0.0)

        # Find best component
        if not result.component_metrics.empty:
            best_component = result.component_metrics.loc[
                result.component_metrics["sharpe"].idxmax()
            ]
            best_comp_name = best_component["component"]
            best_comp_sharpe = best_component["sharpe"]

            # Compare ensemble to best component
            ensemble_improvement = ensemble_sharpe - best_comp_sharpe

            if ensemble_improvement > 0:
                comparison = f"The ensemble **outperforms** the best individual component ({best_comp_name}) by {ensemble_improvement:.4f} Sharpe points."
            else:
                comparison = f"The ensemble **underperforms** the best individual component ({best_comp_name}) by {abs(ensemble_improvement):.4f} Sharpe points."

        else:
            comparison = "Unable to compare ensemble to components."

        if ensemble_type == "weighted":
            specific_recommendations = """
**Weighted Ensemble Recommendations**:
1. Consider dynamic reweighting based on recent performance
2. Experiment with different weighting methods (Sharpe, inverse-vol, return)
3. Add weight constraints to prevent over-concentration
4. Monitor component correlation to avoid redundancy
"""
        else:  # bandit
            specific_recommendations = """
**Bandit Ensemble Recommendations**:
1. Tune exploration parameter (epsilon, UCB c parameter)
2. Experiment with different reward metrics (return, Sharpe, win rate)
3. Consider contextual bandits (use market regime features)
4. Increase update frequency for faster adaptation
"""

        content = f"""
{comparison}

**General Recommendations**:
1. Validate ensemble on out-of-sample data
2. Test across multiple symbols and timeframes
3. Monitor ensemble vs individual components over time
4. Consider adding/removing components based on performance
5. Document decision rationale and parameter choices

{specific_recommendations.strip()}

**Next Steps**:
- Run robustness checks on different time periods
- Test ensemble in paper trading environment
- Compare weighted vs bandit ensemble approaches
- Analyze regime-dependent performance
"""

        section = ReportSection(
            title="Recommendations",
            content=content.strip(),
        )

        return section

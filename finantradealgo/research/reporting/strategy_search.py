"""
Strategy Search Report Generator.

Creates comprehensive reports from strategy parameter search results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from finantradealgo.research.reporting.base import (
    Report,
    ReportGenerator,
    ReportSection,
)


class StrategySearchReportGenerator(ReportGenerator):
    """
    Generate reports for strategy parameter search jobs.

    Creates a comprehensive report including:
    - Job metadata and configuration
    - Top performers (by various metrics)
    - Parameter sensitivity analysis
    - Performance distribution
    - Recommendations
    """

    def generate(
        self,
        job_dir: Path,
        job_id: Optional[str] = None,
        top_n: int = 10,
    ) -> Report:
        """
        Generate strategy search report.

        Args:
            job_dir: Directory containing job results
            job_id: Job identifier (optional, inferred from dir if not provided)
            top_n: Number of top performers to highlight

        Returns:
            Generated report
        """
        job_dir = Path(job_dir)

        if job_id is None:
            job_id = job_dir.name

        # Load results
        results_path = job_dir / "results.parquet"
        if not results_path.exists():
            results_path = job_dir / "results.csv"

        if not results_path.exists():
            raise FileNotFoundError(f"No results file found in {job_dir}")

        if str(results_path).endswith(".parquet"):
            results_df = pd.read_parquet(results_path)
        else:
            results_df = pd.read_csv(results_path)

        # Load metadata
        import json
        meta_path = job_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        # Create report
        report = Report(
            title=f"Strategy Search Report: {job_id}",
            description=f"Parameter search results for {meta.get('strategy', 'unknown')} strategy on {meta.get('symbol', 'unknown')}/{meta.get('timeframe', 'unknown')}",
            metadata={
                "job_id": job_id,
                "strategy": meta.get("strategy"),
                "symbol": meta.get("symbol"),
                "timeframe": meta.get("timeframe"),
                "n_samples": meta.get("n_samples"),
                "search_type": meta.get("search_type"),
                "git_sha": meta.get("git_sha"),
            },
        )

        # Section 1: Job Overview
        report.add_section(self._create_overview_section(results_df, meta))

        # Section 2: Top Performers
        report.add_section(self._create_top_performers_section(results_df, top_n))

        # Section 3: Performance Distribution
        report.add_section(self._create_distribution_section(results_df))

        # Section 4: Parameter Analysis
        report.add_section(self._create_parameter_analysis_section(results_df))

        # Section 5: Recommendations
        report.add_section(self._create_recommendations_section(results_df, top_n))

        return report

    def _create_overview_section(self, results_df: pd.DataFrame, meta: dict) -> ReportSection:
        """Create job overview section."""
        # Calculate summary statistics
        n_total = len(results_df)
        n_success = (results_df["status"] == "ok").sum() if "status" in results_df.columns else n_total
        n_errors = n_total - n_success

        # Mean/median metrics
        metrics_summary = {}
        for metric in ["sharpe", "cum_return", "max_drawdown", "trade_count", "win_rate"]:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                if len(values) > 0:
                    metrics_summary[metric] = {
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }

        content = f"""
This report summarizes the results of a parameter search for the **{meta.get('strategy', 'unknown')}** strategy.

**Job Configuration**:
- Symbol: {meta.get('symbol', 'N/A')}
- Timeframe: {meta.get('timeframe', 'N/A')}
- Search Type: {meta.get('search_type', 'N/A')}
- Samples: {meta.get('n_samples', 'N/A')}
- Created: {meta.get('created_at', 'N/A')}

**Execution Summary**:
- Total Evaluations: {n_total}
- Successful: {n_success}
- Errors: {n_errors}
- Success Rate: {n_success / n_total * 100:.1f}%
"""

        # Create metrics summary table
        if metrics_summary:
            metrics_df = pd.DataFrame(metrics_summary).T
            metrics_df.index.name = "Metric"
            metrics_df = metrics_df.reset_index()

        section = ReportSection(
            title="Job Overview",
            content=content.strip(),
            data={"Performance Summary Statistics": metrics_df} if metrics_summary else None,
        )

        return section

    def _create_top_performers_section(self, results_df: pd.DataFrame, top_n: int) -> ReportSection:
        """Create top performers section."""
        content = f"Top {top_n} parameter sets ranked by Sharpe ratio."

        # Get top performers by Sharpe
        top_sharpe = results_df.nlargest(top_n, "sharpe") if "sharpe" in results_df.columns else results_df.head(top_n)

        # Select relevant columns
        display_cols = ["sharpe", "cum_return", "max_drawdown", "trade_count", "win_rate"]
        display_cols = [col for col in display_cols if col in top_sharpe.columns]

        # Add parameter columns
        param_cols = [col for col in top_sharpe.columns if col.startswith("param_")]
        display_df = top_sharpe[param_cols + display_cols].copy()

        # Format metrics
        for col in display_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

        section = ReportSection(
            title="Top Performers",
            content=content,
            data={"Top Parameter Sets (by Sharpe)": display_df},
        )

        return section

    def _create_distribution_section(self, results_df: pd.DataFrame) -> ReportSection:
        """Create performance distribution section."""
        content = "Distribution of performance metrics across all parameter sets."

        # Calculate quartiles for key metrics
        metrics = ["sharpe", "cum_return", "max_drawdown", "win_rate"]
        metrics = [m for m in metrics if m in results_df.columns]

        quartiles_data = []
        for metric in metrics:
            values = results_df[metric].dropna()
            if len(values) > 0:
                quartiles_data.append({
                    "Metric": metric,
                    "Min": float(values.min()),
                    "25%": float(values.quantile(0.25)),
                    "50% (Median)": float(values.quantile(0.50)),
                    "75%": float(values.quantile(0.75)),
                    "Max": float(values.max()),
                })

        quartiles_df = pd.DataFrame(quartiles_data) if quartiles_data else None

        section = ReportSection(
            title="Performance Distribution",
            content=content,
            data={"Quartile Analysis": quartiles_df} if quartiles_df is not None else None,
        )

        return section

    def _create_parameter_analysis_section(self, results_df: pd.DataFrame) -> ReportSection:
        """Create parameter sensitivity analysis section."""
        content = "Analysis of parameter impact on performance."

        # Get parameter columns
        param_cols = [col for col in results_df.columns if col.startswith("param_")]

        if not param_cols:
            section = ReportSection(
                title="Parameter Analysis",
                content="No parameter columns found in results.",
            )
            return section

        # Calculate correlation between parameters and Sharpe
        if "sharpe" in results_df.columns:
            correlations = []
            for param in param_cols:
                # Try to convert to numeric
                try:
                    param_values = pd.to_numeric(results_df[param], errors='coerce')
                    sharpe_values = results_df["sharpe"]

                    # Remove NaN pairs
                    mask = ~(param_values.isna() | sharpe_values.isna())
                    if mask.sum() > 1:
                        corr = param_values[mask].corr(sharpe_values[mask])
                        correlations.append({
                            "Parameter": param.replace("param_", ""),
                            "Correlation with Sharpe": float(corr),
                        })
                except Exception:
                    pass

            corr_df = pd.DataFrame(correlations) if correlations else None

            if corr_df is not None:
                corr_df = corr_df.sort_values("Correlation with Sharpe", ascending=False)

        else:
            corr_df = None

        section = ReportSection(
            title="Parameter Analysis",
            content=content,
            data={"Parameter Sensitivity": corr_df} if corr_df is not None else None,
        )

        return section

    def _create_recommendations_section(self, results_df: pd.DataFrame, top_n: int) -> ReportSection:
        """Create recommendations section."""
        # Get best parameter set
        if "sharpe" in results_df.columns and len(results_df) > 0:
            best_row = results_df.loc[results_df["sharpe"].idxmax()]

            best_params = {}
            for col in results_df.columns:
                if col.startswith("param_"):
                    param_name = col.replace("param_", "")
                    best_params[param_name] = best_row[col]

            best_sharpe = best_row.get("sharpe", None)
            best_return = best_row.get("cum_return", None)

            content = f"""
Based on the parameter search results, the following recommendations are provided:

**Best Parameter Set** (Highest Sharpe Ratio):
- Sharpe Ratio: {best_sharpe:.4f}
- Cumulative Return: {best_return:.4f if pd.notna(best_return) else 'N/A'}

**Parameters**:
"""
            for param, value in best_params.items():
                content += f"\n- {param}: {value}"

            content += f"""

**Next Steps**:
1. Validate best parameter set on out-of-sample data
2. Run robustness checks (different symbols, timeframes)
3. Consider top {top_n} parameter sets for ensemble
4. Monitor performance in live paper trading
5. Document parameter rationale and strategy logic
"""

        else:
            content = "Unable to generate recommendations: insufficient data."

        section = ReportSection(
            title="Recommendations",
            content=content.strip(),
        )

        return section

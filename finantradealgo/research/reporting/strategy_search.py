"""
Strategy Search Report Generator.

Creates comprehensive reports from strategy parameter search results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype

from finantradealgo.research.visualization.charts import ChartConfig, ChartType, create_chart, save_chart
from finantradealgo.research.reporting.base import (
    Report,
    ReportGenerator,
    ReportProfile,
    ReportSection,
)


@dataclass
class StrategySearchReportGenerator(ReportGenerator):
    """
    Generate reports for strategy parameter search jobs.
    """

    def generate(
        self,
        job_dir: Path,
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
        profile: Optional[ReportProfile] = None,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        top_n: int = 10,
    ) -> Report:
        """
        Generate strategy search report.

        Args:
            job_dir: Directory containing job results
            job_id: Job identifier (optional, inferred from dir if not provided)
            run_id: Execution identifier (e.g., research run or live session id)
            profile: Execution profile (research/live)
            strategy_id: Strategy identifier (rule, ml, trend_continuation, etc.)
            symbol: Trading symbol
            timeframe: Bar timeframe
            top_n: Number of top performers to highlight

        Returns:
            Generated report
        """
        job_dir = Path(job_dir)
        job_id = job_id or job_dir.name

        results_df, meta = self._load_results_and_meta(job_dir)
        meta = dict(meta or {})
        meta.setdefault("job_id", job_id)

        profile = profile or meta.get("profile")
        strategy_id = strategy_id or meta.get("strategy")
        symbol = symbol or meta.get("symbol")
        timeframe = timeframe or meta.get("timeframe")

        # Top-level metrics for fast access
        status_col = "status" if "status" in results_df.columns else None
        ok_df = results_df[results_df[status_col] != "error"] if status_col else results_df
        n_total = len(results_df)
        n_success = len(ok_df)
        n_errors = n_total - n_success

        best_sharpe = None
        if "sharpe" in ok_df.columns and not ok_df["sharpe"].isna().all():
            best_sharpe = float(ok_df["sharpe"].max())
        best_return = None
        for return_key in ["cum_return", "return", "total_return"]:
            if return_key in ok_df.columns and not ok_df[return_key].isna().all():
                best_return = float(ok_df[return_key].max())
                break

        report_metrics: Dict[str, Any] = {
            "best_sharpe": best_sharpe,
            "best_cum_return": best_return,
            "samples_total": n_total,
            "samples_ok": n_success,
            "samples_failed": n_errors,
        }
        report_metrics = {k: v for k, v in report_metrics.items() if v is not None}

        report = Report(
            title=f"Strategy Search Report: {job_id}",
            description=(
                f"Parameter search results for {strategy_id or 'unknown'} strategy on "
                f"{symbol or 'unknown'}/{timeframe or 'unknown'}"
            ),
            job_id=meta.get("job_id", job_id),
            run_id=run_id or meta.get("run_id"),
            profile=profile if isinstance(profile, ReportProfile) else (
                ReportProfile(profile) if profile in {p.value for p in ReportProfile} else profile
            ),
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            metrics=report_metrics,
            artifacts={
                "results_parquet": str(job_dir / "results.parquet"),
                "results_csv": str(job_dir / "results.csv"),
                "meta_json": str(job_dir / "meta.json"),
            },
            metadata={
                "n_samples": meta.get("n_samples"),
                "search_type": meta.get("search_type"),
                "git_sha": meta.get("git_sha"),
            },
        )

        # Ordered sections
        report.add_section(self._create_overview_section(results_df, meta))
        report.add_section(self._create_top_performers_section(results_df, top_n))
        report.add_section(self._create_distribution_section(results_df))
        report.add_section(self._create_parameter_analysis_section(results_df))
        report.add_section(self._create_recommendations_section(results_df, top_n))

        # Visualization: parameter sensitivity heatmap (if possible)
        heatmap_section = self._create_heatmap_section(results_df, job_dir)
        if heatmap_section:
            report.add_section(heatmap_section)
            report.artifacts["heatmap_html"] = str((job_dir / "param_heatmap.html").as_posix())

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_results_and_meta(self, job_dir: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Load results (parquet/csv) and meta.json if present."""
        results_path = job_dir / "results.parquet"
        if not results_path.exists():
            results_path = job_dir / "results.csv"

        if not results_path.exists():
            raise FileNotFoundError(f"No results file found in {job_dir}")

        if results_path.suffix == ".parquet":
            results_df = pd.read_parquet(results_path)
        else:
            results_df = pd.read_csv(results_path)

        meta: Dict[str, Any] = {}
        meta_path = job_dir / "meta.json"
        if meta_path.exists():
            import json

            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        return results_df, meta

    def _create_overview_section(self, results_df: pd.DataFrame, meta: dict) -> ReportSection:
        """Create job overview section."""
        n_total = len(results_df)
        n_success = (results_df["status"] == "ok").sum() if "status" in results_df.columns else n_total
        n_errors = n_total - n_success
        success_rate = (n_success / n_total * 100) if n_total > 0 else 0.0

        metrics_summary: Dict[str, Dict[str, float]] = {}
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

        metrics_df = None
        if metrics_summary:
            metrics_df = pd.DataFrame(metrics_summary).T
            metrics_df.index.name = "Metric"
            metrics_df = metrics_df.reset_index()

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
- Success Rate: {success_rate:.1f}%
"""

        return ReportSection(
            title="Job Overview",
            content=content.strip(),
            metrics={
                "evaluations": n_total,
                "success_rate": round(success_rate, 2),
            },
            data={"Performance Summary Statistics": metrics_df} if metrics_df is not None else None,
        )

    def _create_top_performers_section(self, results_df: pd.DataFrame, top_n: int) -> ReportSection:
        """Create top performers section."""
        content = f"Top {top_n} parameter sets ranked by Sharpe ratio."

        df = results_df.copy()
        if "status" in df.columns:
            df = df[df["status"] != "error"]

        if df.empty:
            return ReportSection(title="Top Performers", content="No successful evaluations available.")

        if "sharpe" in df.columns:
            top_sharpe = df.sort_values("sharpe", ascending=False).head(top_n)
        else:
            top_sharpe = df.head(top_n)

        display_cols = ["sharpe", "cum_return", "max_drawdown", "trade_count", "win_rate"]
        display_cols = [col for col in display_cols if col in top_sharpe.columns]
        param_cols = [col for col in top_sharpe.columns if col.startswith("param_")]

        display_df = top_sharpe[param_cols + display_cols].copy()
        for col in display_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

        return ReportSection(
            title="Top Performers",
            content=content,
            data={"Top Parameter Sets (by Sharpe)": display_df},
        )

    def _create_distribution_section(self, results_df: pd.DataFrame) -> ReportSection:
        """Create performance distribution section."""
        content = "Distribution of performance metrics across all parameter sets."

        metrics = ["sharpe", "cum_return", "max_drawdown", "win_rate", "trade_count"]
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

        return ReportSection(
            title="Performance Distribution",
            content=content,
            metrics={"metrics_analyzed": len(metrics)},
            data={"Quartile Analysis": quartiles_df} if quartiles_df is not None else None,
        )

    def _create_parameter_analysis_section(self, results_df: pd.DataFrame) -> ReportSection:
        """Create parameter sensitivity analysis section."""
        content = "Analysis of parameter impact on performance."

        param_cols = [col for col in results_df.columns if col.startswith("param_")]
        if not param_cols:
            return ReportSection(title="Parameter Analysis", content="No parameter columns found in results.")

        corr_df = None
        if "sharpe" in results_df.columns:
            correlations = []
            for param in param_cols:
                param_values = pd.to_numeric(results_df[param], errors="coerce")
                sharpe_values = results_df["sharpe"]
                mask = ~(param_values.isna() | sharpe_values.isna())
                if mask.sum() > 1:
                    corr = param_values[mask].corr(sharpe_values[mask])
                    correlations.append({
                        "Parameter": param.replace("param_", ""),
                        "Correlation with Sharpe": float(corr),
                    })
            if correlations:
                corr_df = pd.DataFrame(correlations).sort_values("Correlation with Sharpe", ascending=False)

        return ReportSection(
            title="Parameter Analysis",
            content=content,
            data={"Parameter Sensitivity": corr_df} if corr_df is not None else None,
        )

    def _create_recommendations_section(self, results_df: pd.DataFrame, top_n: int) -> ReportSection:
        """Create recommendations section."""
        df = results_df.copy()
        if "status" in df.columns:
            df = df[df["status"] != "error"]

        if "sharpe" in df.columns and len(df) > 0:
            best_row = df.loc[df["sharpe"].idxmax()]
            best_params = {}
            for col in df.columns:
                if col.startswith("param_"):
                    best_params[col.replace("param_", "")] = best_row[col]

            best_sharpe = best_row.get("sharpe", None)
            best_return = best_row.get("cum_return", None)

            low_trades = None
            if "trade_count" in df.columns and is_numeric_dtype(df["trade_count"]):
                low_trades = df["trade_count"].median()

            safe_best_return = (
                f"{best_return:.4f}" if (best_return is not None and pd.notna(best_return)) else "N/A"
            )

            content = f"""
Based on the parameter search results, the following recommendations are provided:

**Best Parameter Set** (Highest Sharpe Ratio):
- Sharpe Ratio: {best_sharpe:.4f}
- Cumulative Return: {safe_best_return}

**Parameters**:
"""
            for param, value in best_params.items():
                content += f"\n- {param}: {value}"

            content += f"""

**Next Steps**:
1. Validate best parameter set on out-of-sample data.
2. Run robustness checks across symbols/timeframes.
3. Consider top {top_n} parameter sets for ensemble trials.
4. Monitor performance in paper trading; be cautious with low-trade setups ({low_trades if low_trades is not None else 'n/a'} median trades).
5. Document parameter rationale and strategy logic.
"""
        else:
            content = "Unable to generate recommendations: insufficient data."

        return ReportSection(title="Recommendations", content=content.strip())

    def _create_heatmap_section(self, results_df: pd.DataFrame, job_dir: Path) -> Optional[ReportSection]:
        """Create a parameter heatmap section if 2D parameter grid available."""
        param_cols = [c for c in results_df.columns if c.startswith("param_")]
        if len(param_cols) < 2 or "sharpe" not in results_df.columns:
            return None

        # Take first two params for a simple 2D pivot
        p1, p2 = param_cols[:2]
        df = results_df.copy()
        df = df[df["status"] != "error"] if "status" in df.columns else df
        pivot = df.pivot_table(index=p1, columns=p2, values="sharpe", aggfunc="mean")
        if pivot.empty:
            return None

        fig = create_chart(
            chart_type=ChartType.HEATMAP,
            data=pivot,
            config=ChartConfig(
                title="Sharpe Heatmap",
                xaxis_title=p2.replace("param_", ""),
                yaxis_title=p1.replace("param_", ""),
                height=600,
                width=800,
            ),
        )
        output_path = job_dir / "param_heatmap.html"
        save_chart(fig, output_path, format="html")

        content = f"See interactive parameter heatmap: {output_path.name}"
        return ReportSection(title="Parameter Heatmap", content=content.strip())

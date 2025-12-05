"""
Backtest Report Generator.

Converts backtest outputs (metrics json/dict + equity/trades CSVs) into unified reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from finantradealgo.research.reporting.base import (
    Report,
    ReportFormat,
    ReportGenerator,
    ReportProfile,
    ReportSection,
)

MetricsInput = Union[Dict[str, Any], str, Path]


class BacktestReportGenerator(ReportGenerator):
    """Generate reports for single backtest runs."""

    def generate(
        self,
        metrics: MetricsInput,
        equity_curve_path: Path,
        trades_path: Path,
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
        profile: Optional[ReportProfile] = ReportProfile.RESEARCH,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Report:
        metrics_dict = self._load_metrics(metrics)

        equity_metrics: Dict[str, Any] = metrics_dict.get("equity_metrics") or metrics_dict.get("metrics") or {}
        trade_stats: Dict[str, Any] = metrics_dict.get("trade_stats") or {}
        risk_stats: Dict[str, Any] = metrics_dict.get("risk_stats") or {}

        # Meta fallbacks
        job_id = job_id or metrics_dict.get("job_id")
        run_id = run_id or metrics_dict.get("run_id")
        profile = profile or metrics_dict.get("profile")
        strategy_id = strategy_id or metrics_dict.get("strategy")
        symbol = symbol or metrics_dict.get("symbol")
        timeframe = timeframe or metrics_dict.get("timeframe")

        equity_df = self._load_csv(equity_curve_path)
        trades_df = self._load_csv(trades_path)

        report_metrics = {
            "sharpe": equity_metrics.get("sharpe"),
            "cum_return": equity_metrics.get("cum_return"),
            "max_drawdown": equity_metrics.get("max_drawdown"),
            "win_rate": trade_stats.get("win_rate"),
            "trade_count": trade_stats.get("trade_count"),
        }
        report_metrics = {k: v for k, v in report_metrics.items() if v is not None}

        artifacts = {
            "equity_curve_csv": str(equity_curve_path),
            "trades_csv": str(trades_path),
        }

        report = Report(
            title=f"Backtest Report: {strategy_id or 'strategy'}",
            description=(
                f"Backtest results for {strategy_id or 'strategy'} on {symbol or 'unknown'}/{timeframe or 'unknown'}"
            ),
            job_id=job_id,
            run_id=run_id,
            profile=profile if isinstance(profile, ReportProfile) else (
                ReportProfile(profile) if profile in {p.value for p in ReportProfile} else profile
            ),
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            metrics=report_metrics,
            artifacts=artifacts,
            metadata={
                k: v
                for k, v in metrics_dict.items()
                if k not in {"equity_metrics", "trade_stats", "risk_stats", "equity_curve", "trades"}
            },
        )

        report.add_section(self._build_overview_section(equity_metrics, trade_stats))
        report.add_section(self._build_equity_section(equity_metrics, equity_df, equity_curve_path))
        report.add_section(self._build_trades_section(trade_stats, trades_df, trades_path))

        risk_section = self._build_risk_section(risk_stats)
        if risk_section:
            report.add_section(risk_section)

        return report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load_metrics(self, metrics: MetricsInput) -> Dict[str, Any]:
        if isinstance(metrics, dict):
            return metrics
        metrics_path = Path(metrics)
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Required CSV not found: {path}")
        return pd.read_csv(path)

    def _build_overview_section(
        self,
        equity_metrics: Dict[str, Any],
        trade_stats: Dict[str, Any],
    ) -> ReportSection:
        content = f"""
Overall performance summary.

- Cumulative Return: {equity_metrics.get('cum_return', 'N/A')}
- Sharpe: {equity_metrics.get('sharpe', 'N/A')}
- Max Drawdown: {equity_metrics.get('max_drawdown', 'N/A')}
- Win Rate: {trade_stats.get('win_rate', 'N/A')}
- Trades: {trade_stats.get('trade_count', 'N/A')}
""".strip()

        metrics = {
            "cum_return": equity_metrics.get("cum_return"),
            "sharpe": equity_metrics.get("sharpe"),
            "max_drawdown": equity_metrics.get("max_drawdown"),
            "win_rate": trade_stats.get("win_rate"),
            "trade_count": trade_stats.get("trade_count"),
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}

        return ReportSection(
            title="Overview",
            content=content,
            metrics=metrics,
        )

    def _build_equity_section(
        self,
        equity_metrics: Dict[str, Any],
        equity_df: pd.DataFrame,
        equity_curve_path: Path,
    ) -> ReportSection:
        drawdown = equity_metrics.get("max_drawdown")
        final_equity = equity_metrics.get("final_equity")
        initial_cash = equity_metrics.get("initial_cash")
        content = f"""
Equity curve and drawdown summary. Source: {equity_curve_path}

- Final Equity: {final_equity}
- Initial Cash: {initial_cash}
- Max Drawdown: {drawdown}
""".strip()

        metrics = {
            "final_equity": final_equity,
            "initial_cash": initial_cash,
            "max_drawdown": drawdown,
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}

        data = {"Equity Curve": equity_df} if equity_df is not None else None

        return ReportSection(
            title="Equity & Drawdown",
            content=content,
            metrics=metrics,
            artifacts={"equity_curve_csv": str(equity_curve_path)},
            data=data,
        )

    def _build_trades_section(
        self,
        trade_stats: Dict[str, Any],
        trades_df: pd.DataFrame,
        trades_path: Path,
    ) -> ReportSection:
        pnl_stats: Dict[str, Any] = {}
        if "pnl" in trades_df.columns:
            pnl_series = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
            if not pnl_series.empty:
                pnl_stats = {
                    "min": float(pnl_series.min()),
                    "max": float(pnl_series.max()),
                    "median": float(pnl_series.median()),
                    "mean": float(pnl_series.mean()),
                    "std": float(pnl_series.std()),
                    "positive": int((pnl_series > 0).sum()),
                    "negative": int((pnl_series < 0).sum()),
                }

        side_col = None
        for candidate in ["side", "Side", "position", "direction"]:
            if candidate in trades_df.columns:
                side_col = candidate
                break

        side_breakdown = None
        if side_col:
            side_breakdown = []
            for side, grp in trades_df.groupby(side_col):
                pnl_series = pd.to_numeric(grp["pnl"], errors="coerce").dropna()
                side_breakdown.append(
                    {
                        "side": side,
                        "trades": len(grp),
                        "win_rate": float((pnl_series > 0).mean()) if len(pnl_series) else 0.0,
                        "avg_pnl": float(pnl_series.mean()) if len(pnl_series) else 0.0,
                    }
                )
            side_breakdown = pd.DataFrame(side_breakdown)

        content = "Trade distribution and side breakdown."

        data: Dict[str, Any] = {}
        if pnl_stats:
            pnl_df = pd.DataFrame([pnl_stats]).T.reset_index()
            pnl_df.columns = ["Metric", "Value"]
            data["PnL Distribution"] = pnl_df
        data["Trades"] = trades_df
        if side_breakdown is not None:
            data["Side Breakdown"] = side_breakdown

        metrics = {
            "trade_count": trade_stats.get("trade_count"),
            "win_rate": trade_stats.get("win_rate"),
            "profit_factor": trade_stats.get("profit_factor"),
        }
        metrics.update({k: v for k, v in pnl_stats.items() if k in {"mean", "median", "min", "max"}})
        metrics = {k: v for k, v in metrics.items() if v is not None}

        return ReportSection(
            title="Trades",
            content=content,
            metrics=metrics,
            artifacts={"trades_csv": str(trades_path)},
            data=data,
        )

    def _build_risk_section(self, risk_stats: Dict[str, Any]) -> Optional[ReportSection]:
        if not risk_stats:
            return None

        content_lines = ["Risk controls and events:", ""]
        for key, value in risk_stats.items():
            content_lines.append(f"- {key}: {value}")

        return ReportSection(
            title="Risk",
            content="\n".join(content_lines),
            metrics={k: v for k, v in risk_stats.items() if isinstance(v, (int, float, str, bool))},
        )


__all__ = ["BacktestReportGenerator"]

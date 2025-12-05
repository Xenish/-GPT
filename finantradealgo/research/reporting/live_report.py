"""
Live Performance & Health Report Generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from finantradealgo.research.reporting.base import (
    Report,
    ReportGenerator,
    ReportProfile,
    ReportSection,
)


SnapshotInput = Union[Dict[str, Any], str, Path]
TradesInput = Union[List[Dict[str, Any]], pd.DataFrame, str, Path, None]


class LiveReportGenerator(ReportGenerator):
    """Generate reports for live trading health/performance snapshots."""

    def generate(
        self,
        snapshot: SnapshotInput,
        trades: TradesInput = None,
        snapshot_path: Optional[Path] = None,
    ) -> Report:
        snap = self._load_snapshot(snapshot)
        trades_df = self._load_trades(trades)

        run_id = snap.get("run_id") or snap.get("id")
        mode = snap.get("mode")
        symbol = snap.get("symbol")
        timeframe = snap.get("timeframe")
        equity_now = snap.get("equity_now") or snap.get("equity")
        equity_start = snap.get("equity_start")
        daily_pnl = snap.get("daily_pnl") or snap.get("daily_realized_pnl") or snap.get("daily_unrealized_pnl")
        max_intraday_dd = snap.get("max_intraday_dd")
        kill_switch_triggered = snap.get("kill_switch_triggered") or snap.get("kill_switch", False)
        kill_switch_reason = snap.get("kill_switch_reason")
        validation_issues = snap.get("validation_issues") or []
        last_bar_time = snap.get("last_bar_time")
        heartbeat_age_sec = snap.get("heartbeat_age_sec") or snap.get("stale_data_seconds")

        report = Report(
            title=f"Live Performance Report: {run_id or 'unknown'}",
            description=f"Live status for {symbol or 'N/A'} / {timeframe or 'N/A'}",
            job_id=None,
            run_id=run_id,
            profile=ReportProfile.LIVE,
            strategy_id=snap.get("strategy"),
            symbol=symbol,
            timeframe=timeframe,
            metrics={
                "equity_now": equity_now,
                "equity_start": equity_start,
                "daily_pnl": daily_pnl,
                "max_intraday_dd": max_intraday_dd,
            },
            artifacts={"snapshot_path": str(snapshot_path)} if snapshot_path else {},
            metadata={"mode": mode, "last_bar_time": last_bar_time, "heartbeat_age_sec": heartbeat_age_sec},
        )

        report.add_section(
            self._build_overview_section(
                snap=snap,
                equity_now=equity_now,
                equity_start=equity_start,
                daily_pnl=daily_pnl,
                max_intraday_dd=max_intraday_dd,
            )
        )
        risk_section = self._build_risk_section(kill_switch_triggered, kill_switch_reason, validation_issues)
        if risk_section:
            report.add_section(risk_section)
        report.add_section(self._build_trades_section(trades_df))

        return report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load_snapshot(self, snapshot: SnapshotInput) -> Dict[str, Any]:
        if isinstance(snapshot, dict):
            return snapshot
        path = Path(snapshot)
        if not path.exists():
            raise FileNotFoundError(f"Live snapshot not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_trades(self, trades: TradesInput) -> pd.DataFrame:
        if trades is None:
            return pd.DataFrame()
        if isinstance(trades, pd.DataFrame):
            return trades
        if isinstance(trades, list):
            return pd.DataFrame(trades)
        path = Path(trades)
        if not path.exists():
            raise FileNotFoundError(f"Trades file not found: {path}")
        return pd.read_csv(path)

    def _build_overview_section(
        self,
        snap: Dict[str, Any],
        equity_now: Optional[float],
        equity_start: Optional[float],
        daily_pnl: Optional[float],
        max_intraday_dd: Optional[float],
    ) -> ReportSection:
        content = f"""
Live run overview.

- Run: {snap.get('run_id')}
- Mode: {snap.get('mode')}
- Symbol/TF: {snap.get('symbol')}/{snap.get('timeframe')}
- Last bar: {snap.get('last_bar_time')}
- Heartbeat age (sec): {snap.get('heartbeat_age_sec') or snap.get('stale_data_seconds')}
""".strip()

        metrics = {
            "equity_now": equity_now,
            "equity_start": equity_start,
            "daily_pnl": daily_pnl,
            "max_intraday_dd": max_intraday_dd,
            "open_positions": len(snap.get("open_positions") or []),
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}

        return ReportSection(
            title="Overview",
            content=content,
            metrics=metrics,
        )

    def _build_risk_section(
        self,
        kill_switch_triggered: Optional[bool],
        kill_switch_reason: Optional[str],
        validation_issues: List[Any],
    ) -> Optional[ReportSection]:
        if kill_switch_triggered is None and not validation_issues and not kill_switch_reason:
            return None

        lines = []
        if kill_switch_triggered is not None:
            lines.append(f"- Kill Switch Triggered: {kill_switch_triggered}")
        if kill_switch_reason:
            lines.append(f"- Kill Switch Reason: {kill_switch_reason}")
        if validation_issues:
            lines.append("- Validation Issues:")
            for issue in validation_issues:
                lines.append(f"  - {issue}")

        return ReportSection(
            title="Risk",
            content="\n".join(lines).strip(),
            metrics={"kill_switch_triggered": kill_switch_triggered} if kill_switch_triggered is not None else {},
        )

    def _build_trades_section(self, trades_df: pd.DataFrame) -> ReportSection:
        if trades_df is None:
            trades_df = pd.DataFrame()

        metrics: Dict[str, Any] = {}
        if not trades_df.empty and "pnl" in trades_df.columns:
            pnl_series = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
            if not pnl_series.empty:
                metrics["trade_count"] = len(pnl_series)
                metrics["win_rate"] = float((pnl_series > 0).mean())
                metrics["pnl_mean"] = float(pnl_series.mean())

        return ReportSection(
            title="Recent Trades",
            content="Most recent trades from live session.",
            metrics=metrics,
            data={"trades": trades_df} if not trades_df.empty else None,
        )


__all__ = ["LiveReportGenerator"]

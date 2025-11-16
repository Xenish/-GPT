from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ReportConfig:
    regime_columns: Optional[List[str]] = None


def _compute_trade_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "median_hold_time": None,
        }

    df = trades.copy().dropna(subset=["pnl"])
    if df.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "median_hold_time": None,
        }

    pnl = df["pnl"]
    trade_count = len(pnl)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = len(wins) / trade_count if trade_count > 0 else 0.0
    avg_pnl = float(pnl.mean())
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    if len(losses) > 0 and losses.sum() != 0:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    else:
        profit_factor = 0.0

    if "timestamp_exit" in df.columns:
        hold_times = pd.to_datetime(df["timestamp_exit"]) - pd.to_datetime(df["timestamp"])
        median_hold = hold_times.median()
    else:
        median_hold = None

    return {
        "trade_count": int(trade_count),
        "win_rate": float(win_rate),
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "median_hold_time": median_hold,
    }


def _compute_regime_stats(
    trades: pd.DataFrame,
    df: pd.DataFrame,
    regime_columns: List[str],
) -> Dict[str, Any]:
    if trades is None or trades.empty or df is None or df.empty:
        return {}

    cols = ["timestamp"] + [c for c in regime_columns if c in df.columns]
    df_reg = df[cols].drop_duplicates(subset=["timestamp"]).copy()

    merged = trades.merge(df_reg, on="timestamp", how="left")

    out: Dict[str, Any] = {}
    for col in regime_columns:
        if col not in merged.columns:
            continue

        regime_dict: Dict[str, Any] = {}
        for regime_value, grp in merged.groupby(col):
            key = "UNKNOWN" if pd.isna(regime_value) else str(regime_value)
            stats = _compute_trade_stats(grp)
            regime_dict[key] = stats

        out[col] = regime_dict

    return out


def generate_report(
    backtest_result: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    config: Optional[ReportConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = ReportConfig()

    trades = backtest_result.get("trades", pd.DataFrame())
    equity_curve: pd.Series = backtest_result.get("equity_curve", pd.Series(dtype=float))

    equity_metrics = {
        "initial_cash": backtest_result.get("initial_cash"),
        "final_equity": backtest_result.get("final_equity"),
        "cum_return": backtest_result.get("cum_return"),
        "max_drawdown": backtest_result.get("max_drawdown"),
        "sharpe": backtest_result.get("sharpe"),
    }

    trade_stats = _compute_trade_stats(trades)

    regime_stats = {}
    if config.regime_columns and df is not None:
        regime_stats = _compute_regime_stats(trades, df, config.regime_columns)

    return {
        "equity_metrics": equity_metrics,
        "trade_stats": trade_stats,
        "regime_stats": regime_stats,
        "equity_curve": equity_curve,
        "trades": trades,
    }

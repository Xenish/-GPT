from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DailyLossLimitConfig:
    max_daily_loss_pct: float = 0.03
    lookback_days: int = 1


def compute_daily_pnl(trades: pd.DataFrame) -> pd.Series:
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return pd.Series(dtype=float)

    df = trades.dropna(subset=["pnl"]).copy()
    if df.empty or "timestamp_exit" not in df.columns:
        return pd.Series(dtype=float)

    df["timestamp_exit"] = pd.to_datetime(df["timestamp_exit"])
    df["date"] = df["timestamp_exit"].dt.floor("D")
    daily = df.groupby("date")["pnl"].sum()
    return daily


def is_daily_loss_limit_hit(
    equity_start_of_day: float,
    realized_pnl_today: float,
    cfg: DailyLossLimitConfig,
) -> bool:
    if cfg.max_daily_loss_pct <= 0 or equity_start_of_day <= 0:
        return False
    loss_amount = -float(realized_pnl_today)
    if loss_amount <= 0:
        return False
    loss_pct = loss_amount / equity_start_of_day
    return loss_pct >= cfg.max_daily_loss_pct

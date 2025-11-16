from __future__ import annotations

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


class EMACrossStrategy(BaseStrategy):
    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast EMA must be less than slow EMA")
        self.fast = fast
        self.slow = slow
        self.df: pd.DataFrame | None = None

    def init(self, df: pd.DataFrame) -> None:
        self.df = df
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if self.df is None:
            return None

        i = ctx.index
        if i == 0:
            return None

        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])

        prev = self.df.iloc[i - 1]
        prev_fast = float(prev["ema_fast"])
        prev_slow = float(prev["ema_slow"])

        crossed_up = prev_fast <= prev_slow and ema_fast > ema_slow
        crossed_down = prev_fast >= prev_slow and ema_fast < ema_slow

        if ctx.position is None:
            if crossed_up:
                return "LONG"
        else:
            if crossed_down:
                return "CLOSE"

        return None

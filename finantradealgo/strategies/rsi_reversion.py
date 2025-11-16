from __future__ import annotations

import numpy as np
import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class RSIReversionStrategy(BaseStrategy):
    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        warmup_bars: int = 50,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.warmup_bars = warmup_bars
        self.df: pd.DataFrame | None = None

    def init(self, df: pd.DataFrame) -> None:
        self.df = df
        df["rsi"] = compute_rsi(df["close"], period=self.period)

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if self.df is None:
            return None

        i = ctx.index
        if i < self.warmup_bars:
            return None

        rsi = float(row["rsi"])
        if np.isnan(rsi):
            return None

        if ctx.position is None:
            if rsi <= self.oversold:
                return "LONG"
            if rsi >= self.overbought:
                return "SHORT"
            return None

        side = ctx.position.side

        if side in ("LONG", "SHORT") and self.oversold < rsi < self.overbought:
            return "CLOSE"

        return None

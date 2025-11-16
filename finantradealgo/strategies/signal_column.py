from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


@dataclass
class ColumnSignalStrategyConfig:
    signal_col: str = "ml_signal_long"
    warmup_bars: int = 0


class ColumnSignalStrategy(BaseStrategy):
    def __init__(self, config: ColumnSignalStrategyConfig | None = None):
        self.config = config or ColumnSignalStrategyConfig()
        self.df: pd.DataFrame | None = None

    def init(self, df: pd.DataFrame) -> None:
        self.df = df

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if ctx.index < self.config.warmup_bars:
            return None

        sig = row.get(self.config.signal_col, float("nan"))
        if pd.isna(sig):
            return None

        in_position = ctx.position is not None

        if sig >= 1.0:
            if not in_position:
                return "LONG"
            return None

        if in_position:
            return "CLOSE"

        return None

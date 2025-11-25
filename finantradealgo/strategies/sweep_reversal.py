from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


@dataclass
class SweepReversalConfig:
    swing_high_col: str = "ms_swing_high"
    swing_low_col: str = "ms_swing_low"
    fvg_col: str = "ms_fvg_flag"
    rsi_col: str = "rsi_14"
    max_rsi_for_long: float = 55.0
    min_pullback_ratio: float = 0.003
    use_fvg_filter: bool = True
    warmup_bars: int = 100

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SweepReversalConfig":
        data = data or {}
        return cls(
            swing_high_col=data.get("swing_high_col", cls.swing_high_col),
            swing_low_col=data.get("swing_low_col", cls.swing_low_col),
            fvg_col=data.get("fvg_col", cls.fvg_col),
            rsi_col=data.get("rsi_col", cls.rsi_col),
            max_rsi_for_long=float(data.get("max_rsi_for_long", cls.max_rsi_for_long)),
            min_pullback_ratio=float(data.get("min_pullback_ratio", cls.min_pullback_ratio)),
            use_fvg_filter=bool(data.get("use_fvg_filter", cls.use_fvg_filter)),
            warmup_bars=int(data.get("warmup_bars", cls.warmup_bars)),
        )


class SweepReversalStrategy(BaseStrategy):
    """
    Looks for liquidity sweeps below previous swing lows combined with bullish
    confirmation (RSI, fair value gap) to fade extreme moves.
    """

    def __init__(self, config: Optional[SweepReversalConfig] = None):
        self.config = config or SweepReversalConfig()
        self._df: Optional[pd.DataFrame] = None
        self._in_position = False

    def init(self, df: pd.DataFrame) -> None:
        self._df = df
        self._in_position = False

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if self._df is None or ctx.index < self.config.warmup_bars:
            return None

        swing_low = row.get(self.config.swing_low_col)
        swing_high = row.get(self.config.swing_high_col)
        close = row.get("close")
        low = row.get("low")
        rsi_val = row.get(self.config.rsi_col)

        if any(pd.isna(val) for val in (swing_low, swing_high, close, low, rsi_val)):
            return None

        if not self._in_position:
            sweep_occured = low <= swing_low * (1.0 + self.config.min_pullback_ratio)
            reclaim = close > (swing_low + (swing_high - swing_low) * 0.25)
            if not (sweep_occured and reclaim):
                return None
            if rsi_val > self.config.max_rsi_for_long:
                return None
            if self.config.use_fvg_filter:
                fvg_flag = row.get(self.config.fvg_col, 0)
                if fvg_flag != 1:
                    return None
            self._in_position = True
            return "LONG"

        # exit
        if close >= swing_high or rsi_val >= 65.0:
            self._in_position = False
            return "CLOSE"

        return None


__all__ = ["SweepReversalConfig", "SweepReversalStrategy"]

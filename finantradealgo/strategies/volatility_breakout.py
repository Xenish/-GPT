from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


@dataclass
class VolatilityBreakoutConfig:
    atr_col: str = "atr_14_pct"
    hv_col: str = "hv_20"
    bb_width_col: str = "bb_width_20"
    min_contraction: float = 0.01
    max_contraction: float = 0.05
    atr_entry_min: float = 0.002
    hv_entry_min: float = 0.02
    use_chop_filter: bool = True
    ms_chop_col: str = "ms_chop_score"
    max_chop_score: float = 0.5
    warmup_bars: int = 100

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "VolatilityBreakoutConfig":
        data = data or {}
        return cls(
            atr_col=data.get("atr_col", cls.atr_col),
            hv_col=data.get("hv_col", cls.hv_col),
            bb_width_col=data.get("bb_width_col", cls.bb_width_col),
            min_contraction=float(data.get("min_contraction", cls.min_contraction)),
            max_contraction=float(data.get("max_contraction", cls.max_contraction)),
            atr_entry_min=float(data.get("atr_entry_min", cls.atr_entry_min)),
            hv_entry_min=float(data.get("hv_entry_min", cls.hv_entry_min)),
            use_chop_filter=bool(data.get("use_chop_filter", cls.use_chop_filter)),
            ms_chop_col=data.get("ms_chop_col", cls.ms_chop_col),
            max_chop_score=float(data.get("max_chop_score", cls.max_chop_score)),
            warmup_bars=int(data.get("warmup_bars", cls.warmup_bars)),
        )


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Detects volatility contractions via Bollinger width and rides the breakout
    when ATR/HV expand.
    """

    def __init__(self, config: Optional[VolatilityBreakoutConfig] = None):
        self.config = config or VolatilityBreakoutConfig()
        self._df: Optional[pd.DataFrame] = None
        self._in_position = False
        self._entry_price: Optional[float] = None

    def init(self, df: pd.DataFrame) -> None:
        self._df = df
        self._in_position = False
        self._entry_price = None

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if self._df is None or ctx.index < self.config.warmup_bars:
            return None

        bb_width = row.get(self.config.bb_width_col)
        atr_val = row.get(self.config.atr_col)
        hv_val = row.get(self.config.hv_col)
        close = row.get("close")
        if any(pd.isna(val) for val in (bb_width, atr_val, hv_val, close)):
            return None

        if not self._in_position:
            if not (self.config.min_contraction <= bb_width <= self.config.max_contraction):
                return None
            if atr_val < self.config.atr_entry_min or hv_val < self.config.hv_entry_min:
                return None
            if self.config.use_chop_filter:
                chop = row.get(self.config.ms_chop_col, np.nan)
                if not np.isnan(chop) and chop > self.config.max_chop_score:
                    return None
            self._in_position = True
            self._entry_price = float(close)
            return "LONG"

        # exit if volatility collapses or profit target reached
        if atr_val < self.config.atr_entry_min * 0.8 or bb_width > self.config.max_contraction * 1.5:
            self._in_position = False
            self._entry_price = None
            return "CLOSE"

        if self._entry_price is not None:
            gain = (float(close) - self._entry_price) / self._entry_price
            if gain >= 0.02:  # ~2% pop
                self._in_position = False
                self._entry_price = None
                return "CLOSE"

        return None


__all__ = ["VolatilityBreakoutConfig", "VolatilityBreakoutStrategy"]

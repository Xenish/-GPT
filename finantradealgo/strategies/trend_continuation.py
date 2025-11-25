from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


@dataclass
class TrendContinuationConfig:
    ema_fast_col: str = "ema_20"
    ema_slow_col: str = "ema_50"
    rsi_col: str = "rsi_14"
    rsi_trend_min: float = 50.0
    rsi_trend_max: float = 70.0
    min_trend_score: float = 0.0
    htf_trend_col: str = "htf1h_trend_score"
    use_ms_trend_filter: bool = True
    ms_trend_col: str = "ms_trend_score"
    ms_trend_min: float = -0.2
    ms_trend_max: float = 1.0
    warmup_bars: int = 100

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TrendContinuationConfig":
        data = data or {}
        return cls(
            ema_fast_col=data.get("ema_fast_col", cls.ema_fast_col),
            ema_slow_col=data.get("ema_slow_col", cls.ema_slow_col),
            rsi_col=data.get("rsi_col", cls.rsi_col),
            rsi_trend_min=float(data.get("rsi_trend_min", cls.rsi_trend_min)),
            rsi_trend_max=float(data.get("rsi_trend_max", cls.rsi_trend_max)),
            min_trend_score=float(data.get("min_trend_score", cls.min_trend_score)),
            htf_trend_col=data.get("htf_trend_col", cls.htf_trend_col),
            use_ms_trend_filter=bool(data.get("use_ms_trend_filter", cls.use_ms_trend_filter)),
            ms_trend_col=data.get("ms_trend_col", cls.ms_trend_col),
            ms_trend_min=float(data.get("ms_trend_min", cls.ms_trend_min)),
            ms_trend_max=float(data.get("ms_trend_max", cls.ms_trend_max)),
            warmup_bars=int(data.get("warmup_bars", cls.warmup_bars)),
        )


class TrendContinuationStrategy(BaseStrategy):
    """
    Momentum-following strategy that rides favorable EMA/RSi alignment and optional
    market-structure filters.
    """

    def __init__(self, config: Optional[TrendContinuationConfig] = None):
        self.config = config or TrendContinuationConfig()
        self._df: Optional[pd.DataFrame] = None
        self._in_position = False

    def init(self, df: pd.DataFrame) -> None:
        self._df = df
        self._in_position = False

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if self._df is None or ctx.index < self.config.warmup_bars:
            return None

        fast = row.get(self.config.ema_fast_col)
        slow = row.get(self.config.ema_slow_col)
        rsi = row.get(self.config.rsi_col)
        trend_score = row.get(self.config.htf_trend_col, np.nan)

        if any(pd.isna(val) for val in (fast, slow, rsi)):
            return None

        if not self._in_position:
            if fast <= slow:
                return None
            if not (self.config.rsi_trend_min <= rsi <= self.config.rsi_trend_max):
                return None
            if not pd.isna(trend_score) and trend_score < self.config.min_trend_score:
                return None
            if self.config.use_ms_trend_filter:
                ms_score = row.get(self.config.ms_trend_col, np.nan)
                if not np.isnan(ms_score):
                    if not (self.config.ms_trend_min <= ms_score <= self.config.ms_trend_max):
                        return None
            self._in_position = True
            return "LONG"

        # exit rules
        price = float(row.get("close", np.nan))
        if np.isnan(price):
            return None

        if price < slow or rsi < 45.0:
            self._in_position = False
            return "CLOSE"

        return None


__all__ = ["TrendContinuationConfig", "TrendContinuationStrategy"]

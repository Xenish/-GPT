from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext

@dataclass
class RuleStrategyConfig:
    """
    Rule-based long-only strateji için config.

    entry_col: long açma sinyali (0/1)
    exit_col : long kapama sinyali (0/1)
    warmup_bars: baştaki bar sayısı (pozisyon alma)
    """
    entry_col: str = "rule_long_entry"
    exit_col: str = "rule_long_exit"
    warmup_bars: int = 50


class RuleSignalStrategy(BaseStrategy):
    """
    Long-only kural tabanlı strateji.

    DataFrame'de şu kolonların var olduğunu varsayar:
      - entry_col (varsayılan: "rule_long_entry")
      - exit_col  (varsayılan: "rule_long_exit")

    Bu kolonlardan 'signal' üretir:
      signal = 0 → flat
      signal = 1 → long

    Backtester, df["signal"]’ı kullanarak trade simüle eder.
    """

    def __init__(self, config: Optional[RuleStrategyConfig] = None):
        self.config = config or RuleStrategyConfig()
        self._df: pd.DataFrame | None = None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        entry/exit kolonlarından 'signal' kolonunu üretir.
        df'i in-place günceller ve geri döner.
        """

        if self.config.entry_col not in df.columns or self.config.exit_col not in df.columns:
            raise ValueError(
                f"DataFrame'de '{self.config.entry_col}' ve/veya '{self.config.exit_col}' kolonu yok."
            )

        entry = df[self.config.entry_col].fillna(0).astype(int)
        exit_ = df[self.config.exit_col].fillna(0).astype(int)

        position = 0
        signals = []

        for i, (e, x) in enumerate(zip(entry, exit_)):
            # Warmup döneminde pozisyon alma
            if i < self.config.warmup_bars:
                position = 0
            else:
                if position == 0 and e == 1:
                    position = 1  # long aç
                elif position == 1 and x == 1:
                    position = 0  # long kapa

            signals.append(position)

        df["signal"] = signals
        return df

    def init(self, df: pd.DataFrame) -> None:
        self.generate_signals(df)
        self._df = df
        return None

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if self._df is None:
            return None

        sig = row.get("signal", float("nan"))
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


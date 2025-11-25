from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


@dataclass
class MLStrategyConfig:
    proba_col: str = "ml_proba_long"
    entry_threshold: float = 0.55
    exit_threshold: float = 0.50
    warmup_bars: int = 200
    side: str = "long_only"

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "MLStrategyConfig":
        data = data or {}
        entry = float(data.get("proba_threshold", data.get("entry_threshold", cls.entry_threshold)))
        exit_value = float(
            data.get(
                "proba_exit_threshold",
                data.get("exit_threshold", cls.exit_threshold),
            )
        )
        if exit_value > entry:
            exit_value = entry

        return cls(
            proba_col=data.get("proba_column", data.get("proba_col", cls.proba_col)),
            entry_threshold=entry,
            exit_threshold=exit_value,
            warmup_bars=int(data.get("warmup_bars", cls.warmup_bars)),
            side=data.get("side", cls.side),
        )


class MLSignalStrategy(BaseStrategy):
    def __init__(self, config: Optional[MLStrategyConfig] = None):
        self.config = config or MLStrategyConfig()
        self._df: Optional[pd.DataFrame] = None

    def init(self, df: pd.DataFrame) -> None:
        if self.config.proba_col not in df.columns:
            raise ValueError(
                f"{self.config.proba_col} column missing from DataFrame for ML strategy."
            )
        self._df = df

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if self._df is None:
            return None

        idx = getattr(ctx, "index", row.name)
        if isinstance(idx, (int, float)) and idx < self.config.warmup_bars:
            return None

        proba = row.get(self.config.proba_col, float("nan"))
        if pd.isna(proba):
            return None

        in_position = ctx.position is not None

        if not in_position and proba >= self.config.entry_threshold:
            return "LONG"

        if in_position and proba <= self.config.exit_threshold:
            return "CLOSE"

        return None

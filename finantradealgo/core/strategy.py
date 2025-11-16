from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from .portfolio import Position


SignalType = Literal["LONG", "SHORT", "CLOSE", None]


@dataclass
class StrategyContext:
    equity: float
    position: Optional[Position]
    index: int


class BaseStrategy(ABC):
    @abstractmethod
    def init(self, df: pd.DataFrame) -> None:
        """
        Feature hesabı vs. için ilk setup.
        df üzerinde istediğini ekleyebilirsin (örn: ema kolonları).
        """
        pass

    @abstractmethod
    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        """
        Her bar için çağrılır.
        'LONG', 'SHORT', 'CLOSE' veya None dönebilir.
        """
        pass

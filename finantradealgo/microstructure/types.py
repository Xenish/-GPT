"""
Types for the microstructure module, re-exported from core types for consistency.
"""
from dataclasses import asdict, dataclass, fields

import numpy as np

from finantradealgo.core.types import (
    Bar,
    OrderBookLevel,
    OrderBookSnapshot,
    Trade,
)

__all__ = [
    "Bar",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "Trade",
    "MicrostructureSignals",
]


@dataclass
class MicrostructureSignals:
    """Holds all calculated microstructure signals for a single timestep."""

    imbalance: float = np.nan
    sweep_up: float = np.nan
    sweep_down: float = np.nan
    chop: float = np.nan
    burst_up: float = np.nan
    burst_down: float = np.nan
    vol_regime: float = np.nan
    exhaustion_up: float = np.nan
    exhaustion_down: float = np.nan
    parabolic_trend: float = np.nan

    @staticmethod
    def columns() -> list[str]:
        """Returns the list of official column names for these signals."""
        # Note: we rely on fields being ordered in Python 3.7+
        return [f"ms_{f.name}" for f in fields(MicrostructureSignals)]

    def to_dict(self) -> dict[str, float]:
        """Returns the signals as a dictionary with official column names."""
        return {f"ms_{key}": value for key, value in asdict(self).items()}

    def to_array(self) -> np.ndarray:
        """Returns the signals as a numpy array."""
        return np.array(list(asdict(self).values()))

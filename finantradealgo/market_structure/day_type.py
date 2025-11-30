from dataclasses import dataclass
from enum import IntEnum


@dataclass
class DayTypeConfig:
    trend_body_to_range_min: float = 2.5
    range_body_to_range_max: float = 0.7
    liquidation_wick_to_range_min: float = 1.5
    liquidation_volume_z: float = 2.0


class DAY_TYPE(IntEnum):
    TREND = 1
    NORMAL = 0
    LIQUIDATION = -1
    # RANGE vs NORMAL will be handled by logic
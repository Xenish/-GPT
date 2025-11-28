from dataclasses import dataclass
from typing import Literal

@dataclass
class DayTypeConfig:
    trend_body_to_range_min: float = 0.6
    range_body_to_range_max: float = 0.4
    liquidation_wick_to_range_min: float = 0.7
    liquidation_volume_z: float = 2.0

class DAY_TYPE:
    TREND: Literal[1] = 1
    NORMAL: Literal[0] = 0
    LIQUIDATION: Literal[-1] = -1
    # RANGE could be separated from NORMAL if needed, as per comment in the task.

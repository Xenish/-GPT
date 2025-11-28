from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class EventBarConfig:
    mode: Literal["time", "volume", "dollar", "tick"] = "time"
    target_volume: Optional[float] = None
    target_notional: Optional[float] = None
    target_ticks: Optional[int] = None

@dataclass
class DayTypeConfig:
    trend_body_to_range_min: float = 0.6
    range_body_to_range_max: float = 0.4
    liquidation_wick_to_range_min: float = 0.5
    liquidation_volume_z: float = 2.0

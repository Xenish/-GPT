"""
Core data types for the market_structure module, such as Swing points.
"""
from dataclasses import dataclass
from typing import Literal

# Import the core Bar type for potential future use, ensuring no re-definition.
from finantradealgo.core.types import Bar

SwingKind = Literal["high", "low"]
ZoneType = Literal["demand", "supply"]


@dataclass
class MarketStructureColumns:
    """
    Contract for market structure output column names.

    This dataclass ensures consistent naming across the codebase.
    All market structure features use the 'ms_' prefix.
    """
    price_smooth: str = "price_smooth"
    swing_high: str = "ms_swing_high"
    swing_low: str = "ms_swing_low"
    swing_idx: str = "ms_swing_index"  # Optional: Index tracking
    fvg_up: str = "ms_fvg_up"
    fvg_down: str = "ms_fvg_down"
    trend_regime: str = "ms_trend_regime"
    chop_regime: str = "ms_chop_regime"
    zone_demand: str = "ms_zone_demand"
    zone_supply: str = "ms_zone_supply"
    bos_up: str = "ms_bos_up"
    bos_down: str = "ms_bos_down"
    choch: str = "ms_choch"


@dataclass
class SwingPoint:
    """Represents a single swing high or swing low point."""

    ts: int  # Integer index of the bar in the DataFrame
    price: float
    kind: SwingKind


@dataclass
class Zone:
    """Represents a supply or demand zone."""

    id: int
    type: ZoneType
    low: float
    high: float
    strength: float  # Can be based on number of touches, volume, etc.
    first_ts: int
    last_ts: int


@dataclass
class VolumeProfileBin:
    """Represents a single bin in a volume profile."""

    price_low: float
    price_high: float
    volume: float


@dataclass
class VolumeProfile:
    """Represents the volume profile for a given period."""

    bins: "List[VolumeProfileBin]"
    poc_bin: "VolumeProfileBin"  # Point of Control

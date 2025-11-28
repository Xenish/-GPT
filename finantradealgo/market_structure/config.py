"""
Configuration classes for the market_structure module.
"""
from dataclasses import dataclass, field


@dataclass
class SwingConfig:
    """Configuration for swing point detection."""

    lookback: int = 2
    min_swing_size_pct: float = 0.003  # 0.3%


@dataclass
class TrendRegimeConfig:
    """Configuration for trend regime detection."""

    min_swings: int = 4


@dataclass
class FVGConfig:
    """Configuration for Fair Value Gap (FVG) detection."""

    min_gap_pct: float = 0.001  # 0.1%
    max_bars_ahead: int = 50  # For tracking if the gap is filled


@dataclass
class ZoneConfig:
    """Configuration for supply/demand zone detection."""

    price_proximity_pct: float = 0.003
    min_touches: int = 2
    window_bars: int = 500


@dataclass
class BreakConfig:
    """Configuration for Break of Structure (BoS) / Change of Character (ChoCh)."""

    swing_break_buffer_pct: float = 0.0005  # 0.05% tolerance


@dataclass
class MarketStructureConfig:
    """Main config for all market structure signals."""

    swing: SwingConfig = field(default_factory=SwingConfig)
    trend: TrendRegimeConfig = field(default_factory=TrendRegimeConfig)
    fvg: FVGConfig = field(default_factory=FVGConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    breaks: BreakConfig = field(default_factory=BreakConfig)

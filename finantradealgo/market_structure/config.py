"""
Configuration classes for the market_structure module.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Import ChopConfig from microstructure for regime detection
from finantradealgo.microstructure.config import ChopConfig


@dataclass
class SwingConfig:
    """Configuration for swing point detection."""

    lookback: int = 2
    min_swing_size_pct: float = 0.003  # 0.3%

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SwingConfig":
        data = data or {}
        return cls(
            lookback=int(data.get("lookback", cls.lookback)),
            min_swing_size_pct=float(data.get("min_swing_size_pct", cls.min_swing_size_pct)),
        )


@dataclass
class TrendRegimeConfig:
    """Configuration for trend regime detection."""

    min_swings: int = 4

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrendRegimeConfig":
        data = data or {}
        return cls(min_swings=int(data.get("min_swings", cls.min_swings)))


@dataclass
class FVGConfig:
    """Configuration for Fair Value Gap (FVG) detection."""

    min_gap_pct: float = 0.001  # 0.1%
    max_bars_ahead: int = 50  # For tracking if the gap is filled

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FVGConfig":
        data = data or {}
        return cls(
            min_gap_pct=float(data.get("min_gap_pct", cls.min_gap_pct)),
            max_bars_ahead=int(data.get("max_bars_ahead", cls.max_bars_ahead)),
        )


@dataclass
class ZoneConfig:
    """Configuration for supply/demand zone detection."""

    price_proximity_pct: float = 0.003
    min_touches: int = 2
    window_bars: int = 500

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ZoneConfig":
        data = data or {}
        return cls(
            price_proximity_pct=float(data.get("price_proximity_pct", cls.price_proximity_pct)),
            min_touches=int(data.get("min_touches", cls.min_touches)),
            window_bars=int(data.get("window_bars", cls.window_bars)),
        )


@dataclass
class BreakConfig:
    """Configuration for Break of Structure (BoS) / Change of Character (ChoCh)."""

    swing_break_buffer_pct: float = 0.0005  # 0.05% tolerance

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BreakConfig":
        data = data or {}
        return cls(
            swing_break_buffer_pct=float(data.get("swing_break_buffer_pct", cls.swing_break_buffer_pct)),
        )


@dataclass
class SmoothingConfig:
    """Configuration for price and swing smoothing."""

    enabled: bool = True
    price_ma_window: int = 3
    swing_min_distance: int = 3
    swing_min_zscore: float = 0.5

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SmoothingConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            price_ma_window=int(data.get("price_ma_window", cls.price_ma_window)),
            swing_min_distance=int(data.get("swing_min_distance", cls.swing_min_distance)),
            swing_min_zscore=float(data.get("swing_min_zscore", cls.swing_min_zscore)),
        )


@dataclass
class MarketStructureConfig:
    """Main config for all market structure signals."""

    swing: SwingConfig = field(default_factory=SwingConfig)
    trend: TrendRegimeConfig = field(default_factory=TrendRegimeConfig)
    fvg: FVGConfig = field(default_factory=FVGConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    breaks: BreakConfig = field(default_factory=BreakConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    chop: ChopConfig = field(default_factory=ChopConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MarketStructureConfig":
        data = data or {}

        # Parse chop config from microstructure format
        chop_data = data.get("chop")
        if chop_data:
            chop_cfg = ChopConfig(
                lookback_period=int(chop_data.get("lookback_period", ChopConfig.lookback_period))
            )
        else:
            chop_cfg = ChopConfig()

        return cls(
            swing=SwingConfig.from_dict(data.get("swing")),
            trend=TrendRegimeConfig.from_dict(data.get("trend")),
            fvg=FVGConfig.from_dict(data.get("fvg")),
            zone=ZoneConfig.from_dict(data.get("zone")),
            breaks=BreakConfig.from_dict(data.get("breaks")),
            smoothing=SmoothingConfig.from_dict(data.get("smoothing")),
            chop=chop_cfg,
        )

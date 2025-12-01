from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ImbalanceConfig:
    """Config for order book imbalance signals."""

    depth: int = 5
    threshold: float = 2.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ImbalanceConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            depth=int(data.get("depth", cls.depth)),
            threshold=float(data.get("threshold", cls.threshold)),
        )


@dataclass
class LiquiditySweepConfig:
    """Config for liquidity sweep signals."""

    lookback_ms: int = 5000  # Lookback window in milliseconds from the start of the bar
    notional_threshold: float = 50000.0  # e.g., $50,000 in notional value to trigger a sweep

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LiquiditySweepConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            lookback_ms=int(data.get("lookback_ms", cls.lookback_ms)),
            notional_threshold=float(data.get("notional_threshold", cls.notional_threshold)),
        )


@dataclass
class ChopConfig:
    """Config for chop detection."""

    lookback_period: int = 14

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ChopConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            lookback_period=int(data.get("lookback_period", cls.lookback_period)),
        )


@dataclass
class VolatilityRegimeConfig:
    """Config for volatility regime detection."""

    period: int = 20
    z_score_window: int = 100  # Window for z-score calculation itself
    low_z_threshold: float = -1.5
    high_z_threshold: float = 1.5

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "VolatilityRegimeConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            period=int(data.get("period", cls.period)),
            z_score_window=int(data.get("z_score_window", cls.z_score_window)),
            low_z_threshold=float(data.get("low_z_threshold", cls.low_z_threshold)),
            high_z_threshold=float(data.get("high_z_threshold", cls.high_z_threshold)),
        )


@dataclass
class BurstConfig:
    """Config for momentum burst signals."""

    return_window: int = 5
    z_score_window: int = 100
    z_up_threshold: float = 2.0
    z_down_threshold: float = 2.0  # Note: logic uses abs value of this for down bursts

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BurstConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            return_window=int(data.get("return_window", cls.return_window)),
            z_score_window=int(data.get("z_score_window", cls.z_score_window)),
            z_up_threshold=float(data.get("z_up_threshold", cls.z_up_threshold)),
            z_down_threshold=float(data.get("z_down_threshold", cls.z_down_threshold)),
        )


@dataclass
class ExhaustionConfig:
    """Config for market exhaustion signals."""

    min_consecutive_bars: int = 5
    volume_z_score_window: int = 50
    volume_z_threshold: float = -0.5  # Z-score threshold for what is considered low volume

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExhaustionConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            min_consecutive_bars=int(data.get("min_consecutive_bars", cls.min_consecutive_bars)),
            volume_z_score_window=int(data.get("volume_z_score_window", cls.volume_z_score_window)),
            volume_z_threshold=float(data.get("volume_z_threshold", cls.volume_z_threshold)),
        )


@dataclass
class ParabolicConfig:
    """Config for parabolic trend detection based on price curvature."""

    rolling_std_window: int = 20
    curvature_threshold: float = 1.5

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ParabolicConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            rolling_std_window=int(data.get("rolling_std_window", cls.rolling_std_window)),
            curvature_threshold=float(data.get("curvature_threshold", cls.curvature_threshold)),
        )


@dataclass
class MicrostructureConfig:
    """Main config for all microstructure signals."""

    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    sweep: LiquiditySweepConfig = field(default_factory=LiquiditySweepConfig)
    chop: ChopConfig = field(default_factory=ChopConfig)
    vol_regime: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig)
    burst: BurstConfig = field(default_factory=BurstConfig)
    exhaustion: ExhaustionConfig = field(default_factory=ExhaustionConfig)
    parabolic: ParabolicConfig = field(default_factory=ParabolicConfig)
    enabled: bool = True
    max_lookback_seconds: int = 3600  # Task S2.E2: Max lookback window to prevent latency in live/paper trading

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MicrostructureConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            imbalance=ImbalanceConfig.from_dict(data.get("imbalance")),
            sweep=LiquiditySweepConfig.from_dict(data.get("sweep")),
            chop=ChopConfig.from_dict(data.get("chop")),
            vol_regime=VolatilityRegimeConfig.from_dict(data.get("vol_regime")),
            burst=BurstConfig.from_dict(data.get("burst")),
            exhaustion=ExhaustionConfig.from_dict(data.get("exhaustion")),
            parabolic=ParabolicConfig.from_dict(data.get("parabolic")),
            enabled=bool(data.get("enabled", cls.enabled)),
            max_lookback_seconds=int(data.get("max_lookback_seconds", 3600)),
        )

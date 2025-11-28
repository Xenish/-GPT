from dataclasses import dataclass, field


@dataclass
class ImbalanceConfig:
    """Config for order book imbalance signals."""

    depth: int = 5
    threshold: float = 2.0


@dataclass
class LiquiditySweepConfig:
    """Config for liquidity sweep signals."""

    lookback_ms: int = 5000  # Lookback window in milliseconds from the start of the bar
    notional_threshold: float = 50000.0  # e.g., $50,000 in notional value to trigger a sweep


@dataclass
class ChopConfig:
    """Config for chop detection."""

    lookback_period: int = 14


@dataclass
class VolatilityRegimeConfig:
    """Config for volatility regime detection."""

    period: int = 20
    z_score_window: int = 100  # Window for z-score calculation itself
    low_z_threshold: float = -1.5
    high_z_threshold: float = 1.5


@dataclass
class BurstConfig:
    """Config for momentum burst signals."""

    return_window: int = 5
    z_score_window: int = 100
    z_up_threshold: float = 2.0
    z_down_threshold: float = 2.0  # Note: logic uses abs value of this for down bursts


@dataclass
class ExhaustionConfig:
    """Config for market exhaustion signals."""

    min_consecutive_bars: int = 5
    volume_z_score_window: int = 50
    volume_z_threshold: float = -0.5  # Z-score threshold for what is considered low volume


@dataclass
class ParabolicConfig:
    """Config for parabolic trend detection based on price curvature."""

    rolling_std_window: int = 20
    curvature_threshold: float = 1.5


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

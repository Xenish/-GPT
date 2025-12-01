"""
Data validation configuration.

Task S3.1: DataValidationConfig for OHLCV and data quality validation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OHLCVValidationConfig:
    """Configuration for OHLCV data validation."""

    # Required columns
    required_columns: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )

    # Price validation
    check_negative_prices: bool = True
    check_zero_prices: bool = True
    check_ohlc_relationship: bool = True  # high >= low, open/close within [low, high]

    # Volume validation
    check_negative_volume: bool = True
    check_zero_volume: bool = False  # Allow zero volume (some instruments have low liquidity)

    # Data continuity
    check_missing_bars: bool = True
    max_gap_multiplier: float = 2.0  # Max gap = timeframe * multiplier

    # Outlier detection
    check_price_spikes: bool = True
    price_spike_z_threshold: float = 5.0  # Z-score threshold for spike detection
    price_spike_window: int = 100  # Rolling window for spike detection

    # Data quality
    check_duplicate_timestamps: bool = True
    check_chronological_order: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OHLCVValidationConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        # Create default instance to get default values
        defaults = cls()
        return cls(
            required_columns=data.get("required_columns", defaults.required_columns),
            check_negative_prices=bool(data.get("check_negative_prices", defaults.check_negative_prices)),
            check_zero_prices=bool(data.get("check_zero_prices", defaults.check_zero_prices)),
            check_ohlc_relationship=bool(data.get("check_ohlc_relationship", defaults.check_ohlc_relationship)),
            check_negative_volume=bool(data.get("check_negative_volume", defaults.check_negative_volume)),
            check_zero_volume=bool(data.get("check_zero_volume", defaults.check_zero_volume)),
            check_missing_bars=bool(data.get("check_missing_bars", defaults.check_missing_bars)),
            max_gap_multiplier=float(data.get("max_gap_multiplier", defaults.max_gap_multiplier)),
            check_price_spikes=bool(data.get("check_price_spikes", defaults.check_price_spikes)),
            price_spike_z_threshold=float(data.get("price_spike_z_threshold", defaults.price_spike_z_threshold)),
            price_spike_window=int(data.get("price_spike_window", defaults.price_spike_window)),
            check_duplicate_timestamps=bool(data.get("check_duplicate_timestamps", defaults.check_duplicate_timestamps)),
            check_chronological_order=bool(data.get("check_chronological_order", defaults.check_chronological_order)),
        )


@dataclass
class ExternalValidationConfig:
    """Configuration for external data validation (flow, sentiment, etc.)."""

    check_missing_data: bool = True
    max_missing_pct: float = 0.1  # Allow up to 10% missing data
    check_value_range: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExternalValidationConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            check_missing_data=bool(data.get("check_missing_data", cls.check_missing_data)),
            max_missing_pct=float(data.get("max_missing_pct", cls.max_missing_pct)),
            check_value_range=bool(data.get("check_value_range", cls.check_value_range)),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
        )


@dataclass
class DataValidationConfig:
    """
    Main configuration for data validation.

    Task S3.1: Centralized validation config for all data types.
    """

    # Validation mode: "off", "warn", "strict"
    # - "off": No validation
    # - "warn": Log warnings but continue
    # - "strict": Raise exceptions on validation errors
    mode: str = "warn"

    # OHLCV validation
    ohlcv: OHLCVValidationConfig = field(default_factory=OHLCVValidationConfig)

    # External data validation (flow, sentiment)
    external: ExternalValidationConfig = field(default_factory=ExternalValidationConfig)

    # Multi-timeframe validation
    check_multi_tf_alignment: bool = False  # Task S3.3

    # Live-specific validation
    check_suspect_bars: bool = False  # Task S3.E4
    suspect_bar_volume_z_threshold: float = -3.0  # Very low volume = suspect
    suspect_bar_range_z_threshold: float = -3.0   # Very narrow range = suspect

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DataValidationConfig":
        """Create config from dictionary (for YAML loading)."""
        data = data or {}
        return cls(
            mode=data.get("mode", cls.mode),
            ohlcv=OHLCVValidationConfig.from_dict(data.get("ohlcv")),
            external=ExternalValidationConfig.from_dict(data.get("external")),
            check_multi_tf_alignment=bool(data.get("check_multi_tf_alignment", cls.check_multi_tf_alignment)),
            check_suspect_bars=bool(data.get("check_suspect_bars", cls.check_suspect_bars)),
            suspect_bar_volume_z_threshold=float(
                data.get("suspect_bar_volume_z_threshold", cls.suspect_bar_volume_z_threshold)
            ),
            suspect_bar_range_z_threshold=float(
                data.get("suspect_bar_range_z_threshold", cls.suspect_bar_range_z_threshold)
            ),
        )

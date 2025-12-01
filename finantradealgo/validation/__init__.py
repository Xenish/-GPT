"""
Data validation module.

Task S3.1: Data validation entry-points and configuration.
"""
from finantradealgo.validation.config import (
    DataValidationConfig,
    ExternalValidationConfig,
    OHLCVValidationConfig,
)
from finantradealgo.validation.ohlcv_validator import (
    ValidationIssue,
    ValidationResult,
    validate_ohlcv,
    validate_ohlcv_strict,
)
from finantradealgo.validation.timeframe_utils import (
    TIMEFRAME_TO_SECONDS,
    detect_gaps,
    infer_timeframe,
    timeframe_to_seconds,
)
from finantradealgo.validation.multi_tf_validator import validate_multi_tf_alignment
from finantradealgo.validation.external_validator import (
    validate_external_series,
    validate_flow_features,
    validate_sentiment_features,
)
from finantradealgo.validation.live_validator import (
    detect_suspect_bars,
    validate_live_bar,
)

__all__ = [
    # Config
    "DataValidationConfig",
    "OHLCVValidationConfig",
    "ExternalValidationConfig",
    # Validators
    "validate_ohlcv",
    "validate_ohlcv_strict",
    "validate_multi_tf_alignment",
    "validate_external_series",
    "validate_flow_features",
    "validate_sentiment_features",
    "detect_suspect_bars",
    "validate_live_bar",
    "ValidationResult",
    "ValidationIssue",
    # Timeframe utils
    "TIMEFRAME_TO_SECONDS",
    "timeframe_to_seconds",
    "detect_gaps",
    "infer_timeframe",
]

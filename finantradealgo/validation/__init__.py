"""
Data validation module.

Task S3.1: Data validation entry-points and configuration.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

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


class DataQualitySeverity(str, Enum):
    """Severity levels for detected data quality issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataQualityIssueType(str, Enum):
    """Categorization for data quality issues."""

    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    DUPLICATE = "duplicate"
    INCONSISTENCY = "inconsistency"
    GAP = "gap"
    ANOMALY = "anomaly"
    OTHER = "other"


@dataclass(slots=True)
class DataSliceRef:
    """
    References a specific slice of data where an issue or observation applies.
    All fields are optional to accommodate partial context.
    """

    symbol: str | None
    timeframe: str | None  # e.g. "1m", "5m", "1h"
    column: str | None
    start: datetime | None
    end: datetime | None
    indices: list[int] | None  # optional index positions


@dataclass(slots=True)
class DataQualityIssue:
    """Represents a single detected data quality issue."""

    issue_type: DataQualityIssueType
    severity: DataQualitySeverity
    message: str
    slice_ref: DataSliceRef | None
    details: dict[str, Any] | None
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class DataQualityCheckConfig:
    """Configuration flags and thresholds controlling data quality checks."""

    enable_missing_check: bool = True
    enable_outlier_check: bool = True
    enable_duplicate_check: bool = True
    enable_consistency_check: bool = True
    enable_gap_check: bool = True
    max_missing_ratio: float = 0.05
    outlier_zscore_threshold: float = 4.0
    max_gap_length: int | None = None  # in bars
    columns_in_scope: list[str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class DataQualityReport:
    """Aggregated report containing all detected issues and summary statistics."""

    issues: list[DataQualityIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    data_ref: DataSliceRef | None = None


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
    # Data quality shared types
    "DataQualitySeverity",
    "DataQualityIssueType",
    "DataSliceRef",
    "DataQualityIssue",
    "DataQualityCheckConfig",
    "DataQualityReport",
]

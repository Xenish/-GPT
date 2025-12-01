"""
OHLCV data validation functions.

Task S3.1: validate_ohlcv() function for OHLCV data quality checks.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from finantradealgo.validation.config import OHLCVValidationConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    check_name: str
    severity: str  # "error" or "warning"
    message: str
    affected_rows: Optional[List[int]] = None
    affected_count: int = 0


@dataclass
class ValidationResult:
    """Result of validation checks."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0

    def add_issue(
        self,
        check_name: str,
        severity: str,
        message: str,
        affected_rows: Optional[List[int]] = None,
    ) -> None:
        """Add a validation issue."""
        affected_count = len(affected_rows) if affected_rows else 0
        issue = ValidationIssue(
            check_name=check_name,
            severity=severity,
            message=message,
            affected_rows=affected_rows,
            affected_count=affected_count,
        )
        self.issues.append(issue)

        if severity == "error":
            self.errors_count += 1
            self.is_valid = False
        elif severity == "warning":
            self.warnings_count += 1

    def summary(self) -> str:
        """Get a summary of validation results."""
        lines = []
        lines.append(f"Validation Result: {'PASS' if self.is_valid else 'FAIL'}")
        lines.append(f"Errors: {self.errors_count}, Warnings: {self.warnings_count}")

        if self.issues:
            lines.append("\nIssues:")
            for issue in self.issues:
                prefix = "  [ERROR]" if issue.severity == "error" else "  [WARN] "
                lines.append(f"{prefix} {issue.check_name}: {issue.message}")
                if issue.affected_count > 0:
                    lines.append(f"         Affected rows: {issue.affected_count}")

        return "\n".join(lines)


def validate_ohlcv(
    df: pd.DataFrame,
    cfg: Optional[OHLCVValidationConfig] = None,
    timeframe: Optional[str] = None,
) -> ValidationResult:
    """
    Validate OHLCV DataFrame for data quality issues.

    Task S3.1: Main entry-point for OHLCV validation.

    Args:
        df: OHLCV DataFrame to validate
        cfg: Validation configuration. If None, uses defaults.
        timeframe: Optional timeframe string (e.g., "15m", "1h") for gap detection

    Returns:
        ValidationResult with is_valid flag and list of issues

    Example:
        >>> df = pd.read_csv("ohlcv.csv", index_col=0, parse_dates=True)
        >>> result = validate_ohlcv(df, timeframe="15m")
        >>> if not result.is_valid:
        ...     print(result.summary())
    """
    if cfg is None:
        cfg = OHLCVValidationConfig()

    result = ValidationResult(is_valid=True)

    # Basic checks
    if df.empty:
        result.add_issue("empty_dataframe", "error", "DataFrame is empty")
        return result

    # Check required columns
    missing_cols = [col for col in cfg.required_columns if col not in df.columns]
    if missing_cols:
        result.add_issue(
            "missing_columns",
            "error",
            f"Missing required columns: {missing_cols}",
        )
        return result  # Can't proceed without required columns

    # Check for duplicate timestamps
    if cfg.check_duplicate_timestamps and isinstance(df.index, pd.DatetimeIndex):
        duplicates = df.index.duplicated()
        if duplicates.any():
            dup_indices = df.index[duplicates].tolist()
            result.add_issue(
                "duplicate_timestamps",
                "error",
                f"Found {duplicates.sum()} duplicate timestamps",
                affected_rows=dup_indices,
            )

    # Check chronological order
    if cfg.check_chronological_order and isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            result.add_issue(
                "chronological_order",
                "error",
                "Timestamps are not in chronological order",
            )

    # Price validation
    if cfg.check_negative_prices:
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                negative_mask = df[col] < 0
                if negative_mask.any():
                    negative_indices = df.index[negative_mask].tolist()
                    result.add_issue(
                        f"negative_{col}",
                        "error",
                        f"Found {negative_mask.sum()} negative {col} prices",
                        affected_rows=negative_indices,
                    )

    if cfg.check_zero_prices:
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                zero_mask = df[col] == 0
                if zero_mask.any():
                    zero_indices = df.index[zero_mask].tolist()
                    result.add_issue(
                        f"zero_{col}",
                        "error",
                        f"Found {zero_mask.sum()} zero {col} prices",
                        affected_rows=zero_indices,
                    )

    # OHLC relationship validation
    if cfg.check_ohlc_relationship:
        required = ["open", "high", "low", "close"]
        if all(col in df.columns for col in required):
            # high >= low
            invalid_hl = df["high"] < df["low"]
            if invalid_hl.any():
                invalid_indices = df.index[invalid_hl].tolist()
                result.add_issue(
                    "high_low_relationship",
                    "error",
                    f"Found {invalid_hl.sum()} bars where high < low",
                    affected_rows=invalid_indices,
                )

            # open within [low, high]
            invalid_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
            if invalid_open.any():
                invalid_indices = df.index[invalid_open].tolist()
                result.add_issue(
                    "open_range",
                    "error",
                    f"Found {invalid_open.sum()} bars where open is outside [low, high]",
                    affected_rows=invalid_indices,
                )

            # close within [low, high]
            invalid_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
            if invalid_close.any():
                invalid_indices = df.index[invalid_close].tolist()
                result.add_issue(
                    "close_range",
                    "error",
                    f"Found {invalid_close.sum()} bars where close is outside [low, high]",
                    affected_rows=invalid_indices,
                )

    # Volume validation
    if cfg.check_negative_volume and "volume" in df.columns:
        negative_vol = df["volume"] < 0
        if negative_vol.any():
            negative_indices = df.index[negative_vol].tolist()
            result.add_issue(
                "negative_volume",
                "error",
                f"Found {negative_vol.sum()} negative volume values",
                affected_rows=negative_indices,
            )

    if cfg.check_zero_volume and "volume" in df.columns:
        zero_vol = df["volume"] == 0
        if zero_vol.any():
            zero_indices = df.index[zero_vol].tolist()
            result.add_issue(
                "zero_volume",
                "warning",
                f"Found {zero_vol.sum()} zero volume bars",
                affected_rows=zero_indices,
            )

    # Price spike detection
    if cfg.check_price_spikes and "close" in df.columns and len(df) > cfg.price_spike_window:
        returns = df["close"].pct_change()
        rolling_mean = returns.rolling(window=cfg.price_spike_window).mean()
        rolling_std = returns.rolling(window=cfg.price_spike_window).std()

        # Calculate z-scores
        z_scores = (returns - rolling_mean) / rolling_std
        spikes = np.abs(z_scores) > cfg.price_spike_z_threshold

        if spikes.any():
            spike_indices = df.index[spikes].tolist()
            result.add_issue(
                "price_spikes",
                "warning",
                f"Found {spikes.sum()} potential price spikes (|z| > {cfg.price_spike_z_threshold})",
                affected_rows=spike_indices,
            )

    # Missing bars detection (requires timeframe)
    if cfg.check_missing_bars and isinstance(df.index, pd.DatetimeIndex) and timeframe:
        from finantradealgo.validation.timeframe_utils import (
            TIMEFRAME_TO_SECONDS,
            detect_gaps,
        )

        if timeframe in TIMEFRAME_TO_SECONDS:
            gaps = detect_gaps(df.index, timeframe, cfg.max_gap_multiplier)
            if gaps:
                result.add_issue(
                    "missing_bars",
                    "warning",
                    f"Found {len(gaps)} gaps in data (> {cfg.max_gap_multiplier}x expected interval)",
                )

    return result


def validate_ohlcv_strict(
    df: pd.DataFrame,
    cfg: Optional[OHLCVValidationConfig] = None,
    timeframe: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate OHLCV DataFrame and raise exception if validation fails.

    This is a convenience wrapper around validate_ohlcv() that raises
    an exception for strict validation mode.

    Args:
        df: OHLCV DataFrame to validate
        cfg: Validation configuration
        timeframe: Optional timeframe string

    Returns:
        The input DataFrame if validation passes

    Raises:
        ValueError: If validation fails
    """
    result = validate_ohlcv(df, cfg, timeframe)

    if not result.is_valid:
        raise ValueError(f"OHLCV validation failed:\n{result.summary()}")

    return df

"""
External data validation functions.

Task S3.E3: validate_external_series() for flow/sentiment data validation.
"""
import logging
from typing import Optional

import pandas as pd

from finantradealgo.validation.config import ExternalValidationConfig
from finantradealgo.validation.ohlcv_validator import ValidationResult

logger = logging.getLogger(__name__)


def validate_external_series(
    series: pd.Series,
    cfg: Optional[ExternalValidationConfig] = None,
    series_name: str = "external",
) -> ValidationResult:
    """
    Validate external data series (flow, sentiment, etc.).

    Task S3.E3: Validation for external data sources.

    Checks:
    1. Missing data percentage
    2. Value range constraints
    3. Data quality (NaN, Inf values)

    Args:
        series: External data series to validate
        cfg: Validation configuration. If None, uses defaults.
        series_name: Name of the series for error messages

    Returns:
        ValidationResult with validation issues

    Example:
        >>> flow_series = pd.Series([0.5, 0.3, np.nan, 0.7])
        >>> result = validate_external_series(flow_series, series_name="flow")
        >>> if not result.is_valid:
        ...     print(result.summary())
    """
    if cfg is None:
        cfg = ExternalValidationConfig()

    result = ValidationResult(is_valid=True)

    # Basic checks
    if series.empty:
        result.add_issue(
            "empty_series",
            "error",
            f"Series '{series_name}' is empty"
        )
        return result

    # Check missing data
    if cfg.check_missing_data:
        missing_count = series.isna().sum()
        total_count = len(series)
        missing_pct = missing_count / total_count if total_count > 0 else 0

        if missing_pct > cfg.max_missing_pct:
            result.add_issue(
                "excessive_missing_data",
                "error",
                f"Series '{series_name}' has {missing_pct:.1%} missing data "
                f"(threshold: {cfg.max_missing_pct:.1%})"
            )

    # Check for infinite values
    inf_count = 0
    if series.dtype in ['float64', 'float32']:
        inf_mask = pd.isna(series) | (series == float('inf')) | (series == float('-inf'))
        # Actually check for inf separately from NaN
        finite_mask = pd.isna(series) | pd.Series(pd.to_numeric(series, errors='coerce')).apply(lambda x: not pd.isna(x) and not (x == float('inf') or x == float('-inf')))
        try:
            inf_count = (~finite_mask).sum() - series.isna().sum()
        except:
            # Fallback: use numpy
            import numpy as np
            inf_count = np.isinf(series.dropna()).sum()

        if inf_count > 0:
            result.add_issue(
                "infinite_values",
                "error",
                f"Series '{series_name}' contains {inf_count} infinite values"
            )

    # Check value range if configured
    if cfg.check_value_range:
        non_na_series = series.dropna()
        if not non_na_series.empty:
            min_val = non_na_series.min()
            max_val = non_na_series.max()

            if cfg.min_value is not None and min_val < cfg.min_value:
                result.add_issue(
                    "value_below_min",
                    "error",
                    f"Series '{series_name}' has values below minimum "
                    f"({min_val:.4f} < {cfg.min_value:.4f})"
                )

            if cfg.max_value is not None and max_val > cfg.max_value:
                result.add_issue(
                    "value_above_max",
                    "error",
                    f"Series '{series_name}' has values above maximum "
                    f"({max_val:.4f} > {cfg.max_value:.4f})"
                )

    return result


def validate_flow_features(
    df: pd.DataFrame,
    cfg: Optional[ExternalValidationConfig] = None,
) -> ValidationResult:
    """
    Validate flow features DataFrame.

    Task S3.E3: Flow-specific validation.

    Args:
        df: DataFrame with flow features
        cfg: Validation configuration

    Returns:
        ValidationResult with validation issues
    """
    if cfg is None:
        cfg = ExternalValidationConfig()

    result = ValidationResult(is_valid=True)

    if df.empty:
        result.add_issue("empty_dataframe", "error", "Flow DataFrame is empty")
        return result

    # Common flow columns to check
    flow_columns = [col for col in df.columns if 'flow' in col.lower()]

    if not flow_columns:
        result.add_issue(
            "no_flow_columns",
            "warning",
            "No flow-related columns found in DataFrame"
        )
        return result

    # Validate each flow column
    for col in flow_columns:
        col_result = validate_external_series(df[col], cfg, series_name=col)
        if not col_result.is_valid or col_result.warnings_count > 0:
            # Merge issues
            for issue in col_result.issues:
                result.issues.append(issue)
                if issue.severity == "error":
                    result.errors_count += 1
                    result.is_valid = False
                elif issue.severity == "warning":
                    result.warnings_count += 1

    return result


def validate_sentiment_features(
    df: pd.DataFrame,
    cfg: Optional[ExternalValidationConfig] = None,
) -> ValidationResult:
    """
    Validate sentiment features DataFrame.

    Task S3.E3: Sentiment-specific validation.

    Args:
        df: DataFrame with sentiment features
        cfg: Validation configuration

    Returns:
        ValidationResult with validation issues
    """
    if cfg is None:
        cfg = ExternalValidationConfig()

    result = ValidationResult(is_valid=True)

    if df.empty:
        result.add_issue("empty_dataframe", "error", "Sentiment DataFrame is empty")
        return result

    # Common sentiment columns to check
    sentiment_columns = [col for col in df.columns if 'sentiment' in col.lower() or 'fear' in col.lower()]

    if not sentiment_columns:
        result.add_issue(
            "no_sentiment_columns",
            "warning",
            "No sentiment-related columns found in DataFrame"
        )
        return result

    # Validate each sentiment column
    for col in sentiment_columns:
        col_result = validate_external_series(df[col], cfg, series_name=col)
        if not col_result.is_valid or col_result.warnings_count > 0:
            # Merge issues
            for issue in col_result.issues:
                result.issues.append(issue)
                if issue.severity == "error":
                    result.errors_count += 1
                    result.is_valid = False
                elif issue.severity == "warning":
                    result.warnings_count += 1

    return result

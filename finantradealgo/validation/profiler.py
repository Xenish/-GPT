"""
Lightweight data profiling utilities for tabular datasets (e.g., OHLCV or feature matrices).

The profiler summarizes column statistics, captures distribution snapshots,
computes correlations, and reports coverage metrics useful for validation and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ColumnProfile:
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_ratio: float
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    percentiles: dict[str, float]
    unique_count: int | None
    example_values: list[Any] | None


@dataclass(slots=True)
class CorrelationProfile:
    method: str
    matrix: dict[tuple[str, str], float]


@dataclass(slots=True)
class CoverageProfile:
    row_count: int
    start: datetime | None
    end: datetime | None
    time_coverage_ratio: float | None
    expected_frequency: str | None


@dataclass(slots=True)
class DataProfileReport:
    columns: list[ColumnProfile]
    correlation: CorrelationProfile | None
    coverage: CoverageProfile | None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] | None = None


class DataProfiler:
    """
    Creates a structured profile of a pandas DataFrame, including per-column stats,
    distribution snapshots, correlations, and coverage metrics.
    """

    def __init__(self, expected_frequency: str | None = None, index_name: str = "timestamp") -> None:
        self.expected_frequency = expected_frequency
        self.index_name = index_name

    def profile(self, df: pd.DataFrame) -> DataProfileReport:
        """
        Generate a profile report for the provided DataFrame.
        """
        column_profiles = self._profile_columns(df)
        corr_profile = self._profile_correlation(df)
        coverage_profile = self._profile_coverage(df)

        return DataProfileReport(
            columns=column_profiles,
            correlation=corr_profile,
            coverage=coverage_profile,
        )

    # --- Column profiling -----------------------------------------------------
    def _profile_columns(self, df: pd.DataFrame) -> list[ColumnProfile]:
        profiles: list[ColumnProfile] = []
        percentiles_to_compute = [0.05, 0.25, 0.5, 0.75, 0.95]

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            null_count = int(series.isna().sum())
            non_null_count = int(len(series) - null_count)
            null_ratio = float(null_count / len(series)) if len(series) else 0.0

            is_numeric = pd.api.types.is_numeric_dtype(series)
            percentiles: dict[str, float] = {}
            col_min = col_max = col_mean = col_std = None
            unique_count = None
            example_values: list[Any] | None = None

            if is_numeric and not non_null.empty:
                col_min = float(non_null.min())
                col_max = float(non_null.max())
                col_mean = float(non_null.mean())
                col_std = float(non_null.std())
                q = non_null.quantile(percentiles_to_compute)
                percentiles = {f"{int(p * 100)}%": float(q.loc[p]) for p in percentiles_to_compute}
            else:
                unique_count = int(non_null.nunique()) if not non_null.empty else 0

            if not non_null.empty:
                example_values = non_null.iloc[:3].tolist()

            profiles.append(
                ColumnProfile(
                    name=str(col),
                    dtype=str(series.dtype),
                    non_null_count=non_null_count,
                    null_count=null_count,
                    null_ratio=null_ratio,
                    min=col_min,
                    max=col_max,
                    mean=col_mean,
                    std=col_std,
                    percentiles=percentiles,
                    unique_count=unique_count,
                    example_values=example_values,
                )
            )

        return profiles

    # --- Correlation profiling ------------------------------------------------
    def _profile_correlation(self, df: pd.DataFrame) -> CorrelationProfile | None:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return None

        corr_method = "pearson"
        corr_matrix = numeric_df.corr(method=corr_method)
        matrix_dict: dict[tuple[str, str], float] = {}

        for i, col_i in enumerate(corr_matrix.columns):
            for j, col_j in enumerate(corr_matrix.columns):
                if j < i:
                    continue  # store upper triangle including diagonal
                value = corr_matrix.loc[col_i, col_j]
                matrix_dict[(str(col_i), str(col_j))] = float(value)

        return CorrelationProfile(method=corr_method, matrix=matrix_dict)

    # --- Coverage profiling ---------------------------------------------------
    def _profile_coverage(self, df: pd.DataFrame) -> CoverageProfile | None:
        row_count = len(df)
        start = end = None
        time_coverage_ratio = None
        expected_freq = self.expected_frequency

        if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
            start = df.index.min().to_pydatetime()
            end = df.index.max().to_pydatetime()

            if expected_freq:
                expected_index = pd.date_range(start=start, end=end, freq=expected_freq)
                expected_rows = len(expected_index)
                if expected_rows > 0:
                    time_coverage_ratio = row_count / expected_rows
        elif row_count == 0:
            expected_freq = expected_freq or None

        return CoverageProfile(
            row_count=row_count,
            start=start,
            end=end,
            time_coverage_ratio=time_coverage_ratio,
            expected_frequency=expected_freq,
        )


__all__ = [
    "ColumnProfile",
    "CorrelationProfile",
    "CoverageProfile",
    "DataProfileReport",
    "DataProfiler",
]

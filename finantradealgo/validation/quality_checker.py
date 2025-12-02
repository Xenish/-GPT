"""
Data quality checker for tabular time-series data (e.g., OHLCV).

The checker runs a configurable set of validations, emits structured issues,
and produces a summary report that can be consumed by downstream systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from finantradealgo.validation import (
    DataQualityCheckConfig,
    DataQualityIssue,
    DataQualityIssueType,
    DataQualityReport,
    DataQualitySeverity,
    DataSliceRef,
)


@dataclass(slots=True)
class DataContext:
    """
    Optional contextual metadata for the dataset under validation.

    Attributes:
        symbol: Logical identifier for the asset/instrument.
        timeframe: Bar timeframe string (e.g., "1m", "5m", "1h").
        index_name: Name of the index column (default: "timestamp").
        metadata: Arbitrary metadata to carry into issue details.
    """

    symbol: str | None
    timeframe: str | None
    index_name: str = "timestamp"
    metadata: dict[str, Any] | None = None


class DataQualityChecker:
    """
    Runs configurable data quality checks and produces a structured report.

    Checks include:
    - Missing data ratios per column.
    - Outlier detection via z-score thresholding on numeric columns.
    - Duplicate index or row detection.
    - Basic OHLC consistency checks (high/low/open/close relationships).
    - Gap detection for time-based indexes with inferable frequency.
    """

    def __init__(
        self,
        config: DataQualityCheckConfig | None = None,
        context: DataContext | None = None,
    ) -> None:
        self.config = config or DataQualityCheckConfig()
        self.context = context or DataContext(symbol=None, timeframe=None)

    def run_checks(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Run all enabled data quality checks on the given DataFrame.
        """
        issues: list[DataQualityIssue] = []

        if self.config.enable_missing_check:
            issues.extend(self._check_missing(df))
        if self.config.enable_outlier_check:
            issues.extend(self._check_outliers(df))
        if self.config.enable_duplicate_check:
            issues.extend(self._check_duplicates(df))
        if self.config.enable_consistency_check:
            issues.extend(self._check_consistency(df))
        if self.config.enable_gap_check:
            issues.extend(self._check_gaps(df))

        summary = self._build_summary(issues)
        data_ref = self._build_data_ref(df)

        return DataQualityReport(issues=issues, summary=summary, data_ref=data_ref)

    # --- Check implementations -------------------------------------------------
    def _check_missing(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        columns = self._columns_in_scope(df)

        for col in columns:
            missing_ratio = float(df[col].isna().mean())
            if missing_ratio == 0.0:
                continue

            severity = (
                DataQualitySeverity.ERROR
                if self.config.max_missing_ratio is not None
                and missing_ratio > self.config.max_missing_ratio
                else DataQualitySeverity.WARNING
            )
            slice_ref = self._make_slice_ref(df, column=col)
            issues.append(
                DataQualityIssue(
                    issue_type=DataQualityIssueType.MISSING_DATA,
                    severity=severity,
                    message=(
                        f"Missing values detected in column '{col}' "
                        f"(ratio={missing_ratio:.4f})."
                    ),
                    slice_ref=slice_ref,
                    details={
                        "missing_ratio": missing_ratio,
                        "missing_count": int(df[col].isna().sum()),
                    },
                )
            )
        return issues

    def _check_outliers(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        columns = self._numeric_columns_in_scope(df)

        for col in columns:
            series = df[col]
            clean = series.dropna()
            if clean.empty:
                continue

            std = float(clean.std())
            if std == 0.0:
                continue

            mean = float(clean.mean())
            zscores = (clean - mean) / std
            outlier_mask = zscores.abs() > self.config.outlier_zscore_threshold
            if not outlier_mask.any():
                continue

            outlier_index_values = outlier_mask[outlier_mask].index
            outlier_positions = self._indices_to_positions(df, outlier_index_values)
            ratio = len(outlier_positions) / len(series)
            severity = (
                DataQualitySeverity.ERROR
                if self.config.max_missing_ratio is not None
                and ratio > self.config.max_missing_ratio
                else DataQualitySeverity.WARNING
            )

            issues.append(
                DataQualityIssue(
                    issue_type=DataQualityIssueType.OUTLIER,
                    severity=severity,
                    message=(
                        f"Outliers detected in column '{col}' "
                        f"(|z| > {self.config.outlier_zscore_threshold})."
                    ),
                    slice_ref=self._make_slice_ref(
                        df,
                        column=col,
                        indices=self._truncate_indices(outlier_positions),
                    ),
                    details={
                        "outlier_count": len(outlier_positions),
                        "outlier_ratio": ratio,
                        "sample_positions": self._truncate_indices(
                            outlier_positions, limit=10
                        ),
                        "sample_index_values": self._truncate_indices(
                            outlier_index_values, limit=10
                        ),
                    },
                )
            )
        return issues

    def _check_duplicates(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []

        if not df.index.is_unique:
            dup_mask = df.index.duplicated(keep=False)
            dup_positions = [int(i) for i, flag in enumerate(dup_mask) if flag]
            issues.append(
                DataQualityIssue(
                    issue_type=DataQualityIssueType.DUPLICATE,
                    severity=DataQualitySeverity.ERROR,
                    message="Duplicate index entries detected.",
                    slice_ref=self._make_slice_ref(df, indices=dup_positions),
                    details={
                        "duplicate_index_values": df.index[dup_mask].tolist(),
                        "duplicate_positions": self._truncate_indices(
                            dup_positions, limit=25
                        ),
                    },
                )
            )

        row_dup_mask = df.duplicated(keep=False)
        if row_dup_mask.any():
            dup_positions = [int(i) for i, flag in enumerate(row_dup_mask) if flag]
            issues.append(
                DataQualityIssue(
                    issue_type=DataQualityIssueType.DUPLICATE,
                    severity=DataQualitySeverity.WARNING,
                    message="Duplicate rows detected.",
                    slice_ref=self._make_slice_ref(df, indices=dup_positions),
                    details={
                        "duplicate_row_count": int(row_dup_mask.sum()),
                        "duplicate_positions": self._truncate_indices(
                            dup_positions, limit=25
                        ),
                    },
                )
            )

        return issues

    def _check_consistency(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []

        # Case-insensitive mapping for typical OHLC names.
        lower_map = {col.lower(): col for col in df.columns}
        required = {"high", "low", "open", "close"}
        if not required.issubset(lower_map.keys()):
            return issues

        h = df[lower_map["high"]]
        l = df[lower_map["low"]]
        o = df[lower_map["open"]]
        c = df[lower_map["close"]]

        checks = {
            "high_below_low": h < l,
            "high_below_open": h < o,
            "high_below_close": h < c,
            "low_above_open": l > o,
            "low_above_close": l > c,
        }

        for name, mask in checks.items():
            if not mask.any():
                continue
            violating_index_values = mask[mask].index
            violating_positions = self._indices_to_positions(
                df, violating_index_values
            )
            issues.append(
                DataQualityIssue(
                    issue_type=DataQualityIssueType.INCONSISTENCY,
                    severity=DataQualitySeverity.ERROR,
                    message=f"OHLC consistency violation: {name}.",
                    slice_ref=self._make_slice_ref(
                        df, indices=self._truncate_indices(violating_positions)
                    ),
                    details={
                        "violation": name,
                        "violation_count": int(mask.sum()),
                        "sample_positions": self._truncate_indices(
                            violating_positions, limit=10
                        ),
                        "sample_index_values": self._truncate_indices(
                            violating_index_values, limit=10
                        ),
                    },
                )
            )
        return issues

    def _check_gaps(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        if not isinstance(df.index, pd.DatetimeIndex):
            return issues
        if df.index.empty or len(df.index) < 2:
            return issues

        freq = df.index.freqstr or pd.infer_freq(df.index)
        if not freq:
            return issues

        expected_index = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=freq
        )
        missing = expected_index.difference(df.index)
        if missing.empty:
            return issues

        missing_count = len(missing)
        max_gap = self.config.max_gap_length
        if max_gap is None:
            severity = DataQualitySeverity.WARNING
        elif missing_count > max_gap * 2:
            severity = DataQualitySeverity.CRITICAL
        elif missing_count > max_gap:
            severity = DataQualitySeverity.ERROR
        else:
            severity = DataQualitySeverity.WARNING

        issues.append(
            DataQualityIssue(
                issue_type=DataQualityIssueType.GAP,
                severity=severity,
                message="Time index gaps detected.",
                slice_ref=self._make_slice_ref(df),
                details={
                    "missing_count": missing_count,
                    "first_missing": missing.min(),
                    "last_missing": missing.max(),
                    "sample_missing": self._truncate_indices(missing, limit=10),
                    "inferred_freq": freq,
                },
            )
        )
        return issues

    # --- Helpers ---------------------------------------------------------------
    def _columns_in_scope(self, df: pd.DataFrame) -> list[str]:
        if self.config.columns_in_scope:
            return [c for c in df.columns if c in self.config.columns_in_scope]
        return list(df.columns)

    def _numeric_columns_in_scope(self, df: pd.DataFrame) -> list[str]:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if self.config.columns_in_scope:
            numeric_cols = [c for c in numeric_cols if c in self.config.columns_in_scope]
        return numeric_cols

    def _make_slice_ref(
        self, df: pd.DataFrame, column: str | None = None, indices: list[int] | None = None
    ) -> DataSliceRef:
        start = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None
        end = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
        return DataSliceRef(
            symbol=self.context.symbol,
            timeframe=self.context.timeframe,
            column=column,
            start=start,
            end=end,
            indices=indices,
        )

    def _build_data_ref(self, df: pd.DataFrame) -> DataSliceRef | None:
        if df.empty:
            return DataSliceRef(
                symbol=self.context.symbol,
                timeframe=self.context.timeframe,
                column=None,
                start=None,
                end=None,
                indices=None,
            )
        return self._make_slice_ref(df)

    def _indices_to_positions(
        self, df: pd.DataFrame, index_values: Any
    ) -> list[int]:
        """Translate index labels into positional offsets."""
        if isinstance(index_values, pd.Index):
            positions = df.index.get_indexer(index_values)
        else:
            try:
                positions = df.index.get_indexer(list(index_values))
            except TypeError:
                positions = df.index.get_indexer([index_values])
        return [int(pos) for pos in positions if pos != -1]

    def _truncate_indices(self, indices: Any, limit: int = 50) -> list[Any]:
        """
        Return a truncated list of positional indices (or inferred positions).
        """
        if isinstance(indices, pd.Index):
            indices_list = list(indices.to_numpy())
        elif isinstance(indices, list):
            indices_list = indices
        else:
            try:
                indices_list = list(indices)
            except TypeError:
                indices_list = [indices]

        return [int(i) if isinstance(i, (int, float)) else i for i in indices_list[:limit]]

    def _build_summary(self, issues: list[DataQualityIssue]) -> dict[str, Any]:
        counts_by_type: dict[str, int] = {}
        counts_by_column: dict[str, int] = {}
        severity_order = {
            DataQualitySeverity.INFO: 0,
            DataQualitySeverity.WARNING: 1,
            DataQualitySeverity.ERROR: 2,
            DataQualitySeverity.CRITICAL: 3,
        }
        max_severity = DataQualitySeverity.INFO

        for issue in issues:
            counts_by_type.setdefault(issue.issue_type.value, 0)
            counts_by_type[issue.issue_type.value] += 1

            if issue.slice_ref and issue.slice_ref.column:
                counts_by_column.setdefault(issue.slice_ref.column, 0)
                counts_by_column[issue.slice_ref.column] += 1

            if severity_order[issue.severity] > severity_order[max_severity]:
                max_severity = issue.severity

        return {
            "total_issues": len(issues),
            "counts_by_type": counts_by_type,
            "counts_by_column": counts_by_column,
            "max_severity": max_severity.value,
        }


__all__ = ["DataQualityChecker", "DataContext"]

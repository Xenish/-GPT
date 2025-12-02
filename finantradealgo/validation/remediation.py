"""
Automated remediation layer for data quality issues.

The remediator applies conservative, configurable fixes (fill, interpolate,
flag, drop, or fallback) to address detected issues while aiming to remain
idempotent when possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import pandas as pd

from finantradealgo.validation import (
    DataQualityIssue,
    DataQualityIssueType,
    DataQualityReport,
    DataSliceRef,
)


class RemediationActionType(str, Enum):
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    INTERPOLATE = "interpolate"
    FLAG_ANOMALY = "flag_anomaly"
    FALLBACK_SOURCE = "fallback_source"
    DROP_ROWS = "drop_rows"
    DROP_COLUMN = "drop_column"


@dataclass(slots=True)
class RemediationRule:
    issue_type: DataQualityIssueType
    action: RemediationActionType
    columns: list[str] | None = None
    parameters: dict[str, Any] | None = None


@dataclass(slots=True)
class RemediationPlan:
    rules: list[RemediationRule] = field(default_factory=list)
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class RemediationResult:
    df: pd.DataFrame
    applied_actions: list[dict[str, Any]]
    unresolved_issues: list[DataQualityIssue]


class DataRemediator:
    """
    Applies a remediation plan to a dataset using detected data quality issues.

    Design notes:
    - Idempotent when possible: actions like fill/flag should not distort data on repeated runs.
    - Conservative by default: destructive actions (drops) require explicit rules.
    - Fallback loading is pluggable via a user-provided loader; not implemented here.
    """

    def __init__(
        self,
        plan: RemediationPlan,
        fallback_loader: Callable[[DataSliceRef], pd.DataFrame] | None = None,
    ) -> None:
        self.plan = plan
        self.fallback_loader = fallback_loader

    def apply(
        self,
        df: pd.DataFrame,
        report: DataQualityReport | None = None,
        issues: list[DataQualityIssue] | None = None,
    ) -> RemediationResult:
        """
        Apply remediation actions to the DataFrame based on the provided issues or report.
        """
        if report is None and issues is None:
            raise ValueError("Either 'report' or 'issues' must be provided.")

        issue_list = issues if issues is not None else report.issues if report else []
        working_df = df.copy()
        applied_actions: list[dict[str, Any]] = []
        unresolved: list[DataQualityIssue] = []

        for issue in issue_list:
            matching_rules = self._match_rules(issue)
            if not matching_rules:
                unresolved.append(issue)
                continue

            for rule in matching_rules:
                working_df, action_record = self._apply_rule(working_df, issue, rule)
                if action_record:
                    applied_actions.append(action_record)
                else:
                    unresolved.append(issue)

        return RemediationResult(
            df=working_df, applied_actions=applied_actions, unresolved_issues=unresolved
        )

    # --- Rule matching and dispatch ------------------------------------------
    def _match_rules(self, issue: DataQualityIssue) -> list[RemediationRule]:
        rules: list[RemediationRule] = []
        for rule in self.plan.rules:
            if rule.issue_type != issue.issue_type:
                continue
            if rule.columns and issue.slice_ref and issue.slice_ref.column:
                if issue.slice_ref.column not in rule.columns:
                    continue
            rules.append(rule)
        return rules

    def _apply_rule(
        self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if rule.action == RemediationActionType.FILL_FORWARD:
            return self._apply_fill_forward(df, issue, rule)
        if rule.action == RemediationActionType.FILL_BACKWARD:
            return self._apply_fill_backward(df, issue, rule)
        if rule.action == RemediationActionType.INTERPOLATE:
            return self._apply_interpolate(df, issue, rule)
        if rule.action == RemediationActionType.FLAG_ANOMALY:
            return self._apply_flag_anomaly(df, issue, rule)
        if rule.action == RemediationActionType.FALLBACK_SOURCE:
            return self._apply_fallback_source(df, issue, rule)
        if rule.action == RemediationActionType.DROP_ROWS:
            return self._apply_drop_rows(df, issue, rule)
        if rule.action == RemediationActionType.DROP_COLUMN:
            return self._apply_drop_column(df, issue, rule)

        return df, None

    # --- Action implementations ----------------------------------------------
    def _apply_fill_forward(
        self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        target_cols = self._target_columns(df, issue, rule)
        filled = df.copy()
        limit = None
        if rule.parameters:
            limit = rule.parameters.get("limit")
        filled[target_cols] = filled[target_cols].ffill(limit=limit)
        return filled, self._make_action_record(issue, rule, {"columns": target_cols, "limit": limit})

    def _apply_fill_backward(
        self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        target_cols = self._target_columns(df, issue, rule)
        filled = df.copy()
        limit = None
        if rule.parameters:
            limit = rule.parameters.get("limit")
        filled[target_cols] = filled[target_cols].bfill(limit=limit)
        return filled, self._make_action_record(issue, rule, {"columns": target_cols, "limit": limit})

    def _apply_interpolate(
        self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        target_cols = self._target_columns(df, issue, rule)
        method = "linear"
        limit_direction = "both"
        limit = None
        if rule.parameters:
            method = rule.parameters.get("method", method)
            limit_direction = rule.parameters.get("limit_direction", limit_direction)
            limit = rule.parameters.get("limit", limit)

        interpolated = df.copy()
        interpolated[target_cols] = interpolated[target_cols].interpolate(
            method=method, limit_direction=limit_direction, limit=limit
        )
        return interpolated, self._make_action_record(
            issue,
            rule,
            {"columns": target_cols, "method": method, "limit_direction": limit_direction, "limit": limit},
        )

    def _apply_flag_anomaly(
        self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        flagged = df.copy()
        flag_col = rule.parameters.get("flag_column", "anomaly_flag") if rule.parameters else "anomaly_flag"
        flagged[flag_col] = False if flag_col not in flagged else flagged[flag_col]

        indices = self._extract_indices(issue, df)
        if indices:
            flagged.loc[indices, flag_col] = True

        return flagged, self._make_action_record(issue, rule, {"flag_column": flag_col, "indices": indices})

    def _apply_fallback_source(
        self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if self.fallback_loader is None:
            # Cannot apply fallback without a loader; mark unresolved
            return df, None

        target_cols = self._target_columns(df, issue, rule)
        slice_ref = issue.slice_ref or DataSliceRef(
            symbol=None, timeframe=None, column=None, start=None, end=None, indices=None
        )
        fallback_df = self.fallback_loader(slice_ref)
        merged = df.copy()
        indices = self._extract_indices(issue, df)

        if indices:
            merged.loc[indices, target_cols] = merged.loc[indices, target_cols].combine_first(
                fallback_df[target_cols]
                if not fallback_df.empty and set(target_cols).issubset(fallback_df.columns)
                else merged.loc[indices, target_cols]
            )

        return merged, self._make_action_record(
            issue,
            rule,
            {"columns": target_cols, "indices": indices, "fallback_rows": len(fallback_df)},
        )

    def _apply_drop_rows(
        self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        indices = self._extract_indices(issue, df)
        dropped = df.copy()
        if indices:
            dropped = dropped.drop(index=indices)
        return dropped, self._make_action_record(issue, rule, {"dropped_count": len(indices), "indices": indices})

    def _apply_drop_column(
        self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        target_cols = self._target_columns(df, issue, rule)
        dropped = df.copy()
        for col in target_cols:
            if col in dropped:
                dropped = dropped.drop(columns=[col])
        return dropped, self._make_action_record(issue, rule, {"dropped_columns": target_cols})

    # --- Helpers --------------------------------------------------------------
    def _target_columns(self, df: pd.DataFrame, issue: DataQualityIssue, rule: RemediationRule) -> list[str]:
        if rule.columns:
            return [c for c in rule.columns if c in df.columns]
        if issue.slice_ref and issue.slice_ref.column and issue.slice_ref.column in df.columns:
            return [issue.slice_ref.column]
        return list(df.columns)

    def _extract_indices(self, issue: DataQualityIssue, df: pd.DataFrame) -> list[Any]:
        if issue.slice_ref and issue.slice_ref.indices is not None:
            # Assume indices are positional; translate to index labels.
            try:
                return df.index[issue.slice_ref.indices].tolist()
            except Exception:
                return []
        return []

    def _make_action_record(
        self, issue: DataQualityIssue, rule: RemediationRule, details: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "issue_type": issue.issue_type.value,
            "action_type": rule.action.value,
            "column": issue.slice_ref.column if issue.slice_ref else None,
            "indices": issue.slice_ref.indices if issue.slice_ref else None,
            "details": details,
        }


__all__ = [
    "RemediationActionType",
    "RemediationRule",
    "RemediationPlan",
    "RemediationResult",
    "DataRemediator",
]

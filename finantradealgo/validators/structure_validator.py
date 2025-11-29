"""
Consistency validator for market structure signals.

This module provides lightweight validation to catch logical inconsistencies
in market structure computations, helping prevent regressions during refactoring.
"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from finantradealgo.market_structure.config import MarketStructureConfig
from finantradealgo.market_structure.types import Zone


@dataclass
class ValidationViolation:
    """
    Represents a single consistency violation found during validation.

    Attributes:
        rule: Name of the validation rule that was violated
        timestamp: Index/timestamp where the violation occurred
        message: Human-readable description of the violation
        severity: Severity level ('error', 'warning')
    """
    rule: str
    timestamp: any
    message: str
    severity: str = 'error'

    def __str__(self):
        return f"[{self.severity.upper()}] {self.rule} at {self.timestamp}: {self.message}"


def validate_market_structure(
    df: pd.DataFrame,
    swings: Optional[List] = None,
    zones: Optional[List[Zone]] = None,
    cfg: Optional[MarketStructureConfig] = None,
) -> List[ValidationViolation]:
    """
    Validates market structure signals for logical consistency.

    This function performs lightweight checks to ensure market structure
    signals follow their expected rules. It's designed to catch bugs
    introduced during refactoring.

    Args:
        df: DataFrame with market structure columns (ms_*)
        swings: Optional list of SwingPoint objects (not currently used)
        zones: Optional list of Zone objects (not currently used)
        cfg: Optional MarketStructureConfig for threshold values

    Returns:
        List of ValidationViolation objects (empty if no violations found)

    Expected columns in df:
        - ms_swing_high, ms_swing_low
        - ms_trend_regime
        - ms_bos_up, ms_bos_down, ms_choch
        - ms_fvg_up, ms_fvg_down
        - ms_zone_demand, ms_zone_supply
    """
    violations = []

    if cfg is None:
        cfg = MarketStructureConfig()

    # Required columns check
    required_cols = [
        'ms_swing_high', 'ms_swing_low', 'ms_trend_regime',
        'ms_bos_up', 'ms_bos_down', 'ms_choch',
        'ms_fvg_up', 'ms_fvg_down',
        'ms_zone_demand', 'ms_zone_supply'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        violations.append(ValidationViolation(
            rule='required_columns',
            timestamp=None,
            message=f"Missing required columns: {missing_cols}",
            severity='error'
        ))
        return violations  # Can't proceed without required columns

    # --- Rule 1: Swing points should not overlap ---
    # A bar cannot be both a swing high AND swing low
    swing_overlap = df[
        (df['ms_swing_high'] == 1) & (df['ms_swing_low'] == 1)
    ]

    for idx in swing_overlap.index:
        violations.append(ValidationViolation(
            rule='swing_overlap',
            timestamp=idx,
            message="Bar marked as both swing_high and swing_low",
            severity='error'
        ))

    # --- Rule 2: FVG should not overlap ---
    # A bar cannot have both FVG up AND FVG down (both > 0)
    fvg_overlap = df[
        (df['ms_fvg_up'] > 0) & (df['ms_fvg_down'] > 0)
    ]

    for idx in fvg_overlap.index:
        fvg_up = df.loc[idx, 'ms_fvg_up']
        fvg_down = df.loc[idx, 'ms_fvg_down']
        violations.append(ValidationViolation(
            rule='fvg_overlap',
            timestamp=idx,
            message=f"Bar has both fvg_up ({fvg_up}) and fvg_down ({fvg_down})",
            severity='error'
        ))

    # --- Rule 3: BoS up requires uptrend ---
    # If ms_bos_up == 1, then ms_trend_regime should be 1 (uptrend)
    bos_up_violations = df[
        (df['ms_bos_up'] == 1) & (df['ms_trend_regime'] != 1)
    ]

    for idx in bos_up_violations.index:
        trend = df.loc[idx, 'ms_trend_regime']
        violations.append(ValidationViolation(
            rule='bos_up_trend',
            timestamp=idx,
            message=f"BOS up detected but trend_regime is {trend} (expected 1)",
            severity='error'
        ))

    # --- Rule 4: BoS down requires downtrend ---
    # If ms_bos_down == 1, then ms_trend_regime should be -1 (downtrend)
    bos_down_violations = df[
        (df['ms_bos_down'] == 1) & (df['ms_trend_regime'] != -1)
    ]

    for idx in bos_down_violations.index:
        trend = df.loc[idx, 'ms_trend_regime']
        violations.append(ValidationViolation(
            rule='bos_down_trend',
            timestamp=idx,
            message=f"BOS down detected but trend_regime is {trend} (expected -1)",
            severity='error'
        ))

    # --- Rule 5: BoS should not happen on same bar ---
    # A bar cannot have both BoS up AND BoS down
    bos_overlap = df[
        (df['ms_bos_up'] == 1) & (df['ms_bos_down'] == 1)
    ]

    for idx in bos_overlap.index:
        violations.append(ValidationViolation(
            rule='bos_overlap',
            timestamp=idx,
            message="Bar marked as both bos_up and bos_down",
            severity='error'
        ))

    # --- Rule 6: ChoCh should correspond to trend change ---
    # If ms_choch == 1, trend_regime should differ from previous bar
    if len(df) > 1:
        trend_diff = df['ms_trend_regime'].diff() != 0
        choch_flags = df['ms_choch'] == 1

        # ChoCh without trend change (except first bar)
        false_choch = df.iloc[1:][(choch_flags.iloc[1:]) & (~trend_diff.iloc[1:])]

        for idx in false_choch.index:
            violations.append(ValidationViolation(
                rule='choch_trend_change',
                timestamp=idx,
                message="ChoCh detected but trend_regime did not change from previous bar",
                severity='error'
            ))

    # --- Rule 7: Trend regime should be valid values ---
    # ms_trend_regime should only be -1, 0, or 1
    invalid_trend = df[
        ~df['ms_trend_regime'].isin([-1.0, 0.0, 1.0])
    ]

    for idx in invalid_trend.index:
        trend = df.loc[idx, 'ms_trend_regime']
        violations.append(ValidationViolation(
            rule='trend_regime_values',
            timestamp=idx,
            message=f"Invalid trend_regime value: {trend} (expected -1, 0, or 1)",
            severity='error'
        ))

    # --- Rule 8: Binary flags should be 0 or 1 ---
    # Note: FVG columns are NOT binary - they contain the gap size as a float
    binary_cols = [
        'ms_swing_high', 'ms_swing_low',
        'ms_bos_up', 'ms_bos_down', 'ms_choch',
    ]

    for col in binary_cols:
        invalid_values = df[~df[col].isin([0, 1, 0.0, 1.0])]

        for idx in invalid_values.index:
            value = df.loc[idx, col]
            violations.append(ValidationViolation(
                rule='binary_flag_values',
                timestamp=idx,
                message=f"{col} has invalid value {value} (expected 0 or 1)",
                severity='error'
            ))

    # --- Rule 8b: FVG values should be non-negative ---
    # FVG columns contain gap size (0 or positive float)
    negative_fvg_up = df[df['ms_fvg_up'] < 0]
    for idx in negative_fvg_up.index:
        value = df.loc[idx, 'ms_fvg_up']
        violations.append(ValidationViolation(
            rule='fvg_value_positive',
            timestamp=idx,
            message=f"ms_fvg_up is negative: {value}",
            severity='error'
        ))

    negative_fvg_down = df[df['ms_fvg_down'] < 0]
    for idx in negative_fvg_down.index:
        value = df.loc[idx, 'ms_fvg_down']
        violations.append(ValidationViolation(
            rule='fvg_value_positive',
            timestamp=idx,
            message=f"ms_fvg_down is negative: {value}",
            severity='error'
        ))

    # --- Rule 9: Zone strengths should be non-negative ---
    # ms_zone_demand and ms_zone_supply should be >= 0
    negative_demand = df[df['ms_zone_demand'] < 0]
    for idx in negative_demand.index:
        value = df.loc[idx, 'ms_zone_demand']
        violations.append(ValidationViolation(
            rule='zone_strength_positive',
            timestamp=idx,
            message=f"ms_zone_demand is negative: {value}",
            severity='error'
        ))

    negative_supply = df[df['ms_zone_supply'] < 0]
    for idx in negative_supply.index:
        value = df.loc[idx, 'ms_zone_supply']
        violations.append(ValidationViolation(
            rule='zone_strength_positive',
            timestamp=idx,
            message=f"ms_zone_supply is negative: {value}",
            severity='error'
        ))

    # --- Rule 10: Swing points should be relatively sparse ---
    # Warning if more than 20% of bars are swing points
    total_bars = len(df)
    swing_high_count = (df['ms_swing_high'] == 1).sum()
    swing_low_count = (df['ms_swing_low'] == 1).sum()

    if swing_high_count > total_bars * 0.2:
        violations.append(ValidationViolation(
            rule='swing_density',
            timestamp=None,
            message=f"Too many swing highs: {swing_high_count}/{total_bars} ({100*swing_high_count/total_bars:.1f}%)",
            severity='warning'
        ))

    if swing_low_count > total_bars * 0.2:
        violations.append(ValidationViolation(
            rule='swing_density',
            timestamp=None,
            message=f"Too many swing lows: {swing_low_count}/{total_bars} ({100*swing_low_count/total_bars:.1f}%)",
            severity='warning'
        ))

    # --- Rule 11: FVG should be relatively sparse ---
    # Warning if more than 15% of bars have FVG (value > 0)
    fvg_up_count = (df['ms_fvg_up'] > 0).sum()
    fvg_down_count = (df['ms_fvg_down'] > 0).sum()

    if fvg_up_count > total_bars * 0.15:
        violations.append(ValidationViolation(
            rule='fvg_density',
            timestamp=None,
            message=f"Too many FVG ups: {fvg_up_count}/{total_bars} ({100*fvg_up_count/total_bars:.1f}%)",
            severity='warning'
        ))

    if fvg_down_count > total_bars * 0.15:
        violations.append(ValidationViolation(
            rule='fvg_density',
            timestamp=None,
            message=f"Too many FVG downs: {fvg_down_count}/{total_bars} ({100*fvg_down_count/total_bars:.1f}%)",
            severity='warning'
        ))

    return violations


def validate_and_raise(
    df: pd.DataFrame,
    swings: Optional[List] = None,
    zones: Optional[List[Zone]] = None,
    cfg: Optional[MarketStructureConfig] = None,
    allow_warnings: bool = True,
) -> None:
    """
    Validates market structure and raises an exception if errors are found.

    Args:
        df: DataFrame with market structure columns
        swings: Optional list of SwingPoint objects
        zones: Optional list of Zone objects
        cfg: Optional MarketStructureConfig
        allow_warnings: If False, warnings will also raise an exception

    Raises:
        ValueError: If any validation errors (or warnings if allow_warnings=False) are found
    """
    violations = validate_market_structure(df, swings, zones, cfg)

    if not violations:
        return

    # Filter violations by severity
    errors = [v for v in violations if v.severity == 'error']
    warnings = [v for v in violations if v.severity == 'warning']

    # Build error message
    messages = []

    if errors:
        messages.append(f"Found {len(errors)} validation error(s):")
        for error in errors:
            messages.append(f"  - {error}")

    if warnings and not allow_warnings:
        messages.append(f"Found {len(warnings)} validation warning(s):")
        for warning in warnings:
            messages.append(f"  - {warning}")

    if errors or (warnings and not allow_warnings):
        raise ValueError("\n".join(messages))

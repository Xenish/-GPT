"""
Dependency checking for features required by strategies.

Task S1.E4: Validate that strategies have required features available.
"""
import logging
from typing import List, Set
import pandas as pd

from finantradealgo.market_structure.types import MarketStructureColumns

logger = logging.getLogger(__name__)


def check_market_structure_dependencies(
    df: pd.DataFrame,
    strategy_name: str,
    requires_market_structure: bool,
    strict: bool = False,
) -> bool:
    """
    Check if DataFrame has required market structure features for a strategy.

    Args:
        df: Feature DataFrame to validate
        strategy_name: Name of the strategy for logging
        requires_market_structure: Whether strategy requires market structure features
        strict: If True, raise ValueError on missing features. If False, only log warning.

    Returns:
        True if all dependencies are satisfied, False otherwise

    Raises:
        ValueError: If strict=True and required features are missing
    """
    if not requires_market_structure:
        # Strategy doesn't need market structure
        return True

    cols = MarketStructureColumns()
    required_columns = [
        cols.swing_high,
        cols.swing_low,
        cols.trend_regime,
        cols.fvg_up,
        cols.fvg_down,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        msg = (
            f"Strategy '{strategy_name}' requires market structure features, "
            f"but the following columns are missing from DataFrame: {missing_columns}. "
            f"Enable market structure features in config (features.use_market_structure=true) "
            f"or ensure the market structure engine is running."
        )

        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    logger.debug(
        f"Strategy '{strategy_name}' market structure dependencies satisfied. "
        f"Found columns: {required_columns}"
    )
    return True


def check_microstructure_dependencies(
    df: pd.DataFrame,
    strategy_name: str,
    requires_microstructure: bool,
    strict: bool = False,
) -> bool:
    """
    Check if DataFrame has required microstructure features for a strategy.

    Args:
        df: Feature DataFrame to validate
        strategy_name: Name of the strategy for logging
        requires_microstructure: Whether strategy requires microstructure features
        strict: If True, raise ValueError on missing features. If False, only log warning.

    Returns:
        True if all dependencies are satisfied, False otherwise

    Raises:
        ValueError: If strict=True and required features are missing
    """
    if not requires_microstructure:
        # Strategy doesn't need microstructure
        return True

    # Common microstructure feature prefixes
    microstructure_prefixes = ["imb_", "sweep_", "burst_", "exhaustion_", "chop_"]

    found_microstructure = any(
        any(col.startswith(prefix) for prefix in microstructure_prefixes)
        for col in df.columns
    )

    if not found_microstructure:
        msg = (
            f"Strategy '{strategy_name}' requires microstructure features, "
            f"but no microstructure columns were found in DataFrame. "
            f"Enable microstructure features in config (features.use_microstructure=true) "
            f"or ensure the microstructure engine is running."
        )

        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    logger.debug(
        f"Strategy '{strategy_name}' microstructure dependencies satisfied."
    )
    return True


def validate_strategy_dependencies(
    df: pd.DataFrame,
    strategy_name: str,
    uses_market_structure: bool = False,
    uses_microstructure: bool = False,
    strict: bool = False,
) -> bool:
    """
    Validate all feature dependencies for a strategy.

    Args:
        df: Feature DataFrame to validate
        strategy_name: Name of the strategy for logging
        uses_market_structure: Whether strategy uses market structure features
        uses_microstructure: Whether strategy uses microstructure features
        strict: If True, raise ValueError on missing features. If False, only log warning.

    Returns:
        True if all dependencies are satisfied, False otherwise

    Raises:
        ValueError: If strict=True and required features are missing

    Example:
        >>> from finantradealgo.strategies.strategy_engine import STRATEGY_SPECS
        >>>
        >>> strategy_spec = STRATEGY_SPECS["rule"]
        >>> meta = strategy_spec.meta
        >>>
        >>> validate_strategy_dependencies(
        ...     df=feature_df,
        ...     strategy_name=meta.name,
        ...     uses_market_structure=meta.uses_market_structure,
        ...     uses_microstructure=meta.uses_microstructure,
        ...     strict=False,
        ... )
    """
    ms_ok = check_market_structure_dependencies(
        df, strategy_name, uses_market_structure, strict
    )

    micro_ok = check_microstructure_dependencies(
        df, strategy_name, uses_microstructure, strict
    )

    return ms_ok and micro_ok


def get_missing_market_structure_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of missing market structure feature columns.

    Args:
        df: DataFrame to check

    Returns:
        List of missing market structure column names
    """
    cols = MarketStructureColumns()
    expected_columns = [
        cols.price_smooth,
        cols.swing_high,
        cols.swing_low,
        cols.trend_regime,
        cols.chop_regime,
        cols.fvg_up,
        cols.fvg_down,
        cols.zone_demand,
        cols.zone_supply,
        cols.bos_up,
        cols.bos_down,
        cols.choch,
    ]

    return [col for col in expected_columns if col not in df.columns]


def has_market_structure_features(df: pd.DataFrame, minimal: bool = True) -> bool:
    """
    Check if DataFrame has market structure features.

    Args:
        df: DataFrame to check
        minimal: If True, check for minimal set of features. If False, check for all.

    Returns:
        True if market structure features are present
    """
    if minimal:
        cols = MarketStructureColumns()
        minimal_columns = [cols.swing_high, cols.swing_low, cols.trend_regime]
        return all(col in df.columns for col in minimal_columns)
    else:
        missing = get_missing_market_structure_features(df)
        return len(missing) == 0

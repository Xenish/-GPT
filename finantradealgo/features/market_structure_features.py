import pandas as pd
from typing import Optional

from finantradealgo.market_structure.engine import MarketStructureEngine, MarketStructureResult
from finantradealgo.market_structure.config import MarketStructureConfig


def add_market_structure_features(
    df: pd.DataFrame,
    cfg: Optional[MarketStructureConfig] = None,
) -> pd.DataFrame:
    """
    Computes and adds market structure features to the main DataFrame
    using the dedicated MarketStructureEngine.

    Args:
        df: Input OHLCV DataFrame
        cfg: Optional MarketStructureConfig, defaults to MarketStructureConfig()

    Returns:
        DataFrame with added market structure columns (ms_*)

    Note:
        If you need access to the Zone objects, use
        compute_market_structure_with_zones() instead.
    """
    cfg = cfg or MarketStructureConfig()
    engine = MarketStructureEngine(cfg)
    result = engine.compute_df(df)

    return pd.concat([df, result.features], axis=1)


def compute_market_structure_with_zones(
    df: pd.DataFrame,
    cfg: Optional[MarketStructureConfig] = None,
) -> MarketStructureResult:
    """
    Computes market structure features and returns both DataFrame and Zone objects.

    Args:
        df: Input OHLCV DataFrame
        cfg: Optional MarketStructureConfig, defaults to MarketStructureConfig()

    Returns:
        MarketStructureResult containing:
            - features: DataFrame with market structure columns (ms_*)
            - zones: List of Zone objects (supply/demand)

    Note:
        Use this function when you need access to the Zone objects.
        For simple feature addition, use add_market_structure_features().
    """
    cfg = cfg or MarketStructureConfig()
    engine = MarketStructureEngine(cfg)
    return engine.compute_df(df)


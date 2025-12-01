"""
Microstructure features wrapper for feature pipeline integration.

Task S2.5: Feature pipeline integration - connects microstructure engine to pipeline.
"""
import pandas as pd
from typing import Optional

from finantradealgo.microstructure.config import MicrostructureConfig
from finantradealgo.microstructure.microstructure_engine import compute_microstructure_df


def add_microstructure_features(
    df: pd.DataFrame,
    cfg: Optional[MicrostructureConfig] = None
) -> pd.DataFrame:
    """
    Compute and join microstructure features to the main DataFrame.

    This is the feature pipeline integration point for microstructure signals.
    Called by build_feature_pipeline() when use_microstructure=True.

    Task S2.5: Feature pipeline integration.

    Args:
        df: Input OHLCV DataFrame
        cfg: Microstructure configuration. If None, uses defaults.

    Returns:
        DataFrame with original columns + microstructure feature columns (ms_*)

    Example:
        >>> df = pd.read_csv("ohlcv.csv", index_col=0, parse_dates=True)
        >>> cfg = MicrostructureConfig()
        >>> df_with_ms = add_microstructure_features(df, cfg)
        >>> assert "ms_chop" in df_with_ms.columns
        >>> assert "ms_burst_up" in df_with_ms.columns
    """
    # Delegate to single entry-point
    ms_df = compute_microstructure_df(df, cfg)
    return pd.concat([df, ms_df], axis=1)
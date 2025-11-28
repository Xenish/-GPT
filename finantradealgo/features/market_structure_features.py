import pandas as pd
from typing import Optional, Tuple, List

from finantradealgo.market_structure.engine import MarketStructureEngine
from finantradealgo.market_structure.config import MarketStructureConfig
from finantradealgo.market_structure.types import Zone


def add_market_structure_features(
    df: pd.DataFrame,
    cfg: Optional[MarketStructureConfig] = None,
) -> Tuple[pd.DataFrame, List[Zone]]:
    """
    Computes and adds market structure features to the main DataFrame
    using the dedicated MarketStructureEngine.

    Returns:
        A tuple of (DataFrame with features, list of Zone objects).
    """
    cfg = cfg or MarketStructureConfig()
    engine = MarketStructureEngine(cfg)
    ms_df, zones = engine.compute_df(df)
    
    combined_df = pd.concat([df, ms_df], axis=1)
    return combined_df, zones


import pandas as pd
from typing import Optional

from finantradealgo.microstructure.config import MicrostructureConfig
from finantradealgo.microstructure.microstructure_engine import compute_microstructure_df

def add_microstructure_features(df: pd.DataFrame, cfg: Optional[MicrostructureConfig] = None) -> pd.DataFrame:
    """
    Wrapper to compute and join microstructure features to the main DataFrame.
    """
    ms_df = compute_microstructure_df(df, cfg)
    return pd.concat([df, ms_df], axis=1)
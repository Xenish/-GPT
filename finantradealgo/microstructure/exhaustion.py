from typing import Tuple

import numpy as np
import pandas as pd

from finantradealgo.microstructure.config import ExhaustionConfig


def compute_exhaustion(
    close: pd.Series, volume: pd.Series, cfg: ExhaustionConfig
) -> Tuple[pd.Series, pd.Series]:
    """
    Computes trend exhaustion signals for upward and downward moves.

    Exhaustion is defined as a sustained trend (a minimum number of
    consecutive bars in the same direction) occurring on declining volume
    (volume z-score is below a certain threshold).

    Args:
        close: Series of close prices.
        volume: Series of volume data.
        cfg: Configuration object for the calculation.

    Returns:
        A tuple of two boolean series: (exhaustion_up, exhaustion_down).
    """
    # 1. Calculate price direction
    direction = np.sign(close.diff()).fillna(0)

    # 2. Calculate consecutive up/down bars using the groupby-cumsum trick
    # A new group is formed every time the direction changes
    blocks = (direction.ne(direction.shift())).cumsum()
    consecutive_counts = direction.groupby(blocks).cumsum().abs()

    consec_up = consecutive_counts.where(direction > 0, 0)
    consec_down = consecutive_counts.where(direction < 0, 0)

    # 3. Calculate rolling volume z-score
    vol_mean = volume.rolling(
        window=cfg.volume_z_score_window, min_periods=cfg.volume_z_score_window
    ).mean()
    vol_std = volume.rolling(
        window=cfg.volume_z_score_window, min_periods=cfg.volume_z_score_window
    ).std()
    
    volume_z_score = (volume - vol_mean) / (vol_std + 1e-9)
    
    # 4. Identify exhaustion points
    is_low_volume = volume_z_score <= cfg.volume_z_threshold

    exhaustion_up = (
        (consec_up >= cfg.min_consecutive_bars) & is_low_volume
    ).astype(int)
    
    exhaustion_down = (
        (consec_down >= cfg.min_consecutive_bars) & is_low_volume
    ).astype(int)

    return exhaustion_up, exhaustion_down

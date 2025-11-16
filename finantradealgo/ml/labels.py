from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LabelConfig:
    horizon: int = 5
    pos_threshold: float = 0.003
    fee_slippage: float = 0.001


def add_long_only_labels(
    df: pd.DataFrame,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = LabelConfig()

    df = df.copy()

    fwd = df["close"].shift(-config.horizon) / df["close"] - 1.0
    df["fwd_return"] = fwd

    thr = config.pos_threshold + config.fee_slippage
    label = (fwd > thr).astype(float)

    if config.horizon > 0:
        label.iloc[-config.horizon :] = np.nan

    df["label_long"] = label
    return df

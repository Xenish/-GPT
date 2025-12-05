from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LabelConfig:
    """
    Label configuration for long-only classification/regression targets.

    Attributes:
        horizon: Forward-looking bars to compute returns.
        pos_threshold: Threshold above which return is considered positive.
        neg_threshold: Threshold below which return is negative (reserved for future use).
        fee_slippage: Cost buffer added to thresholds.
        method: Labeling method (currently 'simple'; extendable for other schemes).
    """

    horizon: int = 5
    pos_threshold: float = 0.003
    neg_threshold: float = -0.003
    fee_slippage: float = 0.001
    method: str = "simple"

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "LabelConfig":
        data = data or {}
        return cls(
            horizon=int(data.get("horizon_bars", data.get("horizon", cls.horizon))),
            pos_threshold=float(
                data.get(
                    "up_threshold",
                    data.get("pos_threshold", cls.pos_threshold),
                )
            ),
            neg_threshold=float(
                data.get(
                    "down_threshold",
                    data.get("neg_threshold", cls.neg_threshold),
                )
            ),
            fee_slippage=float(data.get("fee_slippage", cls.fee_slippage)),
            method=str(data.get("method", cls.method)),
        )


def add_long_only_labels(
    df: pd.DataFrame,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """
    Add forward return and long-only labels to an OHLCV DataFrame.

    Args:
        df: DataFrame with at least a 'close' column.
        config: LabelConfig describing horizon/thresholds.

    Returns:
        DataFrame with:
          - fwd_return: forward return over horizon bars
          - label_long: 1.0 if fwd_return > (pos_threshold + fee_slippage), 0.0 otherwise
                        last `horizon` rows set to NaN
    """
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

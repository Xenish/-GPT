from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


@dataclass
class DataConfig:
    timestamp_col: str = "timestamp"
    tz: str | None = None


def load_ohlcv_csv(path: str, config: DataConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = DataConfig()

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], utc=True)
    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    return df

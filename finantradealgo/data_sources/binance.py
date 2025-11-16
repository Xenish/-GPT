from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


BINANCE_FUTURES_BASE = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"


@dataclass
class BinanceKlinesConfig:
    """
    Basic config for Binance klines fetch.
    """

    symbol: str = "AIAUSDT"
    interval: str = "1h"
    limit: int = 1000
    base_url: str = BINANCE_FUTURES_BASE
    endpoint: str = KLINES_ENDPOINT

    def params(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol.upper().replace(".", ""),
            "interval": self.interval,
            "limit": self.limit,
        }


def _request_klines(
    config: BinanceKlinesConfig,
    extra_params: Optional[Dict[str, Any]] = None,
) -> List[List[Any]]:
    params = config.params()
    if extra_params:
        params.update(extra_params)

    url = f"{config.base_url}{config.endpoint}"
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list):
        raise ValueError(f"Unexpected response: {data}")
    return data


def _klines_to_dataframe(data: List[List[Any]]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(data, columns=columns)
    if df.empty:
        return df

    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.drop(columns=["open_time"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def fetch_klines(config: BinanceKlinesConfig) -> pd.DataFrame:
    """
    Fetch klines from Binance futures API and return OHLCV DataFrame.
    """

    data = _request_klines(config)
    return _klines_to_dataframe(data)


def fetch_klines_series(
    config: BinanceKlinesConfig,
    total_limit: int,
) -> pd.DataFrame:
    """
    Fetch multiple klines requests (most recent data) until total_limit candles collected.
    """
    if total_limit <= 0:
        raise ValueError("total_limit must be positive")

    frames: list[pd.DataFrame] = []
    remaining = total_limit
    end_time: Optional[int] = None

    while remaining > 0:
        batch_limit = min(config.limit, remaining)
        extra: Dict[str, Any] = {"limit": batch_limit}
        if end_time is not None:
            extra["endTime"] = end_time

        data = _request_klines(config, extra_params=extra)
        if not data:
            break

        df_chunk = _klines_to_dataframe(data)
        if df_chunk.empty:
            break

        frames.append(df_chunk)
        remaining -= len(df_chunk)

        first_open_time = int(data[0][0])
        end_time = first_open_time - 1

        if len(data) < batch_limit:
            # No more historical data available
            break

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df_all) > total_limit:
        df_all = df_all.iloc[-total_limit:].reset_index(drop=True)

    return df_all


def fetch_and_save_klines(
    config: BinanceKlinesConfig,
    output_path: str | Path,
) -> Path:
    """
    Convenience helper to fetch data and persist as CSV.
    """

    df = fetch_klines(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

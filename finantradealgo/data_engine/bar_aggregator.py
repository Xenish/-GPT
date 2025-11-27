from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .live_data_source import Bar


def _parse_tf(tf: str) -> pd.Timedelta:
    unit = tf[-1].lower()
    value = int(tf[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "s":
        return pd.Timedelta(seconds=value)
    raise ValueError(f"Unsupported timeframe: {tf}")


@dataclass
class _Bucket:
    symbol: str
    open_time: pd.Timestamp
    bars: list


class BarAggregator:
    def __init__(self, target_timeframe: str) -> None:
        self.target_tf = target_timeframe
        self.tf_delta = _parse_tf(target_timeframe)
        self._buckets: Dict[str, _Bucket] = {}
        self._active_keys: Dict[str, str] = {}

    def _bucket_key(self, bar: Bar) -> Tuple[str, pd.Timestamp]:
        bucket_start = pd.Timestamp(bar.open_time).floor(self.tf_delta)
        return bar.symbol, bucket_start

    def add_bar(self, bar: Bar) -> Optional[Bar]:
        symbol, bucket_start = self._bucket_key(bar)
        key = f"{symbol}_{bucket_start.value}"

        prev_key = self._active_keys.get(symbol)
        completed_from_prev = None
        if prev_key and prev_key != key:
            old_bucket = self._buckets.pop(prev_key, None)
            if old_bucket and old_bucket.bars:
                completed_from_prev = self._finalize_bucket(
                    old_bucket, old_bucket.bars[-1].close_time
                )

        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(symbol=symbol, open_time=bucket_start, bars=[])
            self._buckets[key] = bucket
            self._active_keys[symbol] = key

        bucket.bars.append(bar)
        expected_close = bucket_start + self.tf_delta
        if pd.Timestamp(bar.close_time) < expected_close:
            return completed_from_prev

        completed_bar = self._finalize_bucket(bucket, expected_close)
        self._buckets.pop(key, None)
        self._active_keys[symbol] = None
        return completed_bar or completed_from_prev

    def _finalize_bucket(self, bucket: _Bucket, close_time: pd.Timestamp) -> Bar:
        bars = bucket.bars
        open_price = bars[0].open
        high_price = max(b.high for b in bars)
        low_price = min(b.low for b in bars)
        close_price = bars[-1].close
        volume = sum(b.volume for b in bars)
        extras = dict(bars[-1].extras or {})
        extras["source_bars"] = [
            {
                "open_time": b.open_time.isoformat(),
                "close_time": b.close_time.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        return Bar(
            symbol=bucket.symbol,
            timeframe=self.target_tf,
            open_time=bucket.open_time,
            close_time=close_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            extras=extras,
        )

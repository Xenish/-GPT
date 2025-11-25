from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class LiveDataSource(ABC):
    """
    Base interface for live data feeds so trading engines can
    swap between replay, websocket, or REST implementations.
    """

    @abstractmethod
    def connect(self) -> None:
        """Initialize network/file handles prior to streaming data."""

    @abstractmethod
    def next_bar(self) -> Optional[pd.Series]:
        """Return the next bar/tick. Should return None when no data remains."""

    @abstractmethod
    def close(self) -> None:
        """Tear down any open resources."""


class FileReplayDataSource(LiveDataSource):
    """
    Simple replay data source backed by a pandas DataFrame. Each call to
    next_bar returns the next row so that we can test live components
    offline using historical features. Supports optional bars_limit and
    start offsets for partial replays.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        bars_limit: Optional[int] = None,
        start_index: int = 0,
        start_timestamp: Optional[str] = None,
        timestamp_col: str = "timestamp",
    ):
        if df is None or df.empty:
            raise ValueError("Replay DataFrame must not be empty.")
        self._df = df.reset_index(drop=True)
        self._bars_limit = bars_limit if bars_limit is None or bars_limit > 0 else None
        self._start_index = max(start_index, 0)
        self._start_timestamp = (
            pd.to_datetime(start_timestamp) if start_timestamp else None
        )
        self._timestamp_col = timestamp_col
        self._current_idx = 0
        self._end_idx = len(self._df)
        self._connected = False

    def connect(self) -> None:
        self._current_idx = self._resolve_start_index()
        if self._bars_limit is not None:
            self._end_idx = min(self._current_idx + self._bars_limit, len(self._df))
        else:
            self._end_idx = len(self._df)
        self._connected = True

    def _resolve_start_index(self) -> int:
        idx = min(self._start_index, len(self._df))
        if self._start_timestamp is not None and self._timestamp_col in self._df.columns:
            ts_series = pd.to_datetime(self._df[self._timestamp_col])
            matches = ts_series >= self._start_timestamp
            try:
                first_idx = matches[matches].index[0]
                idx = first_idx
            except IndexError:
                idx = len(self._df)
        return idx

    def next_bar(self) -> Optional[pd.Series]:
        if not self._connected:
            self.connect()
        if self._current_idx >= self._end_idx:
            return None
        row = self._df.iloc[self._current_idx]
        self._current_idx += 1
        return row

    def close(self) -> None:
        self._connected = False

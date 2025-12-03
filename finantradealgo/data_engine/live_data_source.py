from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
from typing import Optional, Callable

import pandas as pd

from finantradealgo.core.types import Bar
from finantradealgo.data_engine.data_backend import build_backend
from finantradealgo.data_engine.ingestion.ohlcv import timeframe_to_seconds
from finantradealgo.data_engine.ingestion.state import BaseStateStore
from finantradealgo.system.config_loader import DataConfig


class AbstractLiveDataSource(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Initialize underlying sockets/files."""

    @abstractmethod
    def next_bar(self) -> Optional[Bar]:
        """Return the next bar (blocking if necessary)."""

    @abstractmethod
    def close(self) -> None:
        """Close connections / cleanup."""


class FileReplayDataSource(AbstractLiveDataSource):
    """
    Simple replay data source backed by a pandas DataFrame. Each call to
    next_bar returns the next row as a Bar so we can test live components
    offline using historical features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        symbol: str = "UNKNOWN",
        timeframe: str = "15m",
        bars_limit: Optional[int] = None,
        start_index: int = 0,
        start_timestamp: Optional[str] = None,
        timestamp_col: str = "timestamp",
    ):
        if df is None or df.empty:
            raise ValueError("Replay DataFrame must not be empty.")
        self._df = df.reset_index(drop=True)
        self.symbol = symbol
        self.timeframe = timeframe
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

    def next_bar(self) -> Optional[Bar]:
        if not self._connected:
            self.connect()
        if self._current_idx >= self._end_idx:
            return None
        row = self._df.iloc[self._current_idx]
        self._current_idx += 1
        open_time = (
            pd.to_datetime(row[self._timestamp_col])
            if self._timestamp_col in row
            else pd.Timestamp.utcnow()
        )
        extras = row.to_dict()
        return Bar(
            ts=open_time,
            open=float(row.get("open", row.get("close"))),
            high=float(row.get("high", row.get("close"))),
            low=float(row.get("low", row.get("close"))),
            close=float(row.get("close")),
            volume=float(row.get("volume", 0.0)),
            symbol=self.symbol,
            timeframe=self.timeframe,
            extras=extras,
        )

    def close(self) -> None:
        self._connected = False


class DataBackendReplaySource(AbstractLiveDataSource):
    """
    Replay/live source that streams bars from a DataBackend (Timescale/DuckDB/CSV).
    Uses ingestion watermarks to detect staleness/gaps and can trigger backfill via
    an optional gap_handler callable.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        symbol: str,
        timeframe: str,
        *,
        watermark_store: BaseStateStore | None = None,
        watermark_job: str = "live_poll",
        lookback_bars: int = 2000,
        gap_handler: Callable[[str, str, pd.Timestamp, pd.Timestamp], None] | None = None,
    ) -> None:
        if not symbol or not timeframe:
            raise ValueError("symbol and timeframe are required for DataBackendReplaySource")
        self.data_cfg = data_cfg
        self.symbol = symbol
        self.timeframe = timeframe
        self.watermark_store = watermark_store
        self.watermark_job = watermark_job
        self.lookback_bars = lookback_bars
        self.gap_handler = gap_handler
        self._df: pd.DataFrame | None = None
        self._idx = 0
        self._connected = False
        self._last_ts: pd.Timestamp | None = None
        self._step_seconds = timeframe_to_seconds(timeframe)

    def connect(self) -> None:
        self._reload_data()
        self._connected = True

    def _reload_data(self) -> None:
        backend = build_backend(self.data_cfg)
        end_ts = pd.Timestamp.utcnow().tz_localize("UTC")
        start_ts = None
        if self.watermark_store:
            wm = self.watermark_store.get_watermark(self.watermark_job, f"{self.symbol}:{self.timeframe}")
            if wm is not None:
                start_ts = wm - pd.Timedelta(seconds=self.lookback_bars * self._step_seconds)
        if start_ts is None:
            start_ts = end_ts - pd.Timedelta(seconds=self.lookback_bars * self._step_seconds)
        df = backend.load_ohlcv(
            self.symbol,
            self.timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        df = df.sort_values("timestamp").reset_index(drop=True)
        self._df = df
        self._idx = 0

    def _maybe_gap_fill(self, current_ts: pd.Timestamp) -> None:
        if self._last_ts is None:
            return
        expected_next = self._last_ts + pd.Timedelta(seconds=self._step_seconds)
        if current_ts <= expected_next:
            return
        if self.gap_handler:
            try:
                self.gap_handler(self.symbol, self.timeframe, expected_next, current_ts)
                # After backfill, reload data window
                self._reload_data()
            except Exception:
                # Log and continue replaying existing data
                pass

    def next_bar(self, timeout: Optional[float] = None) -> Optional[Bar]:
        if not self._connected:
            self.connect()
        if self._df is None or self._idx >= len(self._df):
            self._reload_data()
            if self._df is None or self._idx >= len(self._df):
                return None
        row = self._df.iloc[self._idx]
        ts = pd.to_datetime(row.get("timestamp"), utc=True)
        self._maybe_gap_fill(ts)
        self._last_ts = ts
        if self.watermark_store and ts is not None:
            try:
                self.watermark_store.upsert_watermark(self.watermark_job, f"{self.symbol}:{self.timeframe}", ts)
            except Exception:
                pass
        self._idx += 1
        return Bar(
            symbol=self.symbol,
            timeframe=self.timeframe,
            open=float(row.get("open", row.get("close"))),
            high=float(row.get("high", row.get("close"))),
            low=float(row.get("low", row.get("close"))),
            close=float(row.get("close")),
            volume=float(row.get("volume", 0.0)),
            open_time=ts,
            close_time=ts,
            extras=row.to_dict(),
        )

    def close(self) -> None:
        self._connected = False

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Protocol, Sequence

import pandas as pd

from finantradealgo.core.types import Bar
from finantradealgo.data_engine.ingestion.models import IngestCandle, ensure_timestamp
from finantradealgo.data_engine.ingestion.writer import TimescaleWarehouse
from finantradealgo.execution.exchange_client import BinanceFuturesClient
from finantradealgo.system.config_loader import ExchangeConfig, LiveConfig

logger = logging.getLogger(__name__)


def timeframe_to_seconds(tf: str) -> int:
    """Parse timeframe strings like '1m', '15m', '1h', '1d' into seconds."""
    tf = tf.strip().lower()
    unit = tf[-1]
    value = int(tf[:-1])
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    raise ValueError(f"Unsupported timeframe: {tf}")


class CandleSource(Protocol):
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_ts: pd.Timestamp,
        end_ts: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
    ) -> Sequence[IngestCandle]:
        ...


class WsBarSource(Protocol):
    def connect(self) -> None: ...
    def next_bar(self, timeout: float | None = None) -> Optional[Bar]: ...
    def close(self) -> None: ...


class BinanceRESTCandleSource:
    """
    REST-based OHLCV fetcher for Binance Futures.
    Includes simple retry/backoff.
    """

    def __init__(
        self,
        exchange_cfg: ExchangeConfig,
        api_key: str,
        secret: str,
        *,
        max_batch: int = 1000,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.client = BinanceFuturesClient(exchange_cfg, api_key=api_key, secret=secret)
        self.max_batch = max_batch
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_ts: pd.Timestamp,
        end_ts: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
    ) -> Sequence[IngestCandle]:
        start_ms = int(ensure_timestamp(start_ts).timestamp() * 1000)
        end_ms = int(ensure_timestamp(end_ts).timestamp() * 1000) if end_ts else None
        batch = min(limit or self.max_batch, self.max_batch)
        attempts = 0
        while True:
            try:
                klines = self.client.get_klines(
                    symbol,
                    interval=timeframe,
                    limit=batch,
                    start_time=start_ms,
                    end_time=end_ms,
                )
                return [
                    IngestCandle.from_binance_kline(symbol, timeframe, kline)
                    for kline in klines
                ]
            except Exception as exc:
                attempts += 1
                if attempts > self.max_retries:
                    raise
                sleep_for = self.backoff_seconds * (2 ** (attempts - 1))
                logger.warning(
                    "Binance REST fetch retry %s/%s for %s %s: %s",
                    attempts,
                    self.max_retries,
                    symbol,
                    timeframe,
                    exc,
                )
                time.sleep(sleep_for)


class HistoricalOHLCVIngestor:
    """
    Batch/backfill loader that can:
    - Fetch historical ranges
    - Detect gaps in recent history and repair them
    - Start from DB latest ts and catch-up to now
    """

    def __init__(
        self,
        source: CandleSource,
        warehouse: TimescaleWarehouse,
        *,
        chunk_size_bars: int = 900,
    ) -> None:
        self.source = source
        self.warehouse = warehouse
        self.chunk_size_bars = chunk_size_bars

    def backfill_range(
        self,
        symbol: str,
        timeframe: str,
        start_ts: datetime | str | pd.Timestamp,
        end_ts: datetime | str | pd.Timestamp,
    ) -> int:
        start = ensure_timestamp(start_ts)
        end = ensure_timestamp(end_ts)
        step_sec = timeframe_to_seconds(timeframe)
        cursor = start
        written = 0
        while cursor <= end:
            batch_end = min(cursor + timedelta(seconds=step_sec * self.chunk_size_bars), end)
            candles = self.source.fetch(symbol, timeframe, cursor, batch_end, self.chunk_size_bars)
            if candles:
                written += self.warehouse.upsert_ohlcv(candles)
                cursor = candles[-1].ts + timedelta(seconds=step_sec)
            else:
                cursor = batch_end + timedelta(seconds=step_sec)
        return written

    def catch_up_from_latest(
        self,
        symbol: str,
        timeframe: str,
        *,
        lookback_bars: int = 500,
    ) -> int:
        """
        Resume from the last stored candle. If none exists, use lookback_bars from now.
        """
        latest = self.warehouse.get_latest_ts(symbol, timeframe)
        step_sec = timeframe_to_seconds(timeframe)
        end = pd.Timestamp.utcnow().tz_localize("UTC")
        if latest is None:
            start = end - timedelta(seconds=lookback_bars * step_sec)
        else:
            start = latest + timedelta(seconds=step_sec)
        return self.backfill_range(symbol, timeframe, start, end)

    def repair_recent_gaps(
        self,
        symbol: str,
        timeframe: str,
        *,
        lookback_hours: int = 24,
        max_gaps: int = 10,
    ) -> int:
        """
        Detect missing candles over a recent window and patch them.
        """
        end = pd.Timestamp.utcnow().tz_localize("UTC")
        start = end - timedelta(hours=lookback_hours)
        step = timeframe_to_seconds(timeframe)
        gaps = self.warehouse.detect_gaps(symbol, timeframe, start, end, step_seconds=step, max_gaps=max_gaps)
        repaired = 0
        for gap_start, gap_end in gaps:
            candles = self.source.fetch(symbol, timeframe, gap_start, gap_end, self.chunk_size_bars)
            if candles:
                repaired += self.warehouse.upsert_ohlcv(candles)
        return repaired


class LiveOHLCVIngestor:
    """
    WebSocket live ingestor with REST catch-up and DB sink.
    - Streams final bars via WS
    - Detects gaps relative to last stored ts and fills them via REST
    """

    def __init__(
        self,
        ws_source: WsBarSource,
        rest_source: CandleSource,
        warehouse: TimescaleWarehouse,
        live_cfg: LiveConfig,
        *,
        flush_every: int = 50,
    ) -> None:
        self.ws_source = ws_source
        self.rest_source = rest_source
        self.warehouse = warehouse
        self.live_cfg = live_cfg
        self.flush_every = flush_every
        self._buffer: list[IngestCandle] = []
        self._last_ts: dict[str, pd.Timestamp] = {}
        self._step_sec = timeframe_to_seconds(self.live_cfg.timeframe)

    def _to_candle(self, bar: Bar) -> IngestCandle:
        ts = ensure_timestamp(bar.close_time or bar.ts or bar.open_time)
        return IngestCandle(
            ts=ts,
            symbol=bar.symbol,
            timeframe=self.live_cfg.timeframe,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )

    def _flush(self) -> None:
        if not self._buffer:
            return
        self.warehouse.upsert_ohlcv(self._buffer)
        self._buffer.clear()

    def _maybe_repair_gap(self, symbol: str, incoming_ts: pd.Timestamp) -> None:
        prev = self._last_ts.get(symbol)
        if prev is None:
            db_latest = self.warehouse.get_latest_ts(symbol, self.live_cfg.timeframe)
            if db_latest:
                prev = db_latest
        if prev is None:
            self._last_ts[symbol] = incoming_ts
            return
        expected_next = prev + timedelta(seconds=self._step_sec)
        if incoming_ts <= expected_next:
            self._last_ts[symbol] = incoming_ts
            return
        # Gap detected -> backfill from REST
        gap_end = incoming_ts - timedelta(seconds=self._step_sec)
        candles = self.rest_source.fetch(symbol, self.live_cfg.timeframe, expected_next, gap_end)
        if candles:
            self.warehouse.upsert_ohlcv(candles)
            self._last_ts[symbol] = candles[-1].ts
        else:
            self._last_ts[symbol] = incoming_ts

    def run(self, *, max_messages: Optional[int] = None) -> None:
        """
        Consume WS stream and persist bars. max_messages can be used for tests.
        """
        self.ws_source.connect()
        processed = 0
        try:
            while True:
                bar = self.ws_source.next_bar(timeout=1.0)
                if bar is None:
                    self._flush()
                    continue

                candle = self._to_candle(bar)
                self._maybe_repair_gap(candle.symbol, candle.ts)
                self._buffer.append(candle)
                self._last_ts[candle.symbol] = candle.ts
                processed += 1

                if len(self._buffer) >= self.flush_every:
                    self._flush()

                if max_messages is not None and processed >= max_messages:
                    break
        finally:
            self._flush()
            self.ws_source.close()

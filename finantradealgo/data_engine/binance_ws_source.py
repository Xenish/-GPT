from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Dict, List, Optional

import pandas as pd
from websocket import WebSocketApp

from finantradealgo.data_engine.bar_aggregator import BarAggregator
from finantradealgo.data_engine.live_data_source import AbstractLiveDataSource, Bar
from finantradealgo.execution.exchange_client import BinanceFuturesClient
from finantradealgo.system.config_loader import ExchangeConfig, LiveConfig


class BinanceWsDataSource(AbstractLiveDataSource):
    def __init__(self, cfg: Dict[str, any], symbols: List[str]) -> None:
        self.cfg = cfg
        self.symbols = [s.upper() for s in symbols]
        self.exchange_cfg: ExchangeConfig = cfg.get("exchange_cfg")
        if not isinstance(self.exchange_cfg, ExchangeConfig):
            self.exchange_cfg = ExchangeConfig.from_dict(cfg.get("exchange", {}))
        self.live_cfg: LiveConfig = LiveConfig.from_dict(
            cfg.get("live"),
            default_symbol=cfg.get("symbol"),
            default_timeframe=cfg.get("timeframe"),
        )
        self.logger = logging.getLogger(__name__)
        self.use_1m_stream = getattr(self.live_cfg, "ws_use_1m_stream", True)
        self.aggregate_tf = getattr(
            self.live_cfg, "ws_aggregate_to_tf", self.live_cfg.timeframe
        )
        self.resync_lookback = getattr(
            self.live_cfg, "ws_resync_lookback_bars", 200
        )
        self.max_stale_seconds = getattr(
            self.live_cfg, "ws_max_stale_seconds", 30
        )
        self.max_ws_reconnects = getattr(
            self.live_cfg, "ws_max_ws_reconnects", 10
        )

        self.ws_url = self._build_ws_url()
        self._queue: queue.Queue[Bar] = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_app: Optional[WebSocketApp] = None
        self._last_message_time: Optional[float] = None
        self._aggregators: Dict[str, BarAggregator] = {}
        self._rest_client = BinanceFuturesClient(
            self.exchange_cfg,
            api_key="",
            secret="",
        )
        self._last_close_time: Dict[str, Optional[pd.Timestamp]] = {
            sym: None for sym in self.symbols
        }
        if self.use_1m_stream:
            for symbol in self.symbols:
                self._aggregators[symbol] = BarAggregator(self.aggregate_tf)

    def _build_ws_url(self) -> str:
        base = (
            self.exchange_cfg.base_url_ws_testnet
            if self.exchange_cfg.testnet
            else self.exchange_cfg.base_url_ws
        ).rstrip("/")
        interval = "1m" if self.use_1m_stream else self.live_cfg.timeframe.lower()
        streams = [f"{s.lower()}@kline_{interval}" for s in self.symbols]
        stream_path = "/".join(streams)
        return f"{base}/stream?streams={stream_path}"

    def connect(self) -> None:
        self._initial_resync()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()

    def _initial_resync(self) -> None:
        interval = "1m" if self.use_1m_stream else self.live_cfg.timeframe.lower()
        for symbol in self.symbols:
            exch_symbol = self.exchange_cfg.symbol_mapping.get(symbol, symbol)
            try:
                raw_bars = self._rest_client.get_klines(
                    exch_symbol,
                    interval=interval,
                    limit=self.resync_lookback,
                )
            except Exception as exc:
                self.logger.exception("Failed to resync %s: %s", symbol, exc)
                continue
            bars = [self._bar_from_kline(symbol, kline) for kline in raw_bars]
            for bar in bars:
                self._process_unit_bar(symbol, bar, from_backfill=True)

    def _ws_loop(self) -> None:
        reconnects = 0
        while not self._stop_event.is_set():
            try:
                self._run_ws_once()
                reconnects = 0
            except Exception as exc:
                self.logger.exception("WebSocket loop error: %s", exc)
                reconnects += 1
                if (
                    self.max_ws_reconnects
                    and reconnects >= self.max_ws_reconnects
                ):
                    self.logger.error("Max reconnect limit reached, stopping WS.")
                    break
                time.sleep(min(2 ** reconnects, 30))

    def _run_ws_once(self) -> None:
        self.logger.info("Connecting Binance WS: %s", self.ws_url)

        def on_message(ws, message):
            self._last_message_time = time.time()
            try:
                data = json.loads(message)
                payload = data.get("data", data)
                kline = payload.get("k")
                if not kline:
                    return
                self._handle_kline(kline)
            except Exception as exc:
                self.logger.exception("Failed to process WS message: %s", exc)

        def on_error(ws, error):
            self.logger.error("WS error: %s", error)

        def on_close(ws, status_code, msg):
            self.logger.warning("WS closed: %s %s", status_code, msg)

        ws_app = WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws_app = ws_app
        ws_app.run_forever()

    def _handle_kline(self, kline: Dict[str, any]) -> None:
        symbol = kline["s"].upper()
        is_final = bool(kline.get("x"))
        bar = Bar(
            symbol=symbol,
            timeframe=kline.get("i", "1m"),
            open_time=pd.to_datetime(kline["t"], unit="ms"),
            close_time=pd.to_datetime(kline["T"], unit="ms"),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            extras=kline,
        )
        if not is_final and self.use_1m_stream:
            return
        self._process_unit_bar(symbol, bar, from_backfill=False)

    def _process_unit_bar(self, symbol: str, bar: Bar, *, from_backfill: bool) -> None:
        if self.use_1m_stream:
            if not from_backfill:
                self._backfill_if_needed(symbol, bar)
            self._last_close_time[symbol] = bar.close_time
            aggregator = self._aggregators.get(symbol)
            if not aggregator:
                aggregator = BarAggregator(self.aggregate_tf)
                self._aggregators[symbol] = aggregator
            aggregated = aggregator.add_bar(bar)
            if aggregated:
                self._enqueue_bar(aggregated)
        else:
            if from_backfill or getattr(bar, "extras", {}).get("x", True):
                self._enqueue_bar(bar)

    def _backfill_if_needed(self, symbol: str, bar: Bar) -> None:
        last = self._last_close_time.get(symbol)
        if last is None:
            return
        expected_next = last + pd.Timedelta(minutes=1)
        if bar.open_time <= expected_next:
            return
        gap_end = bar.open_time - pd.Timedelta(minutes=1)
        if expected_next > gap_end:
            return
        missing = self._backfill_gap(symbol, expected_next, gap_end)
        for missing_bar in missing:
            self._process_unit_bar(symbol, missing_bar, from_backfill=True)
            self._last_close_time[symbol] = missing_bar.close_time

    def _backfill_gap(
        self, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> List[Bar]:
        exch_symbol = self.exchange_cfg.symbol_mapping.get(symbol, symbol)
        try:
            raw = self._rest_client.get_klines(
                exch_symbol,
                interval="1m",
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
            )
        except Exception as exc:
            self.logger.exception("Gap fetch failed for %s: %s", symbol, exc)
            return []
        return [self._bar_from_kline(symbol, kline) for kline in raw]

    def _bar_from_kline(self, symbol: str, kline: Dict[str, any]) -> Bar:
        return Bar(
            symbol=symbol,
            timeframe=kline.get("i", "1m"),
            open_time=pd.to_datetime(kline["t"], unit="ms"),
            close_time=pd.to_datetime(kline["T"], unit="ms"),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            extras=kline,
        )

    def _enqueue_bar(self, bar: Bar) -> None:
        try:
            self._queue.put_nowait(bar)
        except queue.Full:
            self.logger.warning("Dropping bar due to full queue.")

    def next_bar(self, timeout: Optional[float] = None) -> Optional[Bar]:
        if self._stop_event.is_set():
            return None
        try:
            return self._queue.get(timeout=timeout or 1.0)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop_event.set()
        if self._ws_app:
            try:
                self._ws_app.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def get_last_message_time(self) -> Optional[float]:
        return self._last_message_time

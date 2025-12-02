from __future__ import annotations

"""
Exchange adapter skeletons focused on normalization and shared interfaces.

Real HTTP/WebSocket calls are intentionally omitted; plug in httpx/aiohttp/requests
or your preferred client in `_request` per adapter.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import time

from finantradealgo.exchanges import (
    ExchangeAdapter,
    ExchangeHealth,
    ExchangeId,
    ExchangeStatus,
    MarketType,
    NormalizedOrderBook,
    NormalizedTicker,
    NormalizedTrade,
    SymbolMapping,
)


@dataclass
class BaseHTTPExchangeAdapter(ExchangeAdapter):
    exchange: ExchangeId
    base_url: str
    timeout_seconds: float = 5.0
    rate_limit_per_sec: float | None = None
    metadata: dict[str, Any] | None = None

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Placeholder for HTTP requests. Replace with real client (httpx/aiohttp/requests).
        Should handle auth, retries, and rate limiting.
        """
        raise NotImplementedError("Plug in real HTTP client logic here.")

    def _now_ts(self) -> float:
        return time.time()

    def get_health(self) -> ExchangeHealth:
        # TODO: Replace with real health metrics (latency, heartbeat, data quality).
        return ExchangeHealth(
            exchange=self.exchange,
            status=ExchangeStatus.HEALTHY,
            last_heartbeat_ts=self._now_ts(),
            metadata={"note": "Placeholder health; implement real checks"},
        )


class BinanceAdapter(BaseHTTPExchangeAdapter):
    def __init__(self, base_url: str = "https://api.binance.com"):
        super().__init__(exchange=ExchangeId.BINANCE, base_url=base_url)

    def get_ticker(self, symbol: SymbolMapping) -> NormalizedTicker:
        # TODO: Fetch /api/v3/ticker/bookTicker or /api/v3/ticker/24hr
        # Example normalization:
        #   data = self._request("/api/v3/ticker/bookTicker", {"symbol": symbol.exchange_symbol})
        #   return NormalizedTicker(
        #       exchange=self.exchange,
        #       symbol=symbol.internal_symbol,
        #       bid=float(data["bidPrice"]),
        #       ask=float(data["askPrice"]),
        #       last_price=float(data.get("lastPrice", data["bidPrice"])),
        #       timestamp=normalize_timestamp(data.get("closeTime", self._now_ts())),
        #       volume_24h=float(data.get("volume", 0)),
        #       metadata={"exchange_symbol": symbol.exchange_symbol},
        #   )
        return NormalizedTicker(
            exchange=self.exchange,
            symbol=symbol.internal_symbol,
            bid=0.0,
            ask=0.0,
            last_price=0.0,
            timestamp=self._now_ts(),
            metadata={"todo": "Implement Binance ticker normalization"},
        )

    def get_order_book(self, symbol: SymbolMapping, depth: int = 50) -> NormalizedOrderBook:
        # TODO: Fetch /api/v3/depth with limit=depth
        # Normalize bids/asks as [(price, qty)], convert timestamp to unix seconds.
        # Consider market type (spot vs futures) for contract size scaling if needed.
        return NormalizedOrderBook(
            exchange=self.exchange,
            symbol=symbol.internal_symbol,
            bids=[],
            asks=[],
            timestamp=self._now_ts(),
            metadata={"todo": "Implement Binance order book normalization"},
        )

    def get_recent_trades(self, symbol: SymbolMapping, limit: int = 100) -> List[NormalizedTrade]:
        # TODO: Fetch /api/v3/trades or /fapi/v1/trades for futures.
        # Convert price/qty to floats, map buyerMaker to side, normalize timestamp (ms -> s).
        return [
            # Example placeholder trade
            NormalizedTrade(
                exchange=self.exchange,
                symbol=symbol.internal_symbol,
                price=0.0,
                qty=0.0,
                side="buy",
                timestamp=self._now_ts(),
                metadata={"todo": "Implement Binance trades normalization"},
            )
        ]

    def get_ohlcv(
        self,
        symbol: SymbolMapping,
        timeframe: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int | None = None,
    ):
        # TODO: Fetch /api/v3/klines or futures equivalent.
        # Normalize timestamps (ms -> s) and return DataFrame-like structure as project expects.
        raise NotImplementedError("Implement Binance OHLCV normalization")


class BybitAdapter(BaseHTTPExchangeAdapter):
    def __init__(self, base_url: str = "https://api.bybit.com"):
        super().__init__(exchange=ExchangeId.BYBIT, base_url=base_url)

    def get_ticker(self, symbol: SymbolMapping) -> NormalizedTicker:
        # TODO: Fetch /v5/market/tickers
        # Normalize bid1Price/ask1Price, lastPrice, and turn ts (ms) into unix seconds.
        # Hook: contract size normalization for linear/inverse contracts.
        return NormalizedTicker(
            exchange=self.exchange,
            symbol=symbol.internal_symbol,
            bid=0.0,
            ask=0.0,
            last_price=0.0,
            timestamp=self._now_ts(),
            metadata={"todo": "Implement Bybit ticker normalization"},
        )

    def get_order_book(self, symbol: SymbolMapping, depth: int = 50) -> NormalizedOrderBook:
        # TODO: Fetch /v5/market/orderbook with category based on MarketType.
        # Normalize to list of (price, size) and ensure timestamp in seconds.
        return NormalizedOrderBook(
            exchange=self.exchange,
            symbol=symbol.internal_symbol,
            bids=[],
            asks=[],
            timestamp=self._now_ts(),
            metadata={"todo": "Implement Bybit order book normalization"},
        )

    def get_recent_trades(self, symbol: SymbolMapping, limit: int = 100) -> List[NormalizedTrade]:
        # TODO: Fetch /v5/market/recent-trade.
        # Map side ("Buy"/"Sell") -> "buy"/"sell"; normalize ts (ms -> s).
        # Hook: adjust qty for contract size if derivatives.
        return [
            NormalizedTrade(
                exchange=self.exchange,
                symbol=symbol.internal_symbol,
                price=0.0,
                qty=0.0,
                side="buy",
                timestamp=self._now_ts(),
                metadata={"todo": "Implement Bybit trades normalization"},
            )
        ]

    def get_ohlcv(
        self,
        symbol: SymbolMapping,
        timeframe: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int | None = None,
    ):
        # TODO: Fetch /v5/market/kline.
        # Normalize start/end to ms, parse candles, and return DataFrame-like structure.
        raise NotImplementedError("Implement Bybit OHLCV normalization")


class OKXAdapter(BaseHTTPExchangeAdapter):
    def __init__(self, base_url: str = "https://www.okx.com"):
        super().__init__(exchange=ExchangeId.OKX, base_url=base_url)

    def get_ticker(self, symbol: SymbolMapping) -> NormalizedTicker:
        # TODO: Fetch /api/v5/market/ticker.
        # OKX timestamps are in ms; convert to seconds.
        # Hook: contract value normalization for swaps/futures (size vs qty in coin).
        return NormalizedTicker(
            exchange=self.exchange,
            symbol=symbol.internal_symbol,
            bid=0.0,
            ask=0.0,
            last_price=0.0,
            timestamp=self._now_ts(),
            metadata={"todo": "Implement OKX ticker normalization"},
        )

    def get_order_book(self, symbol: SymbolMapping, depth: int = 50) -> NormalizedOrderBook:
        # TODO: Fetch /api/v5/market/books?sz=depth.
        # Convert bids/asks to floats; ts in ms -> s.
        # Hook: adjust size if contract size differs from 1.
        return NormalizedOrderBook(
            exchange=self.exchange,
            symbol=symbol.internal_symbol,
            bids=[],
            asks=[],
            timestamp=self._now_ts(),
            metadata={"todo": "Implement OKX order book normalization"},
        )

    def get_recent_trades(self, symbol: SymbolMapping, limit: int = 100) -> List[NormalizedTrade]:
        # TODO: Fetch /api/v5/market/trades with limit.
        # Map side ("buy"/"sell" from OKX) directly; ts in ms -> s.
        return [
            NormalizedTrade(
                exchange=self.exchange,
                symbol=symbol.internal_symbol,
                price=0.0,
                qty=0.0,
                side="buy",
                timestamp=self._now_ts(),
                metadata={"todo": "Implement OKX trades normalization"},
            )
        ]

    def get_ohlcv(
        self,
        symbol: SymbolMapping,
        timeframe: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int | None = None,
    ):
        # TODO: Fetch /api/v5/market/candles.
        # Convert ms timestamps to seconds; return DataFrame-like structure.
        raise NotImplementedError("Implement OKX OHLCV normalization")


class KrakenAdapter(BaseHTTPExchangeAdapter):
    def __init__(self, base_url: str = "https://api.kraken.com"):
        super().__init__(exchange=ExchangeId.KRAKEN, base_url=base_url)

    def get_ticker(self, symbol: SymbolMapping) -> NormalizedTicker:
        # TODO: Fetch /0/public/Ticker
        # Kraken symbols may differ (e.g., XBT/USD); use symbol.exchange_symbol mapping.
        # Normalize bid[0], ask[0], c[0], timestamp via server time or response metadata.
        return NormalizedTicker(
            exchange=self.exchange,
            symbol=symbol.internal_symbol,
            bid=0.0,
            ask=0.0,
            last_price=0.0,
            timestamp=self._now_ts(),
            metadata={"todo": "Implement Kraken ticker normalization"},
        )

    def get_order_book(self, symbol: SymbolMapping, depth: int = 50) -> NormalizedOrderBook:
        # TODO: Fetch /0/public/Depth with count=depth.
        # Bids/asks entries as [price, volume, timestamp]; timestamp already in seconds.
        return NormalizedOrderBook(
            exchange=self.exchange,
            symbol=symbol.internal_symbol,
            bids=[],
            asks=[],
            timestamp=self._now_ts(),
            metadata={"todo": "Implement Kraken order book normalization"},
        )

    def get_recent_trades(self, symbol: SymbolMapping, limit: int = 100) -> List[NormalizedTrade]:
        # TODO: Fetch /0/public/Trades.
        # Each trade: [price, volume, time, side, orderType, misc].
        # Normalize side ("b"/"s") -> "buy"/"sell"; time already in seconds.
        return [
            NormalizedTrade(
                exchange=self.exchange,
                symbol=symbol.internal_symbol,
                price=0.0,
                qty=0.0,
                side="buy",
                timestamp=self._now_ts(),
                metadata={"todo": "Implement Kraken trades normalization"},
            )
        ]

    def get_ohlcv(
        self,
        symbol: SymbolMapping,
        timeframe: str,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int | None = None,
    ):
        # TODO: Fetch /0/public/OHLC with interval mapped from timeframe.
        # Kraken returns [time, open, high, low, close, vwap, volume, count]; time in seconds.
        raise NotImplementedError("Implement Kraken OHLCV normalization")


def normalize_timestamp(ts: Any) -> float:
    """
    Convert various timestamp formats to unix seconds.
    Accepts seconds, milliseconds, or ISO-8601 strings.
    """
    if ts is None:
        return time.time()
    if isinstance(ts, (int, float)):
        # Heuristic: treat large numbers as ms.
        return ts / 1000 if ts > 1e12 else float(ts)
    if isinstance(ts, str):
        try:
            # Attempt ISO-8601 parsing; fallback to float.
            return time.mktime(time.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            return float(ts)
    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


def normalize_side(raw_side: str) -> str:
    """
    Map exchange-specific side representations to 'buy' or 'sell'.
    """
    side_map = {
        "b": "buy",
        "s": "sell",
        "buy": "buy",
        "sell": "sell",
        "BUY": "buy",
        "SELL": "sell",
        "Bid": "buy",
        "Ask": "sell",
        "ask": "sell",
        "bid": "buy",
    }
    return side_map.get(raw_side, raw_side).lower()


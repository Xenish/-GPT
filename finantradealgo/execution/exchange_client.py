from __future__ import annotations

import hmac
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from finantradealgo.system.config_loader import ExchangeConfig


class ExchangeClientError(Exception):
    """Base exception for exchange client errors."""


class ExchangeRateLimitError(ExchangeClientError):
    """Raised when exchange signals rate limiting."""


class ExchangeClientBase(ABC):
    @abstractmethod
    def get_server_time(self) -> int:
        ...

    @abstractmethod
    def get_exchange_info(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        *,
        price: Optional[float] = None,
        time_in_force: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: int | str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List[Any]]:
        ...


class BinanceFuturesClient(ExchangeClientBase):
    RATE_LIMIT_STATUS = 429
    RATE_LIMIT_CODES = {-1003}

    def __init__(
        self,
        cfg: ExchangeConfig,
        api_key: str,
        secret: str,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.cfg = cfg
        self.api_key = api_key
        self.secret = secret
        self.session = session or requests.Session()
        self.rest_base = (
            cfg.base_url_rest_testnet if cfg.testnet else cfg.base_url_rest
        ).rstrip("/")
        self.time_offset_ms = 0
        if self.cfg.time_sync:
            self._sync_time()

    def _sync_time(self) -> None:
        server_time = self._get_server_time_raw()
        local_time = int(time.time() * 1000)
        self.time_offset_ms = server_time - local_time

    def _get_server_time_raw(self) -> int:
        data = self._request("GET", "/fapi/v1/time", signed=False, auth=False)
        return int(data["serverTime"])

    def _map_symbol(self, symbol: str) -> str:
        return self.cfg.symbol_mapping.get(symbol, symbol)

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        timestamp = int(time.time() * 1000 + self.time_offset_ms)
        params.setdefault("timestamp", timestamp)
        params.setdefault("recvWindow", self.cfg.recv_window_ms)
        query = urlencode(params, doseq=True)
        signature = hmac.new(
            self.secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        auth: bool = True,
        max_retries: int = 3,
    ) -> Any:
        url = f"{self.rest_base}{path}"
        params = params or {}
        data = data or {}
        headers: Dict[str, str] = {}
        if signed:
            params = self._sign(params)
        if auth:
            headers["X-MBX-APIKEY"] = self.api_key

        backoff = 0.5
        last_error: Optional[Exception] = None

        for _ in range(max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params if method.upper() == "GET" else None,
                    data=data if method.upper() != "GET" else None,
                    headers=headers,
                    timeout=10,
                )

                if response.status_code == self.RATE_LIMIT_STATUS:
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                payload = response.json()
                if 200 <= response.status_code < 300:
                    # Some endpoints return list, others dict
                    return payload

                code = payload.get("code")
                msg = payload.get("msg", response.text)
                if code in self.RATE_LIMIT_CODES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise ExchangeClientError(f"Binance error {code}: {msg}")
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(backoff)
                backoff *= 2
            except ExchangeClientError as exc:
                last_error = exc
                break

        if isinstance(last_error, ExchangeClientError):
            raise last_error
        if last_error:
            raise ExchangeRateLimitError("Rate limit exceeded") from last_error
        raise ExchangeClientError("Unknown exchange error")

    def get_server_time(self) -> int:
        return self._get_server_time_raw()

    def get_exchange_info(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v1/exchangeInfo", signed=False)

    def get_account_info(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        return data if isinstance(data, list) else []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {}
        if symbol:
            params["symbol"] = self._map_symbol(symbol)
        data = self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)
        return data if isinstance(data, list) else []

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        *,
        price: Optional[float] = None,
        time_in_force: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": self._map_symbol(symbol),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if price is not None:
            params["price"] = price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if reduce_only is not None:
            params["reduceOnly"] = "true" if reduce_only else "false"
        params.update({k: v for k, v in extra.items() if v is not None})
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int | str) -> Dict[str, Any]:
        params = {"symbol": self._map_symbol(symbol), "orderId": order_id}
        return self._request("DELETE", "/fapi/v1/order", params=params, signed=True)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List[Any]]:
        params: Dict[str, Any] = {
            "symbol": self._map_symbol(symbol),
            "interval": interval,
        }
        if limit is not None:
            params["limit"] = limit
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        data = self._request("GET", "/fapi/v1/klines", params=params, signed=False)
        return data if isinstance(data, list) else []


__all__ = [
    "ExchangeClientBase",
    "ExchangeClientError",
    "ExchangeRateLimitError",
    "BinanceFuturesClient",
]

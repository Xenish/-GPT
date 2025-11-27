from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finantradealgo.execution.exchange_client import (
    BinanceFuturesClient,
    ExchangeClientBase as RawExchangeClient,
)
from finantradealgo.execution.execution_client import ExchangeExecutionClient
from finantradealgo.system.config_loader import ExchangeConfig


def _build_cfg(**override):
    data = {
        "name": "binance_futures",
        "testnet": True,
        "base_url_rest": "https://fapi.binance.com",
        "base_url_rest_testnet": "https://testnet.binancefuture.com",
        "base_url_ws": "wss://fstream.binance.com",
        "base_url_ws_testnet": "wss://stream.binancefuture.com",
        "api_key_env": "BINANCE_FUTURES_API_KEY",
        "secret_key_env": "BINANCE_FUTURES_API_SECRET",
        "recv_window_ms": 5000,
        "time_sync": False,
        "max_time_skew_ms": 1000,
        "symbol_mapping": {"AIAUSDT": "AIAUSDT"},
        "default_leverage": 5,
        "position_mode": "one_way",
        "dry_run": True,
    }
    data.update(override)
    return ExchangeConfig.from_dict(data)


def test_sign_helper_adds_signature_and_timestamp(monkeypatch):
    cfg = _build_cfg(time_sync=False)
    client = BinanceFuturesClient(cfg, api_key="key", secret="secret")
    params = {"symbol": "BTCUSDT", "side": "BUY"}
    signed = client._sign(params.copy())
    assert "timestamp" in signed and "recvWindow" in signed and "signature" in signed
    assert isinstance(signed["signature"], str)


def test_request_retries_on_rate_limit(monkeypatch):
    cfg = _build_cfg(time_sync=False)
    client = BinanceFuturesClient(cfg, api_key="key", secret="secret")

    class DummyResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    session = MagicMock()
    session.request.side_effect = [
        DummyResp(429, {"code": -1003, "msg": "Rate limit"}),
        DummyResp(200, {"ok": True}),
    ]
    client.session = session
    with patch("time.sleep", return_value=None) as sleep_mock:
        result = client._request("GET", "/fapi/v1/time", signed=False)
    assert result == {"ok": True}
    assert sleep_mock.call_count == 1


def test_get_klines_returns_list(monkeypatch):
    cfg = _build_cfg(time_sync=False)
    client = BinanceFuturesClient(cfg, api_key="key", secret="secret")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [[0, 1, 2]]
    client.session.request = MagicMock(return_value=mock_resp)
    klines = client.get_klines("AIAUSDT", "15m", limit=2)
    assert klines == [[0, 1, 2]]


def test_place_order_calls_endpoint(monkeypatch):
    cfg = _build_cfg(time_sync=False)
    client = BinanceFuturesClient(cfg, api_key="key", secret="secret")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"orderId": 123}
    client.session.request = MagicMock(return_value=mock_resp)
    resp = client.place_order("AIAUSDT", "BUY", "LIMIT", 0.001, price=1.0)
    assert resp["orderId"] == 123


def test_exchange_execution_client_submits_orders():
    raw = MagicMock(spec=RawExchangeClient)
    cfg = _build_cfg(symbol_mapping={"FOO": "FOOUSDT"})
    exec_client = ExchangeExecutionClient(raw, cfg, run_id="run")
    raw.place_order.return_value = {"orderId": 55}
    exec_client.submit_order("FOO", "BUY", 1.0, "MARKET", price=10.0)
    args, kwargs = raw.place_order.call_args
    assert kwargs["symbol"] == "FOOUSDT"
    assert kwargs["side"] == "BUY"
    assert kwargs["order_type"] == "MARKET"
    assert "newClientOrderId" in kwargs

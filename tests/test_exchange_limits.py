from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from finantradealgo.execution.exchange_client import ExchangeClientBase
from finantradealgo.execution.execution_client import ExchangeExecutionClient, ExchangeRiskLimitError
from finantradealgo.system.config_loader import ExchangeConfig, ExchangeRiskConfig


class DummyRawExchangeClient(ExchangeClientBase):
    def __init__(self):
        self.orders: List[Dict[str, Any]] = []

    def place_order(self, **payload):
        self.orders.append(payload)
        qty = payload.get("quantity", 0)
        price = payload.get("price", 0) or 0
        return {
            "orderId": len(self.orders),
            "clientOrderId": payload.get("newClientOrderId", ""),
            "origQty": str(qty),
            "executedQty": str(qty),
            "avgPrice": str(price),
            "type": payload.get("order_type", "MARKET"),
        }

    def get_server_time(self) -> int:
        return 0

    def get_exchange_info(self) -> Dict[str, Any]:
        return {}

    def get_account_info(self) -> Dict[str, Any]:
        return {"totalWalletBalance": "1000"}

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def cancel_order(self, symbol: str, order_id: int | str) -> Dict[str, Any]:
        return {}

    def get_klines(self, *args, **kwargs) -> List[List[Any]]:
        return []


def make_client(risk_cfg: ExchangeRiskConfig) -> ExchangeExecutionClient:
    exch_cfg = ExchangeConfig()
    raw = DummyRawExchangeClient()
    return ExchangeExecutionClient(raw, exch_cfg, risk_cfg, run_id="test")


def test_exchange_client_rejects_high_leverage():
    risk_cfg = ExchangeRiskConfig(max_leverage=3, max_position_notional=0.0, max_position_contracts=0.0)
    client = make_client(risk_cfg)
    with pytest.raises(ExchangeRiskLimitError):
        client.submit_order(
            symbol="BTCUSDT",
            side="BUY",
            qty=0.1,
            order_type="MARKET",
            price=50000,
            leverage=5,
        )


def test_exchange_client_enforces_notional_limit():
    risk_cfg = ExchangeRiskConfig(max_leverage=5, max_position_notional=1000.0, max_position_contracts=1.0)
    client = make_client(risk_cfg)
    with pytest.raises(ExchangeRiskLimitError):
        client.submit_order(
            symbol="BTCUSDT",
            side="BUY",
            qty=0.1,
            order_type="MARKET",
            price=20000,
            leverage=2,
        )


def test_exchange_client_enforces_contract_limit():
    risk_cfg = ExchangeRiskConfig(max_leverage=5, max_position_notional=0.0, max_position_contracts=0.01)
    client = make_client(risk_cfg)
    with pytest.raises(ExchangeRiskLimitError):
        client.submit_order(
            symbol="BTCUSDT",
            side="BUY",
            qty=0.1,
            order_type="MARKET",
            price=1000,
        )

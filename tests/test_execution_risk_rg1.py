from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.execution.execution_client import (
    ExchangeExecutionClient,
    ExchangeRiskLimitError,
)
from finantradealgo.execution.paper_client import PaperExecutionClient
from finantradealgo.execution.exchange_client import ExchangeClientBase
from finantradealgo.system.config_loader import ExchangeConfig, ExchangeRiskConfig


class DummyRawExchangeClient(ExchangeClientBase):
    def place_order(self, **kwargs):
        return {"orderId": "1", "clientOrderId": kwargs.get("newClientOrderId"), "status": "FILLED", "origQty": kwargs.get("quantity", 0)}

    def cancel_order(self, symbol: str, order_id: str | int) -> None:
        return None

    def get_open_positions(self):
        return []

    def get_open_orders(self, symbol=None):
        return []

    def get_account_info(self):
        return {"totalWalletBalance": 1000}

    def get_klines(self, symbol: str, interval: str, limit: int = 500):
        return []

    def get_exchange_info(self):
        return {}

    def get_server_time(self):
        return int(pd.Timestamp.utcnow().timestamp() * 1000)


def test_exchange_execution_risk_limits_raise():
    exch_cfg = ExchangeConfig.from_dict({"symbol_mapping": {}})
    risk_cfg = ExchangeRiskConfig(max_leverage=3, max_position_notional=100, max_position_contracts=1)
    client = ExchangeExecutionClient(DummyRawExchangeClient(), exch_cfg, risk_cfg)
    with pytest.raises(ExchangeRiskLimitError):
        client.submit_order(
            symbol="BTCUSDT",
            side="BUY",
            qty=2.0,  # exceeds max_position_contracts
            order_type="LIMIT",
            price=100.0,
            leverage=5,  # exceeds max_leverage
        )


def test_paper_client_slippage_applied():
    paper = PaperExecutionClient(
        initial_cash=1000.0,
        fee_pct=0.001,
        slippage_pct=0.01,
        simple_mode=True,
    )
    paper.mark_to_market(price=100.0, timestamp=pd.Timestamp("2024-01-01"))
    order = paper.submit_order("TEST", "BUY", qty=1.0, order_type="MARKET", price=100.0)
    assert order is not None
    assert order["entry_price"] >= 101.0  # slippage applied on entry for BUY

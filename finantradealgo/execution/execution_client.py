from __future__ import annotations

import time
from typing import Any, Dict, Optional

from finantradealgo.execution.client_base import ExecutionClientBase
from finantradealgo.execution.paper_client import PaperExecutionClient
from finantradealgo.execution.exchange_client import (
    BinanceFuturesClient,
    ExchangeClientBase as RawExchangeClient,
)
from finantradealgo.system.config_loader import (
    ExchangeConfig,
    LiveConfig,
    load_exchange_credentials,
    ExchangeRiskConfig,
)


class ExchangeRiskLimitError(ValueError):
    """Raised when an order violates configured exchange risk limits."""


class ExchangeExecutionClient(ExecutionClientBase):
    def __init__(
        self,
        exchange_client: RawExchangeClient,
        cfg: ExchangeConfig,
        risk_cfg: ExchangeRiskConfig,
        *,
        run_id: Optional[str] = None,
    ) -> None:
        self.exchange_client = exchange_client
        self.cfg = cfg
        self.risk_cfg = risk_cfg
        self.run_id = run_id or "exchange"
        self._order_log: list[Dict[str, Any]] = []

    def _map_symbol(self, symbol: str) -> str:
        return self.cfg.symbol_mapping.get(symbol, symbol)

    def _client_order_id(self, symbol: str, side: str) -> str:
        return f"{self.run_id}_{symbol}_{int(time.time() * 1000)}_{side}"

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        *,
        price: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        leverage: int = 1,
        **extra: Any,
    ) -> Dict[str, Any]:
        payload = {
            "symbol": self._map_symbol(symbol),
            "side": side.upper(),
            "order_type": order_type.upper(),
            "quantity": qty,
        }
        if price is not None:
            payload["price"] = price
        if reduce_only:
            payload["reduce_only"] = True
        if client_order_id is None:
            client_order_id = self._client_order_id(symbol, side)
        payload["newClientOrderId"] = client_order_id
        if payload["order_type"] == "LIMIT" and "timeInForce" not in extra:
            extra["timeInForce"] = "GTC"
        payload.update(extra)
        check_price = price if price is not None else 0.0
        self._check_limits(symbol, qty, check_price, leverage)
        resp = self.exchange_client.place_order(**payload)
        order = self._normalize_order_response(
            symbol=symbol,
            side=payload["side"],
            qty=qty,
            price=price,
            reduce_only=reduce_only,
            response=resp,
        )
        self._order_log.append(order)
        max_log = 50
        if len(self._order_log) > max_log:
            self._order_log = self._order_log[-max_log:]
        return order

    def cancel_order(self, symbol: str, order_id: str | int) -> None:
        self.exchange_client.cancel_order(self._map_symbol(symbol), order_id)

    def get_open_positions(self) -> list[dict]:
        return self.exchange_client.get_open_positions()

    def get_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        if symbol:
            symbol = self._map_symbol(symbol)
        return self.exchange_client.get_open_orders(symbol)

    def mark_to_market(self, price: float, timestamp) -> None:
        return None

    def get_portfolio(self) -> Dict[str, Any]:
        try:
            account = self.exchange_client.get_account_info()
        except Exception:
            return {}
        total_equity = float(account.get("totalWalletBalance", 0.0))
        return {"equity": total_equity, "cash": total_equity}

    def export_logs(self, timeframe: str):
        return {}

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "open_positions": self.get_open_positions(),
            "open_orders": self.get_open_orders(),
            "recent_orders": list(self._order_log[-20:]),
        }

    def close(self) -> None:
        session = getattr(self.exchange_client, "session", None)
        if session:
            try:
                session.close()
            except Exception:
                pass

    def _normalize_order_response(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float],
        reduce_only: bool,
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        order_id = response.get("orderId")
        client_order_id = response.get("clientOrderId") or response.get("newClientOrderId")
        status = response.get("status")
        orig_qty = response.get("origQty", qty)
        executed_qty = response.get("executedQty", orig_qty)
        avg_price = response.get("avgPrice")
        try:
            orig_qty = float(orig_qty)
        except (TypeError, ValueError):
            orig_qty = float(qty)
        try:
            executed_qty = float(executed_qty)
        except (TypeError, ValueError):
            executed_qty = float(orig_qty)
        if executed_qty <= 0 and response.get("type", "").upper() == "MARKET":
            executed_qty = float(orig_qty)
        try:
            avg_price = float(avg_price) if avg_price not in (None, "") else float(price or 0.0)
        except (TypeError, ValueError):
            avg_price = float(price or 0.0)
        ts = response.get("transactTime") or int(time.time() * 1000)
        return {
            "symbol": symbol,
            "side": side,
            "order_type": response.get("type", "").upper(),
            "order_id": order_id,
            "client_order_id": client_order_id,
            "status": status,
            "orig_qty": orig_qty,
            "executed_qty": executed_qty,
            "avg_price": avg_price,
            "reduce_only": reduce_only,
            "timestamp": ts,
        }

    def _check_limits(self, symbol: str, qty: float, price: float, leverage: int) -> None:
        notional = abs(qty * price)
        if self.risk_cfg.max_leverage and leverage > self.risk_cfg.max_leverage:
            raise ExchangeRiskLimitError(
                f"Requested leverage {leverage} exceeds max {self.risk_cfg.max_leverage}"
            )
        if (
            self.risk_cfg.max_position_notional > 0
            and notional > self.risk_cfg.max_position_notional
        ):
            raise ExchangeRiskLimitError(
                f"Order notional {notional} exceeds max {self.risk_cfg.max_position_notional}"
            )
        if (
            self.risk_cfg.max_position_contracts > 0
            and abs(qty) > self.risk_cfg.max_position_contracts
        ):
            raise ExchangeRiskLimitError(
                f"Order qty {qty} exceeds max contracts {self.risk_cfg.max_position_contracts}"
            )


def create_execution_client(
    system_cfg: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
) -> ExecutionClientBase:
    live_cfg = LiveConfig.from_dict(
        system_cfg.get("live"),
        default_symbol=system_cfg.get("symbol"),
        default_timeframe=system_cfg.get("timeframe"),
    )
    mode = (live_cfg.mode or "replay").lower()

    if mode in {"exchange", "live"}:
        exchange_cfg: ExchangeConfig = system_cfg.get("exchange_cfg")
        if not isinstance(exchange_cfg, ExchangeConfig):
            exchange_cfg = ExchangeConfig.from_dict(system_cfg.get("exchange", {}))
        exchange_risk_cfg: ExchangeRiskConfig = system_cfg.get("exchange_risk_cfg")
        if not isinstance(exchange_risk_cfg, ExchangeRiskConfig):
            exchange_risk_cfg = ExchangeRiskConfig.from_dict(system_cfg.get("exchange", {}))
        api_key, secret = load_exchange_credentials(exchange_cfg)
        exch_client = BinanceFuturesClient(exchange_cfg, api_key, secret)
        return ExchangeExecutionClient(exch_client, exchange_cfg, exchange_risk_cfg, run_id=run_id)

    paper_settings = live_cfg.paper
    return PaperExecutionClient(
        initial_cash=paper_settings.initial_cash,
        fee_pct=getattr(paper_settings, "fee_pct", 0.0004),
        slippage_pct=getattr(paper_settings, "slippage_pct", 0.0005),
        output_dir=paper_settings.output_dir,
        state_path=paper_settings.state_path,
        symbol=live_cfg.symbol,
    )


__all__ = [
    "ExecutionClientBase",
    "ExchangeExecutionClient",
    "ExchangeRiskLimitError",
    "create_execution_client",
]

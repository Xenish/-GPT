from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ExecutionClientBase(ABC):
    @abstractmethod
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Submit an order and return the exchange response."""

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str | int) -> None:
        """Cancel an order by id."""

    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return currently open positions."""

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return currently open/pending orders."""

    def has_position(self) -> bool:
        return bool(self.get_open_positions())

    def get_position(self) -> Optional[Any]:
        positions = self.get_open_positions()
        return positions[0] if positions else None

    def mark_to_market(self, price: float, timestamp) -> None:
        """Optional hook for paper trading clients."""
        return None

    def get_portfolio(self) -> Dict[str, Any]:
        """Optional portfolio snapshot."""
        return {}

    def get_trade_log(self) -> List[Dict[str, Any]]:
        return []

    def export_logs(self, timeframe: str):
        return {}

    def to_state_dict(self) -> Dict[str, Any]:
        return {}

    def close(self) -> None:
        """Optional cleanup hook."""
        return None

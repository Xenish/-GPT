from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    stop_loss_pct: float = 0.01
    max_leverage: float = 1.0


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def get_position_size(self, entry_price: float, equity: float) -> float:
        risk_amount = equity * self.config.risk_per_trade
        if risk_amount <= 0:
            return 0.0

        if self.config.stop_loss_pct <= 0:
            max_notional = equity * self.config.max_leverage
            return max_notional / entry_price

        stop_distance = entry_price * self.config.stop_loss_pct
        if stop_distance <= 0:
            return 0.0

        qty = risk_amount / stop_distance

        max_notional = equity * self.config.max_leverage
        notional = qty * entry_price
        if notional > max_notional:
            qty = max_notional / entry_price

        return max(qty, 0.0)

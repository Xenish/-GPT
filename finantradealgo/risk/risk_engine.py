from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import pandas as pd

from finantradealgo.risk import RiskMetricConfig
from finantradealgo.risk.daily_loss_limit import (
    DailyLossLimitConfig,
    is_daily_loss_limit_hit,
)
from finantradealgo.risk.leverage_scheduler import (
    LeverageScheduleConfig,
    compute_leverage,
)
from finantradealgo.risk.position_sizing import (
    PositionSizingInput,
    calc_size_atr_stop,
)
from finantradealgo.risk.tail_risk import compute_tail_risk_metrics
from finantradealgo.risk.var_calculator import VaRCalculator


@dataclass
class RiskConfig:
    capital_risk_pct_per_trade: float = 0.01
    max_leverage: float = 5.0
    max_notional_per_symbol: Optional[float] = None
    max_daily_loss_pct: float = 0.03
    daily_loss_lookback_days: int = 1
    min_hold_bars: int = 2
    use_tail_risk_guard: bool = True
    tail_risk_hv_threshold: float = 0.25
    tail_risk_max_leverage_in_crash: float = 1.0
    risk_per_trade: float = 0.01  # backward compatibility
    stop_loss_pct: float = 0.01

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RiskConfig":
        data = data or {}
        return cls(
            capital_risk_pct_per_trade=data.get(
                "capital_risk_pct_per_trade",
                data.get("risk_per_trade", cls.capital_risk_pct_per_trade),
            ),
            max_leverage=data.get("max_leverage", cls.max_leverage),
            max_notional_per_symbol=data.get("max_notional_per_symbol", cls.max_notional_per_symbol),
            max_daily_loss_pct=data.get("max_daily_loss_pct", cls.max_daily_loss_pct),
            daily_loss_lookback_days=data.get("daily_loss_lookback_days", cls.daily_loss_lookback_days),
            min_hold_bars=data.get("min_hold_bars", cls.min_hold_bars),
            use_tail_risk_guard=data.get("use_tail_risk_guard", cls.use_tail_risk_guard),
            tail_risk_hv_threshold=data.get("tail_risk_hv_threshold", cls.tail_risk_hv_threshold),
            tail_risk_max_leverage_in_crash=data.get(
                "tail_risk_max_leverage_in_crash", cls.tail_risk_max_leverage_in_crash
            ),
            risk_per_trade=data.get("risk_per_trade", cls.risk_per_trade),
            stop_loss_pct=data.get("stop_loss_pct", cls.stop_loss_pct),
        )


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self._tail_risk_active: bool = False
        self._leverage_cfg = LeverageScheduleConfig(max_leverage=self.config.max_leverage)
        self._daily_limit_cfg = DailyLossLimitConfig(
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            lookback_days=self.config.daily_loss_lookback_days,
        )

    def calc_position_size(
        self,
        equity: float,
        price: float,
        atr: Optional[float] = None,
        row: Optional[pd.Series] = None,
        *,
        current_notional: float = 0.0,
        max_position_notional: Optional[float] = None,
    ) -> float:
        if equity <= 0 or price <= 0:
            return 0.0

        atr_value = None
        if atr is not None:
            try:
                atr_value = float(atr)
            except (TypeError, ValueError):
                atr_value = None

        ps_input = PositionSizingInput(
            equity=equity,
            price=price,
            atr=atr_value,
            capital_risk_pct_per_trade=self.config.capital_risk_pct_per_trade,
            max_notional_per_symbol=self.config.max_notional_per_symbol,
        )
        base_size = calc_size_atr_stop(ps_input, atr_mult_for_stop=2.0)
        if base_size <= 0:
            return 0.0

        lev = 1.0
        if row is not None:
            lev = compute_leverage(row, self._leverage_cfg)

        if self._tail_risk_active:
            lev = min(lev, max(1.0, self.config.tail_risk_max_leverage_in_crash))

        lev = max(1.0, min(lev, self.config.max_leverage))
        final_size = base_size * lev

        if self.config.max_notional_per_symbol:
            max_size = self.config.max_notional_per_symbol / price
            final_size = min(final_size, max_size)

        if max_position_notional is not None:
            remaining = float(max_position_notional) - float(current_notional or 0.0)
            if remaining <= 0:
                return 0.0
            final_size = min(final_size, remaining / price)

        return max(final_size, 0.0)

    def can_open_new_trade(
        self,
        current_date: pd.Timestamp,
        equity_start_of_day: float,
        realized_pnl_today: float,
        row: Optional[pd.Series] = None,
        *,
        open_positions: Optional[List[Dict[str, Any]]] = None,
        max_open_trades: Optional[int] = None,
        tail_guard_returns: Optional[pd.Series] = None,
        tail_guard_config: Optional[RiskMetricConfig] = None,
    ) -> bool:
        if equity_start_of_day <= 0:
            return False

        if is_daily_loss_limit_hit(equity_start_of_day, realized_pnl_today, self._daily_limit_cfg):
            return False

        if (
            max_open_trades is not None
            and max_open_trades > 0
            and open_positions is not None
            and len(open_positions) >= max_open_trades
        ):
            return False

        self._tail_risk_active = False
        if self.config.use_tail_risk_guard and row is not None:
            hv_value = row.get("hv_20")
            try:
                hv_value = float(hv_value) if hv_value is not None else None
            except (TypeError, ValueError):
                hv_value = None

            if hv_value is not None and hv_value >= self.config.tail_risk_hv_threshold:
                self._tail_risk_active = True
                if self.config.tail_risk_max_leverage_in_crash <= 0:
                    return False

        # Optional VaR/tail guard using provided returns
        if tail_guard_returns is not None and tail_guard_config is not None:
            try:
                metrics = compute_tail_risk_metrics(
                    tail_guard_returns,
                    tail_guard_config,
                    use_parametric=False,
                )
                var_calc = VaRCalculator(tail_guard_config)
                var_res = var_calc.historical_var(tail_guard_returns)
                var_loss = float(var_res.var_value)
                if var_loss > equity_start_of_day * max(self.config.capital_risk_pct_per_trade, 0.0):
                    return False
                if metrics.expected_max_drawdown and metrics.expected_max_drawdown > self.config.max_daily_loss_pct:
                    return False
            except Exception:
                # Tail guard failures should not silently permit risk; block conservatively
                return False

        return True

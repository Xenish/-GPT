from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from finantradealgo.system.config_loader import KillSwitchConfig


class KillSwitchReason(str, Enum):
    NONE = "none"
    DAILY_PNL = "daily_pnl"
    DRAWDOWN = "drawdown"
    MIN_EQUITY = "min_equity"
    EXCEPTIONS = "exceptions"


@dataclass
class KillSwitchState:
    is_triggered: bool = False
    reason: KillSwitchReason = KillSwitchReason.NONE
    trigger_time: Optional[dt.datetime] = None
    last_eval_time: Optional[dt.datetime] = None
    recent_exceptions: List[dt.datetime] = field(default_factory=list)
    peak_equity: float = 0.0


class KillSwitch:
    def __init__(self, cfg: KillSwitchConfig) -> None:
        self.cfg = cfg
        self.state = KillSwitchState()

    def register_exception(self, ts: dt.datetime) -> None:
        self.state.recent_exceptions.append(ts)
        cutoff = ts - dt.timedelta(hours=1)
        self.state.recent_exceptions = [t for t in self.state.recent_exceptions if t >= cutoff]

    def evaluate(
        self,
        now: dt.datetime,
        *,
        equity: float,
        daily_realized_pnl: float,
    ) -> KillSwitchState:
        if not self.cfg.enabled or self.state.is_triggered:
            return self.state

        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        if daily_realized_pnl <= self.cfg.daily_realized_pnl_limit:
            return self._trigger(now, KillSwitchReason.DAILY_PNL)

        if self.state.peak_equity > 0:
            drawdown_pct = (self.state.peak_equity - equity) / self.state.peak_equity * 100
            if drawdown_pct >= self.cfg.max_equity_drawdown_pct:
                return self._trigger(now, KillSwitchReason.DRAWDOWN)

        if equity <= self.cfg.min_equity:
            return self._trigger(now, KillSwitchReason.MIN_EQUITY)

        if len(self.state.recent_exceptions) > self.cfg.max_exceptions_per_hour:
            return self._trigger(now, KillSwitchReason.EXCEPTIONS)

        self.state.last_eval_time = now
        return self.state

    def _trigger(self, now: dt.datetime, reason: KillSwitchReason) -> KillSwitchState:
        self.state.is_triggered = True
        self.state.reason = reason
        self.state.trigger_time = now
        self.state.last_eval_time = now
        return self.state

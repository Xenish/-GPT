from __future__ import annotations

import datetime as dt

from finantradealgo.system.kill_switch import KillSwitch, KillSwitchConfig, KillSwitchReason


def test_kill_switch_triggers_on_daily_pnl():
    cfg = KillSwitchConfig(enabled=True, daily_realized_pnl_limit=-10.0)
    ks = KillSwitch(cfg)
    state = ks.evaluate(dt.datetime.utcnow(), equity=1000.0, daily_realized_pnl=-15.0)
    assert state.is_triggered
    assert state.reason == KillSwitchReason.DAILY_PNL


def test_kill_switch_triggers_on_drawdown():
    cfg = KillSwitchConfig(enabled=True, max_equity_drawdown_pct=30.0)
    ks = KillSwitch(cfg)
    now = dt.datetime.utcnow()
    ks.evaluate(now, equity=1000.0, daily_realized_pnl=0.0)
    state = ks.evaluate(now + dt.timedelta(seconds=1), equity=650.0, daily_realized_pnl=0.0)
    assert state.is_triggered
    assert state.reason == KillSwitchReason.DRAWDOWN


def test_kill_switch_exception_window():
    cfg = KillSwitchConfig(enabled=True, max_exceptions_per_hour=5, daily_realized_pnl_limit=-100.0)
    ks = KillSwitch(cfg)
    now = dt.datetime.utcnow()
    for i in range(6):
        ks.register_exception(now - dt.timedelta(minutes=i * 5))
    state = ks.evaluate(now, equity=1000.0, daily_realized_pnl=0.0)
    assert state.is_triggered
    assert state.reason == KillSwitchReason.EXCEPTIONS

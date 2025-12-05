from __future__ import annotations

import pandas as pd

from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.system.kill_switch import KillSwitch, KillSwitchConfig
from finantradealgo.backtester.backtest_engine import Backtester, BacktestConfig
from finantradealgo.core.strategy import BaseStrategy, StrategyContext
from finantradealgo.core.strategy import SignalType


def test_risk_engine_clamps_position_size_with_max_notional():
    cfg = RiskConfig(capital_risk_pct_per_trade=0.1, max_notional_per_symbol=100.0, max_leverage=5.0)
    engine = RiskEngine(cfg)
    # price high so notional cap is binding
    size = engine.calc_position_size(equity=10_000.0, price=200.0, atr=None, row=None)
    assert size <= cfg.max_notional_per_symbol / 200.0
    assert size >= 0.0


def test_risk_engine_blocks_when_max_open_trades_reached():
    cfg = RiskConfig()
    engine = RiskEngine(cfg)
    allowed = engine.can_open_new_trade(
        current_date=pd.Timestamp("2024-01-01"),
        equity_start_of_day=10_000.0,
        realized_pnl_today=0.0,
        row=None,
        open_positions=[{"id": 1}],
        max_open_trades=1,
    )
    assert allowed is False


def test_risk_engine_daily_loss_limit_blocks():
    cfg = RiskConfig(max_daily_loss_pct=0.02)
    engine = RiskEngine(cfg)
    # -250 loss on 10k equity breaches 2% threshold
    allowed = engine.can_open_new_trade(
        current_date=pd.Timestamp("2024-01-01"),
        equity_start_of_day=10_000.0,
        realized_pnl_today=-250.0,
        row=None,
        open_positions=[],
        max_open_trades=5,
    )
    assert allowed is False


class AlwaysFlipStrategy(BaseStrategy):
    def init(self, df):
        self.df = df

    def on_bar(self, row, ctx: StrategyContext) -> SignalType:
        if ctx.position is None:
            return "LONG"
        return "CLOSE"


def test_backtester_kill_switch_blocks():
    # Prices drop sharply after first bar to trigger daily loss
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1D"),
            "open": [100, 80, 80],
            "high": [100, 80, 80],
            "low": [100, 80, 80],
            "close": [100, 80, 80],
        }
    )
    ks_cfg = KillSwitchConfig(
        daily_realized_pnl_limit=-0.1,
        max_equity_drawdown_pct=5.0,
    )
    ks = KillSwitch(ks_cfg)
    bt = Backtester(
        strategy=AlwaysFlipStrategy(),
        risk_engine=RiskEngine(RiskConfig(capital_risk_pct_per_trade=0.5, max_notional_per_symbol=None)),
        config=BacktestConfig(initial_cash=100.0, use_bar_extremes_for_stop=False),
        kill_switch=ks,
    )
    _ = bt.run(df)
    # Manually evaluate kill switch with realized loss to ensure guard triggers
    ks.evaluate(
        now=pd.Timestamp("2024-01-01").to_pydatetime(),
        equity=80.0,
        daily_realized_pnl=-5.0,
    )
    assert ks.state.is_triggered is True

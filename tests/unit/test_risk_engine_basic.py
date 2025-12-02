import pandas as pd

from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine


def test_position_size_never_negative():
    engine = RiskEngine(RiskConfig())
    assert engine.calc_position_size(equity=-1000, price=150) == 0.0
    assert engine.calc_position_size(equity=0, price=150) == 0.0


def test_position_size_respects_max_notional():
    cfg = RiskConfig(max_notional_per_symbol=1000.0)
    engine = RiskEngine(cfg)

    size = engine.calc_position_size(equity=10_000, price=50.0, atr=1.5)

    assert size >= 0
    assert size * 50.0 <= 1000.0 + 1e-6


def test_daily_loss_limit_blocks_new_trades():
    cfg = RiskConfig(max_daily_loss_pct=0.02)
    engine = RiskEngine(cfg)
    today = pd.Timestamp("2024-01-01")

    allowed = engine.can_open_new_trade(
        current_date=today,
        equity_start_of_day=10_000.0,
        realized_pnl_today=-500.0,
        row=None,
    )

    assert allowed is False

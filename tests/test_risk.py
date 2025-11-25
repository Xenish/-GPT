from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext
from finantradealgo.risk.position_sizing import PositionSizingInput, calc_size_fixed_risk_pct
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine


class OneBarStrategy(BaseStrategy):
    def __init__(self) -> None:
        self._in_position = False

    def init(self, df: pd.DataFrame) -> None:
        self._in_position = False

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if ctx.position is None:
            return "LONG"
        return "CLOSE"


def _make_test_df() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=8, freq="15min")
    prices = [100, 99, 98, 97, 96, 95, 94, 93]
    data = {
        "timestamp": ts,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1.0] * len(ts),
        "atr_14": [1.0] * len(ts),
        "hv_20": [0.05] * len(ts),
    }
    return pd.DataFrame(data)


def test_daily_loss_limit_blocks_entries():
    cfg = RiskConfig(
        capital_risk_pct_per_trade=0.1,
        max_leverage=2.0,
        max_daily_loss_pct=0.0001,
        tail_risk_hv_threshold=0.9,
        use_tail_risk_guard=False,
    )
    engine = RiskEngine(cfg)
    bt = Backtester(
        strategy=OneBarStrategy(),
        risk_engine=engine,
        config=BacktestConfig(initial_cash=1_000.0, fee_pct=0.0, slippage_pct=0.0),
    )
    result = bt.run(_make_test_df())
    risk_stats = result["risk_stats"]
    blocked_total = sum(risk_stats.get("blocked_entries", {}).values())
    assert blocked_total > 0, "Risk engine should block entries after daily loss limit is hit"


def test_position_size_respects_notional_limit():
    cfg = RiskConfig(
        capital_risk_pct_per_trade=0.01,
        max_leverage=2.0,
        max_notional_per_symbol=15.0,
        use_tail_risk_guard=False,
    )
    engine = RiskEngine(cfg)
    row = pd.Series({"ms_trend_score": 0.4, "hv_20": 0.1})
    size = engine.calc_position_size(
        equity=1_000.0,
        price=1.0,
        atr=0.5,
        row=row,
    )
    assert pytest.approx(size, rel=0.01) == 15.0


def test_fixed_risk_pct_calculation():
    inp = PositionSizingInput(equity=10_000.0, price=100.0, capital_risk_pct_per_trade=0.01)
    size = calc_size_fixed_risk_pct(inp)
    assert pytest.approx(size, rel=0.01) == 100.0

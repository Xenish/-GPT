import pandas as pd
import pytest

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.core.strategy import BaseStrategy, StrategyContext, SignalType


class MiniEndToEndStrategy(BaseStrategy):
    """Opens once, holds briefly, then closes to exercise full flow."""

    def init(self, df: pd.DataFrame) -> None:
        self.length = len(df)

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if ctx.position is None and ctx.index == 0:
            return "LONG"
        if ctx.position is not None and ctx.index >= 2:
            return "CLOSE"
        return None


def _tiny_dataset() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=6, freq="H")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 101, 102, 103, 102, 101],
            "high": [101, 102, 103, 104, 103, 102],
            "low": [99, 100, 101, 102, 101, 100],
            "close": [100, 102, 101, 103, 102, 101],
            "atr_14": [1.0] * 6,
            "hv_20": [0.05] * 6,
        }
    )


@pytest.mark.slow
@pytest.mark.integration
def test_backtest_end_to_end_small_sample():
    df = _tiny_dataset()
    engine = BacktestEngine(strategy=MiniEndToEndStrategy())

    result = engine.run(df)

    assert result["equity_curve"] is not None
    assert len(result["equity_curve"]) > 0
    assert "metrics" in result
    assert result["metrics"]["trade_count"] >= 1
    assert "cum_return" in result["metrics"]
    assert "max_drawdown" in result["metrics"]

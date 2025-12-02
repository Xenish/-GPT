import pandas as pd

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.core.strategy import BaseStrategy, StrategyContext, SignalType


class SimpleFlipStrategy(BaseStrategy):
    """Opens a long position on the first bar and closes after the second bar."""

    def init(self, df: pd.DataFrame) -> None:
        self.length = len(df)

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if ctx.position is None and ctx.index == 0:
            return "LONG"
        if ctx.position is not None and ctx.index >= 2:
            return "CLOSE"
        return None


def _make_minimal_data() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 102.0, 101.0, 103.0],
            "high": [101.0, 103.0, 102.0, 104.0],
            "low": [99.0, 101.0, 100.0, 102.0],
            "close": [100.0, 102.0, 101.0, 103.0],
            "atr_14": [1.0, 1.0, 1.0, 1.0],
            "hv_20": [0.05, 0.05, 0.05, 0.05],
        }
    )


def test_backtest_engine_runs_minimal_long_cycle():
    df = _make_minimal_data()
    engine = BacktestEngine(strategy=SimpleFlipStrategy())

    result = engine.run(df)

    assert "equity_curve" in result
    assert "trades" in result
    assert "metrics" in result
    assert len(result["equity_curve"]) > 0
    assert result["metrics"]["trade_count"] >= 1
    assert "pnl" in result["trades"].columns

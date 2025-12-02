import time

import pandas as pd
import pytest

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.core.strategy import BaseStrategy, StrategyContext, SignalType

pytestmark = [pytest.mark.slow, pytest.mark.performance]


class PerfSmokeStrategy(BaseStrategy):
    """Minimal strategy to keep performance run deterministic and fast."""

    def init(self, df: pd.DataFrame) -> None:
        pass

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if ctx.position is None:
            return "LONG"
        if ctx.index % 3 == 0:
            return "CLOSE"
        return None


def _small_perf_dataset() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=30, freq="T")
    base = pd.Series(range(len(ts)), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100 + base,
            "high": 100.5 + base,
            "low": 99.5 + base,
            "close": 100.2 + base,
            "atr_14": [1.0] * len(ts),
            "hv_20": [0.05] * len(ts),
        }
    )


def test_backtest_speed_small_case():
    df = _small_perf_dataset()
    engine = BacktestEngine(strategy=PerfSmokeStrategy())

    start = time.perf_counter()
    result = engine.run(df)
    elapsed = time.perf_counter() - start

    assert result["equity_curve"] is not None
    assert elapsed < 10.0

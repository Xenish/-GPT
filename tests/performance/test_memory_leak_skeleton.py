import tracemalloc

import pandas as pd
import pytest

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.core.strategy import BaseStrategy, StrategyContext, SignalType

pytestmark = [pytest.mark.slow, pytest.mark.performance]


class LeakProbeStrategy(BaseStrategy):
    def init(self, df: pd.DataFrame) -> None:
        pass

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:  # pragma: no cover - simple probe
        if ctx.position is None:
            return "LONG"
        if ctx.index % 5 == 0:
            return "CLOSE"
        return None


def _dataset():
    ts = pd.date_range("2024-01-01", periods=20, freq="T")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100 + i for i in range(len(ts))],
            "high": [100.5 + i for i in range(len(ts))],
            "low": [99.5 + i for i in range(len(ts))],
            "close": [100.2 + i for i in range(len(ts))],
            "atr_14": [1.0] * len(ts),
            "hv_20": [0.05] * len(ts),
        }
    )


@pytest.mark.xfail(reason="Memory leak detection not fully implemented yet")
def test_backtest_memory_usage_stable(tmp_path):
    engine = BacktestEngine(strategy=LeakProbeStrategy())
    df = _dataset()

    tracemalloc.start()
    snapshots = []

    for _ in range(3):
        engine.run(df)
        snapshots.append(tracemalloc.take_snapshot())

    tracemalloc.stop()

    # TODO: analyze snapshots and assert on differences when leak detection is ready.
    assert len(snapshots) == 3

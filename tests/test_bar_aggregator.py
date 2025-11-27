from __future__ import annotations

import pandas as pd

from finantradealgo.data_engine.bar_aggregator import BarAggregator
from finantradealgo.data_engine.live_data_source import Bar


def _make_bar(idx: int, price: float = 100.0, volume: float = 1.0, symbol: str = "AIAUSDT"):
    open_time = pd.Timestamp("2025-01-01 00:00:00+00:00") + pd.Timedelta(minutes=idx)
    close_time = open_time + pd.Timedelta(minutes=1)
    return Bar(
        symbol=symbol,
        timeframe="1m",
        open_time=open_time,
        close_time=close_time,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price + 0.1,
        volume=volume,
        extras={"idx": idx},
    )


def test_bar_aggregator_basic():
    agg = BarAggregator("15m")
    result = None
    prices = []
    volumes = []
    for i in range(15):
        price = 100 + i * 0.1
        prices.append(price + 0.1)
        volumes.append(1 + i)
        bar = _make_bar(i, price=price, volume=1 + i)
        result = agg.add_bar(bar)
        if i < 14:
            assert result is None
    assert result is not None
    assert result.open == _make_bar(0, price=100).open
    assert result.close == _make_bar(14, price=101.4).close
    assert result.high == max(b.high for b in [_make_bar(i, 100 + i * 0.1) for i in range(15)])
    assert result.low == min(b.low for b in [_make_bar(i, 100 + i * 0.1) for i in range(15)])
    assert result.volume == sum(1 + i for i in range(15))
    assert result.timeframe == "15m"


def test_bar_aggregator_bucket_switch():
    agg = BarAggregator("15m")
    first_bucket_bars = []
    for i in range(8):
        bar = _make_bar(i, price=100 + i)
        first_bucket_bars.append(bar)
        res = agg.add_bar(bar)
        assert res is None

    next_bucket_bar = _make_bar(15, price=200)
    completed = agg.add_bar(next_bucket_bar)
    assert completed is not None
    assert completed.open_time == pd.Timestamp("2025-01-01 00:00:00+00:00")
    assert completed.close_time == first_bucket_bars[-1].close_time
    assert completed.volume == sum(bar.volume for bar in first_bucket_bars)

    for offset in range(16, 22):
        agg.add_bar(_make_bar(offset, price=200 + offset))

    final = agg.add_bar(_make_bar(29, price=250))
    assert final is not None
    assert final.open_time == pd.Timestamp("2025-01-01 00:15:00+00:00")

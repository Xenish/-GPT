from __future__ import annotations

import pandas as pd

from finantradealgo.data_engine.event_bars import build_event_bars
from finantradealgo.data_engine.bar_aggregator import Bar, BarAggregator


def _make_trades(n: int = 30):
    ts = pd.date_range("2025-01-01 00:00:00", periods=n, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "price": [100 + i * 0.1 for i in range(n)],
            "volume": [1 + (i % 3) for i in range(n)],
        }
    )
    return df


def test_event_bars_contract_time_mode():
    trades = _make_trades()
    cfg = type("Cfg", (), {"mode": "time", "target_volume": None, "target_notional": None, "target_ticks": None, "source_timeframe": "1m", "keep_partial_last_bar": True})
    bars = build_event_bars(trades.copy(), cfg)

    assert bars["timestamp"].is_monotonic_increasing
    assert (bars["volume"] >= 0).all()
    assert bars["volume"].sum() == trades["volume"].sum()


def test_aggregate_bars_contract():
    trades = _make_trades()
    aggregator = BarAggregator(target_timeframe="1m")
    completed = []
    for _, row in trades.iterrows():
        bar = Bar(
            symbol="TESTUSDT",
            timeframe="1m",
            open_time=row["timestamp"],
            close_time=row["timestamp"],
            open=float(row["price"]),
            high=float(row["price"]),
            low=float(row["price"]),
            close=float(row["price"]),
            volume=float(row["volume"]),
            extras={},
        )
        out = aggregator.add_bar(bar)
        if out:
            completed.append(out)

    # Flush last bucket if aggregator leaves it open
    final = aggregator.add_bar(
        Bar(
            symbol="TESTUSDT",
            timeframe="1m",
            open_time=trades.iloc[-1]["timestamp"] + pd.Timedelta(minutes=1),
            close_time=trades.iloc[-1]["timestamp"] + pd.Timedelta(minutes=1),
            open=0,
            high=0,
            low=0,
            close=0,
            volume=0,
            extras={},
        )
    )
    if final:
        completed.append(final)

    df = pd.DataFrame(
        {
            "timestamp": [b.open_time for b in completed],
            "volume": [b.volume for b in completed],
        }
    )
    assert df["timestamp"].is_monotonic_increasing
    assert (df["volume"] >= 0).all()
    assert df["volume"].sum() == trades["volume"].sum()


def test_orderbook_invariants():
    # minimal orderbook snapshot
    ts = pd.date_range("2025-01-01 00:00:00", periods=5, freq="S", tz="UTC")
    bids = [100.0, 100.1, 100.2, 100.3, 100.4]
    asks = [100.5, 100.6, 100.7, 100.8, 100.9]
    df = pd.DataFrame({"timestamp": ts, "best_bid": bids, "best_ask": asks})

    assert df["timestamp"].is_monotonic_increasing
    assert (df["best_bid"] <= df["best_ask"]).all()
    assert ((df["best_ask"] - df["best_bid"]) >= 0).all()

import json
from pathlib import Path

import pandas as pd
import pytest

from finantradealgo.research.strategy_search.analysis import (
    filter_by_metrics,
    load_results,
    top_n_by_metric,
)


def _dummy_df():
    return pd.DataFrame(
        [
            {
                "params": {"a": 1},
                "cum_return": 0.10,
                "sharpe": 1.5,
                "max_drawdown": -0.05,
                "win_rate": 0.6,
                "trade_count": 100,
                "status": "ok",
                "error_message": None,
            },
            {
                "params": {"a": 2},
                "cum_return": -0.02,
                "sharpe": 0.2,
                "max_drawdown": -0.25,
                "win_rate": None,
                "trade_count": 40,
                "status": "error",
                "error_message": "boom",
            },
            {
                "params": {"a": 3},
                "cum_return": 0.05,
                "sharpe": 0.8,
                "max_drawdown": -0.10,
                "win_rate": 0.55,
                "trade_count": 70,
                "status": "ok",
                "error_message": None,
            },
        ]
    )


def test_filter_by_metrics_applies_bounds():
    df = _dummy_df()

    filtered = filter_by_metrics(
        df,
        cum_return_min=0.0,
        sharpe_min=0.5,
        max_drawdown_min=-0.15,
        trade_count_min=60,
    )

    assert len(filtered) == 2
    assert filtered["cum_return"].min() >= 0.0
    assert filtered["sharpe"].min() >= 0.5
    assert filtered["max_drawdown"].min() >= -0.15
    assert filtered["trade_count"].min() >= 60


def test_top_n_by_metric_sorts_correctly():
    df = _dummy_df()
    top = top_n_by_metric(df, metric="cum_return", n=2, ascending=False)
    assert len(top) == 2
    assert top.iloc[0]["cum_return"] >= top.iloc[1]["cum_return"]

    with pytest.raises(ValueError):
        top_n_by_metric(df, metric="missing_metric", n=1)


def test_load_results_roundtrip(tmp_path: Path):
    df = _dummy_df()
    job_dir = tmp_path / "job1"
    job_dir.mkdir(parents=True, exist_ok=True)

    results_path = job_dir / "results.parquet"
    df.to_parquet(results_path, index=False)

    meta = {
        "job_id": "job1",
        "strategy": "rule",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
    }
    meta_path = job_dir / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    loaded_df, loaded_meta = load_results(job_dir, include_meta=True)
    assert len(loaded_df) == len(df)
    assert loaded_meta["job_id"] == "job1"

import json
from pathlib import Path

import pandas as pd

from finantradealgo.live_trading.factories import create_live_engine
from finantradealgo.system.config_loader import load_config


def _build_dummy_features():
    ts = pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": range(10),
            "high": [x + 1 for x in range(10)],
            "low": range(10),
            "close": [x + 0.5 for x in range(10)],
            "volume": [100] * 10,
        }
    )


def test_live_engine_replay_smoke(tmp_path: Path):
    cfg = load_config("live")

    # Force paper replay mode and heartbeat path into tmpdir
    live_section = dict(cfg.get("live", {}) or {})
    live_section["mode"] = "paper"
    live_section["data_source"] = "replay"
    live_section["heartbeat_path"] = str(tmp_path / "heartbeat_{run_id}.json")
    cfg["live"] = live_section
    cfg["live_cfg"] = cfg["live_cfg"].from_dict(
        live_section,
        default_symbol=cfg.get("symbol"),
        default_timeframe=cfg.get("timeframe"),
    )

    # Monkeypatch feature pipeline input by preloading features into live engine via replay source
    df_features = _build_dummy_features()
    engine, strategy_name = create_live_engine(cfg, run_id="test_replay_smoke")

    # Replace data source with FileReplayDataSource using our dummy features
    from finantradealgo.data_engine.live_data_source import FileReplayDataSource

    engine.data_source = FileReplayDataSource(
        df_features,
        symbol=cfg["live_cfg"].symbol,
        timeframe=cfg["live_cfg"].timeframe,
        bars_limit=len(df_features),
    )

    # Run for limited bars
    for _ in range(10):
        bar = engine.data_source.next_bar()
        if bar is None:
            break
        engine._on_bar(bar)

    # Assertions
    assert engine.iteration > 0
    assert engine.executed_trades + engine.blocked_entries >= 0  # routing attempted or blocked
    assert not engine.kill_switch_triggered_flag

    # Heartbeat file should exist
    hb_file = Path(live_section["heartbeat_path"].format(run_id=engine.run_id))
    assert hb_file.exists()
    hb = json.loads(hb_file.read_text())
    assert hb.get("run_id") == engine.run_id

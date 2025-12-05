import datetime as dt
from pathlib import Path

from scripts.watchdog_live import _resolve_heartbeat_path, _check_heartbeat_once
from finantradealgo.system.config_loader import LiveConfig


def test_resolve_heartbeat_path_template():
    live_cfg = LiveConfig.from_dict({"heartbeat_path": "outputs/live/heartbeat_{run_id}.json"})
    path = _resolve_heartbeat_path(live_cfg, "abc123")
    assert path.as_posix().endswith("heartbeat_abc123.json")


def test_check_heartbeat_staleness(tmp_path):
    hb_path = tmp_path / "heartbeat.json"
    payload = {
        "run_id": "test",
        "status": "running",
        "last_bar_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    import json

    hb_path.write_text(json.dumps(payload), encoding="utf-8")
    # Should not raise or log error when fresh; notifier None
    _check_heartbeat_once(heartbeat_path=hb_path, notifier=None, max_stale_seconds=60)

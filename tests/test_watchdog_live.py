from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import List

from scripts.watchdog_live import run_watchdog_loop


class DummyNotifier:
    def __init__(self) -> None:
        self.warn_calls: List[str] = []
        self.info_calls: List[str] = []
        self.critical_calls: List[str] = []

    def info(self, msg: str) -> None:
        self.info_calls.append(msg)

    def warn(self, msg: str) -> None:
        self.warn_calls.append(msg)

    def critical(self, msg: str) -> None:
        self.critical_calls.append(msg)


def test_watchdog_warns_for_stale_heartbeat(tmp_path):
    heartbeat_path = Path(tmp_path) / "heartbeat.json"
    stale_time = dt.datetime.utcnow() - dt.timedelta(seconds=600)
    heartbeat_path.write_text(json.dumps({"run_id": "run123", "updated_at": stale_time.isoformat()}))
    notifier = DummyNotifier()

    run_watchdog_loop(
        heartbeat_path=heartbeat_path,
        notifier=notifier,
        max_stale_seconds=10,
        interval=0,
        iterations=1,
        sleep_fn=lambda *_: None,
    )

    assert notifier.warn_calls, "Expected notifier.warn to be called for stale heartbeat"
    assert "run123" in notifier.warn_calls[0]

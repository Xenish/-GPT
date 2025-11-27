from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from finantradealgo.system.config_loader import load_system_config
from finantradealgo.system.notifications import create_notification_manager


def main() -> None:
    cfg = load_system_config()
    live_cfg = cfg.get("live_cfg")
    notifier = create_notification_manager(cfg.get("notifications_cfg")) if cfg.get("notifications_cfg") else None
    heartbeat_path = Path(
        getattr(live_cfg, "heartbeat_path", "outputs/live/heartbeat.json")
    )
    max_stale_seconds = int(cfg.get("watchdog", {}).get("max_stale_seconds", 300))
    interval = int(cfg.get("watchdog", {}).get("interval_seconds", 60))

    print(f"[WATCHDOG] Monitoring heartbeat at {heartbeat_path} (max stale {max_stale_seconds}s)")
    while True:
        try:
            with heartbeat_path.open("r", encoding="utf-8") as fh:
                hb = json.load(fh)
            updated_at_raw = hb.get("updated_at")
            if updated_at_raw is None:
                print("[WATCHDOG] Heartbeat missing updated_at field.")
            else:
                updated_at = dt.datetime.fromisoformat(updated_at_raw)
                delta = (dt.datetime.utcnow() - updated_at).total_seconds()
                if delta > max_stale_seconds:
                    msg = (
                        f"Heartbeat stale ({delta:.0f}s). Live process may be stalled "
                        f"or stopped. run_id={hb.get('run_id')}"
                    )
                    print(f"[WATCHDOG] {msg}")
                    if notifier:
                        try:
                            notifier.warn(msg)
                        except Exception:
                            pass
                else:
                    print(f"[WATCHDOG] OK (age {delta:.0f}s). Run={hb.get('run_id')}")
        except FileNotFoundError:
            print(f"[WATCHDOG] Heartbeat file not found at {heartbeat_path}")
        except Exception as exc:
            print(f"[WATCHDOG] Error reading heartbeat: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    main()

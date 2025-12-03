from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Callable, Optional

from finantradealgo.system.config_loader import LiveConfig, load_config
from finantradealgo.system.notifications import create_notification_manager
from finantradealgo.system.notifier import Notifier


def _resolve_heartbeat_path(live_cfg: LiveConfig, run_id: str) -> Path:
    template = getattr(live_cfg, "heartbeat_path", None) or "outputs/live/heartbeat_{run_id}.json"
    try:
        resolved = template.format(run_id=run_id)
    except (KeyError, IndexError):
        resolved = template
    return Path(resolved)


def _check_heartbeat_once(
    *,
    heartbeat_path: Path,
    notifier: Optional[Notifier],
    max_stale_seconds: int,
) -> None:
    try:
        with heartbeat_path.open("r", encoding="utf-8") as fh:
            hb = json.load(fh)
    except FileNotFoundError:
        print(f"[WATCHDOG] Heartbeat file not found at {heartbeat_path}")
        return
    except Exception as exc:
        print(f"[WATCHDOG] Error reading heartbeat: {exc}")
        return

    updated_at_raw = hb.get("updated_at")
    if updated_at_raw is None:
        print("[WATCHDOG] Heartbeat missing updated_at field.")
        return
    try:
        updated_at = dt.datetime.fromisoformat(updated_at_raw)
    except ValueError:
        print(f"[WATCHDOG] Invalid heartbeat timestamp: {updated_at_raw}")
        return
    delta = (dt.datetime.now(dt.timezone.utc) - updated_at).total_seconds()
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


def run_watchdog_loop(
    *,
    heartbeat_path: Path,
    notifier: Optional[Notifier],
    max_stale_seconds: int,
    interval: int,
    sleep_fn: Callable[[float], None] = time.sleep,
    iterations: Optional[int] = None,
) -> None:
    iteration = 0
    while iterations is None or iteration < iterations:
        _check_heartbeat_once(
            heartbeat_path=heartbeat_path,
            notifier=notifier,
            max_stale_seconds=max_stale_seconds,
        )
        iteration += 1
        if iterations is not None and iteration >= iterations:
            break
        sleep_fn(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watchdog for live heartbeat health.")
    parser.add_argument(
        "--run-id",
        default="live_engine",
        help="Run id used to resolve heartbeat file path template.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional number of iterations before exiting (default: run forever).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("live")
    live_cfg: LiveConfig = cfg.get("live_cfg")
    notifier = (
        create_notification_manager(cfg.get("notifications_cfg"))
        if cfg.get("notifications_cfg")
        else None
    )
    heartbeat_path = _resolve_heartbeat_path(live_cfg, args.run_id)
    max_stale_seconds = int(cfg.get("watchdog", {}).get("max_stale_seconds", 300))
    interval = int(cfg.get("watchdog", {}).get("interval_seconds", 60))

    print(f"[WATCHDOG] Monitoring heartbeat at {heartbeat_path} (max stale {max_stale_seconds}s)")
    run_watchdog_loop(
        heartbeat_path=heartbeat_path,
        notifier=notifier,
        max_stale_seconds=max_stale_seconds,
        interval=interval,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()

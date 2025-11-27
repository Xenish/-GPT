from __future__ import annotations

from typing import Dict, Optional

import requests

from .notifier import Notifier

_LEVEL_ORDER = {"info": 0, "warn": 1, "critical": 2}


class FCMNotificationChannel(Notifier):
    def __init__(self, server_key: str, topic: str, min_level: str = "info") -> None:
        self.server_key = server_key
        self.topic = topic
        self.min_level = min_level

    def _should_send(self, level: str) -> bool:
        return _LEVEL_ORDER.get(level, 0) >= _LEVEL_ORDER.get(self.min_level, 0)

    def _send(
        self,
        title: str,
        body: str,
        level: str,
        extra_data: Optional[Dict[str, str]] = None,
    ) -> None:
        if not self._should_send(level):
            return

        url = "https://fcm.googleapis.com/fcm/send"
        headers = {
            "Authorization": f"key={self.server_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "to": f"/topics/{self.topic}",
            "notification": {"title": title, "body": body},
            "data": {
                "level": level,
                **(extra_data or {}),
            },
        }

        try:
            requests.post(url, json=payload, headers=headers, timeout=5)
        except Exception:
            # Optional: log instead of swallowing
            pass

    def info(self, msg: str) -> None:
        self._send("FinanTrade Info", msg, "info")

    def warn(self, msg: str) -> None:
        self._send("FinanTrade Warning", msg, "warn")

    def critical(self, msg: str) -> None:
        self._send("FinanTrade CRITICAL", msg, "critical")
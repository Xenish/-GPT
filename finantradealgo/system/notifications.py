from __future__ import annotations

from typing import List, Optional

from .notifier import Notifier
from .notification_fcm import FCMNotificationChannel
from .config_loader import NotificationsConfig, resolve_env_placeholders


class NotificationManager(Notifier):
    def __init__(self, channels: List[Notifier]):
        self.channels = channels

    def info(self, msg: str) -> None:
        for ch in self.channels:
            ch.info(msg)

    def warn(self, msg: str) -> None:
        for ch in self.channels:
            ch.warn(msg)

    def critical(self, msg: str) -> None:
        for ch in self.channels:
            ch.critical(msg)


def create_notification_manager(cfg: NotificationsConfig) -> Optional[NotificationManager]:
    if not cfg.enabled:
        return None

    channels: List[Notifier] = []

    if cfg.fcm.enabled:
        server_key = resolve_env_placeholders(cfg.fcm.server_key)
        channels.append(
            FCMNotificationChannel(
                server_key=server_key,
                topic=cfg.fcm.topic,
                min_level=cfg.fcm.min_level,
            )
        )

    if not channels:
        return None

    return NotificationManager(channels)

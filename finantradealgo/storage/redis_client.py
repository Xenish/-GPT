from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import logging

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """
    Configuration for Redis cache layer.

    Attributes:
        host:
            Redis host.
        port:
            Redis port.
        db:
            Logical database number.
        password:
            Optional password.
        decode_responses:
            Whether to decode bytes to str automatically.
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    decode_responses: bool = True


class RedisClient:
    """
    Thin wrapper around redis-py client.

    Responsibilities:
    - Simple get/set/delete for caching.
    - Pub/Sub helpers for real-time updates.
    - Optional rate limiting (to be added later).
    """

    def __init__(self, config: RedisConfig) -> None:
        self.config = config

        try:
            import redis  # type: ignore
        except Exception as exc:
            logger.error("redis package is required for RedisClient: %s", exc)
            raise

        self._redis = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            decode_responses=self.config.decode_responses,
        )

    # ---- Basic KV API ------------------------------------------------------

    def get(self, key: str) -> Any:
        return self._redis.get(key)

    def set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
    ) -> bool:
        """
        Set a value with optional expiry (seconds).
        """
        return bool(self._redis.set(name=key, value=value, ex=ex))

    def delete(self, key: str) -> int:
        return int(self._redis.delete(key))

    def exists(self, key: str) -> bool:
        return bool(self._redis.exists(key))

    # ---- Pub/Sub API -------------------------------------------------------

    def publish(self, channel: str, message: str) -> int:
        """
        Publish a message to a channel.
        """
        return int(self._redis.publish(channel, message))

    def subscribe(
        self,
        channel: str,
        handler: Callable[[str], None],
        *,
        run_forever: bool = False,
        max_messages: int | None = None,
    ) -> None:
        """
        Subscribe to a channel and call handler(message) for each message.

        This is a simple, blocking implementation intended for background workers.
        """
        pubsub = self._redis.pubsub()
        pubsub.subscribe(channel)

        count = 0
        for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            data = msg["data"]
            handler(str(data))
            count += 1
            if not run_forever and max_messages is not None and count >= max_messages:
                break

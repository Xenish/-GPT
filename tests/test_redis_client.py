from __future__ import annotations

import os

import pytest

from finantradealgo.storage.redis_client import RedisClient, RedisConfig


@pytest.mark.db
def test_redis_client_smoke():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        pytest.skip("REDIS_URL not set")

    pytest.importorskip("redis")

    # Parse basic redis://host:port/db
    import urllib.parse

    parsed = urllib.parse.urlparse(redis_url)
    cfg = RedisConfig(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        db=int((parsed.path or "/0").lstrip("/") or 0),
        password=parsed.password,
        decode_responses=True,
    )
    client = RedisClient(cfg)
    key = "ft_test_key"
    client.set(key, "value")
    assert client.get(key) == "value"
    client.delete(key)

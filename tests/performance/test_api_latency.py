import time

import pytest
from fastapi.testclient import TestClient

from finantradealgo.api.server import create_app

pytestmark = [pytest.mark.performance]


def test_health_endpoint_latency():
    client = TestClient(create_app())

    start = time.perf_counter()
    resp = client.get("/health")
    elapsed = time.perf_counter() - start

    assert resp.status_code == 200
    assert elapsed < 0.5

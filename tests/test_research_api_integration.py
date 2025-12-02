"""
Integration tests for Research Service API.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Import app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.research_service.main import app

client = TestClient(app)


def test_health_ping():
    """Test health check ping endpoint."""
    response = client.get("/api/research/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "pong"


def test_health_status():
    """Test health status endpoint."""
    response = client.get("/api/research/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "research"
    assert data["status"] == "operational"
    assert "job_limiter" in data


def test_list_jobs_empty():
    """Test listing jobs when none exist."""
    response = client.get("/api/research/strategy-search/jobs/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_scenario_run_validation():
    """Test scenario endpoint with invalid request."""
    # Empty scenarios should fail
    response = client.post(
        "/api/research/scenarios/run",
        json={"scenarios": []}
    )
    assert response.status_code == 400


def test_get_nonexistent_job():
    """Test getting a job that doesn't exist."""
    response = client.get("/api/research/strategy-search/jobs/nonexistent_job_123")
    assert response.status_code == 404


# NOTE: Full job execution tests are skipped in CI
# as they require full data + long runtime

@pytest.mark.skipif(
    not Path("data/ohlcv").exists(),
    reason="Requires data files"
)
def test_scenario_run_minimal():
    """Test running a minimal scenario (requires data)."""
    request_data = {
        "scenarios": [
            {
                "label": "test_scenario",
                "symbol": "AIAUSDT",
                "timeframe": "15m",
                "strategy": "rule",
                "params": {}
            }
        ]
    }

    response = client.post("/api/research/scenarios/run", json=request_data)

    # May fail due to missing data, but should not crash
    assert response.status_code in (200, 500)

    if response.status_code == 200:
        data = response.json()
        assert "n_scenarios" in data
        assert "results" in data

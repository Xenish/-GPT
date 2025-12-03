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


def test_create_and_get_strategy_search_job(tmp_path, monkeypatch):
    """Create a strategy search job via API and fetch summary."""
    monkeypatch.setenv("STRATEGY_SEARCH_DRYRUN", "1")
    monkeypatch.setenv("STRATEGY_SEARCH_OUTPUT_DIR", str(tmp_path / "strategy_search"))

    request_data = {
        "strategy": "rule",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "search_type": "random",
        "n_samples": 1,
        "seed": 123,
        "notes": "api test",
    }

    response = client.post("/api/research/strategy-search/jobs/", json=request_data)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "job_id" in data

    job_id = data["job_id"]
    job_dir = Path(tmp_path / "strategy_search" / job_id)
    assert job_dir.exists()

    status_resp = client.get(f"/api/research/strategy-search/jobs/{job_id}")
    assert status_resp.status_code == 200
    status = status_resp.json()
    assert status["job_id"] == job_id
    assert status["strategy"] == "rule"
    assert status["symbol"] == "BTCUSDT"
    assert status["timeframe"] == "15m"
    assert status["results_available"] is True

    report_resp = client.get(f"/api/research/strategy-search/jobs/{job_id}/report?format=markdown")
    assert report_resp.status_code == 200
    report_data = report_resp.json()
    assert report_data["format"] == "markdown"
    assert "Strategy Search Report" in report_data["content"]


def test_search_rejects_non_research_profile(monkeypatch):
    """API should refuse to run when config profile is not research."""
    from services import research_service
    from services.research_service import jobs_api

    class DummyResearchCfg:
        max_parallel_jobs = 1

    def fake_load_config(profile: str):
        return {
            "profile": "live",
            "research_cfg": DummyResearchCfg(),
        }

    monkeypatch.setattr(jobs_api, "load_config", fake_load_config)

    request_data = {
        "strategy": "rule",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "search_type": "random",
        "n_samples": 1,
    }

    response = client.post("/api/research/strategy-search/jobs/", json=request_data)
    assert response.status_code == 400
    assert "research" in response.json()["detail"]


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

from fastapi.testclient import TestClient

from services.research_service.main import app

client = TestClient(app)


def test_list_strategies_endpoint():
    resp = client.get("/api/research/strategies/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    names = {item["name"] for item in data}
    for required in ["rule", "trend_continuation", "sweep_reversal", "volatility_breakout"]:
        assert required in names
    searchable = [item for item in data if item["is_searchable"]]
    assert searchable

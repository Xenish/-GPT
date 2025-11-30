"""
Tests for /api/meta endpoint with multi-timeframe/multi-symbol configuration.

Sprint T3.5 - Validate that API correctly returns multi-TF config including
lookback_days and ml_targets.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Set dummy FCM key to avoid validation errors
if not os.getenv("FCM_SERVER_KEY"):
    os.environ["FCM_SERVER_KEY"] = "dummy_test_key"

from finantradealgo.api.server import create_app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    app = create_app()
    return TestClient(app)


class TestApiMetaMultiTF:
    """Test suite for /api/meta endpoint with multi-timeframe configuration."""

    def test_meta_returns_multiple_symbols(self, client):
        """Test that /api/meta returns multiple configured symbols."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        assert "symbols" in data
        assert isinstance(data["symbols"], list)
        assert len(data["symbols"]) >= 5, f"Expected at least 5 symbols, got {len(data['symbols'])}"

        # Check that expected symbols are present
        expected_symbols = {"AIAUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"}
        actual_symbols = set(data["symbols"])
        assert expected_symbols.issubset(actual_symbols), \
            f"Expected symbols {expected_symbols} not all present in {actual_symbols}"

    def test_meta_returns_multiple_timeframes(self, client):
        """Test that /api/meta returns multiple configured timeframes."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        assert "timeframes" in data
        assert isinstance(data["timeframes"], list)
        assert len(data["timeframes"]) >= 4, \
            f"Expected at least 4 timeframes, got {len(data['timeframes'])}"

        # Check that expected timeframes are present
        expected_tfs = {"1m", "5m", "15m", "1h"}
        actual_tfs = set(data["timeframes"])
        assert expected_tfs.issubset(actual_tfs), \
            f"Expected timeframes {expected_tfs} not all present in {actual_tfs}"

    def test_meta_returns_lookback_days(self, client):
        """Test that /api/meta returns per-timeframe lookback configuration."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        assert "lookback_days" in data

        lookback = data["lookback_days"]
        if lookback is not None:  # Optional field
            assert isinstance(lookback, dict)

            # Validate specific lookback configurations from system.yml
            assert lookback.get("1m") == 90, "1m lookback should be 90 days"
            assert lookback.get("5m") == 90, "5m lookback should be 90 days"
            assert lookback.get("15m") == 180, "15m lookback should be 180 days"
            assert lookback.get("1h") == 365, "1h lookback should be 365 days"

    def test_meta_returns_default_lookback_days(self, client):
        """Test that /api/meta returns default_lookback_days."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        assert "default_lookback_days" in data

        default_lookback = data["default_lookback_days"]
        if default_lookback is not None:  # Optional field
            assert isinstance(default_lookback, int)
            assert default_lookback == 365, "Default lookback should be 365 days"

    def test_meta_returns_ml_targets(self, client):
        """Test that /api/meta returns ML training targets."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        assert "ml_targets" in data

        ml_targets = data["ml_targets"]
        if ml_targets is not None:  # Optional field
            assert isinstance(ml_targets, list)

            # Should have at least 2 targets from system.yml (BTCUSDT/15m, AIAUSDT/15m)
            assert len(ml_targets) >= 2, f"Expected at least 2 ML targets, got {len(ml_targets)}"

            # Validate structure of targets
            for target in ml_targets:
                assert "symbol" in target
                assert "timeframe" in target
                assert isinstance(target["symbol"], str)
                assert isinstance(target["timeframe"], str)

            # Check specific targets from config
            target_pairs = {(t["symbol"], t["timeframe"]) for t in ml_targets}
            assert ("BTCUSDT", "15m") in target_pairs, "BTCUSDT/15m should be in ML targets"
            assert ("AIAUSDT", "15m") in target_pairs, "AIAUSDT/15m should be in ML targets"

    def test_meta_returns_strategies(self, client):
        """Test that /api/meta returns available strategies."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        assert "strategies" in data
        assert isinstance(data["strategies"], list)
        assert len(data["strategies"]) > 0

    def test_meta_returns_scenario_presets(self, client):
        """Test that /api/meta returns scenario presets."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        assert "scenario_presets" in data
        assert isinstance(data["scenario_presets"], list)

    def test_meta_all_fields_present(self, client):
        """Test that /api/meta returns all expected fields."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()

        # Required fields
        required_fields = ["symbols", "timeframes", "strategies", "scenario_presets"]
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from response"

        # Optional fields (should be present even if null)
        optional_fields = ["lookback_days", "default_lookback_days", "ml_targets"]
        for field in optional_fields:
            assert field in data, f"Optional field '{field}' should be present (can be null)"

    def test_meta_consistency_symbols_in_ml_targets(self, client):
        """Test that ML targets only reference valid symbols."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        valid_symbols = set(data["symbols"])

        ml_targets = data.get("ml_targets")
        if ml_targets:
            for target in ml_targets:
                assert target["symbol"] in valid_symbols, \
                    f"ML target symbol '{target['symbol']}' not in valid symbols {valid_symbols}"

    def test_meta_consistency_timeframes_in_ml_targets(self, client):
        """Test that ML targets only reference valid timeframes."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        valid_timeframes = set(data["timeframes"])

        ml_targets = data.get("ml_targets")
        if ml_targets:
            for target in ml_targets:
                assert target["timeframe"] in valid_timeframes, \
                    f"ML target timeframe '{target['timeframe']}' not in valid timeframes {valid_timeframes}"

    def test_meta_consistency_lookback_keys_are_timeframes(self, client):
        """Test that lookback_days keys match configured timeframes."""
        response = client.get("/api/meta")
        assert response.status_code == 200

        data = response.json()
        valid_timeframes = set(data["timeframes"])

        lookback = data.get("lookback_days")
        if lookback:
            for tf in lookback.keys():
                assert tf in valid_timeframes, \
                    f"Lookback key '{tf}' not in valid timeframes {valid_timeframes}"

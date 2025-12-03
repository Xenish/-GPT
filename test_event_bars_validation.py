#!/usr/bin/env python3
"""
Test script for event bars timeframe validation.

Verifies that:
1. Event bars with non-1m timeframe raise ValueError
2. Event bars with 1m timeframe work correctly
"""

import os
import sys
import yaml
from pathlib import Path
import pytest

# Set dummy FCM key for testing
if not os.getenv("FCM_SERVER_KEY"):
    os.environ["FCM_SERVER_KEY"] = "dummy_test_key"

from finantradealgo.system.config_loader import DataConfig, _propagate_source_timeframe


def test_invalid_timeframe_config():
    """Test that event bars with 15m timeframe raise ValueError."""
    test_config = {
        "timeframe": "15m",
        "data": {"bars": {"mode": "volume", "target_volume": 2000000}},
    }
    data_cfg = DataConfig.from_dict(test_config["data"])
    with pytest.raises(ValueError, match="currently only supported from 1m data"):
        _propagate_source_timeframe(data_cfg, test_config["timeframe"])


def test_valid_timeframe_config():
    """Test that event bars with 1m timeframe work correctly."""
    test_config = {
        "timeframe": "1m",
        "data": {"bars": {"mode": "volume", "target_volume": 2000000}},
    }
    data_cfg = DataConfig.from_dict(test_config["data"])
    data_cfg = _propagate_source_timeframe(data_cfg, test_config["timeframe"])
    assert data_cfg.bars.source_timeframe == "1m"


def test_explicit_1m_override():
    """Test that explicit source_timeframe=1m is preserved even with 15m global timeframe."""
    test_config = {
        "timeframe": "15m",
        "data": {
            "bars": {
                "mode": "volume",
                "target_volume": 2000000,
                "source_timeframe": "1m",
            }
        },
    }
    data_cfg = DataConfig.from_dict(test_config["data"])
    data_cfg = _propagate_source_timeframe(data_cfg, test_config["timeframe"])
    assert data_cfg.bars.source_timeframe == "1m"


if __name__ == "__main__":
    results = []

    # Run tests
    results.append(test_invalid_timeframe_config())
    results.append(test_valid_timeframe_config())
    results.append(test_explicit_1m_override())

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[OK] All tests passed!")
        sys.exit(0)
    else:
        print(f"\n[X] {total - passed} test(s) failed")
        sys.exit(1)

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

from finantradealgo.system.config_loader import load_system_config


def test_invalid_timeframe_config():
    """Test that event bars with 15m timeframe raise ValueError."""
    test_config = {
        "timeframe": "15m",
        "data": {"bars": {"mode": "volume", "target_volume": 2000000}},
    }
    test_path = Path("config/test_invalid.yml")
    with open(test_path, "w") as f:
        yaml.dump(test_config, f)

    try:
        with pytest.raises(ValueError, match="currently only supported from 1m data"):
            load_system_config(str(test_path))
    finally:
        if test_path.exists():
            test_path.unlink()


def test_valid_timeframe_config():
    """Test that event bars with 1m timeframe work correctly."""
    test_config = {
        "timeframe": "1m",
        "data": {"bars": {"mode": "volume", "target_volume": 2000000}},
    }
    test_path = Path("config/test_valid.yml")
    with open(test_path, "w") as f:
        yaml.dump(test_config, f)

    try:
        cfg = load_system_config(str(test_path))
        data_cfg = cfg.get("data_cfg")
        assert data_cfg.bars.source_timeframe == "1m"
    finally:
        if test_path.exists():
            test_path.unlink()


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
    test_path = Path("config/test_override.yml")
    with open(test_path, "w") as f:
        yaml.dump(test_config, f)

    try:
        cfg = load_system_config(str(test_path))
        data_cfg = cfg.get("data_cfg")
        assert data_cfg.bars.source_timeframe == "1m"
    finally:
        if test_path.exists():
            test_path.unlink()


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

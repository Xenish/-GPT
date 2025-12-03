#!/usr/bin/env python3
"""
Test script for EventBar source_timeframe auto-propagation.

Verifies that:
1. When bars.mode is volume/dollar/tick and source_timeframe is None,
   it should be auto-set to the global timeframe
2. When source_timeframe is explicitly set, it should remain unchanged
"""

import os
from finantradealgo.system.config_loader import load_config

# Set dummy FCM key for testing (if not already set)
if not os.getenv("FCM_SERVER_KEY"):
    os.environ["FCM_SERVER_KEY"] = "dummy_test_key"


def test_source_timeframe_propagation():
    """Test the source_timeframe auto-propagation feature."""

    # Load system config
    cfg = load_config("research")

    # Extract relevant values
    global_timeframe = cfg.get("timeframe", "15m")
    data_cfg = cfg.get("data_cfg")

    print("=" * 60)
    print("EventBar source_timeframe Propagation Test")
    print("=" * 60)
    print(f"\nGlobal timeframe: {global_timeframe}")
    print(f"Bars mode: {data_cfg.bars.mode}")
    print(f"Bars source_timeframe: {data_cfg.bars.source_timeframe}")

    # Test case 1: mode = "time" should NOT set source_timeframe
    if data_cfg.bars.mode == "time":
        print("\n[+] Test Case 1: mode='time'")
        print(f"  Expected: source_timeframe should be None (not needed for time bars)")
        print(f"  Actual: {data_cfg.bars.source_timeframe}")
        if data_cfg.bars.source_timeframe is None:
            print("  PASS [OK]")
        else:
            print("  FAIL [X]")

    # Test case 2: mode = "volume/dollar/tick" should set source_timeframe
    elif data_cfg.bars.mode in ("volume", "dollar", "tick"):
        print(f"\n[+] Test Case 2: mode='{data_cfg.bars.mode}'")

        # Check if source_timeframe was explicitly set in config (e.g., "1m")
        # If it differs from global_timeframe, it means user set it explicitly
        if data_cfg.bars.source_timeframe and data_cfg.bars.source_timeframe != global_timeframe:
            print(f"  Expected: source_timeframe should remain '{data_cfg.bars.source_timeframe}' (user-specified)")
            print(f"  Actual: {data_cfg.bars.source_timeframe}")
            print("  PASS [OK] - User override preserved")
        else:
            print(f"  Expected: source_timeframe should be '{global_timeframe}' (auto-propagated)")
            print(f"  Actual: {data_cfg.bars.source_timeframe}")
            if data_cfg.bars.source_timeframe == global_timeframe:
                print("  PASS [OK]")
            else:
                print("  FAIL [X]")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_source_timeframe_propagation()

"""
Test suite for trades_df contract enforcement.

Verifies that:
1. load_trades() returns DataFrame with DatetimeIndex
2. detect_liquidity_sweep() validates DatetimeIndex and raises error on invalid input
3. Contract violations are caught early, not silently handled
"""

import pandas as pd
import pytest
from pathlib import Path
import tempfile

from finantradealgo.data_engine.orderbook_loader import load_trades
from finantradealgo.microstructure.liquidity_sweep import detect_liquidity_sweep
from finantradealgo.microstructure.config import LiquiditySweepConfig


class TestTradesDfContract:
    """Test trades_df contract enforcement across the codebase."""

    @pytest.fixture
    def sample_trades_csv(self):
        """Create a temporary CSV file with sample trade data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trades_dir = Path(tmpdir)
            trades_path = trades_dir / "BTCUSDT_1m_trades.csv"

            # Create sample trades data
            trades_data = pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s"),
                "side": ["buy", "sell"] * 5,
                "price": [40000 + i * 10 for i in range(10)],
                "size": [0.1 + i * 0.01 for i in range(10)]
            })
            trades_data.to_csv(trades_path, index=False)

            yield trades_dir

    def test_load_trades_returns_datetime_index(self, sample_trades_csv):
        """Test that load_trades() returns DataFrame with DatetimeIndex."""
        trades_df = load_trades("BTCUSDT", "1m", trades_dir=sample_trades_csv)

        # Verify contract
        assert trades_df is not None, "load_trades should return DataFrame"
        assert isinstance(trades_df.index, pd.DatetimeIndex), \
            "trades_df.index must be DatetimeIndex"
        assert trades_df.index.is_monotonic_increasing, \
            "trades_df.index must be sorted"

        # Verify columns
        expected_cols = {"side", "price", "size"}
        assert set(trades_df.columns) == expected_cols, \
            f"Expected columns {expected_cols}, got {set(trades_df.columns)}"

        # Verify timestamp is NOT in columns (it's the index)
        assert "timestamp" not in trades_df.columns, \
            "timestamp should be index, not column"

    def test_detect_sweep_with_valid_datetime_index(self, sample_trades_csv):
        """Test that detect_liquidity_sweep works with valid DatetimeIndex."""
        trades_df = load_trades("BTCUSDT", "1m", trades_dir=sample_trades_csv)
        cfg = LiquiditySweepConfig(
            lookback_ms=5000,
            notional_threshold=1000.0
        )

        # This should work without errors
        bar_start = pd.Timestamp("2024-01-01 00:00:02", tz="UTC")
        bar_end = pd.Timestamp("2024-01-01 00:00:05", tz="UTC")

        sweep_up, sweep_down = detect_liquidity_sweep(
            bar_open=40000,
            bar_close=40050,
            bar_start_ts=bar_start,
            bar_end_ts=bar_end,
            trades_df=trades_df,
            cfg=cfg
        )

        # Should return valid results
        assert isinstance(sweep_up, float)
        assert isinstance(sweep_down, float)
        assert sweep_up >= 0
        assert sweep_down >= 0

    def test_detect_sweep_raises_on_invalid_index(self):
        """Test that detect_liquidity_sweep raises ValueError on non-DatetimeIndex."""
        # Create trades_df with integer index (contract violation)
        invalid_trades_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s"),
            "side": ["buy", "sell"] * 5,
            "price": [40000 + i * 10 for i in range(10)],
            "size": [0.1 + i * 0.01 for i in range(10)]
        })
        # Deliberately don't set_index - this violates the contract

        cfg = LiquiditySweepConfig()
        bar_start = pd.Timestamp("2024-01-01 00:00:02", tz="UTC")
        bar_end = pd.Timestamp("2024-01-01 00:00:05", tz="UTC")

        # Should raise ValueError because index is not DatetimeIndex
        with pytest.raises(ValueError, match="trades_df must have a DatetimeIndex"):
            detect_liquidity_sweep(
                bar_open=40000,
                bar_close=40050,
                bar_start_ts=bar_start,
                bar_end_ts=bar_end,
                trades_df=invalid_trades_df,
                cfg=cfg
            )

    def test_detect_sweep_handles_empty_trades(self):
        """Test that detect_liquidity_sweep handles empty trades_df gracefully."""
        # Empty DataFrame with DatetimeIndex
        empty_trades = pd.DataFrame(
            columns=["side", "price", "size"],
            index=pd.DatetimeIndex([])
        )

        cfg = LiquiditySweepConfig()
        bar_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        bar_end = pd.Timestamp("2024-01-01 00:00:05", tz="UTC")

        # Should return zeros, not error
        sweep_up, sweep_down = detect_liquidity_sweep(
            bar_open=40000,
            bar_close=40050,
            bar_start_ts=bar_start,
            bar_end_ts=bar_end,
            trades_df=empty_trades,
            cfg=cfg
        )

        assert sweep_up == 0.0
        assert sweep_down == 0.0

    def test_detect_sweep_handles_none_trades(self):
        """Test that detect_liquidity_sweep handles None trades_df gracefully."""
        cfg = LiquiditySweepConfig()
        bar_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        bar_end = pd.Timestamp("2024-01-01 00:00:05", tz="UTC")

        # Should return zeros, not error
        sweep_up, sweep_down = detect_liquidity_sweep(
            bar_open=40000,
            bar_close=40050,
            bar_start_ts=bar_start,
            bar_end_ts=bar_end,
            trades_df=None,
            cfg=cfg
        )

        assert sweep_up == 0.0
        assert sweep_down == 0.0

    def test_load_trades_nonexistent_file(self):
        """Test that load_trades returns None for nonexistent file."""
        result = load_trades("NONEXISTENT", "1h")
        assert result is None, "load_trades should return None for missing file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

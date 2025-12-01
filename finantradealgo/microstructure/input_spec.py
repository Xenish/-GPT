"""
Input specification and validation for microstructure features.

Task S2.E1: Clarify orderbook vs trades input requirements.
"""
from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class MicrostructureInputSpec:
    """
    Specification for microstructure feature input data.

    This class documents the expected format and columns for:
    - OHLCV DataFrame (required)
    - Trades DataFrame (optional, for sweep features)
    - Order Book DataFrame (optional, for imbalance features)

    Task S2.E1: Orderbook vs Trades input specification.
    """

    # --- Required OHLCV DataFrame ---
    ohlcv_required_columns: List[str] = None
    ohlcv_index_type: str = "DatetimeIndex or RangeIndex"
    ohlcv_description: str = (
        "Standard OHLCV candlestick data. "
        "Required columns: open, high, low, close, volume. "
        "Index should ideally be DatetimeIndex for time-based bars."
    )

    # --- Optional Trades DataFrame ---
    trades_required_columns: List[str] = None
    trades_index_type: str = "DatetimeIndex"
    trades_description: str = (
        "Individual trade executions for computing liquidity sweeps. "
        "Required columns: side, price, size. "
        "Index must be DatetimeIndex with trade timestamps. "
        "Column 'side' should contain 'buy' or 'sell' strings."
    )

    # --- Optional Order Book DataFrame ---
    book_required_columns_pattern: str = "bid_price_{i}, ask_price_{i}, bid_size_{i}, ask_size_{i}"
    book_index_type: str = "DatetimeIndex"
    book_description: str = (
        "Order book snapshots for computing imbalance. "
        "Required columns depend on depth (configurable via ImbalanceConfig.depth). "
        "For depth=5: bid_price_0...4, ask_price_0...4, bid_size_0...4, ask_size_0...4. "
        "Index must be DatetimeIndex with snapshot timestamps. "
        "Level 0 is the best bid/ask, higher levels are further from midpoint."
    )

    def __post_init__(self):
        """Initialize default column lists."""
        if self.ohlcv_required_columns is None:
            self.ohlcv_required_columns = ["open", "high", "low", "close", "volume"]

        if self.trades_required_columns is None:
            self.trades_required_columns = ["side", "price", "size"]

    def validate_ohlcv(self, df: pd.DataFrame) -> None:
        """
        Validate OHLCV DataFrame against specification.

        Args:
            df: OHLCV DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("OHLCV must be a pandas DataFrame")

        if df.empty:
            raise ValueError("OHLCV DataFrame cannot be empty")

        missing_cols = [col for col in self.ohlcv_required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"OHLCV DataFrame missing required columns: {missing_cols}. "
                f"Found: {list(df.columns)}"
            )

    def validate_trades(self, trades_df: pd.DataFrame) -> None:
        """
        Validate trades DataFrame against specification.

        Args:
            trades_df: Trades DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(trades_df, pd.DataFrame):
            raise ValueError("Trades must be a pandas DataFrame")

        if trades_df.empty:
            raise ValueError("Trades DataFrame cannot be empty")

        missing_cols = [col for col in self.trades_required_columns if col not in trades_df.columns]
        if missing_cols:
            raise ValueError(
                f"Trades DataFrame missing required columns: {missing_cols}. "
                f"Found: {list(trades_df.columns)}"
            )

        if not isinstance(trades_df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Trades DataFrame must have DatetimeIndex. "
                f"Found: {type(trades_df.index).__name__}"
            )

    def validate_book(self, book_df: pd.DataFrame, depth: int = 5) -> None:
        """
        Validate order book DataFrame against specification.

        Args:
            book_df: Order book DataFrame to validate
            depth: Number of price levels to validate (default 5)

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(book_df, pd.DataFrame):
            raise ValueError("Order book must be a pandas DataFrame")

        if book_df.empty:
            raise ValueError("Order book DataFrame cannot be empty")

        if not isinstance(book_df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Order book DataFrame must have DatetimeIndex. "
                f"Found: {type(book_df.index).__name__}"
            )

        # Check for required columns at each depth level
        required_prefixes = ["bid_price", "ask_price", "bid_size", "ask_size"]
        missing_cols = []

        for i in range(depth):
            for prefix in required_prefixes:
                col_name = f"{prefix}_{i}"
                if col_name not in book_df.columns:
                    missing_cols.append(col_name)

        if missing_cols:
            raise ValueError(
                f"Order book DataFrame missing required columns for depth={depth}: {missing_cols}. "
                f"Found: {list(book_df.columns)}"
            )

    def get_summary(self) -> str:
        """
        Get a human-readable summary of input specifications.

        Returns:
            String summary of all input requirements
        """
        lines = ["=" * 70]
        lines.append("MICROSTRUCTURE INPUT SPECIFICATION")
        lines.append("=" * 70)

        lines.append("\n--- OHLCV DataFrame (Required) ---")
        lines.append(f"Description: {self.ohlcv_description}")
        lines.append(f"Required columns: {self.ohlcv_required_columns}")
        lines.append(f"Index type: {self.ohlcv_index_type}")

        lines.append("\n--- Trades DataFrame (Optional) ---")
        lines.append(f"Description: {self.trades_description}")
        lines.append(f"Required columns: {self.trades_required_columns}")
        lines.append(f"Index type: {self.trades_index_type}")
        lines.append("Used for: Liquidity sweep detection (ms_sweep_up, ms_sweep_down)")

        lines.append("\n--- Order Book DataFrame (Optional) ---")
        lines.append(f"Description: {self.book_description}")
        lines.append(f"Column pattern: {self.book_required_columns_pattern}")
        lines.append(f"Index type: {self.book_index_type}")
        lines.append("Used for: Order book imbalance (ms_imbalance)")

        lines.append("\n--- Feature Dependencies ---")
        lines.append("OHLCV-only features:")
        lines.append("  - ms_vol_regime, ms_chop, ms_burst_up, ms_burst_down")
        lines.append("  - ms_exhaustion_up, ms_exhaustion_down, ms_parabolic_trend")
        lines.append("\nTrades-dependent features:")
        lines.append("  - ms_sweep_up, ms_sweep_down")
        lines.append("\nBook-dependent features:")
        lines.append("  - ms_imbalance")

        lines.append("=" * 70)

        return "\n".join(lines)


# Singleton instance for easy access
DEFAULT_INPUT_SPEC = MicrostructureInputSpec()

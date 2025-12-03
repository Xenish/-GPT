"""
Script to fetch OHLCV data from exchange using lookback_days configuration.

This script demonstrates how to:
1. Load timeframes and lookback_days from system.yml
2. Fetch data for each symbol/timeframe pair
3. Use per-timeframe lookback periods to optimize API calls

Usage:
    python scripts/fetch_ohlcv.py
    python scripts/fetch_ohlcv.py --symbols BTCUSDT ETHUSDT
    python scripts/fetch_ohlcv.py --timeframes 15m 1h
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finantradealgo.system.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_ohlcv_from_exchange(
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from exchange API.

    This is a placeholder function. Replace with actual exchange API calls:
    - Binance: client.get_historical_klines()
    - Bybit: client.query_kline()
    - CCXT: exchange.fetch_ohlcv()

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "15m", "1h")
        start_time: Start datetime (inclusive)
        end_time: End datetime (inclusive)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    logger.warning(
        f"fetch_ohlcv_from_exchange is a placeholder. "
        f"Implement actual exchange API call for {symbol} {timeframe}"
    )

    # Placeholder: Return empty DataFrame with correct schema
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_and_save_ohlcv(
    symbol: str,
    timeframe: str,
    lookback_days: int,
    output_template: str,
) -> None:
    """
    Fetch OHLCV data and save to CSV.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        lookback_days: Number of days to fetch
        output_template: Output path template (e.g., "data/ohlcv/{symbol}_{timeframe}.csv")
    """
    logger.info(f"Fetching {symbol} {timeframe} (lookback: {lookback_days} days)")

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)

    # Fetch data from exchange
    df = fetch_ohlcv_from_exchange(symbol, timeframe, start_time, end_time)

    if df.empty:
        logger.warning(f"No data fetched for {symbol} {timeframe}")
        return

    # Ensure timestamp is UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save to CSV
    output_path = output_template.format(symbol=symbol, timeframe=timeframe)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} bars to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data using system config")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to fetch (default: from system.yml)"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Timeframes to fetch (default: from system.yml)"
    )
    parser.add_argument(
        "--profile",
        choices=["research", "live"],
        default="research",
        help="Config profile to load (default: research)",
    )
    args = parser.parse_args()

    # Set dummy FCM key for config loading
    if not os.getenv("FCM_SERVER_KEY"):
        os.environ["FCM_SERVER_KEY"] = "dummy_fetch_key"

    # Load system config
    logger.info(f"Loading config profile '{args.profile}'")
    cfg = load_config(args.profile)
    data_cfg = cfg["data_cfg"]

    # Get symbols and timeframes
    symbols = args.symbols if args.symbols else data_cfg.symbols
    timeframes = args.timeframes if args.timeframes else data_cfg.timeframes

    if not symbols:
        logger.error("No symbols specified. Add symbols to system.yml or use --symbols")
        return

    if not timeframes:
        logger.error("No timeframes specified. Add timeframes to system.yml or use --timeframes")
        return

    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Lookback config: {data_cfg.lookback_days}")

    # Fetch data for each symbol/timeframe pair
    total = len(symbols) * len(timeframes)
    count = 0

    for symbol in symbols:
        for timeframe in timeframes:
            count += 1
            logger.info(f"[{count}/{total}] Processing {symbol} {timeframe}")

            # Get lookback days for this timeframe
            lookback = data_cfg.lookback_days.get(
                timeframe,
                data_cfg.default_lookback_days
            )

            try:
                fetch_and_save_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=lookback,
                    output_template=data_cfg.ohlcv_path_template,
                )
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
                continue

    logger.info(f"Completed fetching {count} symbol/timeframe pairs")


if __name__ == "__main__":
    main()

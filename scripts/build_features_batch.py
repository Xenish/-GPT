"""
Batch feature building script for multi-symbol/multi-timeframe workflows.

This script:
1. Loads all symbols and timeframes from system.yml
2. Builds features for each combination using lookback_days config
3. Saves features to parquet/CSV files

Usage:
    python scripts/build_features_batch.py
    python scripts/build_features_batch.py --symbols BTCUSDT ETHUSDT
    python scripts/build_features_batch.py --timeframes 15m 1h
    python scripts/build_features_batch.py --format parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.features.feature_pipeline import (
    build_feature_pipeline_from_system_config,
)
from finantradealgo.system.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def build_and_save_features(
    cfg: dict,
    symbol: str,
    timeframe: str,
    output_dir: Path,
    output_format: str = "parquet",
) -> dict:
    """
    Build features for a single symbol/timeframe pair and save to disk.

    Args:
        cfg: System configuration dict
        symbol: Trading symbol
        timeframe: Timeframe string
        output_dir: Directory to save features
        output_format: Output format ("parquet" or "csv")

    Returns:
        Dictionary with metadata about the build
    """
    logger.info(f"Building features for {symbol} {timeframe}...")
    start_time = time.time()

    try:
        # Build features
        df_feat, meta = build_feature_pipeline_from_system_config(
            cfg,
            symbol=symbol,
            timeframe=timeframe,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Built {len(df_feat)} bars with {len(df_feat.columns)} columns "
            f"in {elapsed:.2f}s"
        )

        # Save features
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "parquet":
            output_path = output_dir / f"{symbol}_{timeframe}_features.parquet"
            df_feat.to_parquet(output_path, index=False)
        else:
            output_path = output_dir / f"{symbol}_{timeframe}_features.csv"
            df_feat.to_csv(output_path, index=False)

        logger.info(f"Saved features to {output_path}")

        # Save metadata
        meta_path = output_dir / f"{symbol}_{timeframe}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df_feat),
            "features": len(df_feat.columns),
            "elapsed": elapsed,
            "output_path": str(output_path),
            "status": "success",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed to build features for {symbol} {timeframe}: {e}")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "elapsed": elapsed,
            "status": "failed",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Build features for all symbol/timeframe combinations"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to process (default: from system.yml)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Timeframes to process (default: from system.yml)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (default: parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: from system.yml features_dir)",
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
        os.environ["FCM_SERVER_KEY"] = "dummy_batch_key"

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

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(data_cfg.features_dir)

    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output format: {args.format}")
    logger.info(f"Lookback config: {data_cfg.lookback_days}")

    # Process all combinations
    total = len(symbols) * len(timeframes)
    results = []
    count = 0

    overall_start = time.time()

    for symbol in symbols:
        for timeframe in timeframes:
            count += 1
            logger.info(f"[{count}/{total}] Processing {symbol} {timeframe}")

            result = build_and_save_features(
                cfg=cfg,
                symbol=symbol,
                timeframe=timeframe,
                output_dir=output_dir,
                output_format=args.format,
            )
            results.append(result)

    overall_elapsed = time.time() - overall_start

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    logger.info(f"Total combinations: {total}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total time: {overall_elapsed:.2f}s")

    if success_count > 0:
        avg_time = sum(r["elapsed"] for r in results if r["status"] == "success") / success_count
        logger.info(f"Average time per combination: {avg_time:.2f}s")

    # Show successful builds
    if success_count > 0:
        logger.info("")
        logger.info("Successful builds:")
        for r in results:
            if r["status"] == "success":
                logger.info(
                    f"  {r['symbol']:10s} {r['timeframe']:5s} -> "
                    f"{r['bars']:6d} bars, {r['features']:3d} features "
                    f"({r['elapsed']:.2f}s)"
                )

    # Show failed builds
    if failed_count > 0:
        logger.info("")
        logger.error("Failed builds:")
        for r in results:
            if r["status"] == "failed":
                logger.error(f"  {r['symbol']:10s} {r['timeframe']:5s} -> {r.get('error', 'Unknown error')}")

    # Save summary
    summary_path = output_dir / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "success": success_count,
            "failed": failed_count,
            "elapsed": overall_elapsed,
            "results": results,
        }, f, indent=2)

    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

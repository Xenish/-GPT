"""
Smoke tests for all symbols/timeframes.

Task S3.E5: CI integration smoke tests for data validation.

This script runs quick validation checks on all configured symbols/timeframes
to catch data quality issues early in CI/CD pipelines.

Usage:
    python scripts/smoke_run_all_symbols.py
    python scripts/smoke_run_all_symbols.py --fail-on-error
    python scripts/smoke_run_all_symbols.py --max-symbols 3 --max-timeframes 2
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from finantradealgo.system.config_loader import load_config
from finantradealgo.validation import (
    validate_ohlcv,
    OHLCVValidationConfig,
)

logger = logging.getLogger(__name__)


def smoke_test_symbol(
    symbol: str,
    timeframe: str,
    data_dir: Path,
    validation_cfg: OHLCVValidationConfig,
) -> Tuple[bool, Dict]:
    """
    Run smoke test for a single symbol/timeframe.

    Args:
        symbol: Symbol name
        timeframe: Timeframe string
        data_dir: Data directory
        validation_cfg: Validation configuration

    Returns:
        Tuple of (success: bool, results: dict)
    """
    results = {
        "symbol": symbol,
        "timeframe": timeframe,
        "file_exists": False,
        "file_path": None,
        "bars_count": 0,
        "validation_passed": False,
        "errors": [],
        "warnings": [],
    }

    # Check file exists
    csv_path = data_dir / f"{symbol}_{timeframe}.csv"
    results["file_path"] = str(csv_path)

    if not csv_path.exists():
        results["errors"].append(f"File not found: {csv_path}")
        return False, results

    results["file_exists"] = True

    try:
        # Load data (limit to last 1000 bars for speed)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        if len(df) > 1000:
            df = df.tail(1000)

        if df.empty:
            results["errors"].append("DataFrame is empty")
            return False, results

        results["bars_count"] = len(df)

        # Run validation
        validation_result = validate_ohlcv(df, validation_cfg, timeframe=timeframe)

        # Collect issues
        for issue in validation_result.issues:
            if issue.severity == "error":
                results["errors"].append(f"{issue.check_name}: {issue.message}")
            else:
                results["warnings"].append(f"{issue.check_name}: {issue.message}")

        results["validation_passed"] = validation_result.is_valid

        return validation_result.is_valid, results

    except Exception as e:
        logger.exception(f"Exception during smoke test for {symbol}_{timeframe}")
        results["errors"].append(f"Exception: {str(e)}")
        return False, results


def run_smoke_tests(
    symbols: List[str],
    timeframes: List[str],
    data_dir: Path,
    fail_on_error: bool = False,
    max_symbols: int = None,
    max_timeframes: int = None,
) -> Tuple[bool, Dict]:
    """
    Run smoke tests for all symbol/timeframe combinations.

    Args:
        symbols: List of symbols
        timeframes: List of timeframes
        data_dir: Data directory
        fail_on_error: If True, exit with error code on first failure
        max_symbols: Limit number of symbols (for faster CI runs)
        max_timeframes: Limit number of timeframes (for faster CI runs)

    Returns:
        Tuple of (all_passed: bool, summary: dict)
    """
    # Limit symbols/timeframes if specified
    if max_symbols:
        symbols = symbols[:max_symbols]
    if max_timeframes:
        timeframes = timeframes[:max_timeframes]

    logger.info(f"Running smoke tests for {len(symbols)} symbols x {len(timeframes)} timeframes")

    # Create validation config (lenient for smoke tests)
    validation_cfg = OHLCVValidationConfig(
        check_price_spikes=False,  # Skip expensive spike detection
        check_missing_bars=True,
        max_gap_multiplier=3.0,  # More lenient for smoke tests
    )

    all_results = []
    passed_count = 0
    failed_count = 0

    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Testing {symbol}_{timeframe}...")

            success, results = smoke_test_symbol(
                symbol, timeframe, data_dir, validation_cfg
            )

            all_results.append(results)

            if success:
                passed_count += 1
                logger.info(f"  ✓ PASS ({results['bars_count']} bars)")
            else:
                failed_count += 1
                logger.error(f"  ✗ FAIL")
                for error in results["errors"]:
                    logger.error(f"    - {error}")

                if fail_on_error:
                    logger.error("Stopping on first failure (--fail-on-error)")
                    break

        if fail_on_error and failed_count > 0:
            break

    # Summary
    total = len(all_results)
    summary = {
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": (passed_count / total * 100) if total > 0 else 0,
        "results": all_results,
    }

    return failed_count == 0, summary


def print_summary(summary: Dict):
    """Print test summary."""
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {summary['total']}")
    print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)")
    print(f"Failed: {summary['failed']}")
    print("")

    if summary['failed'] > 0:
        print("FAILURES:")
        print("-" * 80)
        for result in summary['results']:
            if not result['validation_passed']:
                print(f"\n{result['symbol']}_{result['timeframe']}:")
                for error in result['errors']:
                    print(f"  ✗ {error}")
        print("")

    print("=" * 80)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run smoke tests for all configured symbols/timeframes"
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with error code if any test fails"
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        help="Limit number of symbols to test (for faster CI)"
    )
    parser.add_argument(
        "--max-timeframes",
        type=int,
        help="Limit number of timeframes to test (for faster CI)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    try:
        config = load_config("research")
        data_cfg = config["data_cfg"]
        symbols = data_cfg.symbols
        timeframes = data_cfg.timeframes
        data_dir = Path(data_cfg.ohlcv_dir)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Run smoke tests
    all_passed, summary = run_smoke_tests(
        symbols=symbols,
        timeframes=timeframes,
        data_dir=data_dir,
        fail_on_error=args.fail_on_error,
        max_symbols=args.max_symbols,
        max_timeframes=args.max_timeframes,
    )

    # Print summary
    print_summary(summary)

    # Exit with appropriate code
    if not all_passed:
        logger.error("Some smoke tests failed!")
        sys.exit(1)
    else:
        logger.info("All smoke tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

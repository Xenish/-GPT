"""
Multi-symbol data coverage report generator.

Task S3.E2: Generate comprehensive coverage reports for multiple symbols/timeframes.

Usage:
    python scripts/report_data_coverage.py
    python scripts/report_data_coverage.py --symbols BTCUSDT ETHUSDT --timeframes 15m 1h
    python scripts/report_data_coverage.py --output outputs/coverage_report.txt
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from finantradealgo.system.config_loader import load_system_config
from finantradealgo.validation import (
    validate_ohlcv,
    detect_gaps,
    infer_timeframe,
)

logger = logging.getLogger(__name__)


def analyze_symbol_coverage(
    symbol: str,
    timeframe: str,
    data_dir: Path,
) -> Dict:
    """
    Analyze data coverage for a single symbol/timeframe combination.

    Args:
        symbol: Symbol name (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "15m", "1h")
        data_dir: Data directory path

    Returns:
        Dictionary with coverage statistics
    """
    stats = {
        "symbol": symbol,
        "timeframe": timeframe,
        "file_exists": False,
        "total_bars": 0,
        "date_range_start": None,
        "date_range_end": None,
        "days_covered": 0,
        "gaps_count": 0,
        "validation_errors": 0,
        "validation_warnings": 0,
        "missing_data_pct": 0.0,
    }

    # Construct file path
    csv_path = data_dir / f"{symbol}_{timeframe}.csv"

    if not csv_path.exists():
        return stats

    stats["file_exists"] = True

    try:
        # Load data
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        if df.empty:
            return stats

        stats["total_bars"] = len(df)
        stats["date_range_start"] = df.index[0]
        stats["date_range_end"] = df.index[-1]

        # Calculate days covered
        date_diff = df.index[-1] - df.index[0]
        stats["days_covered"] = date_diff.days

        # Detect gaps
        try:
            gaps = detect_gaps(df.index, timeframe, max_gap_multiplier=2.0)
            stats["gaps_count"] = len(gaps)
        except ValueError:
            # Unknown timeframe
            pass

        # Run validation
        validation_result = validate_ohlcv(df, timeframe=timeframe)
        stats["validation_errors"] = validation_result.errors_count
        stats["validation_warnings"] = validation_result.warnings_count

        # Calculate missing data percentage (NaN values)
        if not df.empty:
            total_values = df.size
            missing_values = df.isna().sum().sum()
            stats["missing_data_pct"] = (missing_values / total_values) * 100 if total_values > 0 else 0.0

    except Exception as e:
        logger.error(f"Error analyzing {symbol}_{timeframe}: {e}")
        stats["error"] = str(e)

    return stats


def generate_coverage_report(
    symbols: List[str],
    timeframes: List[str],
    data_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate multi-symbol coverage report.

    Args:
        symbols: List of symbols to analyze
        timeframes: List of timeframes to analyze
        data_dir: Data directory (defaults to data/ohlcv)
        output_path: Output file path (if None, returns string only)

    Returns:
        Report string
    """
    if data_dir is None:
        data_dir = Path("data/ohlcv")

    # Collect statistics for all combinations
    all_stats = []
    for symbol in symbols:
        for timeframe in timeframes:
            stats = analyze_symbol_coverage(symbol, timeframe, data_dir)
            all_stats.append(stats)

    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append("DATA COVERAGE REPORT")
    lines.append("=" * 80)
    lines.append(f"Data directory: {data_dir}")
    lines.append(f"Symbols analyzed: {len(symbols)}")
    lines.append(f"Timeframes analyzed: {len(timeframes)}")
    lines.append(f"Total combinations: {len(all_stats)}")
    lines.append("")

    # Summary statistics
    files_exist = sum(1 for s in all_stats if s["file_exists"])
    total_errors = sum(s["validation_errors"] for s in all_stats)
    total_warnings = sum(s["validation_warnings"] for s in all_stats)
    total_gaps = sum(s["gaps_count"] for s in all_stats)

    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Files found: {files_exist}/{len(all_stats)}")
    lines.append(f"Total validation errors: {total_errors}")
    lines.append(f"Total validation warnings: {total_warnings}")
    lines.append(f"Total gaps detected: {total_gaps}")
    lines.append("")

    # Detailed breakdown by symbol/timeframe
    lines.append("DETAILED BREAKDOWN")
    lines.append("-" * 80)

    for stats in all_stats:
        symbol = stats["symbol"]
        tf = stats["timeframe"]
        lines.append(f"\n{symbol} / {tf}:")

        if not stats["file_exists"]:
            lines.append("  ✗ File not found")
            continue

        if "error" in stats:
            lines.append(f"  ✗ Error: {stats['error']}")
            continue

        # Status indicator
        status = "✓"
        if stats["validation_errors"] > 0:
            status = "✗"
        elif stats["validation_warnings"] > 0:
            status = "⚠"

        lines.append(f"  {status} Total bars: {stats['total_bars']:,}")

        if stats["date_range_start"] and stats["date_range_end"]:
            lines.append(f"  └─ Date range: {stats['date_range_start'].date()} to {stats['date_range_end'].date()}")
            lines.append(f"  └─ Days covered: {stats['days_covered']}")

        if stats["gaps_count"] > 0:
            lines.append(f"  └─ Gaps detected: {stats['gaps_count']}")

        if stats["missing_data_pct"] > 0:
            lines.append(f"  └─ Missing data: {stats['missing_data_pct']:.2f}%")

        if stats["validation_errors"] > 0:
            lines.append(f"  └─ Validation errors: {stats['validation_errors']}")

        if stats["validation_warnings"] > 0:
            lines.append(f"  └─ Validation warnings: {stats['validation_warnings']}")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Write to file if specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info(f"Report written to: {output_path}")

    return report


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate data coverage report for multiple symbols/timeframes"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="List of symbols to analyze (default: from system.yml)"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="List of timeframes to analyze (default: from system.yml)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory path (default: data/ohlcv)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: print to console)"
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    config = load_system_config()

    # Get symbols and timeframes
    symbols = args.symbols if args.symbols else config.data.symbols
    timeframes = args.timeframes if args.timeframes else config.data.timeframes
    data_dir = args.data_dir if args.data_dir else Path(config.data.ohlcv_dir)

    logger.info(f"Analyzing {len(symbols)} symbols x {len(timeframes)} timeframes")

    # Generate report
    report = generate_coverage_report(
        symbols=symbols,
        timeframes=timeframes,
        data_dir=data_dir,
        output_path=args.output
    )

    # Print to console if no output file specified
    if not args.output:
        print(report)


if __name__ == "__main__":
    main()

"""
Feature health summary and metrics script.

Task CRITICAL-3: Generate comprehensive health reports for feature data.

This script analyzes feature CSV files and reports:
- Missing/NaN value statistics
- Infinite value detection
- Value range and distribution metrics
- Constant/suspicious column detection
- Feature category breakdowns (MS, Micro, HTF, etc.)

Usage:
    python scripts/print_feature_health_summary.py
    python scripts/print_feature_health_summary.py --symbol BTCUSDT --timeframe 15m
    python scripts/print_feature_health_summary.py --file data/features/BTCUSDT_features_15m.csv
    python scripts/print_feature_health_summary.py --output outputs/health_report.txt
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_column_health(series: pd.Series, col_name: str) -> Dict:
    """
    Analyze health metrics for a single column.

    Args:
        series: Column data as pd.Series
        col_name: Column name

    Returns:
        Dictionary with health metrics
    """
    metrics = {
        "name": col_name,
        "dtype": str(series.dtype),
        "count": len(series),
        "missing_count": 0,
        "missing_pct": 0.0,
        "inf_count": 0,
        "zero_count": 0,
        "zero_pct": 0.0,
        "unique_count": 0,
        "is_constant": False,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "issues": [],
    }

    # Missing values
    missing_count = series.isna().sum()
    metrics["missing_count"] = int(missing_count)
    metrics["missing_pct"] = (missing_count / len(series)) * 100 if len(series) > 0 else 0

    if metrics["missing_pct"] > 50:
        metrics["issues"].append(f"HIGH_MISSING: {metrics['missing_pct']:.1f}%")
    elif metrics["missing_pct"] > 10:
        metrics["issues"].append(f"MISSING: {metrics['missing_pct']:.1f}%")

    # For numeric columns
    if pd.api.types.is_numeric_dtype(series):
        # Infinite values
        if series.dtype in ['float64', 'float32']:
            inf_mask = np.isinf(series)
            metrics["inf_count"] = int(inf_mask.sum())
            if metrics["inf_count"] > 0:
                metrics["issues"].append(f"INF_VALUES: {metrics['inf_count']}")

        # Non-NaN data for further analysis
        non_na = series.dropna()
        if not non_na.empty:
            # Basic statistics
            metrics["min"] = float(non_na.min())
            metrics["max"] = float(non_na.max())
            metrics["mean"] = float(non_na.mean())
            metrics["std"] = float(non_na.std())

            # Zero values
            zero_count = (non_na == 0).sum()
            metrics["zero_count"] = int(zero_count)
            metrics["zero_pct"] = (zero_count / len(non_na)) * 100 if len(non_na) > 0 else 0

            if metrics["zero_pct"] > 90:
                metrics["issues"].append(f"MOSTLY_ZEROS: {metrics['zero_pct']:.1f}%")

            # Unique values
            metrics["unique_count"] = int(non_na.nunique())

            # Constant column check
            if metrics["unique_count"] == 1:
                metrics["is_constant"] = True
                metrics["issues"].append(f"CONSTANT: value={metrics['mean']:.4f}")
            elif metrics["std"] == 0:
                metrics["is_constant"] = True
                metrics["issues"].append("CONSTANT: std=0")

    return metrics


def categorize_features(columns: List[str]) -> Dict[str, List[str]]:
    """
    Categorize feature columns by type.

    Args:
        columns: List of column names

    Returns:
        Dictionary mapping category to column names
    """
    categories = {
        "ohlcv": [],
        "market_structure": [],
        "microstructure": [],
        "htf": [],
        "external": [],
        "rule_signals": [],
        "other": [],
    }

    ohlcv_base = ["open", "high", "low", "close", "volume"]

    for col in columns:
        col_lower = col.lower()

        if col in ohlcv_base:
            categories["ohlcv"].append(col)
        elif col_lower.startswith("micro_") or col_lower.startswith("ms_micro_"):
            # Check microstructure first (before ms_ check)
            categories["microstructure"].append(col)
        elif col_lower.startswith("ms_"):
            categories["market_structure"].append(col)
        elif col_lower.startswith("htf"):
            categories["htf"].append(col)
        elif "flow" in col_lower or "sentiment" in col_lower:
            categories["external"].append(col)
        elif col_lower.startswith("rule_"):
            categories["rule_signals"].append(col)
        else:
            categories["other"].append(col)

    return categories


def generate_feature_health_summary(
    df: pd.DataFrame,
    file_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate comprehensive feature health summary.

    Args:
        df: Feature DataFrame to analyze
        file_path: Source file path (for reporting)
        output_path: Output file path (if None, returns string only)

    Returns:
        Report string
    """
    # Analyze all columns
    all_metrics = []
    for col in df.columns:
        metrics = analyze_column_health(df[col], col)
        all_metrics.append(metrics)

    # Categorize features
    categories = categorize_features(df.columns.tolist())

    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("FEATURE HEALTH SUMMARY")
    lines.append("=" * 80)

    if file_path:
        lines.append(f"File: {file_path}")

    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Total columns: {len(df.columns):,}")

    # Only show date range if DataFrame is not empty
    if not df.empty and len(df.index) > 0:
        lines.append(f"Date range: {df.index[0]} to {df.index[-1]}")
    else:
        lines.append("Date range: N/A (empty DataFrame)")

    lines.append("")

    # Overall statistics
    total_issues = sum(1 for m in all_metrics if m["issues"])
    total_missing = sum(m["missing_count"] for m in all_metrics)
    total_inf = sum(m["inf_count"] for m in all_metrics)
    constant_cols = [m["name"] for m in all_metrics if m["is_constant"]]

    lines.append("OVERALL STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Columns with issues: {total_issues}/{len(all_metrics)}")
    lines.append(f"Total missing values: {total_missing:,}")
    lines.append(f"Total infinite values: {total_inf:,}")
    lines.append(f"Constant columns: {len(constant_cols)}")
    lines.append("")

    # Feature category breakdown
    lines.append("FEATURE CATEGORIES")
    lines.append("-" * 80)
    for category, cols in categories.items():
        if cols:
            lines.append(f"{category.upper()}: {len(cols)} columns")

    lines.append("")

    # Issues by category
    lines.append("ISSUES BY CATEGORY")
    lines.append("-" * 80)

    for category, cols in categories.items():
        if not cols:
            continue

        cat_metrics = [m for m in all_metrics if m["name"] in cols]
        cat_issues = [m for m in cat_metrics if m["issues"]]

        if cat_issues:
            lines.append(f"\n{category.upper()} ({len(cat_issues)}/{len(cols)} with issues):")

            for m in cat_issues:
                issues_str = ", ".join(m["issues"])
                lines.append(f"  X {m['name']}: {issues_str}")
        else:
            lines.append(f"\n{category.upper()}: OK - No issues")

    # Detailed column statistics
    lines.append("")
    lines.append("")
    lines.append("DETAILED COLUMN STATISTICS")
    lines.append("-" * 80)
    lines.append(f"{'Column':<30} {'Missing%':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    lines.append("-" * 80)

    for m in all_metrics:
        if m["min"] is not None:  # Numeric columns only
            lines.append(
                f"{m['name']:<30} "
                f"{m['missing_pct']:>9.2f}% "
                f"{m['min']:>11.4f} "
                f"{m['max']:>11.4f} "
                f"{m['mean']:>11.4f} "
                f"{m['std']:>11.4f}"
            )

    # Constant columns detail
    if constant_cols:
        lines.append("")
        lines.append("CONSTANT COLUMNS (WARNING)")
        lines.append("-" * 80)
        for col in constant_cols:
            m = next(m for m in all_metrics if m["name"] == col)
            lines.append(f"  {col}: value={m['mean']:.4f}")

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
        description="Generate feature health summary for debugging and monitoring"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--file",
        type=Path,
        help="Direct path to feature CSV file"
    )
    input_group.add_argument(
        "--symbol",
        help="Symbol to analyze (requires --timeframe)"
    )

    parser.add_argument(
        "--timeframe",
        help="Timeframe to analyze (used with --symbol)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/features"),
        help="Feature data directory (default: data/features)"
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

    # Determine input file
    if args.file:
        csv_path = args.file
    elif args.symbol and args.timeframe:
        csv_path = args.data_dir / f"{args.symbol}_features_{args.timeframe}.csv"
    else:
        # Default: find first available feature file
        feature_files = list(args.data_dir.glob("*_features_*.csv"))
        if not feature_files:
            logger.error(f"No feature files found in {args.data_dir}")
            return
        csv_path = feature_files[0]
        logger.info(f"No input specified, using: {csv_path}")

    # Check file exists
    if not csv_path.exists():
        logger.error(f"Feature file not found: {csv_path}")
        return

    # Load feature data
    logger.info(f"Loading feature data from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if df.empty:
        logger.error("Feature DataFrame is empty")
        return

    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Generate report
    report = generate_feature_health_summary(
        df=df,
        file_path=csv_path,
        output_path=args.output
    )

    # Print to console if no output file specified
    if not args.output:
        print(report)


if __name__ == "__main__":
    main()

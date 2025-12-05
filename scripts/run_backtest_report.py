"""
Generate backtest reports (HTML/Markdown/JSON) from a job directory.

Usage:
    python scripts/run_backtest_report.py --job-dir path/to/job
    python scripts/run_backtest_report.py --job-id my_job_id
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from finantradealgo.research.reporting import BacktestReportGenerator, ReportFormat


def _parse_formats(raw: str) -> List[str]:
    allowed = {"html", "markdown", "md", "json"}
    formats = [f.strip().lower() for f in raw.split(",") if f.strip()]
    for fmt in formats:
        if fmt not in allowed:
            raise ValueError(f"Unsupported format: {fmt}")
    return formats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backtest reports.")
    parser.add_argument("--job-dir", type=str, help="Path to backtest job directory")
    parser.add_argument("--job-id", type=str, help="Backtest job id (under outputs/backtests)")
    parser.add_argument(
        "--formats",
        type=str,
        default="html,markdown",
        help="Comma-separated formats: html, markdown, json",
    )
    args = parser.parse_args()

    if not args.job_dir and not args.job_id:
        raise SystemExit("Please provide --job-dir or --job-id")

    base_dir = Path(os.environ.get("BACKTEST_REPORT_BASE_DIR", Path("outputs") / "backtests"))
    job_dir = Path(args.job_dir) if args.job_dir else base_dir / args.job_id

    if not job_dir.exists():
        raise SystemExit(f"Job directory not found: {job_dir}")

    metrics_path = job_dir / "metrics.json"
    equity_path = job_dir / "equity_curve.csv"
    trades_path = job_dir / "trades.csv"

    for required in [metrics_path, equity_path, trades_path]:
        if not required.exists():
            raise SystemExit(f"Missing required file: {required}")

    formats = _parse_formats(args.formats)

    generator = BacktestReportGenerator()
    report = generator.generate(
        metrics=metrics_path,
        equity_curve_path=equity_path,
        trades_path=trades_path,
        job_id=args.job_id or job_dir.name,
    )

    # Default save location inside job dir
    saved_paths = []
    for fmt in formats:
        if fmt in ("markdown", "md"):
            path = job_dir / "report.md"
            report.save(path, ReportFormat.MARKDOWN)
        elif fmt == "json":
            path = job_dir / "report.json"
            report.save(path, ReportFormat.JSON)
        else:
            path = job_dir / "report.html"
            report.save(path, ReportFormat.HTML)
        saved_paths.append(path)
        print(path)

    # Helpful summary
    print(f"[OK] Backtest report generated: {', '.join(str(p) for p in saved_paths)}")


if __name__ == "__main__":
    main()

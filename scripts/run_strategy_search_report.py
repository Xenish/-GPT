"""
Generate a strategy search report from job outputs.

Usage:
    python -m scripts.run_strategy_search_report --job-id <job_id> [--format html] [--top-n 10]
    python -m scripts.run_strategy_search_report --job-dir outputs/strategy_search/<job_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from finantradealgo.research.reporting.strategy_search import StrategySearchReportGenerator
from finantradealgo.research.reporting.base import ReportFormat
from finantradealgo.research.strategy_search.search_engine import BASE_OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate strategy search report from job outputs.")
    parser.add_argument("--job-id", type=str, help="Job ID (under outputs/strategy_search).")
    parser.add_argument("--job-dir", type=str, help="Explicit job directory path.")
    parser.add_argument("--format", type=str, choices=["html", "markdown"], default="html", help="Report format.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top performers to include.")
    parser.add_argument("--output", type=str, help="Output file path (defaults to job_dir/report.<ext>).")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    job_dir = Path(args.job_dir) if args.job_dir else None
    if job_dir is None:
        if not args.job_id:
            parser.error("Either --job-id or --job-dir must be provided.")
        job_dir = BASE_OUTPUT_DIR / args.job_id

    if not job_dir.exists():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    gen = StrategySearchReportGenerator()
    report = gen.generate(job_dir=job_dir, job_id=args.job_id, top_n=args.top_n)

    fmt = ReportFormat.HTML if args.format == "html" else ReportFormat.MARKDOWN
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = ".html" if fmt == ReportFormat.HTML else ".md"
        output_path = job_dir / f"report{suffix}"

    report.save(output_path, format=fmt)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()

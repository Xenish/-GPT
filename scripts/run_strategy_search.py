"""
Run Strategy Parameter Search.

This script performs parameter search for a strategy and persists results
with full reproducibility (git SHA, job metadata).

Usage:
    python -m scripts.run_strategy_search --profile research --strategy rule --symbol BTCUSDT --timeframe 15m --n-samples 50

Output:
    outputs/strategy_search/{job_id}/
    - results.parquet       # Search results
    - meta.json             # Job metadata + git_sha
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import finantradealgo.research.strategy_search.search_engine as se
from finantradealgo.research.strategy_search.jobs import StrategySearchJobConfig
from finantradealgo.strategies.strategy_engine import get_strategy_meta, get_searchable_strategies
from finantradealgo.system.config_loader import load_config


def build_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run strategy parameter search against research profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="research",
        choices=["research"],
        help="Config profile to load (only 'research' is allowed).",
    )
    parser.add_argument(
        "--strategy",
        "--strategy-name",
        dest="strategy",
        type=str,
        required=True,
        help="Strategy name (e.g., rule, trend_continuation).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., 'BTCUSDT', 'AIAUSDT').",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        required=True,
        help="Timeframe (e.g., '5m', '15m', '1h').",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default="random",
        choices=["random", "grid"],
        help="Search mode (default: random).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of parameter samples to evaluate (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional).",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes about the job.",
    )
    parser.add_argument(
        "--list-searchable",
        action="store_true",
        help="List all searchable strategies and exit",
    )

    return parser


def _persist_results(job, results, sys_cfg, base_output_dir: Path) -> Path:
    """Persist results and metadata similar to run_random_search."""
    job_dir = base_output_dir / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    se._validate_results(results_df)

    results_parquet_path = job_dir / "results.parquet"
    results_csv_path = job_dir / "results.csv"
    results_df.to_parquet(results_parquet_path, index=False)
    results_df.to_csv(results_csv_path, index=False)

    git_sha = se._get_git_sha()
    n_success = (results_df["status"] == "ok").sum() if "status" in results_df.columns else len(results_df)
    n_errors = (results_df["status"] == "error").sum() if "status" in results_df.columns else 0

    data_cfg = sys_cfg.get("data_cfg") if sys_cfg else None
    data_snapshot = {}
    if data_cfg:
        data_snapshot = {
            "ohlcv_dir": getattr(data_cfg, "ohlcv_dir", "unknown"),
            "lookback_days": getattr(data_cfg, "lookback_days", {}),
        }

    meta_dict = job.to_dict()
    meta_dict["git_sha"] = git_sha
    meta_dict["results_path_parquet"] = str(results_parquet_path.relative_to(job_dir))
    meta_dict["results_path_csv"] = str(results_csv_path.relative_to(job_dir))
    meta_dict["n_results"] = len(results_df)
    meta_dict["n_success"] = int(n_success)
    meta_dict["n_errors"] = int(n_errors)
    meta_dict["data_snapshot"] = data_snapshot
    meta_dict["profile"] = getattr(job, "profile", "research")

    meta_path = job_dir / "meta.json"
    meta_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    return job_dir


def main(argv=None):
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # List searchable strategies if requested
    if args.list_searchable:
        searchable = get_searchable_strategies()
        print("Searchable Strategies:")
        print("=" * 60)
        for name, meta in searchable.items():
            print(f"  * {name:<25} (family: {meta.family})")
            if meta.param_space:
                param_names = list(meta.param_space.keys())
                print(f"    Parameters: {', '.join(param_names[:5])}")
                if len(param_names) > 5:
                    print(f"                ... and {len(param_names) - 5} more")
        print("=" * 60)
        return

    # Load system config
    sys_cfg = load_config(args.profile)

    # Validate profile
    cfg_profile = sys_cfg.get("profile", sys_cfg.get("mode", "unknown"))
    if cfg_profile != "research":
        raise RuntimeError(
            "Strategy search must run with the 'research' profile. Use --profile research."
        )

    # Determine output base dir override for tests
    output_override = os.environ.get("STRATEGY_SEARCH_OUTPUT_DIR")
    base_output_dir = Path(output_override) if output_override else se.BASE_OUTPUT_DIR
    if output_override:
        se.BASE_OUTPUT_DIR = base_output_dir

    # Get strategy metadata and param_space
    try:
        meta = get_strategy_meta(args.strategy)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nUse --list-searchable to see available strategies.")
        sys.exit(1)

    # Validate strategy is searchable
    if not meta.is_searchable or meta.param_space is None:
        print(f"Error: Strategy '{args.strategy}' is not searchable (no ParamSpace defined).")
        print("\nUse --list-searchable to see available strategies.")
        sys.exit(1)

    # Build job config and job
    job_name = f"{args.strategy}_{args.symbol}_{args.timeframe}"
    job_config = StrategySearchJobConfig(
        job_name=job_name,
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        search_type=args.search_type,
        n_samples=args.n_samples,
        random_seed=args.seed,
        notes=args.notes,
    )
    job = job_config.to_job(profile=args.profile)

    # Print job info
    print("\n" + "=" * 70)
    print("STRATEGY PARAMETER SEARCH")
    print("=" * 70)
    print(f"  Job ID:      {job.job_id}")
    print(f"  Strategy:    {job.strategy}")
    print(f"  Symbol:      {job.symbol}")
    print(f"  Timeframe:   {job.timeframe}")
    print(f"  Samples:     {job.n_samples}")
    print(f"  Search Type: {job.search_type}")
    print(f"  Profile:     {job.profile}")
    if job.seed is not None:
        print(f"  Seed:        {job.seed}")
    if job.notes:
        print(f"  Notes:       {job.notes}")
    print("=" * 70)
    print()

    # Run search
    print("Starting parameter search...")
    print()

    job_dir: Path
    dry_run = os.environ.get("STRATEGY_SEARCH_DRYRUN") == "1"
    if dry_run:
        dummy_row = {col: None for col in se.REQUIRED_RESULT_COLUMNS}
        dummy_row["params"] = {"_dummy": 0}
        dummy_row["status"] = "ok"
        dummy_row["error_message"] = None
        results = [dummy_row]
        job_dir = _persist_results(job, results, sys_cfg, base_output_dir)
    elif job.search_type == "random":
        job_dir = se.run_random_search(
            job=job,
            param_space=meta.param_space,
            sys_cfg=sys_cfg,
        )
    elif job.search_type == "grid":
        grid_results = se.grid_search(
            strategy_name=job.strategy,
            sys_cfg=sys_cfg,
            param_space=meta.param_space,
            grid_points=None,
        )
        job_dir = _persist_results(job, grid_results, sys_cfg, base_output_dir)
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unsupported search_type: {job.search_type}")

    print()
    print("=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {job_dir}")
    print()
    print("  Files created:")
    print(f"    - {job_dir / 'results.parquet'}")
    print(f"    - {job_dir / 'meta.json'}")
    print()
    print("  Next steps:")
    print("    1. Analyze results:")
    print(f"       python -c \"import pandas as pd; df = pd.read_parquet('{job_dir / 'results.parquet'}'); print(df.head())\"")
    print()
    print("    2. Find top performers:")
    print(f"       python -c \"import pandas as pd; df = pd.read_parquet('{job_dir / 'results.parquet'}'); print(df.nlargest(5, 'sharpe'))\"")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

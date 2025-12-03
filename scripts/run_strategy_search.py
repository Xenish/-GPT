"""
Run Strategy Parameter Search.

This script performs parameter search for a strategy and persists results
with full reproducibility (config snapshot, git SHA, job metadata).

Usage:
    python -m scripts.run_strategy_search --strategy rule --symbol BTCUSDT --timeframe 15m --n-samples 100

    python -m scripts.run_strategy_search \
        --config config/system.research.yml \
        --strategy trend_continuation \
        --symbol AIAUSDT \
        --timeframe 15m \
        --n-samples 50 \
        --job-id custom_job_id \
        --seed 42 \
        --notes "Testing trend strategy with conservative params"

Output:
    outputs/strategy_search/{job_id}/
    - results.parquet       # Search results
    - meta.json            # Job metadata + git_sha
    - config_snapshot.yml  # Config snapshot
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from finantradealgo.research.strategy_search.jobs import StrategySearchJob, create_job_id
from finantradealgo.research.strategy_search.search_engine import run_random_search
from finantradealgo.strategies.strategy_engine import get_strategy_meta, get_searchable_strategies
from finantradealgo.system.config_loader import load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run strategy parameter search with full job persistence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--profile",
        choices=["research", "live"],
        default="research",
        help="Config profile to load (default: research)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy name (e.g., 'rule', 'trend_continuation', 'sweep_reversal')",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (e.g., 'BTCUSDT', 'AIAUSDT')",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Timeframe (e.g., '5m', '15m', '1h')",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of parameter samples to evaluate",
    )

    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Custom job ID (default: auto-generated from strategy/symbol/timeframe/timestamp)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional notes about the job",
    )

    parser.add_argument(
        "--list-searchable",
        action="store_true",
        help="List all searchable strategies and exit",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

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

    # Validate required arguments for search
    missing_args = []
    if not args.strategy:
        missing_args.append("--strategy")
    if not args.symbol:
        missing_args.append("--symbol")
    if not args.timeframe:
        missing_args.append("--timeframe")
    if not args.n_samples:
        missing_args.append("--n-samples")

    if missing_args:
        print(f"Error: Missing required arguments: {', '.join(missing_args)}")
        print("\nUsage: python -m scripts.run_strategy_search --strategy STRATEGY --symbol SYMBOL --timeframe TF --n-samples N")
        print("   Or: python -m scripts.run_strategy_search --list-searchable")
        sys.exit(1)

    # Load system config
    print(f"Loading config: {args.config}")
    sys_cfg = load_config(args.profile)

    # Validate mode
    cfg_mode = sys_cfg.get("mode", "unknown")
    if cfg_mode != "research":
        raise RuntimeError(
            f"Strategy search must run with mode='research' config. "
            f"Got mode='{cfg_mode}'. Use config/system.research.yml or ensure mode='research'."
        )

    # Get strategy metadata and param_space
    strategy_name = args.strategy.lower()
    try:
        meta = get_strategy_meta(strategy_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nUse --list-searchable to see available strategies.")
        sys.exit(1)

    # Validate strategy is searchable
    if not meta.is_searchable or meta.param_space is None:
        print(f"Error: Strategy '{args.strategy}' is not searchable (no ParamSpace defined).")
        print("\nUse --list-searchable to see available strategies.")
        sys.exit(1)

    param_space = meta.param_space

    # Create or use custom job_id
    if args.job_id:
        job_id = args.job_id
    else:
        job_id = create_job_id(
            strategy=strategy_name,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )

    # Create StrategySearchJob
    job = StrategySearchJob(
        job_id=job_id,
        strategy=strategy_name,
        symbol=args.symbol,
        timeframe=args.timeframe,
        search_type="random",
        n_samples=args.n_samples,
        config_path=args.config,
        created_at=datetime.utcnow(),
        seed=args.seed,
        mode="research",
        notes=args.notes,
    )

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
    print(f"  Config:      {job.config_path}")
    print(f"  Mode:        {job.mode}")
    if job.seed is not None:
        print(f"  Seed:        {job.seed}")
    if job.notes:
        print(f"  Notes:       {job.notes}")
    print("=" * 70)
    print()

    # Run search
    print("Starting parameter search...")
    print()

    job_dir = run_random_search(
        job=job,
        param_space=param_space,
        sys_cfg=sys_cfg,
    )

    print()
    print("=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {job_dir}")
    print()
    print("  Files created:")
    print(f"    - {job_dir / 'results.parquet'}")
    print(f"    - {job_dir / 'meta.json'}")
    print(f"    - {job_dir / 'config_snapshot.yml'}")
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

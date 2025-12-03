from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from finantradealgo.backtester.runners import run_backtest_once
from finantradealgo.strategies.param_space import (
    ParamSpace,
    apply_strategy_params_to_cfg,
    sample_params,
)
from finantradealgo.strategies.strategy_engine import get_strategy_meta
from finantradealgo.system.config_loader import load_config

# Import StrategySearchJob from sibling module
import sys
from pathlib import Path as _Path
_current_dir = _Path(__file__).parent
if str(_current_dir.parent) not in sys.path:
    sys.path.insert(0, str(_current_dir.parent))

try:
    from finantradealgo.research.strategy_search.jobs import StrategySearchJob
except ImportError:
    # Fallback for different import paths
    from strategy_search.jobs import StrategySearchJob


def _compute_win_rate(trades: Any) -> Optional[float]:
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return None
    if "pnl" not in trades.columns:
        return None
    closed = trades.dropna(subset=["pnl"])
    if closed.empty:
        return None
    win_rate = (closed["pnl"] > 0).mean()
    return float(win_rate)


def evaluate_strategy_once(
    strategy_name: str,
    params: Optional[Dict[str, Any]] = None,
    sys_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a single backtest evaluation for the given strategy/params pair.
    """
    base_cfg = sys_cfg or load_config("research")
    cfg_with_params = apply_strategy_params_to_cfg(base_cfg, strategy_name, params or {})
    symbol = cfg_with_params.get("symbol", base_cfg.get("symbol"))
    timeframe = cfg_with_params.get("timeframe", base_cfg.get("timeframe"))
    if symbol is None or timeframe is None:
        raise ValueError("System config must provide symbol/timeframe for strategy evaluation.")

    result = run_backtest_once(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        cfg=cfg_with_params,
    )
    metrics = result.get("metrics", {}) or {}
    win_rate = _compute_win_rate(result.get("trades"))

    return {
        "params": dict(params or {}),
        "cum_return": metrics.get("cum_return"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": win_rate,
        "trade_count": metrics.get("trade_count"),
    }


def random_search(
    strategy_name: str,
    n_samples: int,
    sys_cfg: Optional[Dict[str, Any]] = None,
    param_space: Optional[ParamSpace] = None,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample random parameter sets from the strategy's ParamSpace and evaluate each once.

    Args:
        strategy_name: Strategy to search
        n_samples: Number of random samples
        sys_cfg: System configuration
        param_space: Parameter space (uses strategy default if None)
        random_seed: Random seed for reproducibility

    Returns:
        List of evaluation results with status and error handling
    """
    import random
    import numpy as np

    if n_samples <= 0:
        return []

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    cfg = sys_cfg or load_config("research")
    space = param_space or getattr(get_strategy_meta(strategy_name), "param_space", None)
    if not space:
        raise ValueError(f"Strategy '{strategy_name}' has no ParamSpace defined.")

    results: List[Dict[str, Any]] = []
    for i in range(n_samples):
        params = sample_params(space)

        # Error handling: wrap evaluation in try/except
        try:
            result = evaluate_strategy_once(strategy_name, params=params, sys_cfg=cfg)
            result["status"] = "ok"
            result["error_message"] = None
        except Exception as e:
            # On error, create failed result with NaN metrics
            result = {
                "params": params,
                "cum_return": None,
                "sharpe": None,
                "max_drawdown": None,
                "win_rate": None,
                "trade_count": None,
                "status": "error",
                "error_message": str(e)[:120],  # Truncate to 120 chars
            }

        results.append(result)

    return results


def grid_search(
    strategy_name: str,
    sys_cfg: Optional[Dict[str, Any]] = None,
    param_space: Optional[ParamSpace] = None,
    grid_points: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Grid search over parameter space.

    Args:
        strategy_name: Name of strategy to optimize
        sys_cfg: System configuration (optional)
        param_space: Parameter space (optional, uses strategy's default if None)
        grid_points: Number of grid points per parameter (default: 3 for numeric, all for categorical)

    Returns:
        List of evaluation results

    Note:
        Grid search can be computationally expensive. For a param space with N parameters
        and G grid points each, total evaluations = G^N.
        Use with caution for high-dimensional spaces.
    """
    import numpy as np
    from itertools import product

    cfg = sys_cfg or load_config("research")
    space = param_space or getattr(get_strategy_meta(strategy_name), "param_space", None)
    if not space:
        raise ValueError(f"Strategy '{strategy_name}' has no ParamSpace defined.")

    # Generate grid for each parameter
    param_grids: Dict[str, List[Any]] = {}
    default_grid_points = 3

    for param_name, spec in space.items():
        n_points = grid_points.get(param_name, default_grid_points) if grid_points else default_grid_points

        if spec.type == "int":
            low = int(spec.low)
            high = int(spec.high)
            param_grids[param_name] = list(np.linspace(low, high, n_points, dtype=int))

        elif spec.type == "float":
            low = float(spec.low)
            high = float(spec.high)
            if spec.log:
                param_grids[param_name] = list(np.logspace(np.log10(low), np.log10(high), n_points))
            else:
                param_grids[param_name] = list(np.linspace(low, high, n_points))

        elif spec.type == "bool":
            param_grids[param_name] = [False, True]

        elif spec.type == "categorical":
            param_grids[param_name] = list(spec.choices)

    # Generate all combinations
    param_names = list(param_grids.keys())
    param_values = [param_grids[name] for name in param_names]
    combinations = list(product(*param_values))

    # Evaluate each combination
    results: List[Dict[str, Any]] = []
    total_combos = len(combinations)
    print(f"Grid search: {total_combos} combinations to evaluate")

    for combo in combinations:
        params = dict(zip(param_names, combo))
        results.append(evaluate_strategy_once(strategy_name, params=params, sys_cfg=cfg))

    return results


# ============================================================================
# JOB-BASED SEARCH WITH PERSISTENCE
# ============================================================================

# Minimum required columns for results.parquet
REQUIRED_RESULT_COLUMNS = {
    "params", "cum_return", "sharpe", "max_drawdown", "win_rate", "trade_count",
    "status", "error_message"
}

BASE_OUTPUT_DIR = Path("outputs") / "strategy_search"


def _get_git_sha() -> Optional[str]:
    """Get current git commit SHA (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _validate_results(df: pd.DataFrame) -> None:
    """Validate that results DataFrame has required columns.

    Args:
        df: Results DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    if df.empty:
        return  # Empty is OK, just no results

    missing = REQUIRED_RESULT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Results DataFrame missing required columns: {missing}. "
            f"Required: {REQUIRED_RESULT_COLUMNS}"
        )


def run_random_search(
    job: StrategySearchJob,
    param_space: Optional[ParamSpace] = None,
    sys_cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    """Run random parameter search with full job persistence.

    This function:
    1. Runs random_search() to get results
    2. Saves results to results.parquet
    3. Saves job metadata to meta.json with git_sha + data_snapshot
    4. Copies config snapshot to job directory
    5. Validates results format

    Args:
        job: StrategySearchJob specification
        param_space: Optional ParamSpace override
        sys_cfg: Optional system config override

    Returns:
        Path to job output directory

    Raises:
        ValueError: If results validation fails
    """
    # Run the search with seed
    results = random_search(
        strategy_name=job.strategy,
        n_samples=job.n_samples,
        sys_cfg=sys_cfg,
        param_space=param_space,
        random_seed=job.seed,
    )

    # Create job directory
    job_dir = BASE_OUTPUT_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Validate results format
    _validate_results(results_df)

    # Save results in both parquet (efficient) and CSV (human-readable)
    results_parquet_path = job_dir / "results.parquet"
    results_csv_path = job_dir / "results.csv"
    results_df.to_parquet(results_parquet_path, index=False)
    results_df.to_csv(results_csv_path, index=False)

    # Get git SHA
    git_sha = _get_git_sha()

    # Copy config snapshot to job directory
    config_snapshot_name = "config_snapshot.yml"
    config_src = Path(job.config_path)
    if config_src.exists():
        config_dst = job_dir / config_snapshot_name
        shutil.copy2(config_src, config_dst)
        job.config_snapshot_relpath = config_snapshot_name

    # Count successful vs failed evaluations
    n_success = (results_df["status"] == "ok").sum() if "status" in results_df.columns else len(results_df)
    n_errors = (results_df["status"] == "error").sum() if "status" in results_df.columns else 0

    # Get data snapshot info (for reproducibility)
    data_cfg = sys_cfg.get("data_cfg") if sys_cfg else None
    data_snapshot = {}
    if data_cfg:
        data_snapshot = {
            "ohlcv_dir": getattr(data_cfg, "ohlcv_dir", "unknown"),
            "lookback_days": getattr(data_cfg, "lookback_days", {}),
        }

    # Update job with git_sha, data snapshot and save meta.json
    meta_dict = job.to_dict()
    meta_dict["git_sha"] = git_sha
    meta_dict["results_path_parquet"] = str(results_parquet_path.relative_to(job_dir))
    meta_dict["results_path_csv"] = str(results_csv_path.relative_to(job_dir))
    meta_dict["n_results"] = len(results_df)
    meta_dict["n_success"] = int(n_success)
    meta_dict["n_errors"] = int(n_errors)
    meta_dict["data_snapshot"] = data_snapshot

    meta_path = job_dir / "meta.json"
    import json
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    print(f"Strategy search job completed!")
    print(f"   Job ID: {job.job_id}")
    print(f"   Strategy: {job.strategy}")
    print(f"   Samples: {job.n_samples}")
    print(f"   Results: {len(results_df)} evaluations ({n_success} success, {n_errors} errors)")
    print(f"   Output: {job_dir}")
    print(f"   Git SHA: {git_sha or 'N/A'}")
    print(f"   Random Seed: {job.seed or 'N/A'}")

    return job_dir


__all__ = ["evaluate_strategy_once", "random_search", "grid_search", "run_random_search"]

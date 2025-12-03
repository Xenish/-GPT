"""
Strategy Search Results Analysis.

This module provides helper functions for analyzing and filtering strategy search results.

Usage:
    from finantradealgo.research.strategy_search.analysis import (
        load_results,
        filter_by_metrics,
        top_n_by_metric,
        compare_jobs,
    )

    # Load results from a job directory
    df = load_results("outputs/strategy_search/rule_BTCUSDT_15m_20251130_103904")

    # Filter by metrics
    profitable = filter_by_metrics(df, cum_return_min=0.0, sharpe_min=1.0)

    # Get top performers
    top_5 = top_n_by_metric(df, metric="sharpe", n=5)

    # Compare multiple jobs
    comparison = compare_jobs([
        "outputs/strategy_search/job1",
        "outputs/strategy_search/job2",
    ])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from finantradealgo.research.strategy_search.search_engine import REQUIRED_RESULT_COLUMNS


def load_results(
    job_dir: Union[str, Path],
    include_meta: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, Dict[str, Any]]]:
    """Load strategy search results from a job directory.

    Args:
        job_dir: Path to job directory (e.g., "outputs/strategy_search/job_id")
        include_meta: If True, return (df, meta_dict) tuple

    Returns:
        DataFrame with search results, or (df, meta_dict) if include_meta=True

    Raises:
        FileNotFoundError: If results.parquet or meta.json not found
        ValueError: If results format is invalid

    Example:
        >>> df = load_results("outputs/strategy_search/rule_BTCUSDT_15m_20251130_103904")
        >>> print(df.columns)
        Index(['params', 'cum_return', 'sharpe', 'max_drawdown', 'win_rate', 'trade_count'])

        >>> df, meta = load_results("outputs/strategy_search/job1", include_meta=True)
        >>> print(meta['strategy'])
        'rule'
    """
    job_path = Path(job_dir)
    results_path = job_path / "results.parquet"
    meta_path = job_path / "meta.json"

    if not results_path.exists():
        raise FileNotFoundError(f"results.parquet not found in {job_path}")
    if include_meta and not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {job_path}")

    df = pd.read_parquet(results_path)

    # Validate required columns
    missing = REQUIRED_RESULT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Results missing required columns: {missing}")

    if include_meta:
        import json
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        return df, meta

    return df


def filter_by_metrics(
    df: pd.DataFrame,
    cum_return_min: Optional[float] = None,
    cum_return_max: Optional[float] = None,
    sharpe_min: Optional[float] = None,
    sharpe_max: Optional[float] = None,
    max_drawdown_min: Optional[float] = None,
    max_drawdown_max: Optional[float] = None,
    win_rate_min: Optional[float] = None,
    win_rate_max: Optional[float] = None,
    trade_count_min: Optional[int] = None,
    trade_count_max: Optional[int] = None,
) -> pd.DataFrame:
    """Filter results by metric criteria.

    Args:
        df: Results DataFrame
        cum_return_min: Minimum cumulative return (e.g., 0.0 for profitable only)
        cum_return_max: Maximum cumulative return
        sharpe_min: Minimum Sharpe ratio (e.g., 1.0 for good risk-adjusted returns)
        sharpe_max: Maximum Sharpe ratio
        max_drawdown_min: Minimum max drawdown (e.g., -0.2 for max 20% drawdown)
        max_drawdown_max: Maximum max drawdown
        win_rate_min: Minimum win rate (0.0 - 1.0)
        win_rate_max: Maximum win rate (0.0 - 1.0)
        trade_count_min: Minimum number of trades (e.g., 50 for statistical significance)
        trade_count_max: Maximum number of trades

    Returns:
        Filtered DataFrame

    Example:
        >>> # Find profitable strategies with good Sharpe and limited drawdown
        >>> filtered = filter_by_metrics(
        ...     df,
        ...     cum_return_min=0.0,
        ...     sharpe_min=1.0,
        ...     max_drawdown_min=-0.20,
        ...     trade_count_min=50,
        ... )

        >>> # Find strategies with very high Sharpe but moderate returns
        >>> high_sharpe = filter_by_metrics(
        ...     df,
        ...     sharpe_min=2.0,
        ...     cum_return_min=0.05,
        ...     cum_return_max=0.20,
        ... )
    """
    expected_cols = {"cum_return", "sharpe", "max_drawdown", "win_rate", "trade_count"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required metric columns: {missing}")

    result = df.copy()

    # Apply filters
    if cum_return_min is not None:
        result = result[result["cum_return"] >= cum_return_min]
    if cum_return_max is not None:
        result = result[result["cum_return"] <= cum_return_max]

    if sharpe_min is not None:
        result = result[result["sharpe"] >= sharpe_min]
    if sharpe_max is not None:
        result = result[result["sharpe"] <= sharpe_max]

    if max_drawdown_min is not None:
        result = result[result["max_drawdown"] >= max_drawdown_min]
    if max_drawdown_max is not None:
        result = result[result["max_drawdown"] <= max_drawdown_max]

    if win_rate_min is not None:
        # Handle None/NaN win rates
        result = result[result["win_rate"].notna() & (result["win_rate"] >= win_rate_min)]
    if win_rate_max is not None:
        result = result[result["win_rate"].notna() & (result["win_rate"] <= win_rate_max)]

    if trade_count_min is not None:
        result = result[result["trade_count"] >= trade_count_min]
    if trade_count_max is not None:
        result = result[result["trade_count"] <= trade_count_max]

    return result


def top_n_by_metric(
    df: pd.DataFrame,
    metric: str,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """Get top N results sorted by a specific metric.

    Args:
        df: Results DataFrame
        metric: Metric column to sort by (e.g., "sharpe", "cum_return")
        n: Number of top results to return
        ascending: If True, return lowest values (e.g., for max_drawdown)

    Returns:
        DataFrame with top N results

    Example:
        >>> # Get top 5 strategies by Sharpe ratio
        >>> top_sharpe = top_n_by_metric(df, metric="sharpe", n=5)

        >>> # Get 10 strategies with lowest drawdown
        >>> low_dd = top_n_by_metric(df, metric="max_drawdown", n=10, ascending=True)

        >>> # Get top 20 by cumulative return
        >>> top_returns = top_n_by_metric(df, metric="cum_return", n=20)
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame. Available: {list(df.columns)}")

    # Sort and get top N
    sorted_df = df.sort_values(by=metric, ascending=ascending)
    return sorted_df.head(n)


def compare_jobs(
    job_dirs: List[Union[str, Path]],
    metric: str = "sharpe",
    top_n: int = 5,
) -> pd.DataFrame:
    """Compare top results across multiple search jobs.

    Args:
        job_dirs: List of job directory paths
        metric: Metric to compare by (default: "sharpe")
        top_n: Number of top results from each job to compare

    Returns:
        DataFrame with top N results from each job, with job_id column

    Example:
        >>> comparison = compare_jobs(
        ...     job_dirs=[
        ...         "outputs/strategy_search/rule_BTCUSDT_15m_20251130_103904",
        ...         "outputs/strategy_search/rule_BTCUSDT_15m_20251130_110521",
        ...     ],
        ...     metric="sharpe",
        ...     top_n=3,
        ... )
        >>> print(comparison[['job_id', 'sharpe', 'cum_return']])
    """
    all_results = []

    for job_dir in job_dirs:
        try:
            df, meta = load_results(job_dir, include_meta=True)
            top = top_n_by_metric(df, metric=metric, n=top_n)
            top = top.copy()
            top["job_id"] = meta["job_id"]
            top["strategy"] = meta["strategy"]
            top["symbol"] = meta["symbol"]
            top["timeframe"] = meta["timeframe"]
            all_results.append(top)
        except Exception as e:
            print(f"Warning: Failed to load job {job_dir}: {e}")
            continue

    if not all_results:
        raise ValueError("No valid job results found")

    # Combine and sort
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values(by=metric, ascending=False)

    return combined


def get_param_importance(
    df: pd.DataFrame,
    metric: str = "sharpe",
    top_pct: float = 0.1,
) -> pd.DataFrame:
    """Analyze which parameters appear most frequently in top performers.

    Args:
        df: Results DataFrame with 'params' column
        metric: Metric to determine top performers
        top_pct: Percentage of top results to analyze (0.1 = top 10%)

    Returns:
        DataFrame with parameter value frequencies in top performers

    Example:
        >>> importance = get_param_importance(df, metric="sharpe", top_pct=0.1)
        >>> print(importance.head(10))
    """
    # Get top performers
    n_top = max(1, int(len(df) * top_pct))
    top_df = top_n_by_metric(df, metric=metric, n=n_top)

    # Extract parameter values from top performers
    param_counts = {}
    for params_dict in top_df["params"]:
        for key, value in params_dict.items():
            if key not in param_counts:
                param_counts[key] = {}
            value_key = str(value)
            param_counts[key][value_key] = param_counts[key].get(value_key, 0) + 1

    # Convert to DataFrame
    importance_data = []
    for param, value_counts in param_counts.items():
        for value, count in value_counts.items():
            importance_data.append({
                "parameter": param,
                "value": value,
                "count": count,
                "frequency": count / n_top,
            })

    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values(by=["parameter", "frequency"], ascending=[True, False])

    return importance_df


__all__ = [
    "load_results",
    "filter_by_metrics",
    "top_n_by_metric",
    "compare_jobs",
    "get_param_importance",
]

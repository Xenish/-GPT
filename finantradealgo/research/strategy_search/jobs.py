"""
Strategy Search Job Management.

This module defines the StrategySearchJob model and utilities for managing
strategy parameter search jobs with full persistence and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
import json


def create_job_id(
    strategy: str,
    symbol: str,
    timeframe: str,
    timestamp: Optional[datetime] = None
) -> str:
    """Create unique job ID for strategy search.

    Args:
        strategy: Strategy name (e.g., "rule", "trend_continuation")
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "15m")
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Job ID string: "{strategy}_{symbol}_{timeframe}_{timestamp}"

    Example:
        >>> create_job_id("rule", "BTCUSDT", "15m")
        'rule_BTCUSDT_15m_20251130_143022'
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{strategy}_{symbol}_{timeframe}_{ts_str}"


@dataclass
class StrategySearchJob:
    """Strategy parameter search job specification.

    This model defines all metadata needed to run, persist, and reproduce
    a strategy parameter search job.

    Attributes:
        job_id: Unique job identifier
        strategy: Strategy name to search
        symbol: Trading symbol
        timeframe: Trading timeframe
        search_type: Type of search ("random" | "grid")
        n_samples: Number of parameter samples to evaluate
        config_path: Path to system config used for search
        created_at: Job creation timestamp
        notes: Optional notes about the job
        seed: Random seed for reproducibility
        mode: Config mode (should be "research")
        config_snapshot_relpath: Relative path to config snapshot in job dir
    """
    job_id: str
    strategy: str
    symbol: str
    timeframe: str
    search_type: Literal["random", "grid"]
    n_samples: int
    config_path: str
    created_at: datetime
    notes: Optional[str] = None
    seed: Optional[int] = None
    mode: str = "research"
    config_snapshot_relpath: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "search_type": self.search_type,
            "n_samples": self.n_samples,
            "config_path": self.config_path,
            "created_at": self.created_at.isoformat(),
            "notes": self.notes,
            "seed": self.seed,
            "mode": self.mode,
            "config_snapshot_relpath": self.config_snapshot_relpath,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategySearchJob":
        """Create from dictionary (e.g., loaded from JSON)."""
        data = dict(data)  # Copy to avoid mutation
        # Parse datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    def save_meta(self, output_dir: Path) -> Path:
        """Save job metadata to meta.json.

        Args:
            output_dir: Directory to save meta.json

        Returns:
            Path to saved meta.json file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return meta_path

    @classmethod
    def load_meta(cls, output_dir: Path) -> "StrategySearchJob":
        """Load job metadata from meta.json.

        Args:
            output_dir: Directory containing meta.json

        Returns:
            StrategySearchJob instance
        """
        meta_path = Path(output_dir) / "meta.json"
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


__all__ = [
    "StrategySearchJob",
    "create_job_id",
]

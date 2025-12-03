"""
Strategy Search Job Management.

This module defines the StrategySearchJob model and utilities for managing
strategy parameter search jobs with full persistence and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import json
import yaml


def create_job_id(
    strategy: str,
    symbol: str,
    timeframe: str,
    timestamp: Optional[datetime] = None,
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
        timestamp = datetime.now(timezone.utc)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc)
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    def _safe(seg: str) -> str:
        return seg.replace("/", "_").replace("\\", "_")
    return f"{_safe(strategy)}_{_safe(symbol)}_{_safe(timeframe)}_{ts_str}"


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
         profile: Config profile used for search ("research" | "live")
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
    created_at: datetime
    profile: str = "research"
    notes: Optional[str] = None
    seed: Optional[int] = None
    mode: str = "research"
    config_snapshot_relpath: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        created_at = self.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return {
            "job_id": self.job_id,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "search_type": self.search_type,
            "n_samples": self.n_samples,
            "profile": self.profile,
            "created_at": created_at.isoformat(),
            "notes": self.notes,
            "seed": self.seed,
            "mode": self.mode,
            "config_snapshot_relpath": self.config_snapshot_relpath,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategySearchJob":
        """Create from dictionary (e.g., loaded from JSON)."""
        data = dict(data)  # Copy to avoid mutation
        allowed_keys = {f.name for f in fields(cls)}
        # Drop any extra metadata fields that are not part of the dataclass
        data = {k: v for k, v in data.items() if k in allowed_keys}
        created_at = data.get("created_at")
        if created_at is None:
            raise ValueError("StrategySearchJob 'created_at' is required in metadata.")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        data["created_at"] = created_at
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
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {output_dir}")

        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class StrategySearchJobConfig:
    """
    Configuration for strategy search job loaded from YAML.

    This is the user-facing config format for defining search jobs.
    """
    job_name: str
    strategy_name: str
    symbol: str
    timeframe: str
    search_type: Literal["random", "grid"] = "random"
    n_samples: int = 50
    fixed_params: Optional[Dict[str, Any]] = None
    search_space_override: Optional[Dict[str, Any]] = None
    grid_points: Optional[Dict[str, int]] = None  # For grid search
    random_seed: Optional[int] = None
    notes: Optional[str] = None
    config_snapshot_relpath: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StrategySearchJobConfig":
        """Load job config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategySearchJobConfig":
        """Create from dictionary."""
        search_type = data.get("search_type", data.get("mode", "random"))
        return cls(
            job_name=data["job_name"],
            strategy_name=data["strategy_name"],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            search_type=search_type,
            n_samples=int(data.get("n_samples", 50)),
            fixed_params=data.get("fixed_params"),
            search_space_override=data.get("search_space_override"),
            grid_points=data.get("grid_points"),
            random_seed=data.get("random_seed"),
            notes=data.get("notes"),
            config_snapshot_relpath=data.get("config_snapshot_relpath"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_name": self.job_name,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "search_type": self.search_type,
            "n_samples": self.n_samples,
            "fixed_params": self.fixed_params,
            "search_space_override": self.search_space_override,
            "grid_points": self.grid_points,
            "random_seed": self.random_seed,
            "notes": self.notes,
            "config_snapshot_relpath": self.config_snapshot_relpath,
        }

    def to_job(self, profile: str = "research") -> StrategySearchJob:
        """Convert to StrategySearchJob for execution."""
        created_at = datetime.now(timezone.utc)
        job_id = create_job_id(
            self.strategy_name,
            self.symbol,
            self.timeframe,
            timestamp=created_at,
        )
        return StrategySearchJob(
            job_id=job_id,
            strategy=self.strategy_name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            search_type=self.search_type,
            n_samples=self.n_samples,
            profile=profile,
            created_at=created_at,
            notes=self.notes,
            seed=self.random_seed,
            mode="research",
            config_snapshot_relpath=self.config_snapshot_relpath,
        )


__all__ = [
    "StrategySearchJob",
    "StrategySearchJobConfig",
    "create_job_id",
]

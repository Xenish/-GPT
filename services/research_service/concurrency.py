"""
Job concurrency control for research service.

Prevents resource exhaustion by limiting concurrent strategy search jobs.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from finantradealgo.system.config_loader import ResearchConfig


class JobLimiter:
    """
    Semaphore-based job concurrency limiter.

    Ensures that no more than max_parallel_jobs are running simultaneously.
    """

    def __init__(self, max_parallel_jobs: int):
        """
        Initialize job limiter.

        Args:
            max_parallel_jobs: Maximum number of concurrent jobs
        """
        if max_parallel_jobs <= 0:
            raise ValueError(f"max_parallel_jobs must be positive, got {max_parallel_jobs}")

        self._semaphore = asyncio.Semaphore(max_parallel_jobs)
        self._max_parallel_jobs = max_parallel_jobs
        self._active_jobs = 0

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[None, None]:
        """
        Acquire a job slot.

        Usage:
            async with job_limiter.acquire():
                # Run job
                ...
        """
        async with self._semaphore:
            self._active_jobs += 1
            try:
                yield
            finally:
                self._active_jobs -= 1

    @property
    def active_jobs(self) -> int:
        """Get current number of active jobs."""
        return self._active_jobs

    @property
    def max_jobs(self) -> int:
        """Get maximum concurrent jobs."""
        return self._max_parallel_jobs

    @property
    def available_slots(self) -> int:
        """Get number of available job slots."""
        return self._max_parallel_jobs - self._active_jobs


# Global job limiter instance (initialized at startup)
_global_limiter: JobLimiter | None = None


def initialize_job_limiter(research_cfg: ResearchConfig) -> None:
    """
    Initialize global job limiter from research config.

    Args:
        research_cfg: Research configuration with max_parallel_jobs
    """
    global _global_limiter
    _global_limiter = JobLimiter(research_cfg.max_parallel_jobs)


def get_job_limiter() -> JobLimiter:
    """
    Get the global job limiter instance.

    Returns:
        Global JobLimiter instance

    Raises:
        RuntimeError: If limiter not initialized
    """
    if _global_limiter is None:
        raise RuntimeError(
            "Job limiter not initialized. Call initialize_job_limiter() first."
        )
    return _global_limiter


__all__ = [
    "JobLimiter",
    "initialize_job_limiter",
    "get_job_limiter",
]

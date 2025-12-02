"""Champion/challenger A/B testing orchestration.

Provides lightweight A/B/N testing with traffic allocation, metric aggregation,
simple significance testing, and a champion/challenger helper. This does not
execute trades; higher-level orchestration should route orders according to
assignments and persist samples/results.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from finantradealgo.deployment import DeploymentEnvironment, StrategyVersion
from finantradealgo.deployment.version_control import StrategyRegistry


@dataclass
class ABTestArm:
    name: str
    strategy: StrategyVersion
    allocation: float  # fraction of traffic/capital (sums to 1.0 across arms)
    metadata: dict[str, Any] | None = None


@dataclass
class ABTestConfig:
    name: str
    environment: DeploymentEnvironment
    arms: list[ABTestArm]
    metric_name: str = "sharpe"  # or "return", "win_rate"
    min_sample_size: int = 100
    significance_level: float = 0.05
    metadata: dict[str, Any] | None = None


@dataclass
class ABTestSample:
    arm_name: str
    metric_value: float
    weight: float = 1.0
    metadata: dict[str, Any] | None = None


@dataclass
class ABTestResult:
    config: ABTestConfig
    arm_metrics: dict[str, dict[str, float]]  # arm -> metric summary (mean, std, n, etc.)
    p_values: dict[tuple[str, str], float]  # (arm_i, arm_j) -> p_value
    champion: str | None
    is_significant: bool
    metadata: dict[str, Any] | None = None


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """
    Compute weighted mean/std and sample size.
    Note: std is population-style; for small samples this underestimates variance.
    """
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
    w_sum = np.sum(weights)
    mean = float(np.sum(values * weights) / w_sum)
    variance = float(np.sum(weights * (values - mean) ** 2) / w_sum)
    std = math.sqrt(variance)
    return {"mean": mean, "std": std, "n": float(w_sum)}


def _two_sample_t_test(mean_a: float, std_a: float, n_a: float, mean_b: float, std_b: float, n_b: float) -> float:
    """
    Approximate two-sample t-test p-value (two-sided) using a normal approximation.
    Assumes independent samples and large enough n for CLT to hold. For weighted
    samples, n reflects effective sample size (sum of weights).
    """
    if n_a <= 0 or n_b <= 0:
        return 1.0
    se = math.sqrt((std_a ** 2) / n_a + (std_b ** 2) / n_b)
    if se == 0:
        return 1.0
    t_stat = abs(mean_a - mean_b) / se
    # Two-sided p-value using normal approximation.
    return math.erfc(t_stat / math.sqrt(2))


class ABTestingEngine:
    def __init__(
        self,
        registry: StrategyRegistry,
    ) -> None:
        self.registry = registry
        # store samples by test name
        self._samples: dict[str, list[ABTestSample]] = {}

    def assign_arm(self, config: ABTestConfig, *, key: str | None = None) -> ABTestArm:
        """
        Assign a unit (account/trade/session) to an arm using weighted allocation.
        If key is provided, produce a deterministic mapping using a stable hash.
        """
        allocations = np.array([arm.allocation for arm in config.arms], dtype=float)
        if allocations.sum() <= 0:
            raise ValueError("Arm allocations must sum to a positive value.")
        allocations = allocations / allocations.sum()

        if key is None:
            idx = int(np.random.choice(len(config.arms), p=allocations))
            return config.arms[idx]

        digest = hashlib.sha256(key.encode("utf-8")).digest()
        val = int.from_bytes(digest, byteorder="big") / float(2**256)
        cumulative = 0.0
        for i, weight in enumerate(allocations):
            cumulative += weight
            if val <= cumulative:
                return config.arms[i]
        return config.arms[-1]

    def record_sample(self, test_name: str, sample: ABTestSample) -> None:
        """Append a metric observation for a given test."""
        self._samples.setdefault(test_name, []).append(sample)

    def compute_result(self, config: ABTestConfig) -> ABTestResult:
        """
        Aggregate samples, compute weighted stats, run pairwise tests, and pick a champion.
        Uses simple normal-based t-test approximations; for production, consider a more
        robust stats library or Bayesian bandit.
        """
        samples = self._samples.get(config.name, [])
        by_arm: dict[str, list[ABTestSample]] = {}
        for s in samples:
            by_arm.setdefault(s.arm_name, []).append(s)

        arm_metrics: dict[str, dict[str, float]] = {}
        for arm in config.arms:
            arm_samples = by_arm.get(arm.name, [])
            values = np.array([s.metric_value for s in arm_samples], dtype=float)
            weights = np.array([s.weight for s in arm_samples], dtype=float)
            if values.size == 0:
                stats = {"mean": float("nan"), "std": float("nan"), "n": 0.0}
            else:
                stats = _weighted_stats(values, weights)
            arm_metrics[arm.name] = stats

        # Check minimum sample size for all arms.
        if any(stats["n"] < config.min_sample_size for stats in arm_metrics.values()):
            return ABTestResult(
                config=config,
                arm_metrics=arm_metrics,
                p_values={},
                champion=None,
                is_significant=False,
            )

        # Compute pairwise p-values.
        p_values: dict[tuple[str, str], float] = {}
        arm_names = [arm.name for arm in config.arms]
        for i, name_i in enumerate(arm_names):
            for j in range(i + 1, len(arm_names)):
                name_j = arm_names[j]
                stats_i = arm_metrics[name_i]
                stats_j = arm_metrics[name_j]
                p_val = _two_sample_t_test(
                    stats_i["mean"],
                    stats_i["std"],
                    stats_i["n"],
                    stats_j["mean"],
                    stats_j["std"],
                    stats_j["n"],
                )
                p_values[(name_i, name_j)] = p_val
                p_values[(name_j, name_i)] = p_val

        # Choose champion: highest mean for target metric.
        champion = max(arm_metrics.items(), key=lambda kv: kv[1]["mean"])[0]

        # Significant if champion beats any other arm with p < alpha.
        is_significant = False
        for other in arm_names:
            if other == champion:
                continue
            stats_c = arm_metrics[champion]
            stats_o = arm_metrics[other]
            if stats_c["mean"] > stats_o["mean"] and p_values.get((champion, other), 1.0) < config.significance_level:
                is_significant = True
                break

        return ABTestResult(
            config=config,
            arm_metrics=arm_metrics,
            p_values=p_values,
            champion=champion if is_significant else None,
            is_significant=is_significant,
        )

    def promote_champion(self, result: ABTestResult) -> None:
        """
        In a real system this would update registry env mappings (and possibly
        trigger canary promotion) to make the champion strategy active. Left as
        a stub to avoid coupling to live trading here.
        """
        return None

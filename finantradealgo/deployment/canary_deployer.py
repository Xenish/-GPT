"""Canary and blue/green deployment orchestration.

This module defines a `CanaryDeployer` that decides how to progress a strategy
through staged capital allocations based on observed metrics. It does not
execute trades itself; higher-level orchestration should use the produced
configs/records to start/stop strategies and persist history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from finantradealgo.deployment import (
    DeploymentEnvironment,
    DeploymentRecord,
    DeploymentStatus,
    StrategyDeploymentConfig,
    StrategyVersion,
)
from finantradealgo.deployment.version_control import StrategyRegistry


@dataclass
class CanaryStage:
    name: str
    capital_fraction: float  # fraction of total capital at this stage
    min_duration_minutes: int  # minimal runtime before evaluation
    promotion_thresholds: dict[str, float]  # e.g. {"sharpe": 1.0, "max_dd": -0.2}
    max_risk_violations: int = 0
    metadata: dict[str, Any] | None = None


@dataclass
class CanaryConfig:
    stages: list[CanaryStage]
    environment: DeploymentEnvironment = DeploymentEnvironment.LIVE
    max_total_capital_fraction: float = 1.0
    allow_blue_green: bool = True
    blue_green_env: DeploymentEnvironment = DeploymentEnvironment.LIVE
    metadata: dict[str, Any] | None = None


@dataclass
class CanaryState:
    strategy: StrategyVersion
    config: CanaryConfig
    current_stage_index: int = 0
    records: list[DeploymentRecord] = field(default_factory=list)
    risk_violations: int = 0
    is_complete: bool = False
    rolled_back: bool = False
    metadata: dict[str, Any] | None = None


class CanaryDeployer:
    def __init__(
        self,
        registry: StrategyRegistry,
        metrics_provider: Callable[[StrategyVersion, DeploymentEnvironment], dict[str, float]],
    ) -> None:
        """
        metrics_provider:
          Given a strategy version and environment, returns performance metrics
          over the relevant evaluation window (Sharpe, PnL, drawdown, etc.).
        """
        self.registry = registry
        self.metrics_provider = metrics_provider

    def start_canary(self, version: StrategyVersion, config: CanaryConfig) -> CanaryState:
        """Start a canary by deploying stage 0 with the specified capital fraction."""
        if not config.stages:
            raise ValueError("CanaryConfig must define at least one stage.")

        first_stage = config.stages[0]
        deployment_config = StrategyDeploymentConfig(
            strategy=version,
            environment=config.environment,
            capital_fraction=min(first_stage.capital_fraction, config.max_total_capital_fraction),
            risk_limits=None,
            auto_rollout=False,
            metadata=first_stage.metadata,
        )
        record = DeploymentRecord(config=deployment_config, status=DeploymentStatus.DEPLOYED)

        # Mark the canary as active for the target environment.
        self.registry.set_env_mapping(version.id, version.version, config.environment, label="canary")

        return CanaryState(
            strategy=version,
            config=config,
            current_stage_index=0,
            records=[record],
        )

    def evaluate_and_maybe_promote(self, state: CanaryState) -> CanaryState:
        """
        Evaluate current stage metrics, promote to next stage if thresholds pass,
        or rollback on excessive risk violations. When the final stage is passed,
        optionally flip blue/green mapping to make the new version active.
        """
        if state.is_complete or state.rolled_back:
            return state

        stage = state.config.stages[state.current_stage_index]
        metrics = self.metrics_provider(state.strategy, state.config.environment)

        # Simple threshold check: all metrics must meet or exceed the threshold.
        meets_thresholds = True
        for key, threshold in stage.promotion_thresholds.items():
            value = metrics.get(key)
            if value is None or value < threshold:
                meets_thresholds = False
                break

        if meets_thresholds:
            if state.current_stage_index >= len(state.config.stages) - 1:
                state.is_complete = True
                if state.config.allow_blue_green:
                    current_active = self.registry.resolve_for_environment(
                        state.strategy.id, state.config.blue_green_env
                    )
                    if current_active and current_active.version != state.strategy.version:
                        # Tag previous as previous, new as active.
                        self.registry.set_env_mapping(
                            current_active.id, current_active.version, state.config.blue_green_env, label="previous"
                        )
                    self.registry.set_env_mapping(
                        state.strategy.id, state.strategy.version, state.config.blue_green_env, label="active"
                    )
                return state

            # Promote to next stage.
            next_stage_index = state.current_stage_index + 1
            next_stage = state.config.stages[next_stage_index]
            deployment_config = StrategyDeploymentConfig(
                strategy=state.strategy,
                environment=state.config.environment,
                capital_fraction=min(
                    next_stage.capital_fraction, state.config.max_total_capital_fraction
                ),
                risk_limits=None,
                auto_rollout=True,
                metadata=next_stage.metadata,
            )
            state.records.append(DeploymentRecord(config=deployment_config, status=DeploymentStatus.DEPLOYED))
            state.current_stage_index = next_stage_index
            state.risk_violations = 0
            return state

        # Failing thresholds.
        state.risk_violations += 1
        if state.risk_violations > stage.max_risk_violations:
            return self.rollback(state)
        return state

    def rollback(self, state: CanaryState, *, rollback_to: StrategyVersion | None = None) -> CanaryState:
        """
        Mark the canary as rolled back and update registry mappings.
        In a real system, this would also stop the new version and start the
        rollback target in the live environment.
        """
        if not state.records:
            state.rolled_back = True
            return state

        # Mark latest record as rolled back.
        state.records[-1].status = DeploymentStatus.ROLLED_BACK

        target = rollback_to
        if target is None:
            # Prefer current active mapping, otherwise default version.
            active = self.registry.resolve_for_environment(state.strategy.id, state.config.environment)
            target = active if active and active.version != state.strategy.version else self.registry.get_default_version(
                state.strategy.id
            )

        if target:
            self.registry.set_env_mapping(
                target.id,
                target.version,
                state.config.environment,
                label="active",
            )
        state.rolled_back = True
        state.is_complete = True
        return state

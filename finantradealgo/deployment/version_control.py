"""Strategy version control and registry.

This module provides an in-memory registry for strategy versions with optional
hooks to a Git backend. It is designed to sit between the codebase and higher
layers (CI, deployment orchestration, live_trading) so those layers can resolve
which concrete strategy version to run per environment without knowing how the
versions are stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from finantradealgo.deployment import DeploymentEnvironment, StrategyId, StrategyVersion


class GitBackend:
    """
    Thin abstraction to wrap git operations. For now, only define methods and
    docstrings so the registry can be wired to real git commands later.
    """

    def tag_version(self, version: StrategyVersion, *, tag_prefix: str = "strategy") -> str:  # pragma: no cover - placeholder
        """
        Tag the current commit with a strategy-specific tag (e.g., strategy/trend_follow_v2:v1.2.3).
        Returns the tag name.
        """
        raise NotImplementedError

    def get_current_commit(self) -> str:  # pragma: no cover - placeholder
        """Return the current git commit hash."""
        raise NotImplementedError

    def checkout(self, git_ref: str) -> None:  # pragma: no cover - placeholder
        """Checkout the given git reference (commit hash or tag)."""
        raise NotImplementedError


@dataclass
class StrategyRegistryEntry:
    strategy: StrategyVersion
    environments: dict[DeploymentEnvironment, str]
    is_default: bool = False
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None


class StrategyRegistry:
    """
    In-memory registry for strategy versions. Acts as the single source of truth
    for mapping logical strategy IDs to concrete versions, environment
    activations, and tags/metadata. Persistence can be added by serializing
    `_entries` to a file/DB, and git operations can be provided via `git_backend`.
    """

    def __init__(self, git_backend: GitBackend | None = None) -> None:
        # strategy_id -> list[StrategyRegistryEntry] (one per version)
        self._entries: dict[StrategyId, list[StrategyRegistryEntry]] = {}
        self.git_backend = git_backend

    def register_strategy(
        self,
        version: StrategyVersion,
        *,
        is_default: bool = False,
        tags: Optional[list[str]] = None,
    ) -> StrategyRegistryEntry:
        """Add a new version for the strategy ID."""
        entry = StrategyRegistryEntry(
            strategy=version,
            environments={},
            is_default=is_default,
            tags=tags or [],
        )

        entries = self._entries.setdefault(version.id, [])
        if is_default:
            for e in entries:
                e.is_default = False
        entries.append(entry)
        return entry

    def list_versions(self, strategy_id: StrategyId) -> list[StrategyVersion]:
        """Return all known versions for the given strategy."""
        return [entry.strategy for entry in self._entries.get(strategy_id, [])]

    def get_default_version(self, strategy_id: StrategyId) -> StrategyVersion | None:
        """Return the version marked as default, or None."""
        for entry in self._entries.get(strategy_id, []):
            if entry.is_default:
                return entry.strategy
        return None

    def set_default_version(self, strategy_id: StrategyId, version_str: str) -> None:
        """Mark a specific version as default."""
        entries = self._entries.get(strategy_id, [])
        for entry in entries:
            entry.is_default = entry.strategy.version == version_str

    def get_version(self, strategy_id: StrategyId, version_str: str) -> StrategyVersion | None:
        """Fetch a specific version for a strategy."""
        for entry in self._entries.get(strategy_id, []):
            if entry.strategy.version == version_str:
                return entry.strategy
        return None

    def mark_deprecated(self, strategy_id: StrategyId, version_str: str) -> None:
        """Mark an entry as deprecated via tags and metadata."""
        for entry in self._entries.get(strategy_id, []):
            if entry.strategy.version == version_str:
                if "deprecated" not in entry.tags:
                    entry.tags.append("deprecated")
                if entry.metadata is None:
                    entry.metadata = {}
                entry.metadata["deprecated"] = True

    def set_env_mapping(
        self,
        strategy_id: StrategyId,
        version_str: str,
        env: DeploymentEnvironment,
        label: str = "active",
    ) -> None:
        """Record which version is active for which environment (RESEARCH/PAPER/LIVE)."""
        for entry in self._entries.get(strategy_id, []):
            if entry.strategy.version == version_str:
                entry.environments[env] = label

    def resolve_for_environment(
        self,
        strategy_id: StrategyId,
        env: DeploymentEnvironment,
    ) -> StrategyVersion | None:
        """
        Resolve the version to use for the given environment.
        Prefer entry with environments[env] == "active"; otherwise fallback to default.
        """
        for entry in self._entries.get(strategy_id, []):
            if entry.environments.get(env) == "active":
                return entry.strategy
        return self.get_default_version(strategy_id)

    def create_git_tag_for_version(self, version: StrategyVersion, tag_prefix: str = "strategy") -> str | None:
        """
        Create a git tag for the given strategy version using the configured
        backend. Updates version.git_ref if successful.
        """
        if not self.git_backend:
            return None
        tag = self.git_backend.tag_version(version, tag_prefix=tag_prefix)
        version.git_ref = tag
        return tag

    def rollback_to_version(self, strategy_id: StrategyId, version_str: str) -> StrategyVersion | None:
        """
        Resolve and return the requested version. In a real system this would:
        - use git_backend.checkout(version.git_ref)
        - update environment mappings
        For now, only resolves without touching git or environments.
        """
        version = self.get_version(strategy_id, version_str)
        return version

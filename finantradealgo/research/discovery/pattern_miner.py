"""Pattern mining utilities for strategy discovery."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterable, Sequence

from . import DiscoveryResult, StrategyCandidate


@dataclass(slots=True)
class PatternMiningConfig:
    """Configuration for frequent pattern and association-rule mining."""

    min_support: float = 0.05
    min_confidence: float = 0.1
    max_pattern_length: int = 3
    time_window_bars: int | None = None
    random_seed: int | None = None


@dataclass(slots=True)
class DiscoveredPattern:
    """A mined pattern with basic quality metrics."""

    pattern: tuple[str, ...]
    support: float
    confidence: float | None = None
    lift: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": list(self.pattern),
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AssociationRule:
    """Simple association rule derived from frequent patterns."""

    antecedent: tuple[str, ...]
    consequent: tuple[str, ...]
    support: float
    confidence: float
    lift: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "antecedent": list(self.antecedent),
            "consequent": list(self.consequent),
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
        }


class PatternMiner:
    """Mine frequent and sequential patterns from discretized time series."""

    def __init__(
        self,
        data: Sequence[Sequence[Any]] | None,
        config: PatternMiningConfig,
        discretizer: Callable[[Sequence[Sequence[Any]]], list[tuple[str, ...]]] | None = None,
    ) -> None:
        self.config = config
        self._rng = random.Random(config.random_seed)
        self.discretizer = discretizer
        self.transactions: list[tuple[str, ...]] = (
            discretizer(data) if discretizer and data is not None else []
        )

    # --- Public API -----------------------------------------------------

    def discretize_data(
        self, data: Sequence[Sequence[Any]], buckets: int = 3
    ) -> list[tuple[str, ...]]:
        """Discretize continuous features into categorical tokens.

        - If pandas is available and input is a DataFrame, each column is binned into
          quantiles (equal frequency) and encoded as "<col>=q<n>".
        - If input is a sequence of mappings, each mapping's key/value is stringified.
        - If input is already a sequence of sequences, items are coerced to strings.
        """
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            pd = None

        transactions: list[tuple[str, ...]] = []
        if pd is not None and isinstance(data, pd.DataFrame):
            df = data
            quantiles = [i / buckets for i in range(1, buckets)]
            bins = {col: df[col].quantile(quantiles).to_list() for col in df.columns}
            for _, row in df.iterrows():
                tokens: list[str] = []
                for col in df.columns:
                    thresholds = bins[col]
                    val = row[col]
                    bucket_idx = sum(val > t for t in thresholds)
                    tokens.append(f"{col}=q{bucket_idx}")
                transactions.append(tuple(tokens))
            return transactions

        for row in data:
            if isinstance(row, dict):
                tokens = [f"{k}={row[k]}" for k in sorted(row.keys())]
            else:
                tokens = [str(item) for item in row]
            transactions.append(tuple(tokens))
        return transactions

    def extract_frequent_patterns(self) -> list[DiscoveredPattern]:
        """Run a lightweight Apriori-like mining on discretized transactions."""
        if not self.transactions:
            return []

        min_support_count = math.ceil(self.config.min_support * len(self.transactions))
        frequent_patterns: list[DiscoveredPattern] = []

        # 1-item candidates
        item_counts: dict[tuple[str, ...], int] = {}
        for txn in self.transactions:
            for item in set(txn):
                item_counts[(item,)] = item_counts.get((item,), 0) + 1

        current_level = {
            itemset: count
            for itemset, count in item_counts.items()
            if count >= min_support_count
        }

        level = 1
        while current_level and level <= self.config.max_pattern_length:
            for itemset, count in current_level.items():
                support = count / len(self.transactions)
                frequent_patterns.append(
                    DiscoveredPattern(pattern=itemset, support=support, confidence=None, lift=None)
                )

            level += 1
            next_level_candidates: dict[tuple[str, ...], int] = {}
            itemsets = list(current_level.keys())
            for i in range(len(itemsets)):
                for j in range(i + 1, len(itemsets)):
                    union = tuple(sorted(set(itemsets[i]).union(itemsets[j])))
                    if len(union) != level:
                        continue
                    # prune if any subset not frequent
                    subsets = combinations(union, level - 1)
                    if any(tuple(sorted(sub)) not in current_level for sub in subsets):
                        continue
                    next_level_candidates[union] = 0

            # count candidates
            for txn in self.transactions:
                txn_set = set(txn)
                for candidate in list(next_level_candidates.keys()):
                    if set(candidate).issubset(txn_set):
                        next_level_candidates[candidate] += 1

            current_level = {
                itemset: count
                for itemset, count in next_level_candidates.items()
                if count >= min_support_count
            }

        return frequent_patterns

    def mine_sequences(self) -> list[DiscoveredPattern]:
        """Find simple sequential patterns across time using sliding windows."""
        if not self.transactions or self.config.max_pattern_length < 2:
            return []

        window = self.config.time_window_bars or self.config.max_pattern_length
        sequences: dict[tuple[str, ...], int] = {}
        total_windows = 0
        for idx in range(len(self.transactions) - 1):
            max_len = min(window, self.config.max_pattern_length)
            for length in range(2, max_len + 1):
                if idx + length > len(self.transactions):
                    break
                seq_tokens: list[str] = []
                for step in range(length):
                    seq_tokens.extend(self.transactions[idx + step])
                key = tuple(seq_tokens)
                sequences[key] = sequences.get(key, 0) + 1
                total_windows += 1

        min_support_count = math.ceil(self.config.min_support * max(total_windows, 1))
        return [
            DiscoveredPattern(
                pattern=seq,
                support=count / max(total_windows, 1),
                confidence=None,
                lift=None,
                metadata={"type": "sequence"},
            )
            for seq, count in sequences.items()
            if count >= min_support_count
        ]

    def derive_association_rules(self, patterns: list[DiscoveredPattern]) -> list[AssociationRule]:
        """Generate association rules from frequent patterns."""
        support_map = {p.pattern: p.support for p in patterns}
        rules: list[AssociationRule] = []
        for pattern in patterns:
            if len(pattern.pattern) < 2:
                continue
            items = pattern.pattern
            for i in range(1, len(items)):
                for antecedent in combinations(items, i):
                    consequent = tuple(sorted(set(items) - set(antecedent)))
                    antecedent_support = support_map.get(tuple(sorted(antecedent)))
                    consequent_support = support_map.get(consequent)
                    if not antecedent_support or antecedent_support == 0:
                        continue
                    confidence = pattern.support / antecedent_support
                    if confidence < self.config.min_confidence:
                        continue
                    lift = (
                        confidence / consequent_support
                        if consequent_support and consequent_support > 0
                        else None
                    )
                    rules.append(
                        AssociationRule(
                            antecedent=tuple(sorted(antecedent)),
                            consequent=consequent,
                            support=pattern.support,
                            confidence=confidence,
                            lift=lift,
                        )
                    )
        return rules

    def map_pattern_to_candidate(self, pattern: DiscoveredPattern) -> StrategyCandidate:
        """Create a lightweight StrategyCandidate placeholder from a pattern."""
        return StrategyCandidate(
            candidate_id="pattern_" + "_".join(pattern.pattern),
            name="PatternStrategy",
            description=f"Strategy based on pattern {pattern.pattern}",
            tags={"pattern"},
            metrics={"support": pattern.support, "confidence": pattern.confidence or 0.0},
            metadata={"pattern": pattern.to_dict()},
        )

    # --- Utilities ------------------------------------------------------

    def mine_all(self) -> DiscoveryResult[StrategyCandidate]:
        """Convenience end-to-end mining returning a DiscoveryResult."""
        frequent = self.extract_frequent_patterns()
        sequences = self.mine_sequences()
        combined = frequent + sequences
        rules = self.derive_association_rules(frequent)
        candidates = [self.map_pattern_to_candidate(p) for p in combined]
        metadata = {
            "rules": [r.to_dict() for r in rules],
            "config": asdict(self.config),
        }
        return DiscoveryResult(
            method="pattern_mining",
            candidates=candidates,
            metadata=metadata,
        )


__all__ = [
    "AssociationRule",
    "DiscoveredPattern",
    "PatternMiningConfig",
    "PatternMiner",
]

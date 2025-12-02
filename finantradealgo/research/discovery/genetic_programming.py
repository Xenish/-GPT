"""Genetic Programming base representations and concrete engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, Sequence

from . import DiscoveryConfig, DiscoveryResult, StrategyCandidate, TStrategyCandidate


@dataclass(slots=True)
class GeneticProgrammingConfig:
    """Configuration for GP-based strategy discovery.

    These parameters guide population sizing, search depth, and convergence logic.
    They will be consumed by mutation/crossover operators and the evolution loop.
    """

    population_size: int
    n_generations: int
    crossover_rate: float
    mutation_rate: float
    max_tree_depth: int
    elitism_ratio: float = 0.0
    early_stopping_threshold: float | None = None
    random_seed: int | None = None


class GPNodeType(str, Enum):
    """Supported node categories within GP expression trees."""

    OPERATOR = "operator"
    CONDITION = "condition"
    INDICATOR = "indicator"
    CONSTANT = "constant"


@dataclass(slots=True)
class GPNode:
    """A node in the GP expression tree.

    - OPERATOR nodes typically hold boolean/arith operators (e.g., AND, OR, >, <).
    - CONDITION nodes can represent full comparison expressions.
    - INDICATOR nodes wrap feature/indicator references.
    - CONSTANT nodes store literal thresholds or numeric values.
    """

    node_type: GPNodeType
    value: Any
    children: list["GPNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        """Return True if the node has no children."""
        return not self.children

    def to_dict(self) -> dict[str, Any]:
        """Serialize the node recursively for persistence or logging."""
        return {
            "node_type": self.node_type.value,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GPNode":
        """Reconstruct a GPNode tree from a serialized payload."""
        return cls(
            node_type=GPNodeType(payload["node_type"]),
            value=payload["value"],
            children=[cls.from_dict(child) for child in payload.get("children", [])],
        )

    def to_string(self) -> str:
        """Human-readable representation for debugging/trace logging."""
        if self.is_leaf():
            return f"{self.node_type.value}:{self.value}"
        child_str = ", ".join(child.to_string() for child in self.children)
        return f"{self.node_type.value}:{self.value}({child_str})"

    def __repr__(self) -> str:  # pragma: no cover - thin wrapper over to_string
        return f"GPNode({self.to_string()})"


@dataclass(slots=True)
class GPStrategyCandidate(StrategyCandidate):
    """Strategy candidate backed by a GP expression tree."""

    root: GPNode | None = None
    fitness: float | None = None  # shadow base for clarity in GP context
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize candidate including the tree structure."""
        base = super().to_dict()
        base.update(
            {
                "root": self.root.to_dict() if self.root else None,
                "fitness": self.fitness,
                "metadata": self.metadata,
            }
        )
        return base

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GPStrategyCandidate":
        """Reconstruct a candidate from serialized data."""
        root_payload = payload.get("root")
        root = GPNode.from_dict(root_payload) if root_payload else None
        return cls(
            candidate_id=payload["candidate_id"],
            name=payload.get("name"),
            description=payload.get("description"),
            tags=set(payload.get("tags", [])),
            fitness=payload.get("fitness"),
            metrics=payload.get("metrics", {}),
            metadata=payload.get("metadata", {}),
            root=root,
        )

    def to_string(self) -> str:
        """Readable expression for logging/debugging."""
        tree_repr = self.root.to_string() if self.root else "<empty>"
        fitness_repr = f"{self.fitness:.4f}" if self.fitness is not None else "n/a"
        return f"{self.candidate_id} | fitness={fitness_repr} | {tree_repr}"

    def __repr__(self) -> str:  # pragma: no cover - thin wrapper over to_string
        return f"GPStrategyCandidate({self.to_string()})"


class GeneticProgrammingEngine:
    """Simple, extensible GP engine that delegates evaluation to project backtests."""

    def __init__(
        self,
        config: GeneticProgrammingConfig,
        evaluate_candidate: Callable[[GPStrategyCandidate], dict[str, float]],
        fitness_fn: Callable[[dict[str, float]], float] | None = None,
        operator_set: Sequence[str] | None = None,
        indicator_set: Sequence[str] | None = None,
        constant_sampler: Callable[[], float] | None = None,
    ) -> None:
        self._rng = random.Random(config.random_seed)
        self.config = config
        self.evaluate_candidate = evaluate_candidate
        self.fitness_fn = fitness_fn or (lambda metrics: metrics.get("sharpe_ratio", 0.0))
        self.operator_set = tuple(operator_set or ("and", "or", ">", "<"))
        self.indicator_set = tuple(indicator_set or ("sma_10", "sma_50", "rsi_14"))
        self.constant_sampler = constant_sampler or (lambda: self._rng.uniform(-1.0, 1.0))

    # --- Public API -----------------------------------------------------

    def initialize_population(self) -> list[GPStrategyCandidate]:
        """Randomly generate an initial population of GP trees."""
        population: list[GPStrategyCandidate] = []
        for idx in range(self.config.population_size):
            root = self._random_tree(depth=0)
            population.append(
                GPStrategyCandidate(
                    candidate_id=f"gp_{idx}",
                    name=f"gp_candidate_{idx}",
                    root=root,
                )
            )
        return population

    def evaluate_fitness(self, candidate: GPStrategyCandidate) -> float:
        """Evaluate a candidate using the project backtesting/metrics layer."""
        try:
            metrics = self.evaluate_candidate(candidate) or {}
        except Exception as exc:  # pragma: no cover - defensive guard for user callables
            metrics = {"error": str(exc)}
        candidate.metrics = metrics
        candidate.fitness = self.fitness_fn(metrics)
        return candidate.fitness or float("-inf")

    def select_parents(self, population: Sequence[GPStrategyCandidate]) -> list[GPStrategyCandidate]:
        """Tournament selection."""
        selected: list[GPStrategyCandidate] = []
        tournament_size = max(2, int(0.05 * len(population)))
        for _ in range(len(population)):
            tournament = self._rng_choice(population, tournament_size)
            winner = max(tournament, key=lambda c: c.fitness or float("-inf"))
            selected.append(winner)
        return selected

    def crossover(
        self, parent1: GPStrategyCandidate, parent2: GPStrategyCandidate
    ) -> tuple[GPStrategyCandidate, GPStrategyCandidate]:
        """Swap random subtrees between parents."""
        child1 = self._clone_candidate(parent1)
        child2 = self._clone_candidate(parent2)

        node1, parent_ref1 = self._random_node_with_parent(child1.root)
        node2, parent_ref2 = self._random_node_with_parent(child2.root)
        if node1 is None or node2 is None:
            return child1, child2

        if parent_ref1 is None:
            child1.root = node2
        else:
            parent_ref1[parent_ref1.index(node1)] = node2

        if parent_ref2 is None:
            child2.root = node1
        else:
            parent_ref2[parent_ref2.index(node2)] = node1

        return child1, child2

    def mutate(self, candidate: GPStrategyCandidate) -> GPStrategyCandidate:
        """Randomly alter a node or subtree."""
        mutated = self._clone_candidate(candidate)
        target_node, _ = self._random_node_with_parent(mutated.root)
        if target_node is None:
            return mutated

        action = self._rng_choice(["replace_subtree", "swap_operator", "tweak_constant"], 1)[0]
        if action == "replace_subtree":
            new_subtree = self._random_tree(depth=0)
            target_node.node_type = new_subtree.node_type
            target_node.value = new_subtree.value
            target_node.children = new_subtree.children
        elif action == "swap_operator" and target_node.node_type == GPNodeType.OPERATOR:
            target_node.value = self._rng_choice(self.operator_set, 1)[0]
        elif action == "tweak_constant" and target_node.node_type == GPNodeType.CONSTANT:
            target_node.value = self.constant_sampler()
        return mutated

    def evolve(self) -> DiscoveryResult[GPStrategyCandidate]:
        """Run the GP loop and return discovered strategies."""
        population = self.initialize_population()
        best_candidates: list[GPStrategyCandidate] = []

        for generation in range(self.config.n_generations):
            for candidate in population:
                if candidate.fitness is None:
                    self.evaluate_fitness(candidate)

            population.sort(key=lambda c: c.fitness or float("-inf"), reverse=True)
            best_candidates.extend(population[:2])
            best_candidates = sorted(best_candidates, key=lambda c: c.fitness or float("-inf"), reverse=True)[
                : self.config.population_size
            ]

            if (
                self.config.early_stopping_threshold is not None
                and population[0].fitness is not None
                and population[0].fitness >= self.config.early_stopping_threshold
            ):
                break

            new_population: list[GPStrategyCandidate] = []
            elite_count = int(self.config.elitism_ratio * self.config.population_size)
            new_population.extend(self._clone_candidate(c) for c in population[:elite_count])

            parents = self.select_parents(population)
            while len(new_population) < self.config.population_size:
                p1, p2 = self._rng_choice(parents, 2)
                if self._random() < self.config.crossover_rate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = self._clone_candidate(p1), self._clone_candidate(p2)

                if self._random() < self.config.mutation_rate:
                    c1 = self.mutate(c1)
                if self._random() < self.config.mutation_rate:
                    c2 = self.mutate(c2)

                new_population.extend([c1, c2])

            population = new_population[: self.config.population_size]
            for candidate in population:
                candidate.fitness = None  # force re-evaluation

        top_candidates = sorted(best_candidates, key=lambda c: c.fitness or float("-inf"), reverse=True)
        discovery = DiscoveryResult[GPStrategyCandidate](
            method="genetic_programming",
            candidates=top_candidates,
            config=DiscoveryConfig(random_seed=self.config.random_seed),
            total_candidates_evaluated=len({c.candidate_id for c in top_candidates}),
        )
        return discovery

    # --- Internal helpers ------------------------------------------------

    def _random_tree(self, depth: int) -> GPNode:
        if depth >= self.config.max_tree_depth:
            return self._random_terminal()

        if self._random() < 0.5:
            return self._random_terminal()

        op = self._rng_choice(self.operator_set, 1)[0]
        left = self._random_tree(depth + 1)
        right = self._random_tree(depth + 1)
        return GPNode(
            node_type=GPNodeType.OPERATOR,
            value=op,
            children=[left, right],
        )

    def _random_terminal(self) -> GPNode:
        if self._random() < 0.5:
            indicator = self._rng_choice(self.indicator_set, 1)[0]
            return GPNode(node_type=GPNodeType.INDICATOR, value=indicator)
        return GPNode(node_type=GPNodeType.CONSTANT, value=self.constant_sampler())

    def _random_node_with_parent(
        self, node: GPNode | None
    ) -> tuple[GPNode | None, list[GPNode] | None]:
        if node is None:
            return None, None
        nodes_with_parents: list[tuple[GPNode, list[GPNode] | None]] = []
        self._collect_nodes(node, None, nodes_with_parents)
        if not nodes_with_parents:
            return None, None
        return self._rng_choice(nodes_with_parents, 1)[0]

    def _collect_nodes(
        self, node: GPNode, parent_children: list[GPNode] | None, acc: list[tuple[GPNode, list[GPNode] | None]]
    ) -> None:
        acc.append((node, parent_children))
        for child in node.children:
            self._collect_nodes(child, node.children, acc)

    def _clone_candidate(self, candidate: GPStrategyCandidate) -> GPStrategyCandidate:
        return GPStrategyCandidate(
            candidate_id=f"{candidate.candidate_id}_clone",
            name=candidate.name,
            description=candidate.description,
            tags=set(candidate.tags),
            fitness=candidate.fitness,
            metrics=dict(candidate.metrics),
            metadata=dict(candidate.metadata),
            root=self._clone_node(candidate.root) if candidate.root else None,
        )

    def _clone_node(self, node: GPNode | None) -> GPNode | None:
        if node is None:
            return None
        return GPNode(
            node_type=node.node_type,
            value=node.value,
            children=[self._clone_node(child) for child in node.children],
        )

    def _rng_choice(self, seq: Sequence[Any], k: int) -> list[Any]:
        if len(seq) == 0:
            return []
        return self._rng.sample(seq, k) if k <= len(seq) else [self._rng.choice(seq) for _ in range(k)]

    def _random(self) -> float:
        return self._rng.random()


__all__ = [
    "GeneticProgrammingConfig",
    "GeneticProgrammingEngine",
    "GPNode",
    "GPNodeType",
    "GPStrategyCandidate",
]

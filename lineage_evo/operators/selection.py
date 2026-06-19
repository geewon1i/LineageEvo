"""Operator slots and parent selection policies."""

from __future__ import annotations

import random
from dataclasses import dataclass

from lineage_evo.config import SearchConfig
from lineage_evo.factor import FactorExpression
from lineage_evo.lineage import FactorNode, LineageDAG, OperatorType


@dataclass(frozen=True)
class OperatorSlot:
    operator: OperatorType
    slot_index: int


class OperatorSchedule:
    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()

    def slots_for_generation(self) -> list[OperatorSlot]:
        slots: list[OperatorSlot] = []
        for idx in range(self.config.mutation_per_generation):
            slots.append(OperatorSlot(OperatorType.MUTATION, idx))
        for idx in range(self.config.crossover_per_generation):
            slots.append(OperatorSlot(OperatorType.CROSSOVER, idx))
        return slots


class ParentSelector:
    """Select parents by IC strength with train-validation gap penalty."""

    def __init__(self, config: SearchConfig | None = None, rng: random.Random | None = None) -> None:
        self.config = config or SearchConfig()
        self.rng = rng or random.Random(0)

    def select_mutation_parent(self, dag: LineageDAG) -> FactorNode:
        active = self._active_nodes(dag)
        if not active:
            raise ValueError("cannot select mutation parent from empty active pool")
        lineage_id = self._select_lineage(dag.active_by_lineage(), dag)
        return self._score_weighted_choice([dag.nodes[node_id] for node_id in dag.active_by_lineage()[lineage_id]])

    def select_crossover_parents(self, dag: LineageDAG) -> tuple[FactorNode, FactorNode]:
        active = self._active_nodes(dag)
        if len(active) < 2:
            raise ValueError("crossover requires at least two active factors")
        buckets = dag.active_by_lineage()
        primary_lineage_id = self._select_lineage(buckets, dag)
        first = self._score_weighted_choice([dag.nodes[node_id] for node_id in buckets[primary_lineage_id]])
        secondary_buckets = {lineage_id: node_ids for lineage_id, node_ids in buckets.items() if lineage_id != primary_lineage_id}
        if not secondary_buckets:
            secondary_buckets = {
                lineage_id: [node_id for node_id in node_ids if node_id != first.factor_id]
                for lineage_id, node_ids in buckets.items()
            }
            secondary_buckets = {lineage_id: node_ids for lineage_id, node_ids in secondary_buckets.items() if node_ids}
        secondary_lineage_id = self._select_lineage(secondary_buckets, dag)
        candidates = [dag.nodes[node_id] for node_id in secondary_buckets[secondary_lineage_id] if node_id != first.factor_id]
        structurally_different = [node for node in candidates if self._structure_key(node.expression) != self._structure_key(first.expression)]
        if structurally_different:
            candidates = structurally_different
        return first, self._score_weighted_choice(candidates)

    @staticmethod
    def _active_nodes(dag: LineageDAG) -> list[FactorNode]:
        return [dag.nodes[node_id] for node_id in dag.active_ids if dag.nodes[node_id].is_active]

    def parent_score(self, node: FactorNode) -> float:
        if node.evaluation is None:
            return 0.0
        train = abs(node.evaluation.train_ic)
        valid = abs(node.evaluation.validation_ic)
        gap = abs(train - valid)
        return (
            self.config.parent_train_ic_weight * train
            + self.config.parent_validation_ic_weight * valid
            - self.config.parent_gap_penalty_weight * gap
        )

    def _score_weighted_choice(self, nodes: list[FactorNode]) -> FactorNode:
        scores = [self.parent_score(node) for node in nodes]
        if not scores or max(scores) == min(scores):
            return self.rng.choice(nodes)
        min_score = min(scores)
        weights = [score - min_score + self.config.parent_weight_epsilon for score in scores]
        return self.rng.choices(nodes, weights=weights, k=1)[0]

    def _select_lineage(self, buckets: dict[str, list[str]], dag: LineageDAG) -> str:
        lineage_ids = [lineage_id for lineage_id, node_ids in buckets.items() if node_ids]
        if not lineage_ids:
            raise ValueError("cannot select lineage from empty active pool")
        scores = [self._lineage_score(lineage_id, buckets[lineage_id], dag) for lineage_id in lineage_ids]
        if max(scores) == min(scores):
            return self.rng.choice(lineage_ids)
        min_score = min(scores)
        weights = [score - min_score + self.config.parent_weight_epsilon for score in scores]
        return self.rng.choices(lineage_ids, weights=weights, k=1)[0]

    def _lineage_score(self, lineage_id: str, node_ids: list[str], dag: LineageDAG) -> float:
        best = max(self.parent_score(dag.nodes[node_id]) for node_id in node_ids)
        active_total = max(1, len(dag.active_ids))
        active_ratio = len(node_ids) / active_total
        return best - self.config.lineage_concentration_weight * active_ratio

    def _non_dominant_lineage_nodes(self, nodes: list[FactorNode]) -> list[FactorNode]:
        counts: dict[str, int] = {}
        for node in nodes:
            counts[node.lineage_id] = counts.get(node.lineage_id, 0) + 1
        max_count = max(counts.values())
        dominant = {lineage_id for lineage_id, count in counts.items() if count == max_count}
        non_dominant = [node for node in nodes if node.lineage_id not in dominant]
        return non_dominant

    @staticmethod
    def _structure_key(expression: FactorExpression) -> tuple[str, ...]:
        return tuple(token for token in expression.tokens if not token.replace(".", "", 1).lstrip("-").isdigit())

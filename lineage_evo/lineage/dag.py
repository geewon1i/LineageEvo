"""Lineage DAG primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from lineage_evo.evaluation import EvaluationResult
from lineage_evo.factor import ExpressionDiff, FactorExpression


class OperatorType(StrEnum):
    MUTATION = "mutation"
    CROSSOVER = "crossover"


@dataclass(frozen=True)
class FactorNode:
    factor_id: str
    expression: FactorExpression
    lineage_id: str
    generation: int
    evaluation: EvaluationResult | None = None
    is_active: bool = True


@dataclass(frozen=True)
class EvolutionEdge:
    parent_id: str
    child_id: str
    operator: OperatorType
    role: str
    expression_diff: ExpressionDiff | None = None


@dataclass(frozen=True)
class LineageBaseState:
    """论文中的 B_L^t：只描述 lineage 本身，不承担算子级指导。"""

    lineage_id: str
    representative_factor_id: str | None
    representative_expression: str | None
    best_factor_id: str | None
    best_validation_ic: float | None
    best_validation_icir: float | None
    age: int
    size: int
    active_size: int

    def as_dict(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "representative_factor_id": self.representative_factor_id,
            "representative_expression": self.representative_expression,
            "best_factor_id": self.best_factor_id,
            "best_validation_ic": self.best_validation_ic,
            "best_validation_icir": self.best_validation_icir,
            "age": self.age,
            "size": self.size,
            "active_size": self.active_size,
        }


@dataclass
class LineageDAG:
    active_pool_size: int = 50
    elite_archive_size: int = 100
    max_active_lineage_ratio: float = 0.40
    min_active_lineages_before_cap: int = 2
    nodes: dict[str, FactorNode] = field(default_factory=dict)
    edges: list[EvolutionEdge] = field(default_factory=list)
    active_ids: list[str] = field(default_factory=list)
    elite_ids: list[str] = field(default_factory=list)
    lineage_states: dict[str, LineageBaseState] = field(default_factory=dict)

    def add_seed(self, expression: FactorExpression, evaluation: EvaluationResult, generation: int = 0) -> FactorNode:
        factor_id = self._new_id()
        node = FactorNode(
            factor_id=factor_id,
            expression=expression,
            lineage_id=factor_id,
            generation=generation,
            evaluation=evaluation,
        )
        self.nodes[factor_id] = node
        self.active_ids.append(factor_id)
        self.refresh_elite_archive()
        self.prune_active_pool()
        self._refresh_lineage_state(node.lineage_id)
        return node

    def add_mutation_child(
        self,
        parent_id: str,
        child_expression: FactorExpression,
        child_evaluation: EvaluationResult,
        generation: int,
        expression_diff: ExpressionDiff,
    ) -> FactorNode:
        parent = self.nodes[parent_id]
        child_id = self._new_id()
        child = FactorNode(
            factor_id=child_id,
            expression=child_expression,
            lineage_id=parent.lineage_id,
            generation=generation,
            evaluation=child_evaluation,
        )
        self.nodes[child_id] = child
        self.edges.append(EvolutionEdge(parent_id, child_id, OperatorType.MUTATION, "primary", expression_diff))
        self.active_ids.append(child_id)
        self.refresh_elite_archive()
        self.prune_active_pool()
        self._refresh_all_lineage_states()
        return child

    def add_crossover_child(
        self,
        parent_a_id: str,
        parent_b_id: str,
        child_expression: FactorExpression,
        child_evaluation: EvaluationResult,
        generation: int,
        expression_diff: ExpressionDiff,
    ) -> FactorNode:
        parent_a = self.nodes[parent_a_id]
        parent_b = self.nodes[parent_b_id]
        primary, secondary = self._choose_primary(parent_a, parent_b)
        child_id = self._new_id()
        child = FactorNode(
            factor_id=child_id,
            expression=child_expression,
            lineage_id=primary.lineage_id,
            generation=generation,
            evaluation=child_evaluation,
        )
        self.nodes[child_id] = child
        self.edges.append(EvolutionEdge(primary.factor_id, child_id, OperatorType.CROSSOVER, "primary", expression_diff))
        self.edges.append(EvolutionEdge(secondary.factor_id, child_id, OperatorType.CROSSOVER, "secondary", expression_diff))
        self.active_ids.append(child_id)
        self.refresh_elite_archive()
        self.prune_active_pool()
        self._refresh_all_lineage_states()
        return child

    def prune_active_pool(self) -> list[str]:
        removed: list[str] = []
        while len(self.active_ids) > self.active_pool_size:
            worst_id = self._pruning_candidate()
            self.active_ids.remove(worst_id)
            old = self.nodes[worst_id]
            self.nodes[worst_id] = FactorNode(
                factor_id=old.factor_id,
                expression=old.expression,
                lineage_id=old.lineage_id,
                generation=old.generation,
                evaluation=old.evaluation,
                is_active=False,
            )
            removed.append(worst_id)
        if removed:
            self._refresh_all_lineage_states()
        return removed

    def active_by_lineage(self) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {}
        for node_id in self.active_ids:
            node = self.nodes[node_id]
            if not node.is_active:
                continue
            buckets.setdefault(node.lineage_id, []).append(node_id)
        return buckets

    def refresh_elite_archive(self) -> None:
        evaluated_ids = [node_id for node_id, node in self.nodes.items() if node.evaluation is not None]
        ranked = sorted(evaluated_ids, key=lambda node_id: self._abs_validation_ic(node_id), reverse=True)
        self.elite_ids = ranked[: self.elite_archive_size]

    def lineage_state(self, lineage_id: str) -> LineageBaseState:
        if lineage_id not in self.lineage_states:
            self._refresh_lineage_state(lineage_id)
        return self.lineage_states[lineage_id]

    def lineage_state_dict(self, lineage_id: str) -> dict[str, object]:
        return self.lineage_state(lineage_id).as_dict()

    def _refresh_all_lineage_states(self) -> None:
        for lineage_id in {node.lineage_id for node in self.nodes.values()}:
            self._refresh_lineage_state(lineage_id)

    def _refresh_lineage_state(self, lineage_id: str) -> None:
        nodes = [node for node in self.nodes.values() if node.lineage_id == lineage_id]
        if not nodes:
            self.lineage_states.pop(lineage_id, None)
            return

        representative = min(nodes, key=lambda node: (node.generation, node.factor_id))
        evaluated = [node for node in nodes if node.evaluation is not None]
        best = max(evaluated, key=lambda node: abs(node.evaluation.validation_ic)) if evaluated else None
        active_size = sum(1 for node in nodes if node.is_active)
        self.lineage_states[lineage_id] = LineageBaseState(
            lineage_id=lineage_id,
            representative_factor_id=representative.factor_id,
            representative_expression=representative.expression.raw,
            best_factor_id=best.factor_id if best else None,
            best_validation_ic=best.evaluation.validation_ic if best and best.evaluation else None,
            best_validation_icir=best.evaluation.validation_icir if best and best.evaluation else None,
            age=max(node.generation for node in nodes),
            size=len(nodes),
            active_size=active_size,
        )

    def _pruning_candidate(self) -> str:
        buckets = self.active_by_lineage()
        if len(buckets) >= self.min_active_lineages_before_cap and self.active_ids:
            dominant_ids = [
                node_ids
                for node_ids in buckets.values()
                if len(node_ids) / len(self.active_ids) > self.max_active_lineage_ratio
            ]
            if dominant_ids:
                candidate_pool = max(dominant_ids, key=len)
                return min(candidate_pool, key=self._abs_validation_ic)
        return min(self.active_ids, key=self._abs_validation_ic)

    def _abs_validation_ic(self, node_id: str) -> float:
        evaluation = self.nodes[node_id].evaluation
        return abs(evaluation.validation_ic) if evaluation is not None else float("-inf")

    @staticmethod
    def _new_id() -> str:
        return f"f_{uuid4().hex[:12]}"

    @staticmethod
    def _choose_primary(parent_a: FactorNode, parent_b: FactorNode) -> tuple[FactorNode, FactorNode]:
        a_score = abs(parent_a.evaluation.validation_ic) if parent_a.evaluation else float("-inf")
        b_score = abs(parent_b.evaluation.validation_ic) if parent_b.evaluation else float("-inf")
        if b_score > a_score:
            return parent_b, parent_a
        return parent_a, parent_b

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


@dataclass
class LineageDAG:
    active_pool_size: int = 50
    nodes: dict[str, FactorNode] = field(default_factory=dict)
    edges: list[EvolutionEdge] = field(default_factory=list)
    active_ids: list[str] = field(default_factory=list)

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
        self.prune_active_pool()
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
        self.prune_active_pool()
        return child

    def prune_active_pool(self) -> list[str]:
        removed: list[str] = []
        while len(self.active_ids) > self.active_pool_size:
            worst_id = min(
                self.active_ids,
                key=lambda node_id: self.nodes[node_id].evaluation.validation_icir
                if self.nodes[node_id].evaluation is not None
                else float("-inf"),
            )
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
        return removed

    @staticmethod
    def _new_id() -> str:
        return f"f_{uuid4().hex[:12]}"

    @staticmethod
    def _choose_primary(parent_a: FactorNode, parent_b: FactorNode) -> tuple[FactorNode, FactorNode]:
        a_score = parent_a.evaluation.validation_icir if parent_a.evaluation else float("-inf")
        b_score = parent_b.evaluation.validation_icir if parent_b.evaluation else float("-inf")
        if b_score > a_score:
            return parent_b, parent_a
        return parent_a, parent_b


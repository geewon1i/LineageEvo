"""Types shared by prior rewriting components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from lineage_evo.evaluation import EvaluationResult, ScoreDelta
from lineage_evo.factor import ExpressionDiff, FactorExpression
from lineage_evo.lineage import OperatorType


class PriorTarget(StrEnum):
    MUTATION_LINEAGE = "mutation_lineage"
    CROSSOVER_LINEAGE = "crossover_lineage"
    GLOBAL_MUTATION = "global_mutation"
    GLOBAL_CROSSOVER = "global_crossover"


@dataclass(frozen=True)
class PriorRewriteInput:
    run_id: str
    generation: int
    operator: OperatorType
    target_prior_type: PriorTarget
    old_prior: Any
    parent_factors: list[FactorExpression]
    child_factor: FactorExpression | None
    expression_diff: ExpressionDiff | None
    train_score: EvaluationResult | None
    validation_score: EvaluationResult | None
    delta_train_score: float | None
    delta_validation_score: float | None
    validity_info: dict[str, Any]
    parent_ids: list[str] = field(default_factory=list)
    child_id: str | None = None
    lineage_id: str | None = None
    recent_lineage_statistics: dict[str, Any] | None = None

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "operator": self.operator.value,
            "target_prior_type": self.target_prior_type.value,
            "parent_factors": [factor.normalized for factor in self.parent_factors],
            "child_factor": self.child_factor.normalized if self.child_factor else None,
            "train_score": self.train_score.as_dict() if self.train_score else None,
            "validation_score": self.validation_score.as_dict() if self.validation_score else None,
            "delta_train_score": self.delta_train_score,
            "delta_validation_score": self.delta_validation_score,
            "validity_info": self.validity_info,
            "recent_lineage_statistics": self.recent_lineage_statistics,
        }


@dataclass(frozen=True)
class LLMRewriteResponse:
    raw_content: str


@dataclass(frozen=True)
class PriorRewriteResult:
    updated_prior: Any
    update_summary: str
    fields_changed: list[str]
    removed_patterns: list[str]
    warnings: list[str]
    fallback_used: bool
    schema_valid: bool
    raw_llm_output: str

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "updated_prior": self.updated_prior.model_dump(mode="json")
            if hasattr(self.updated_prior, "model_dump")
            else self.updated_prior,
            "update_summary": self.update_summary,
            "fields_changed": self.fields_changed,
            "removed_patterns": self.removed_patterns,
            "warnings": self.warnings,
            "fallback_used": self.fallback_used,
            "schema_valid": self.schema_valid,
            "raw_llm_output": self.raw_llm_output,
        }

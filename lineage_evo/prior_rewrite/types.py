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
    update_trigger: dict[str, Any] | None = None

    def evidence_dict(self, *, compact: bool = False) -> dict[str, Any]:
        child_metrics = self.train_score or self.validation_score
        statistics_key = "lineage_summary" if compact else "recent_lineage_statistics"
        return {
            "generation": self.generation,
            "operator": self.operator.value,
            "target_prior_type": self.target_prior_type.value,
            "parent_factors": [factor.normalized for factor in self.parent_factors],
            "child_factor": self.child_factor.normalized if self.child_factor else None,
            "child_metrics": child_metrics.as_dict() if child_metrics else None,
            "delta_metrics": {
                "train_ic_delta": self.delta_train_score,
                "validation_ic_delta": self.delta_validation_score,
            },
            "validity_info": self.validity_info,
            statistics_key: self._compact_lineage_summary() if compact else self.recent_lineage_statistics,
        }

    def _compact_lineage_summary(self) -> dict[str, Any] | None:
        stats = self.recent_lineage_statistics
        if not stats:
            return None
        keys = [
            "age",
            "size",
            "active_size",
            "best_validation_ic",
            "best_validation_icir",
            "lineage_trend_state",
            "lineage_trend_signal",
            "train_validation_ic_gap",
            "train_validation_icir_gap",
        ]
        summary = {key: stats.get(key) for key in keys if key in stats}
        deltas = stats.get("recent_validation_strength_deltas")
        if isinstance(deltas, list):
            summary["recent_validation_strength_deltas"] = deltas[-3:]
        return summary


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
    deterministic_updates: dict[str, Any] = field(default_factory=dict)

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
            "deterministic_updates": self.deterministic_updates,
        }

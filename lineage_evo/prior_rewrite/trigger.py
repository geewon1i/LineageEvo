"""Deterministic prior update trigger and lineage search-control state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lineage_evo.config import PriorUpdateConfig
from lineage_evo.evaluation import EvaluationResult, ScoreDelta
from lineage_evo.lineage import OperatorType


@dataclass(frozen=True)
class PriorUpdateTriggerDecision:
    should_rewrite_prior: bool
    trigger_reason: str
    operator: str
    generation: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_rewrite_prior": self.should_rewrite_prior,
            "trigger_reason": self.trigger_reason,
            "operator": self.operator,
            "generation": self.generation,
        }


class PriorUpdateTrigger:
    """Decide whether one evaluated evolutionary event is informative enough."""

    _EPSILON = 1e-12

    def __init__(self, config: PriorUpdateConfig | None = None) -> None:
        self.config = config or PriorUpdateConfig()

    def evaluate(
        self,
        *,
        parent: EvaluationResult | None,
        child: EvaluationResult | None,
        delta: ScoreDelta | None,
        validity_info: dict[str, Any],
        operator: OperatorType,
        generation: int,
    ) -> PriorUpdateTriggerDecision:
        if not validity_info.get("is_valid", True):
            return self.invalid(operator=operator, generation=generation)
        if parent is None or child is None or delta is None:
            return self._decision(False, "minor_fluctuation", operator, generation)

        parent_decision_strength = self._decision_strength(parent)
        child_decision_strength = self._decision_strength(child)
        delta_decision_strength = child_decision_strength - parent_decision_strength
        delta_train_strength = abs(child.train_ic) - abs(parent.train_ic)
        delta_validation_strength = abs(child.validation_ic) - abs(parent.validation_ic)
        improvement_threshold = self._improvement_threshold(parent_decision_strength)
        degradation_threshold = self._degradation_threshold(parent_decision_strength)
        train_improvement_threshold = self._improvement_threshold(abs(parent.train_ic))

        if delta_decision_strength >= improvement_threshold - self._EPSILON:
            reason = "significant_train_improvement" if self.config.train_only else "significant_validation_improvement"
            should = True
        elif -delta_decision_strength >= degradation_threshold - self._EPSILON:
            reason = "significant_train_degradation" if self.config.train_only else "significant_validation_degradation"
            should = True
        elif (
            not self.config.train_only
            and delta_train_strength >= train_improvement_threshold - self._EPSILON
            and delta_validation_strength <= self._EPSILON
        ):
            reason = "potential_overfitting"
            should = True
        else:
            reason = "minor_fluctuation"
            should = False

        return PriorUpdateTriggerDecision(
            should_rewrite_prior=should,
            trigger_reason=reason,
            operator=operator.value,
            generation=generation,
        )

    def invalid(self, *, operator: OperatorType, generation: int) -> PriorUpdateTriggerDecision:
        return self._decision(False, "invalid_candidate_logged_only", operator, generation)

    def _decision(
        self,
        should: bool,
        reason: str,
        operator: OperatorType,
        generation: int,
    ) -> PriorUpdateTriggerDecision:
        return PriorUpdateTriggerDecision(
            should_rewrite_prior=should,
            trigger_reason=reason,
            operator=operator.value,
            generation=generation,
        )

    def _improvement_threshold(self, parent_strength: float) -> float:
        return max(self.config.improvement_abs_floor, self.config.improvement_ratio * parent_strength)

    def _degradation_threshold(self, parent_strength: float) -> float:
        return max(self.config.degradation_abs_floor, self.config.degradation_ratio * parent_strength)

    def _decision_strength(self, result: EvaluationResult) -> float:
        value = result.train_ic if self.config.train_only else result.validation_ic
        return abs(value)


@dataclass(frozen=True)
class LineageControlStateDecision:
    quality_trend: str
    stagnation_state: str
    mutation_strength: str
    reason: str
    trend_signal: float
    trend_state: str
    non_improving_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "quality_trend": self.quality_trend,
            "stagnation_state": self.stagnation_state,
            "mutation_strength": self.mutation_strength,
            "reason": self.reason,
            "trend_signal": self.trend_signal,
            "trend_state": self.trend_state,
            "non_improving_count": self.non_improving_count,
        }


class LineageControlStateController:
    """Deterministically set mutation search-control fields from lineage trend."""

    def __init__(self, config: PriorUpdateConfig | None = None) -> None:
        self.config = config or PriorUpdateConfig()

    def decide(self, prior_data: dict[str, Any], lineage_statistics: dict[str, Any] | None) -> LineageControlStateDecision:
        stats = lineage_statistics or {}
        trend_signal = float(stats.get("lineage_trend_signal", 0.0) or 0.0)
        trend_state = self._trend_state(trend_signal, str(stats.get("lineage_trend_state", "flat_or_slowing")))
        gap = float(stats.get("train_validation_ic_gap", stats.get("train_validation_icir_gap", 0.0)) or 0.0)
        bias_risk = str(prior_data.get("bias_risk", "low"))
        strength_deltas = stats.get("recent_validation_strength_deltas", [])
        has_history = isinstance(strength_deltas, list) and len(strength_deltas) > 0
        non_improving_count = self._non_improving_count(strength_deltas)
        stagnation = self._stagnation_state(trend_signal, non_improving_count, has_history)

        if trend_state == "worsening" or stagnation != "not_stagnant" or bias_risk == "high" or gap >= 0.02:
            strength = "exploratory"
            reason = "worsening/stagnant or biased lineage"
        elif trend_state == "improving" and bias_risk == "low" and gap < 0.01:
            strength = "conservative"
            reason = "improving low-risk lineage"
        else:
            strength = "moderate"
            reason = "mixed or insufficient lineage trend evidence"
        return LineageControlStateDecision(
            quality_trend=trend_state,
            stagnation_state=stagnation,
            mutation_strength=strength,
            reason=reason,
            trend_signal=trend_signal,
            trend_state=trend_state,
            non_improving_count=non_improving_count,
        )

    def _trend_state(self, trend_signal: float, fallback: str) -> str:
        if trend_signal >= self.config.trend_epsilon:
            return "improving"
        if trend_signal <= -self.config.trend_epsilon:
            return "worsening"
        return fallback if fallback in {"improving", "worsening"} else "flat_or_slowing"

    def _stagnation_state(self, trend_signal: float, non_improving_count: int, has_history: bool) -> str:
        if not has_history:
            return "not_stagnant"
        if non_improving_count >= 4 or trend_signal <= -2.0 * self.config.trend_epsilon:
            return "severe_stagnation"
        if non_improving_count >= 2 or abs(trend_signal) < self.config.trend_epsilon:
            return "mild_stagnation"
        return "not_stagnant"

    def _non_improving_count(self, values: Any) -> int:
        if not isinstance(values, list):
            return 0
        count = 0
        for value in reversed(values):
            try:
                delta = float(value)
            except (TypeError, ValueError):
                break
            if delta < self.config.improvement_abs_floor:
                count += 1
            else:
                break
        return count


class MutationStrengthController(LineageControlStateController):
    """Backward-compatible name for the deterministic control-state controller."""


def ema(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    value = values[0]
    for item in values[1:]:
        value = (1.0 - alpha) * value + alpha * item
    return value

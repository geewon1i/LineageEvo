"""Local-global prior fusion for Method section 4.7."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lineage_evo.ablation import AblationContext, AblationMode
from lineage_evo.prior_rewrite.trigger import LineageControlStateController
from lineage_evo.priors import (
    CrossoverPrior,
    GlobalCrossoverPrior,
    GlobalMutationPrior,
    MutationPrior,
    render_crossover_prior,
    render_global_crossover_prior,
    render_global_mutation_prior,
    render_mutation_prior,
)


@dataclass(frozen=True)
class PriorFusionDecision:
    local_weight: float
    global_weight: float
    reason: str
    instruction: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_weight": self.local_weight,
            "global_weight": self.global_weight,
            "reason": self.reason,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class PriorFusionInput:
    ablation_context: AblationContext
    lineage_state: dict[str, Any] = field(default_factory=dict)
    parent_lineage_ids: list[str] = field(default_factory=list)
    parent_lineage_priors: list[Any] = field(default_factory=list)
    parent_lineage_states: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FusedPriorContext:
    mode: AblationMode
    operator: str
    lineage_id: str | None
    prompt_context: dict[str, Any]
    prior_updates_enabled: bool = True

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "operator": self.operator,
            "lineage_id": self.lineage_id,
            "structured_priors": self.prompt_context.get("structured_priors", {}),
            "rendered_priors": self.prompt_context.get("rendered_priors", {}),
            "fusion_decision": self.prompt_context.get("fusion_decision", {}),
            "mutation_control_state": self.prompt_context.get("mutation_control_state", {}),
            "prior_updates_enabled": self.prior_updates_enabled,
        }


class PriorFusionPolicy:
    """Render priors and expose explicit local/global trust weights."""

    def __init__(self, control_state_controller: LineageControlStateController | None = None) -> None:
        self.control_state_controller = control_state_controller or LineageControlStateController()

    def fuse(self, fusion_input: PriorFusionInput) -> FusedPriorContext:
        context = fusion_input.ablation_context
        decision = self._decision(context, fusion_input.lineage_state)
        prompt_context = {
            "mode": context.mode.value,
            "operator": context.operator,
            "fusion_decision": decision.as_dict(),
            "mutation_control_state": self._mutation_control_state(context, fusion_input.lineage_state),
            "structured_priors": self._structured_priors(context, fusion_input),
            "rendered_priors": self._rendered_priors(context, fusion_input),
            "raw_ancestral_trace": context.raw_ancestral_trace,
            "global_reflection": context.global_reflection or "",
            "shuffled_from_lineage": context.shuffled_from_lineage,
            "parent_lineage_ids": fusion_input.parent_lineage_ids,
        }
        return FusedPriorContext(
            mode=context.mode,
            operator=context.operator,
            lineage_id=context.lineage_id,
            prompt_context=prompt_context,
            prior_updates_enabled=context.prior_updates_enabled,
        )

    def _decision(self, context: AblationContext, lineage_state: dict[str, Any]) -> PriorFusionDecision:
        mode = context.mode
        if mode == AblationMode.LINEAGE_ONLY:
            return self._fixed(1.0, "lineage_only ablation uses only local lineage prior")
        if mode == AblationMode.GLOBAL_ONLY:
            return self._fixed(0.0, "global_only ablation uses only global operator prior")
        if mode == AblationMode.RAW_ANCESTRAL_TRACE:
            return self._fixed(0.0, "raw_ancestral_trace ablation disables structured priors")
        if mode == AblationMode.REEVO_STYLE_GLOBAL_REFLECTION:
            return self._fixed(0.0, "reevo_style_global_reflection ablation uses reflection text instead of structured priors")

        local_weight = 0.65
        reasons = ["default preference for same-lineage conditional evidence"]

        risk = self._risk_value(context.lineage_prior)
        if risk == "high":
            local_weight -= 0.25
            reasons.append("high lineage risk")
        elif risk == "medium":
            local_weight -= 0.10
            reasons.append("medium lineage risk")

        recent_delta = float(
            lineage_state.get(
                "lineage_trend_signal",
                lineage_state.get(
                    "recent_mean_decision_ic_strength_delta",
                    lineage_state.get("recent_mean_validation_ic_delta", 0.0),
                ),
            )
            or 0.0
        )
        if recent_delta > 0:
            local_weight += 0.10
            reasons.append("lineage validation strength improved")
        elif recent_delta < 0:
            local_weight -= 0.10
            reasons.append("lineage validation strength declined")

        gap = float(lineage_state.get("train_validation_ic_gap", lineage_state.get("train_validation_icir_gap", 0.0)) or 0.0)
        if gap >= 0.02:
            local_weight -= 0.15
            reasons.append("large train-validation IC gap")
        elif gap >= 0.01:
            local_weight -= 0.05
            reasons.append("moderate train-validation IC gap")

        local_weight = round(min(0.85, max(0.20, local_weight)), 2)
        return self._fixed(local_weight, "; ".join(reasons))

    def _mutation_control_state(self, context: AblationContext, lineage_state: dict[str, Any]) -> dict[str, Any]:
        if context.operator != "mutation" or not isinstance(context.lineage_prior, MutationPrior):
            return {}
        return self.control_state_controller.decide(
            context.lineage_prior.model_dump(mode="json"),
            lineage_state,
        ).as_dict()

    @staticmethod
    def _fixed(local_weight: float, reason: str) -> PriorFusionDecision:
        global_weight = round(1.0 - local_weight, 2)
        local_weight = round(local_weight, 2)
        if local_weight >= 0.65:
            instruction = "Prioritize lineage-specific experience, while using the global prior as a bias check."
        elif local_weight <= 0.35:
            instruction = "Rely more on the global operator prior to avoid lineage bias and recover exploration."
        else:
            instruction = "Balance lineage-specific guidance with global operator guidance."
        return PriorFusionDecision(local_weight, global_weight, reason, instruction)

    @staticmethod
    def _risk_value(prior: Any) -> str | None:
        if prior is None:
            return None
        risk = getattr(prior, "bias_risk", None) or getattr(prior, "crossover_risk", None)
        return str(risk) if risk is not None else None

    def _structured_priors(self, context: AblationContext, fusion_input: PriorFusionInput) -> dict[str, Any]:
        return {
            "lineage_prior": self._dump(context.lineage_prior),
            "secondary_lineage_prior": self._dump(context.secondary_lineage_prior),
            "parent_lineage_priors": [self._dump(prior) for prior in fusion_input.parent_lineage_priors],
            "global_prior": self._dump(context.global_prior),
        }

    def _rendered_priors(self, context: AblationContext, fusion_input: PriorFusionInput) -> dict[str, Any]:
        if context.operator == "crossover":
            parent_texts = [self._render_crossover(prior) for prior in fusion_input.parent_lineage_priors]
            return {
                "lineage_prior_text": self._render_crossover(context.lineage_prior),
                "secondary_lineage_prior_text": self._render_crossover(context.secondary_lineage_prior),
                "parent_lineage_prior_texts": parent_texts,
                "global_prior_text": self._render_global_crossover(context.global_prior),
            }
        return {
            "lineage_prior_text": self._render_mutation(context.lineage_prior),
            "global_prior_text": self._render_global_mutation(context.global_prior),
        }

    @staticmethod
    def _dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    @staticmethod
    def _render_mutation(prior: Any) -> str:
        if isinstance(prior, MutationPrior):
            return render_mutation_prior(prior)
        return "Mutation lineage experience is disabled for this ablation mode."

    @staticmethod
    def _render_crossover(prior: Any) -> str:
        if isinstance(prior, CrossoverPrior):
            return render_crossover_prior(prior)
        return "Crossover lineage experience is disabled for this ablation mode."

    @staticmethod
    def _render_global_mutation(prior: Any) -> str:
        if isinstance(prior, GlobalMutationPrior):
            return render_global_mutation_prior(prior)
        return "Global mutation experience is disabled for this ablation mode."

    @staticmethod
    def _render_global_crossover(prior: Any) -> str:
        if isinstance(prior, GlobalCrossoverPrior):
            return render_global_crossover_prior(prior)
        return "Global crossover experience is disabled for this ablation mode."

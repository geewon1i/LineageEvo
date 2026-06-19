"""Search-loop integration for evolutionary factor mining runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from lineage_evo.candidate import CandidateGenerator, CandidateRequest
from lineage_evo.config import SearchConfig
from lineage_evo.evaluation import Evaluator, ScoreDelta
from lineage_evo.factor import FactorExpression, QlibExpressionNormalizer, diff_expressions
from lineage_evo.lineage import FactorNode, LineageDAG, OperatorType
from lineage_evo.operators import OperatorSchedule, ParentSelector
from lineage_evo.ablation import AblationInput, AblationMode, AblationPolicy
from lineage_evo.prior_fusion import FusedPriorContext, PriorFusionInput, PriorFusionPolicy
from lineage_evo.prior_rewrite import (
    LLMRewriteResponse,
    LLMPriorRewriter,
    PriorManager,
    PriorRewriteInput,
    PriorRewriteResult,
    PriorTarget,
    PriorUpdateTrigger,
)
from lineage_evo.prior_rewrite.trigger import PriorUpdateTriggerDecision, ema
from lineage_evo.priors import CrossoverPrior, GlobalCrossoverPrior, GlobalMutationPrior, MutationPrior
from lineage_evo.validation import Validator


@dataclass
class PriorStores:
    mutation_by_lineage: dict[str, MutationPrior]
    crossover_by_lineage: dict[str, CrossoverPrior]
    global_mutation: GlobalMutationPrior
    global_crossover: GlobalCrossoverPrior


@dataclass
class SearchCounters:
    generated_count: int = 0
    valid_evaluated_count: int = 0
    generation_failure_count: int = 0
    validation_failure_count: int = 0
    evaluation_failure_count: int = 0
    duplicate_candidate_count: int = 0
    invalid_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateAcceptanceResult:
    status: str
    child: FactorNode | None = None
    failure_reason: str | None = None
    delta: ScoreDelta | None = None
    update_trigger: PriorUpdateTriggerDecision | None = None
    duplicate_of: str | None = None
    normalized_expression: str | None = None


class SearchEngine:
    """Coordinate generation, deterministic checks, DAG updates, and prior rewrites."""

    def __init__(
        self,
        *,
        run_id: str,
        dag: LineageDAG,
        validator: Validator,
        evaluator: Evaluator,
        prior_stores: PriorStores,
        prior_rewriter: LLMPriorRewriter,
        prior_manager: PriorManager,
        config: SearchConfig | None = None,
        candidate_generator: CandidateGenerator | None = None,
        operator_schedule: OperatorSchedule | None = None,
        parent_selector: ParentSelector | None = None,
        ablation_policy: AblationPolicy | None = None,
        prior_fusion_policy: PriorFusionPolicy | None = None,
        prior_update_trigger: PriorUpdateTrigger | None = None,
        ablation_mode: AblationMode = AblationMode.OURS_FULL,
        fusion_mode: AblationMode | None = None,
        recorder: Any | None = None,
        reporter: Any | None = None,
    ) -> None:
        self.run_id = run_id
        self.dag = dag
        self.validator = validator
        self.evaluator = evaluator
        self.prior_stores = prior_stores
        self.prior_rewriter = prior_rewriter
        self.prior_manager = prior_manager
        self.config = config or SearchConfig()
        self.candidate_generator = candidate_generator
        self.operator_schedule = operator_schedule or OperatorSchedule(self.config)
        self.parent_selector = parent_selector or ParentSelector(self.config)
        self.ablation_policy = ablation_policy or AblationPolicy()
        self.prior_fusion_policy = prior_fusion_policy or PriorFusionPolicy()
        self.prior_update_trigger = prior_update_trigger or PriorUpdateTrigger()
        self.ablation_mode = fusion_mode or ablation_mode
        self.recorder = recorder
        self.reporter = reporter
        self.counters = SearchCounters()
        self.stop_reason: str | None = None
        self.normalizer = QlibExpressionNormalizer()
        self._normalized_expression_index = self._build_normalized_expression_index()

    def run_until_target(self, target_valid_evaluations: int | None = None) -> SearchCounters:
        target = target_valid_evaluations or self.config.target_valid_evaluations
        generation = 1
        start_time = time.monotonic()
        self.stop_reason = None
        while self.counters.valid_evaluated_count < target:
            if self._runtime_exceeded(start_time):
                self.stop_reason = "max_runtime_seconds"
                if self.recorder is not None:
                    self.recorder.log_search(
                        {
                            **self.summary(),
                            "event": "search_stopped",
                            "stop_reason": self.stop_reason,
                            "elapsed_seconds": round(time.monotonic() - start_time, 3),
                        }
                    )
                break
            self.run_generation(generation)
            generation += 1
        if self.stop_reason is None:
            self.stop_reason = "target_reached" if self.counters.valid_evaluated_count >= target else "stopped"
        if self.recorder is not None:
            self.recorder.write_summary(self.summary())
            self.recorder.write_final_factor_pool(self.dag)
        return self.counters

    def run_generation(self, generation: int) -> dict[str, int]:   #单次generation
        if self.candidate_generator is None:
            raise ValueError("run_generation requires a candidate_generator")

        generation_result = {"valid_mutation": 0, "valid_crossover": 0, "generation": generation}
        for slot in self.operator_schedule.slots_for_generation():
            attempts = 0
            slot_filled = False
            retry_parent_nodes: list[FactorNode] | None = None
            retry_fused_context: FusedPriorContext | None = None
            duplicate_feedback: list[dict[str, object]] = []
            while attempts < self.config.max_attempts_per_operator_slot and not slot_filled:
                attempts += 1
                if retry_parent_nodes is not None and retry_fused_context is not None:
                    parent_nodes = retry_parent_nodes
                    fused_context = retry_fused_context
                else:
                    parent_nodes = self._select_parents(slot.operator)
                    fused_context = self._fuse_prior_context(slot.operator, parent_nodes)
                if self.reporter is not None:
                    self.reporter.attempt(
                        generation=generation,
                        operator=slot.operator.value,
                        parents=[node.expression.raw for node in parent_nodes],
                    )
                request = CandidateRequest(
                    operator=slot.operator.value,
                    parent_expressions=[node.expression for node in parent_nodes],
                    fused_prior_context=fused_context,
                    constraints={
                        "factor_prompt_length_limit": self.config.factor_prompt_length_limit,
                        "factor_length_limit": self.config.factor_length_limit,
                        "exactly_one_factor": True,
                    },
                    parent_ids=[node.factor_id for node in parent_nodes],
                    parent_metrics=[node.evaluation.as_dict() if node.evaluation is not None else {} for node in parent_nodes],
                    recent_invalid_or_failed_patterns=list(self.counters.invalid_reason_counts.keys()),
                    duplicate_feedback=duplicate_feedback,
                )

                self.counters.generated_count += 1
                try:
                    generated = self.candidate_generator.generate(request)
                except Exception as exc:  # pragma: no cover - exercised by integration style tests.
                    self.counters.generation_failure_count += 1
                    if self.reporter is not None:
                        self.reporter.failure(status="generation_failure", reason=self._format_exception(exc))
                    self._log_candidate(
                        generation,
                        slot.operator,
                        parent_nodes,
                        "",
                        "generation_failure",
                        self._format_exception(exc),
                        fused_context=fused_context,
                        duplicate_feedback=duplicate_feedback,
                    )
                    retry_parent_nodes = None
                    retry_fused_context = None
                    duplicate_feedback = []
                    continue

                if not generated.parse_result.is_success:
                    self.counters.generation_failure_count += 1
                    if self.reporter is not None:
                        self.reporter.failure(
                            status="generation_failure",
                            reason=generated.parse_result.failure_reason,
                            raw_output=generated.raw_output,
                        )
                    self._log_candidate(
                        generation,
                        slot.operator,
                        parent_nodes,
                        generated.raw_output,
                        "generation_failure",
                        generated.parse_result.failure_reason,
                        fused_context=fused_context,
                        duplicate_feedback=duplicate_feedback,
                    )
                    retry_parent_nodes = None
                    retry_fused_context = None
                    duplicate_feedback = []
                    continue

                if slot.operator == OperatorType.MUTATION:
                    acceptance = self._accept_mutation_candidate_with_status(
                        parent_nodes[0].factor_id,
                        generated.parse_result.factor,
                        generation,
                        prior_updates_enabled=fused_context.prior_updates_enabled,
                    )
                    result_key = "valid_mutation"
                else:
                    acceptance = self._accept_crossover_candidate_with_status(
                        parent_nodes[0].factor_id,
                        parent_nodes[1].factor_id,
                        generated.parse_result.factor,
                        generation,
                        prior_updates_enabled=fused_context.prior_updates_enabled,
                    )
                    result_key = "valid_crossover"

                if acceptance.child is None:
                    if self.reporter is not None:
                        self.reporter.failure(status=acceptance.status, reason=acceptance.failure_reason, raw_output=generated.raw_output)
                    self._log_candidate(
                        generation,
                        slot.operator,
                        parent_nodes,
                        generated.raw_output,
                        acceptance.status,
                        acceptance.failure_reason,
                        fused_context=fused_context,
                        update_trigger=acceptance.update_trigger,
                        duplicate_of=acceptance.duplicate_of,
                        normalized_expression=acceptance.normalized_expression,
                        duplicate_feedback=duplicate_feedback,
                    )
                    if acceptance.status == "duplicate_candidate":
                        retry_parent_nodes = parent_nodes
                        retry_fused_context = fused_context
                        duplicate_feedback = [
                            {
                                "raw_factor": generated.parse_result.factor.raw if generated.parse_result.factor is not None else "",
                                "normalized_expression": acceptance.normalized_expression,
                                "duplicate_of": acceptance.duplicate_of,
                            }
                        ]
                    else:
                        retry_parent_nodes = None
                        retry_fused_context = None
                        duplicate_feedback = []
                    continue

                self.counters.valid_evaluated_count += 1
                generation_result[result_key] += 1
                slot_filled = True
                if self.reporter is not None:
                    self.reporter.valid_factor(
                        expression=acceptance.child.expression.raw,
                        evaluation=acceptance.child.evaluation,
                        delta=acceptance.delta,
                        child_id=acceptance.child.factor_id,
                    )
                self._log_candidate(
                    generation,
                    slot.operator,
                    parent_nodes,
                    generated.raw_output,
                    "valid",
                    None,
                    child_id=acceptance.child.factor_id,
                    fused_context=fused_context,
                    update_trigger=acceptance.update_trigger,
                    duplicate_feedback=duplicate_feedback,
                )
        if self.recorder is not None:
            self.recorder.log_search({**self.summary(), **generation_result})
        if self.reporter is not None:
            self.reporter.generation_summary({**self.summary(), **generation_result})
        return generation_result

    def accept_mutation_candidate(
        self,
        parent_id: str,
        child_expression: FactorExpression,
        generation: int,
        *,
        prior_updates_enabled: bool = True,
    ) -> FactorNode | None:
        return self._accept_mutation_candidate_with_status(
            parent_id,
            child_expression,
            generation,
            prior_updates_enabled=prior_updates_enabled,
        ).child

    def accept_crossover_candidate(
        self,
        parent_a_id: str,
        parent_b_id: str,
        child_expression: FactorExpression,
        generation: int,
        *,
        prior_updates_enabled: bool = True,
    ) -> FactorNode | None:
        return self._accept_crossover_candidate_with_status(
            parent_a_id,
            parent_b_id,
            child_expression,
            generation,
            prior_updates_enabled=prior_updates_enabled,
        ).child

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generated_count": self.counters.generated_count,
            "valid_evaluated_count": self.counters.valid_evaluated_count,
            "generation_failure_count": self.counters.generation_failure_count,
            "validation_failure_count": self.counters.validation_failure_count,
            "evaluation_failure_count": self.counters.evaluation_failure_count,
            "duplicate_candidate_count": self.counters.duplicate_candidate_count,
            "stop_reason": self.stop_reason,
        }

    def _runtime_exceeded(self, start_time: float) -> bool:
        limit = self.config.max_runtime_seconds
        return limit is not None and limit > 0 and (time.monotonic() - start_time) >= limit

    def _accept_mutation_candidate_with_status(
        self,
        parent_id: str,
        child_expression: FactorExpression,
        generation: int,
        *,
        prior_updates_enabled: bool = True,
    ) -> CandidateAcceptanceResult:
        parent = self.dag.nodes[parent_id]
        duplicate = self._duplicate_check(child_expression)
        if duplicate is not None:
            normalized_expression, duplicate_of = duplicate
            self._record_duplicate_candidate()
            return CandidateAcceptanceResult(
                "duplicate_candidate",
                failure_reason="duplicate normalized expression",
                duplicate_of=duplicate_of,
                normalized_expression=normalized_expression,
            )
        validation = self.validator.validate(child_expression)
        expression_diff = diff_expressions(parent.expression, child_expression)
        if not validation.is_valid:
            self._record_validation_failure(validation.reasons)
            return CandidateAcceptanceResult(
                "validation_failure",
                failure_reason=self._join_reasons(validation.reasons),
                update_trigger=self.prior_update_trigger.invalid(operator=OperatorType.MUTATION, generation=generation),
            )

        try:
            child_eval = self.evaluator.evaluate(child_expression)
        except Exception as exc:
            self._record_evaluation_failure(exc)
            return CandidateAcceptanceResult("evaluation_failure", failure_reason=self._format_exception(exc))

        child = self.dag.add_mutation_child(parent_id, child_expression, child_eval, generation, expression_diff)
        self._index_factor_expression(child)
        self._ensure_lineage_priors(child.lineage_id)
        delta = ScoreDelta.from_results(parent.evaluation, child_eval)
        update_trigger = self.prior_update_trigger.evaluate(
            parent=parent.evaluation,
            child=child_eval,
            delta=delta,
            validity_info=validation.as_dict(),
            operator=OperatorType.MUTATION,
            generation=generation,
        )
        if prior_updates_enabled and update_trigger.should_rewrite_prior:
            self._rewrite_mutation_priors(parent, child, expression_diff, delta, validation.as_dict(), generation, update_trigger)
        else:
            self._log_prior_update_trigger(update_trigger, child, [parent_id])
        self._log_dag_event("mutation_child_added", child, [parent_id])
        return CandidateAcceptanceResult("valid", child=child, delta=delta, update_trigger=update_trigger)

    def _accept_crossover_candidate_with_status(
        self,
        parent_a_id: str,
        parent_b_id: str,
        child_expression: FactorExpression,
        generation: int,
        *,
        prior_updates_enabled: bool = True,
    ) -> CandidateAcceptanceResult:
        parent_a = self.dag.nodes[parent_a_id]
        parent_b = self.dag.nodes[parent_b_id]
        duplicate = self._duplicate_check(child_expression)
        if duplicate is not None:
            normalized_expression, duplicate_of = duplicate
            self._record_duplicate_candidate()
            return CandidateAcceptanceResult(
                "duplicate_candidate",
                failure_reason="duplicate normalized expression",
                duplicate_of=duplicate_of,
                normalized_expression=normalized_expression,
            )
        validation = self.validator.validate(child_expression)
        primary_parent = self._primary_parent(parent_a, parent_b)
        expression_diff = diff_expressions(primary_parent.expression, child_expression)
        if not validation.is_valid:
            self._record_validation_failure(validation.reasons)
            return CandidateAcceptanceResult(
                "validation_failure",
                failure_reason=self._join_reasons(validation.reasons),
                update_trigger=self.prior_update_trigger.invalid(operator=OperatorType.CROSSOVER, generation=generation),
            )

        try:
            child_eval = self.evaluator.evaluate(child_expression)
        except Exception as exc:
            self._record_evaluation_failure(exc)
            return CandidateAcceptanceResult("evaluation_failure", failure_reason=self._format_exception(exc))

        child = self.dag.add_crossover_child(parent_a_id, parent_b_id, child_expression, child_eval, generation, expression_diff)
        self._index_factor_expression(child)
        self._ensure_lineage_priors(child.lineage_id)
        delta = ScoreDelta.from_results(primary_parent.evaluation, child_eval)
        update_trigger = self.prior_update_trigger.evaluate(
            parent=primary_parent.evaluation,
            child=child_eval,
            delta=delta,
            validity_info=validation.as_dict(),
            operator=OperatorType.CROSSOVER,
            generation=generation,
        )
        if prior_updates_enabled and update_trigger.should_rewrite_prior:
            self._rewrite_crossover_priors(
                primary_parent,
                parent_a,
                parent_b,
                child,
                expression_diff,
                delta,
                validation.as_dict(),
                generation,
                update_trigger,
            )
        else:
            self._log_prior_update_trigger(update_trigger, child, [parent_a_id, parent_b_id])
        self._log_dag_event("crossover_child_added", child, [parent_a_id, parent_b_id])
        return CandidateAcceptanceResult("valid", child=child, delta=delta, update_trigger=update_trigger)

    def _rewrite_mutation_priors(
        self,
        parent: FactorNode,
        child: FactorNode,
        expression_diff,
        delta: ScoreDelta,
        validity_info: dict,
        generation: int,
        update_trigger: PriorUpdateTriggerDecision,
    ) -> None:
        lineage_id = parent.lineage_id
        lineage_input = PriorRewriteInput(
            run_id=self.run_id,
            generation=generation,
            operator=OperatorType.MUTATION,
            target_prior_type=PriorTarget.MUTATION_LINEAGE,
            old_prior=self.prior_stores.mutation_by_lineage[lineage_id],
            parent_factors=[parent.expression],
            child_factor=child.expression,
            expression_diff=expression_diff,
            train_score=child.evaluation,
            validation_score=child.evaluation,
            delta_train_score=delta.train_ic_delta,
            delta_validation_score=delta.validation_ic_delta,
            validity_info=validity_info,
            parent_ids=[parent.factor_id],
            child_id=child.factor_id,
            lineage_id=lineage_id,
            recent_lineage_statistics=self._lineage_statistics(lineage_id, parent.factor_id),
            update_trigger=update_trigger.as_dict(),
        )
        result = self._safe_prior_rewrite(lineage_input, self.prior_rewriter.rewrite_mutation_prior)
        self.prior_stores.mutation_by_lineage[lineage_id] = result.updated_prior

        global_input = self._global_input_from(lineage_input, PriorTarget.GLOBAL_MUTATION, self.prior_stores.global_mutation)
        global_result = self._safe_prior_rewrite(global_input, self.prior_rewriter.rewrite_global_mutation_prior)
        self.prior_stores.global_mutation = global_result.updated_prior

    def _rewrite_crossover_priors(
        self,
        primary_parent: FactorNode,
        parent_a: FactorNode,
        parent_b: FactorNode,
        child: FactorNode,
        expression_diff,
        delta: ScoreDelta,
        validity_info: dict,
        generation: int,
        update_trigger: PriorUpdateTriggerDecision,
    ) -> None:
        lineage_id = primary_parent.lineage_id
        secondary_parent = parent_b if primary_parent.factor_id == parent_a.factor_id else parent_a
        # Only the primary lineage receives the lineage-level crossover prior update.
        lineage_input = PriorRewriteInput(
            run_id=self.run_id,
            generation=generation,
            operator=OperatorType.CROSSOVER,
            target_prior_type=PriorTarget.CROSSOVER_LINEAGE,
            old_prior=self.prior_stores.crossover_by_lineage[lineage_id],
            parent_factors=[primary_parent.expression, secondary_parent.expression],
            child_factor=child.expression,
            expression_diff=expression_diff,
            train_score=child.evaluation,
            validation_score=child.evaluation,
            delta_train_score=delta.train_ic_delta,
            delta_validation_score=delta.validation_ic_delta,
            validity_info=validity_info,
            parent_ids=[primary_parent.factor_id, secondary_parent.factor_id],
            child_id=child.factor_id,
            lineage_id=lineage_id,
            recent_lineage_statistics=self._lineage_statistics(lineage_id, primary_parent.factor_id),
            update_trigger=update_trigger.as_dict(),
        )
        result = self._safe_prior_rewrite(lineage_input, self.prior_rewriter.rewrite_crossover_prior)
        self.prior_stores.crossover_by_lineage[lineage_id] = result.updated_prior

        global_input = self._global_input_from(lineage_input, PriorTarget.GLOBAL_CROSSOVER, self.prior_stores.global_crossover)
        global_result = self._safe_prior_rewrite(global_input, self.prior_rewriter.rewrite_global_crossover_prior)
        self.prior_stores.global_crossover = global_result.updated_prior

    def _safe_prior_rewrite(
        self,
        rewrite_input: PriorRewriteInput,
        rewrite_call: Callable[[PriorRewriteInput], LLMRewriteResponse],
    ) -> PriorRewriteResult:
        try:
            raw = rewrite_call(rewrite_input)
        except Exception as exc:
            if self.reporter is not None:
                self.reporter.failure(status="prior_rewrite_failure", reason=self._format_exception(exc))
            return self.prior_manager.fallback_on_rewrite_error(rewrite_input, exc)
        return self.prior_manager.accept_rewrite(rewrite_input, raw)

    def _select_parents(self, operator: OperatorType) -> list[FactorNode]:
        if operator == OperatorType.MUTATION:
            return [self.parent_selector.select_mutation_parent(self.dag)]
        first, second = self.parent_selector.select_crossover_parents(self.dag)
        primary = self._primary_parent(first, second)
        secondary = second if primary.factor_id == first.factor_id else first
        return [primary, secondary]

    def _fuse_prior_context(self, operator: OperatorType, parent_nodes: list[FactorNode]) -> FusedPriorContext:
        primary = parent_nodes[0] if operator == OperatorType.MUTATION else self._primary_parent(parent_nodes[0], parent_nodes[1])
        lineage_id = primary.lineage_id
        if operator == OperatorType.MUTATION:
            lineage_prior = self.prior_stores.mutation_by_lineage[lineage_id]
            global_prior = self.prior_stores.global_mutation
            all_lineage_priors = self.prior_stores.mutation_by_lineage
            secondary_lineage_id = None
            secondary_lineage_prior = None
            parent_lineage_ids = [parent_nodes[0].lineage_id]
            parent_lineage_priors = [lineage_prior]
        else:
            lineage_prior = self.prior_stores.crossover_by_lineage[lineage_id]
            global_prior = self.prior_stores.global_crossover
            all_lineage_priors = self.prior_stores.crossover_by_lineage
            secondary = parent_nodes[1] if primary.factor_id == parent_nodes[0].factor_id else parent_nodes[0]
            secondary_lineage_id = secondary.lineage_id
            secondary_lineage_prior = self.prior_stores.crossover_by_lineage[secondary.lineage_id]
            parent_lineage_ids = [node.lineage_id for node in parent_nodes]
            parent_lineage_priors = [self.prior_stores.crossover_by_lineage[node.lineage_id] for node in parent_nodes]

        ablation_context = self.ablation_policy.apply(
            AblationInput(
                mode=self.ablation_mode,
                operator=operator.value,
                lineage_id=lineage_id,
                lineage_prior=lineage_prior,
                global_prior=global_prior,
                secondary_lineage_id=secondary_lineage_id,
                secondary_lineage_prior=secondary_lineage_prior,
                all_lineage_priors=all_lineage_priors,
                raw_ancestral_trace=self._raw_ancestral_trace(primary.factor_id),
            )
        )
        return self.prior_fusion_policy.fuse(
            PriorFusionInput(
                ablation_context=ablation_context,
                lineage_state=self._lineage_statistics(lineage_id, primary.factor_id),
                parent_lineage_ids=parent_lineage_ids,
                parent_lineage_priors=parent_lineage_priors,
                parent_lineage_states=[self._lineage_statistics(node.lineage_id, node.factor_id) for node in parent_nodes],
            )
        )

    def _lineage_statistics(self, lineage_id: str, representative_factor_id: str | None = None) -> dict[str, Any]:
        base_state = self.dag.lineage_state_dict(lineage_id)
        representative = self.dag.nodes.get(representative_factor_id) if representative_factor_id else None
        recent_deltas: list[float] = []
        for edge in self.dag.edges:
            if edge.role != "primary" or edge.child_id not in self.dag.nodes or edge.parent_id not in self.dag.nodes:
                continue
            child = self.dag.nodes[edge.child_id]
            parent = self.dag.nodes[edge.parent_id]
            if child.lineage_id != lineage_id or child.evaluation is None or parent.evaluation is None:
                continue
            recent_deltas.append(child.evaluation.validation_ic - parent.evaluation.validation_ic)
        recent = recent_deltas[-5:]
        gap = 0.0
        if representative is not None and representative.evaluation is not None:
            gap = abs(abs(representative.evaluation.train_ic) - abs(representative.evaluation.validation_ic))
        trend_config = self.prior_update_trigger.config
        trend_window = max(1, int(trend_config.trend_window))
        strength_deltas = self._lineage_strength_deltas(lineage_id)[-trend_window:]
        trend_signal = ema(strength_deltas, trend_config.trend_alpha)
        if trend_signal >= trend_config.trend_epsilon:
            trend_state = "improving"
        elif trend_signal <= -trend_config.trend_epsilon:
            trend_state = "worsening"
        else:
            trend_state = "flat_or_slowing"
        return {
            **base_state,
            # Keep current parent context without replacing B_L^t representative.
            "context_factor_id": representative.factor_id if representative else None,
            "context_expression": representative.expression.raw if representative else None,
            "recent_validation_ic_deltas": recent,
            "recent_mean_validation_ic_delta": sum(recent) / len(recent) if recent else 0.0,
            "recent_validation_strength_deltas": strength_deltas,
            "lineage_trend_signal": trend_signal,
            "lineage_trend_state": trend_state,
            "train_validation_ic_gap": gap,
        }

    def _lineage_strength_deltas(self, lineage_id: str) -> list[float]:
        deltas: list[float] = []
        for edge in self.dag.edges:
            if edge.role != "primary" or edge.child_id not in self.dag.nodes or edge.parent_id not in self.dag.nodes:
                continue
            child = self.dag.nodes[edge.child_id]
            parent = self.dag.nodes[edge.parent_id]
            if child.lineage_id != lineage_id or child.evaluation is None or parent.evaluation is None:
                continue
            deltas.append(abs(child.evaluation.validation_ic) - abs(parent.evaluation.validation_ic))
        return deltas

    def _raw_ancestral_trace(self, factor_id: str) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        current = factor_id
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            node = self.dag.nodes[current]
            trace.append({"factor_id": node.factor_id, "lineage_id": node.lineage_id, "expression": node.expression.raw})
            parent_edges = [edge for edge in self.dag.edges if edge.child_id == current and edge.role == "primary"]
            if not parent_edges:
                break
            current = parent_edges[0].parent_id
        return trace

    def _global_input_from(self, base: PriorRewriteInput, target: PriorTarget, old_prior) -> PriorRewriteInput:
        return PriorRewriteInput(
            run_id=base.run_id,
            generation=base.generation,
            operator=base.operator,
            target_prior_type=target,
            old_prior=old_prior,
            parent_factors=base.parent_factors,
            child_factor=base.child_factor,
            expression_diff=base.expression_diff,
            train_score=base.train_score,
            validation_score=base.validation_score,
            delta_train_score=base.delta_train_score,
            delta_validation_score=base.delta_validation_score,
            validity_info=base.validity_info,
            parent_ids=base.parent_ids,
            child_id=base.child_id,
            lineage_id=None,
            recent_lineage_statistics=base.recent_lineage_statistics,
            update_trigger=base.update_trigger,
        )

    def _record_validation_failure(self, reasons: list[str]) -> None:
        self.counters.validation_failure_count += 1
        for reason in reasons:
            self.counters.invalid_reason_counts[reason] = self.counters.invalid_reason_counts.get(reason, 0) + 1

    def _record_evaluation_failure(self, exc: Exception) -> None:
        self.counters.evaluation_failure_count += 1
        reason = self._format_exception(exc)
        self.counters.invalid_reason_counts[reason] = self.counters.invalid_reason_counts.get(reason, 0) + 1

    def _record_duplicate_candidate(self) -> None:
        self.counters.duplicate_candidate_count += 1
        reason = "duplicate normalized expression"
        self.counters.invalid_reason_counts[reason] = self.counters.invalid_reason_counts.get(reason, 0) + 1

    def _duplicate_check(self, expression: FactorExpression) -> tuple[str, str] | None:
        try:
            normalized = self.normalizer.normalize(expression)
        except Exception:
            return None
        duplicate_of = self._normalized_expression_index.get(normalized)
        if duplicate_of is None:
            return None
        return normalized, duplicate_of

    def _index_factor_expression(self, node: FactorNode) -> None:
        try:
            normalized = self.normalizer.normalize(node.expression)
        except Exception:
            return
        self._normalized_expression_index.setdefault(normalized, node.factor_id)

    def _build_normalized_expression_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for node in self.dag.nodes.values():
            try:
                normalized = self.normalizer.normalize(node.expression)
            except Exception:
                continue
            index.setdefault(normalized, node.factor_id)
        return index

    def _ensure_lineage_priors(self, lineage_id: str) -> None:
        if lineage_id not in self.prior_stores.mutation_by_lineage:
            self.prior_stores.mutation_by_lineage[lineage_id] = self.prior_stores.mutation_by_lineage[next(iter(self.prior_stores.mutation_by_lineage))]
        if lineage_id not in self.prior_stores.crossover_by_lineage:
            self.prior_stores.crossover_by_lineage[lineage_id] = self.prior_stores.crossover_by_lineage[next(iter(self.prior_stores.crossover_by_lineage))]

    def _log_candidate(
        self,
        generation: int,
        operator: OperatorType,
        parents: list[FactorNode],
        raw_output: str,
        status: str,
        failure_reason: str | None,
        child_id: str | None = None,
        fused_context: FusedPriorContext | None = None,
        update_trigger: PriorUpdateTriggerDecision | None = None,
        duplicate_of: str | None = None,
        normalized_expression: str | None = None,
        duplicate_feedback: list[dict[str, object]] | None = None,
    ) -> None:
        if self.recorder is None:
            return
        self.recorder.log_candidate(
            {
                "run_id": self.run_id,
                "generation": generation,
                "operator": operator.value,
                "parent_ids": [parent.factor_id for parent in parents],
                "child_id": child_id,
                "status": status,
                "failure_reason": failure_reason,
                "raw_output": raw_output,
                "duplicate_of": duplicate_of,
                "normalized_expression": normalized_expression,
                "duplicate_feedback": duplicate_feedback or [],
                "prior_context": fused_context.as_log_dict() if fused_context is not None else None,
                "prior_update_trigger": update_trigger.as_dict() if update_trigger is not None else None,
            }
        )

    def _log_prior_update_trigger(self, decision: PriorUpdateTriggerDecision, child: FactorNode, parent_ids: list[str]) -> None:
        if self.recorder is None:
            return
        self.recorder.log_search(
            {
                "run_id": self.run_id,
                "event": "prior_update_trigger",
                "child_id": child.factor_id,
                "parent_ids": parent_ids,
                "lineage_id": child.lineage_id,
                **decision.as_dict(),
            }
        )

    def _log_dag_event(self, event: str, child: FactorNode, parent_ids: list[str]) -> None:
        if self.recorder is None:
            return
        self.recorder.log_dag_event(
            {
                "run_id": self.run_id,
                "event": event,
                "child_id": child.factor_id,
                "parent_ids": parent_ids,
                "lineage_id": child.lineage_id,
                "generation": child.generation,
                "expression": child.expression.raw,
            }
        )

    @staticmethod
    def _join_reasons(reasons: list[str]) -> str:
        return "; ".join(reasons) if reasons else "validation failed"

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        message = str(exc)
        return f"{type(exc).__name__}: {message}" if message else type(exc).__name__

    @staticmethod
    def _primary_parent(parent_a: FactorNode, parent_b: FactorNode) -> FactorNode:
        a_score = abs(parent_a.evaluation.validation_ic) if parent_a.evaluation else float("-inf")
        b_score = abs(parent_b.evaluation.validation_ic) if parent_b.evaluation else float("-inf")
        return parent_b if b_score > a_score else parent_a

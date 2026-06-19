import json

from lineage_evo.factor import FactorExpression, diff_expressions
from lineage_evo.lineage import OperatorType
from lineage_evo.prior_rewrite import LLMRewriteResponse, PriorManager, PriorManagerConfig, PriorRewriteInput, PriorTarget
from lineage_evo.priors import MutationPrior


def old_prior() -> MutationPrior:
    return MutationPrior(
        successful_mutation_patterns=[],
        failed_mutation_patterns=[],
        hint="old concise hint",
        bias_risk="low",
    )


def old_semantic_prior_dict():
    prior = old_prior()
    return {
        "successful_mutation_patterns": [],
        "failed_mutation_patterns": [],
        "hint": prior.hint,
        "bias_risk": prior.bias_risk.value,
    }


def rewrite_input(delta_train=0.03, delta_validation=-0.01):
    parent = FactorExpression("close")
    child = FactorExpression("rank(close)")
    return PriorRewriteInput(
        run_id="run",
        generation=5,
        operator=OperatorType.MUTATION,
        target_prior_type=PriorTarget.MUTATION_LINEAGE,
        old_prior=old_prior(),
        parent_factors=[parent],
        child_factor=child,
        expression_diff=diff_expressions(parent, child),
        train_score=None,
        validation_score=None,
        delta_train_score=delta_train,
        delta_validation_score=delta_validation,
        validity_info={"is_valid": True},
    )


def rewrite_input_with_stats(stats):
    inp = rewrite_input()
    return PriorRewriteInput(
        run_id=inp.run_id,
        generation=inp.generation,
        operator=inp.operator,
        target_prior_type=inp.target_prior_type,
        old_prior=inp.old_prior,
        parent_factors=inp.parent_factors,
        child_factor=inp.child_factor,
        expression_diff=inp.expression_diff,
        train_score=inp.train_score,
        validation_score=inp.validation_score,
        delta_train_score=inp.delta_train_score,
        delta_validation_score=inp.delta_validation_score,
        validity_info=inp.validity_info,
        recent_lineage_statistics=stats,
    )


def test_evidence_dict_can_use_compact_lineage_summary_for_llm_prompt():
    inp = rewrite_input_with_stats(
        {
            "age": 3,
            "size": 8,
            "active_size": 5,
            "best_validation_ic": 0.042,
            "best_validation_icir": 0.42,
            "lineage_trend_state": "improving",
            "lineage_trend_signal": 0.031,
            "train_validation_ic_gap": 0.008,
            "recent_validation_strength_deltas": [0.01, -0.02, 0.03, 0.04],
            "representative_expression": "full log only",
        }
    )

    compact = inp.evidence_dict(compact=True)
    full = inp.evidence_dict()

    assert "lineage_summary" in compact
    assert "recent_lineage_statistics" not in compact
    assert compact["lineage_summary"]["recent_validation_strength_deltas"] == [-0.02, 0.03, 0.04]
    assert "representative_expression" not in compact["lineage_summary"]
    assert full["recent_lineage_statistics"]["representative_expression"] == "full log only"


def test_manager_fallbacks_on_malformed_json():
    manager = PriorManager()
    inp = rewrite_input()
    result = manager.accept_rewrite(inp, LLMRewriteResponse("{not-json"))
    assert result.fallback_used is True
    assert result.updated_prior == inp.old_prior
    assert result.schema_valid is False


def test_manager_downgrades_single_high_confidence_train_only_success():
    candidate = old_semantic_prior_dict()
    candidate["successful_mutation_patterns"] = [
        {
            "pattern": "rank close",
            "evidence": "train improved only",
            "confidence": "high",
            "support_count": 1,
            "last_updated_generation": 5,
        }
    ]
    manager = PriorManager()
    result = manager.accept_rewrite(rewrite_input(), LLMRewriteResponse(json.dumps(candidate)))
    accepted = result.updated_prior
    assert accepted.bias_risk == "medium"
    assert accepted.successful_mutation_patterns[0].confidence == "medium"
    assert any("train improved without validation" in warning for warning in result.warnings)


def test_manager_top_k_pruning_reports_removed_patterns():
    candidate = old_semantic_prior_dict()
    candidate["successful_mutation_patterns"] = [
        {
            "pattern": f"pattern {idx}",
            "evidence": "e",
            "confidence": "low",
            "support_count": idx,
            "last_updated_generation": idx,
        }
        for idx in range(5)
    ]
    manager = PriorManager(PriorManagerConfig(top_k_patterns=2))
    result = manager.accept_rewrite(rewrite_input(delta_validation=0.1), LLMRewriteResponse(json.dumps(candidate)))
    assert len(result.updated_prior.successful_mutation_patterns) == 2
    assert result.removed_patterns


def test_manager_default_top_k_is_five():
    candidate = old_semantic_prior_dict()
    candidate["successful_mutation_patterns"] = [
        {
            "pattern": f"pattern {idx}",
            "evidence": "e",
            "confidence": "medium",
            "support_count": idx,
            "last_updated_generation": idx,
        }
        for idx in range(7)
    ]

    result = PriorManager().accept_rewrite(rewrite_input(delta_validation=0.1), LLMRewriteResponse(json.dumps(candidate)))

    assert len(result.updated_prior.successful_mutation_patterns) == 5
    assert result.removed_patterns


def test_manager_logs_deterministic_control_state_without_storing_it_in_prior():
    candidate = old_semantic_prior_dict()
    inp = rewrite_input_with_stats(
        {
            "lineage_trend_signal": -0.03,
            "lineage_trend_state": "worsening",
            "train_validation_ic_gap": 0.025,
            "recent_validation_strength_deltas": [-0.01, -0.02, -0.03, -0.04],
        }
    )

    result = PriorManager().accept_rewrite(inp, LLMRewriteResponse(json.dumps(candidate)))

    assert result.schema_valid is True
    assert not hasattr(result.updated_prior, "quality_trend")
    assert not hasattr(result.updated_prior, "stagnation_state")
    assert not hasattr(result.updated_prior, "mutation_strength")
    control = result.deterministic_updates["search_control_state"]["accepted"]
    assert control["mutation_strength"] == "exploratory"
    assert control["quality_trend"] == "worsening"


def test_manager_rejects_mutation_semantic_output_with_control_fields():
    candidate = old_semantic_prior_dict()
    candidate["mutation_strength"] = "conservative"

    result = PriorManager().accept_rewrite(rewrite_input(), LLMRewriteResponse(json.dumps(candidate)))

    assert result.fallback_used is True
    assert result.updated_prior == old_prior()


def test_manager_truncates_overlong_hint_with_warning():
    candidate = old_semantic_prior_dict()
    candidate["hint"] = "x" * 50

    result = PriorManager(PriorManagerConfig(max_text_length=10)).accept_rewrite(
        rewrite_input(),
        LLMRewriteResponse(json.dumps(candidate)),
    )

    assert result.updated_prior.hint == "x" * 10
    assert "truncated overlong hint" in result.warnings

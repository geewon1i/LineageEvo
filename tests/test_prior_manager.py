import json

from lineage_evo.factor import FactorExpression, diff_expressions
from lineage_evo.lineage import OperatorType
from lineage_evo.prior_rewrite import LLMRewriteResponse, PriorManager, PriorManagerConfig, PriorRewriteInput, PriorTarget
from lineage_evo.priors import MutationPrior


def old_prior() -> MutationPrior:
    return MutationPrior(
        quality_trend="flat",
        successful_mutation_patterns=[],
        failed_mutation_patterns=[],
        mutation_strength="moderate",
        stagnation_state="not_stagnant",
        bias_risk="low",
    )


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


def test_manager_fallbacks_on_malformed_json():
    manager = PriorManager()
    inp = rewrite_input()
    result = manager.accept_rewrite(inp, LLMRewriteResponse("{not-json"))
    assert result.fallback_used is True
    assert result.updated_prior == inp.old_prior
    assert result.schema_valid is False


def test_manager_downgrades_single_high_confidence_train_only_success():
    candidate = old_prior().model_dump(mode="json")
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
    candidate = old_prior().model_dump(mode="json")
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


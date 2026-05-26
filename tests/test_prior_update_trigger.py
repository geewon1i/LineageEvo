from lineage_evo.config import PriorUpdateConfig
from lineage_evo.evaluation import EvaluationResult, ScoreDelta
from lineage_evo.lineage import OperatorType
from lineage_evo.prior_rewrite import PriorUpdateTrigger


def eval_result(train_icir: float, validation_icir: float) -> EvaluationResult:
    return EvaluationResult(0.0, train_icir, 0.0, validation_icir)


def decide(parent: EvaluationResult, child: EvaluationResult):
    return PriorUpdateTrigger(
        PriorUpdateConfig(
            improvement_abs_floor=0.02,
            improvement_ratio=0.20,
            degradation_abs_floor=0.05,
            degradation_ratio=0.30,
        )
    ).evaluate(
        parent=parent,
        child=child,
        delta=ScoreDelta.from_results(parent, child),
        validity_info={"is_valid": True},
        operator=OperatorType.MUTATION,
        generation=1,
    )


def test_trigger_uses_validation_strength_improvement():
    decision = decide(eval_result(0.1, -0.06), eval_result(0.1, -0.08))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "significant_validation_improvement"


def test_trigger_skips_small_degradation():
    decision = decide(eval_result(0.1, 0.06), eval_result(0.1, 0.04))

    assert decision.should_rewrite_prior is False
    assert decision.trigger_reason == "minor_fluctuation"


def test_trigger_skips_small_relative_improvement_for_strong_parent():
    decision = decide(eval_result(0.1, 0.20), eval_result(0.1, 0.22))

    assert decision.should_rewrite_prior is False
    assert decision.trigger_reason == "minor_fluctuation"


def test_trigger_significant_relative_improvement_for_strong_parent():
    decision = decide(eval_result(0.1, 0.20), eval_result(0.1, 0.24))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "significant_validation_improvement"


def test_trigger_significant_relative_degradation():
    decision = decide(eval_result(0.1, 0.20), eval_result(0.1, 0.13))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "significant_validation_degradation"


def test_trigger_potential_overfitting():
    decision = decide(eval_result(0.10, 0.10), eval_result(0.13, 0.10))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "potential_overfitting"


def test_trigger_minor_fluctuation_skips_rewrite():
    decision = decide(eval_result(0.1, 0.10), eval_result(0.1, 0.105))

    assert decision.should_rewrite_prior is False
    assert decision.trigger_reason == "minor_fluctuation"


def test_invalid_candidate_is_logged_only():
    decision = PriorUpdateTrigger().invalid(operator=OperatorType.CROSSOVER, generation=2)

    assert decision.should_rewrite_prior is False
    assert decision.trigger_reason == "invalid_candidate_logged_only"


def test_trigger_log_shape_is_compact():
    decision = decide(eval_result(0.1, 0.06), eval_result(0.1, 0.08))

    assert decision.as_dict() == {
        "should_rewrite_prior": True,
        "trigger_reason": "significant_validation_improvement",
        "operator": "mutation",
        "generation": 1,
    }

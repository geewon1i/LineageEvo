import pytest

from lineage_evo.config import PriorUpdateConfig
from lineage_evo.evaluation import EvaluationResult, ScoreDelta
from lineage_evo.lineage import OperatorType
from lineage_evo.prior_rewrite import PriorUpdateTrigger


def eval_result(train_ic: float, validation_ic: float) -> EvaluationResult:
    return EvaluationResult(train_ic, train_ic * 10, validation_ic, validation_ic * 10)


def decide(parent: EvaluationResult, child: EvaluationResult):
    return PriorUpdateTrigger(
        PriorUpdateConfig(
            improvement_abs_floor=0.003,
            improvement_ratio=0.30,
            degradation_abs_floor=0.012,
            degradation_ratio=0.60,
        )
    ).evaluate(
        parent=parent,
        child=child,
        delta=ScoreDelta.from_results(parent, child),
        validity_info={"is_valid": True},
        operator=OperatorType.MUTATION,
        generation=1,
    )


def decide_train_only(parent: EvaluationResult, child: EvaluationResult):
    return PriorUpdateTrigger(
        PriorUpdateConfig(
            train_only=True,
            improvement_abs_floor=0.003,
            improvement_ratio=0.30,
            degradation_abs_floor=0.012,
            degradation_ratio=0.60,
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
    decision = decide(eval_result(0.01, -0.010), eval_result(0.01, -0.013))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "significant_validation_improvement"


def test_score_delta_uses_absolute_ic_strength():
    parent = eval_result(-0.06, -0.06)
    stronger_child = eval_result(-0.08, -0.08)
    weaker_child = eval_result(-0.04, -0.04)

    improvement = ScoreDelta.from_results(parent, stronger_child)
    degradation = ScoreDelta.from_results(parent, weaker_child)

    assert improvement.train_ic_strength_delta == pytest.approx(0.02)
    assert improvement.validation_ic_strength_delta == pytest.approx(0.02)
    assert degradation.train_ic_strength_delta == pytest.approx(-0.02)
    assert degradation.validation_ic_strength_delta == pytest.approx(-0.02)


def test_trigger_skips_small_degradation():
    decision = decide(eval_result(0.01, 0.010), eval_result(0.01, 0.008))

    assert decision.should_rewrite_prior is False
    assert decision.trigger_reason == "minor_fluctuation"


def test_trigger_skips_small_relative_improvement_for_strong_parent():
    decision = decide(eval_result(0.01, 0.020), eval_result(0.01, 0.022))

    assert decision.should_rewrite_prior is False
    assert decision.trigger_reason == "minor_fluctuation"


def test_trigger_significant_relative_improvement_for_strong_parent():
    decision = decide(eval_result(0.01, 0.020), eval_result(0.01, 0.026))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "significant_validation_improvement"


def test_trigger_significant_relative_degradation():
    decision = decide(eval_result(0.01, 0.020), eval_result(0.01, 0.008))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "significant_validation_degradation"


def test_trigger_potential_overfitting():
    decision = decide(eval_result(0.010, 0.010), eval_result(0.013, 0.010))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "potential_overfitting"


def test_train_only_trigger_uses_train_strength_and_disables_overfitting():
    decision = decide_train_only(eval_result(0.010, 0.010), eval_result(0.013, 0.010))

    assert decision.should_rewrite_prior is True
    assert decision.trigger_reason == "significant_train_improvement"


def test_trigger_minor_fluctuation_skips_rewrite():
    decision = decide(eval_result(0.01, 0.010), eval_result(0.01, 0.011))

    assert decision.should_rewrite_prior is False
    assert decision.trigger_reason == "minor_fluctuation"


def test_invalid_candidate_is_logged_only():
    decision = PriorUpdateTrigger().invalid(operator=OperatorType.CROSSOVER, generation=2)

    assert decision.should_rewrite_prior is False
    assert decision.trigger_reason == "invalid_candidate_logged_only"


def test_trigger_log_shape_is_compact():
    decision = decide(eval_result(0.01, 0.010), eval_result(0.01, 0.013))

    assert decision.as_dict() == {
        "should_rewrite_prior": True,
        "trigger_reason": "significant_validation_improvement",
        "operator": "mutation",
        "generation": 1,
    }

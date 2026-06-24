import json

from lineage_evo.evaluation import EvaluationResult, ScoreDelta
from lineage_evo.factor import FactorExpression, diff_expressions
from lineage_evo.lineage import OperatorType
from lineage_evo.llm import LLMResponse, MockLLMClient
from lineage_evo.prior_rewrite import LLMPriorRewriter, PriorRewriteInput, PriorTarget
from lineage_evo.priors import MutationPrior


def empty_mutation_prior() -> MutationPrior:
    return MutationPrior(
        successful_mutation_patterns=[],
        failed_mutation_patterns=[],
        hint="No mutation hint yet.",
        bias_risk="low",
    )


def test_rewriter_asks_for_complete_prior_json():
    old_prior = empty_mutation_prior()
    response_json = json.dumps(
        {
            "successful_mutation_patterns": [],
            "failed_mutation_patterns": [],
            "hint": "Prefer compact ranked refinements.",
            "bias_risk": "low",
        }
    )
    client = MockLLMClient([LLMResponse(response_json)])
    rewriter = LLMPriorRewriter(client)
    parent = FactorExpression("close")
    child = FactorExpression("rank(close)")
    rewrite_input = PriorRewriteInput(
        run_id="r1",
        generation=1,
        operator=OperatorType.MUTATION,
        target_prior_type=PriorTarget.MUTATION_LINEAGE,
        old_prior=old_prior,
        parent_factors=[parent],
        child_factor=child,
        expression_diff=diff_expressions(parent, child),
        train_score=None,
        validation_score=None,
        delta_train_score=0.01,
        delta_validation_score=0.02,
        validity_info={"is_valid": True},
    )

    raw = rewriter.rewrite_mutation_prior(rewrite_input)

    assert "quality_trend" not in json.loads(raw.raw_content)
    assert json.loads(raw.raw_content)["hint"] == "Prefer compact ranked refinements."
    prompt = client.calls[0]["user_prompt"]
    assert "Rewrite the complete mutation semantic prior" in prompt
    assert "Do not output quality_trend, stagnation_state, or mutation_strength" in prompt
    assert "hint" in prompt


def test_rewriter_uses_compact_lineage_summary_in_prompt():
    client = MockLLMClient(
        [
            LLMResponse(
                json.dumps(
                    {
                        "successful_mutation_patterns": [],
                        "failed_mutation_patterns": [],
                        "hint": "Stay compact.",
                        "bias_risk": "low",
                    }
                )
            )
        ]
    )
    rewriter = LLMPriorRewriter(client)
    parent = FactorExpression("close")
    child = FactorExpression("rank(close)")
    rewrite_input = PriorRewriteInput(
        run_id="r1",
        generation=1,
        operator=OperatorType.MUTATION,
        target_prior_type=PriorTarget.MUTATION_LINEAGE,
        old_prior=empty_mutation_prior(),
        parent_factors=[parent],
        child_factor=child,
        expression_diff=diff_expressions(parent, child),
        train_score=None,
        validation_score=None,
        delta_train_score=0.01,
        delta_validation_score=0.02,
        validity_info={"is_valid": True},
        recent_lineage_statistics={
            "age": 3,
            "size": 8,
            "active_size": 5,
            "best_validation_ic": 0.042,
            "best_validation_icir": 0.42,
            "lineage_trend_state": "improving",
            "lineage_trend_signal": 0.031,
            "train_validation_ic_gap": 0.008,
            "recent_validation_strength_deltas": [0.01, -0.02, 0.03, 0.04],
            "representative_expression": "should not be in prompt",
            "context_expression": "should not be in prompt",
        },
    )

    rewriter.rewrite_mutation_prior(rewrite_input)
    prompt = client.calls[0]["user_prompt"]

    assert "lineage_summary" in prompt
    assert "recent_lineage_statistics" not in prompt
    assert "representative_expression" not in prompt
    assert "should not be in prompt" not in prompt
    assert "0.04" in prompt


def test_rewriter_exposes_absolute_ic_strength_and_strength_delta():
    client = MockLLMClient(
        [
            LLMResponse(
                json.dumps(
                    {
                        "successful_mutation_patterns": [],
                        "failed_mutation_patterns": [],
                        "hint": "Use direction-free strength.",
                        "bias_risk": "low",
                    }
                )
            )
        ]
    )
    parent_score = EvaluationResult(-0.06, -0.6, -0.05, -0.5)
    child_score = EvaluationResult(-0.08, -0.8, -0.04, -0.4)
    delta = ScoreDelta.from_results(parent_score, child_score)
    parent = FactorExpression("close")
    child = FactorExpression("rank(close)")
    rewrite_input = PriorRewriteInput(
        run_id="r1",
        generation=1,
        operator=OperatorType.MUTATION,
        target_prior_type=PriorTarget.MUTATION_LINEAGE,
        old_prior=empty_mutation_prior(),
        parent_factors=[parent],
        child_factor=child,
        expression_diff=diff_expressions(parent, child),
        train_score=child_score,
        validation_score=child_score,
        delta_train_score=delta.train_ic_strength_delta,
        delta_validation_score=delta.validation_ic_strength_delta,
        validity_info={"is_valid": True},
        parent_scores=[parent_score],
    )

    LLMPriorRewriter(client).rewrite_mutation_prior(rewrite_input)
    prompt = client.calls[0]["user_prompt"]

    assert '"train_ic_strength": 0.06' in prompt
    assert '"train_ic_strength": 0.08' in prompt
    assert '"train_ic_strength_delta": 0.020000000000000004' in prompt
    assert '"validation_ic_strength_delta": -0.010000000000000002' in prompt
    assert "abs(child_IC) - abs(parent_IC)" in prompt
    assert '"train_ic": -0.08' not in prompt

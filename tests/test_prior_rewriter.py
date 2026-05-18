import json

from lineage_evo.factor import FactorExpression, diff_expressions
from lineage_evo.lineage import OperatorType
from lineage_evo.llm import LLMResponse, MockLLMClient
from lineage_evo.prior_rewrite import LLMPriorRewriter, PriorRewriteInput, PriorTarget
from lineage_evo.priors import MutationPrior


def empty_mutation_prior() -> MutationPrior:
    return MutationPrior(
        quality_trend="unknown",
        successful_mutation_patterns=[],
        failed_mutation_patterns=[],
        mutation_strength="moderate",
        stagnation_state="not_stagnant",
        bias_risk="low",
    )


def test_rewriter_asks_for_complete_prior_json():
    old_prior = empty_mutation_prior()
    response_json = old_prior.model_dump_json()
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

    assert json.loads(raw.raw_content)["quality_trend"] == "unknown"
    prompt = client.calls[0]["user_prompt"]
    assert "Rewrite the complete mutation prior" in prompt
    assert "Output strict JSON matching this schema" in prompt

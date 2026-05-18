from lineage_evo.candidate import CandidatePromptBuilder, CandidateRequest, parse_candidate_output
from lineage_evo.factor import FactorExpression
from lineage_evo.prior_fusion import FusedPriorContext, FusionMode


def test_candidate_parse_accepts_one_factor_json():
    parsed = parse_candidate_output('{"factor": "Rank($close)", "rationale": "ok"}')
    assert parsed.is_success is True
    assert parsed.factor.raw == "Rank($close)"


def test_candidate_parse_accepts_expression_alias():
    parsed = parse_candidate_output('{"expression": "Rank($close)", "rationale": "ok"}')
    assert parsed.is_success is True
    assert parsed.factor.raw == "Rank($close)"


def test_candidate_parse_generation_failures():
    assert parse_candidate_output("").failure_reason == "empty output"
    assert parse_candidate_output("not json").failure_reason == "non-json output"
    assert parse_candidate_output('{"factors": ["a", "b"]}').failure_reason == "multiple factors output"
    assert parse_candidate_output('{"rationale": "missing"}').failure_reason == "missing factor field"


def test_candidate_prompt_includes_allowed_dsl_contract():
    _system, user_prompt = CandidatePromptBuilder().build(
        CandidateRequest(
            operator="mutation",
            parent_expressions=[FactorExpression("$close")],
            fused_prior_context=FusedPriorContext(
                mode=FusionMode.OURS_FULL,
                operator="mutation",
                lineage_id="l1",
                prompt_context={},
            ),
            constraints={"factor_length_limit": 40},
        )
    )
    assert "Allowed variables" in user_prompt
    assert "$close" in user_prompt
    assert "Rank(x)" in user_prompt
    assert "rolling_constants" in user_prompt

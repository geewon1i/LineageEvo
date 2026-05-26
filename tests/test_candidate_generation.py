from lineage_evo.candidate import CandidatePromptBuilder, CandidateRequest, parse_candidate_output
from lineage_evo.factor import FactorExpression
from lineage_evo.ablation import AblationMode
from lineage_evo.prior_fusion import FusedPriorContext


def test_candidate_parse_accepts_one_factor_json():
    parsed = parse_candidate_output('{"factor": "Rank($close)", "rationale": "ok"}')
    assert parsed.is_success is True
    assert parsed.factor.raw == "Rank($close)"


def test_candidate_parse_accepts_expression_alias():
    parsed = parse_candidate_output('{"expression": "Rank($close)", "rationale": "ok"}')
    assert parsed.is_success is True
    assert parsed.factor.raw == "Rank($close)"


def test_candidate_parse_extracts_json_from_common_llm_wrappers():
    fenced = parse_candidate_output('```json\n{"factor": "Rank($close)", "rationale": "ok"}\n```')
    with_text = parse_candidate_output('Here is one factor:\n{"factor": "Rank($open)", "rationale": "ok"}\nDone.')

    assert fenced.is_success is True
    assert fenced.factor.raw == "Rank($close)"
    assert with_text.is_success is True
    assert with_text.factor.raw == "Rank($open)"


def test_candidate_parse_generation_failures():
    assert parse_candidate_output("").failure_reason == "empty output"
    assert parse_candidate_output("not json").failure_reason == "non-json output"
    assert parse_candidate_output('{"factor": "Rank($close)"').failure_reason == "non-json output"
    assert parse_candidate_output('{"factors": ["a", "b"]}').failure_reason == "multiple factors output"
    assert parse_candidate_output('{"rationale": "missing"}').failure_reason == "missing factor field"


def test_candidate_prompt_includes_allowed_dsl_contract():
    _system, user_prompt = CandidatePromptBuilder().build(
        CandidateRequest(
            operator="mutation",
            parent_expressions=[FactorExpression("$close")],
            fused_prior_context=FusedPriorContext(
                mode=AblationMode.OURS_FULL,
                operator="mutation",
                lineage_id="l1",
                prompt_context={
                    "rendered_priors": {
                        "lineage_prior_text": "Mutation experience for this lineage:\n- Hint: improving",
                        "global_prior_text": "Global mutation experience across lineages:\n- Hint: stay diverse",
                    },
                    "structured_priors": {
                        "lineage_prior": {"raw_json_marker": "SHOULD_NOT_ENTER_PROMPT"},
                    },
                    "fusion_decision": {
                        "local_weight": 0.7,
                        "global_weight": 0.3,
                        "reason": "recent improvement",
                        "instruction": "Prioritize lineage-specific experience.",
                    },
                },
            ),
            constraints={"factor_length_limit": 40},
            duplicate_feedback=[
                {
                    "raw_factor": "Rank($close)",
                    "normalized_expression": "Rank($close)",
                    "duplicate_of": "f_seed",
                }
            ],
        )
    )
    assert "Allowed variables" in user_prompt
    assert "$close" in user_prompt
    assert "Rank(x)" in user_prompt
    assert "0.0001, 0.001, 0.01, 0.0, 1.0, 2.0" in user_prompt
    assert "TsPctChange(x, d)" in user_prompt
    assert "Mutation experience for this lineage" in user_prompt
    assert '"lineage_prior_weight_lambda": 0.7' in user_prompt
    assert "Previous duplicate candidate to avoid" in user_prompt
    assert "Rank($close)" in user_prompt
    assert "duplicate_of" in user_prompt
    assert "SHOULD_NOT_ENTER_PROMPT" not in user_prompt

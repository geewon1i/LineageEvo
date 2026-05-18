from lineage_evo.prompts import build_candidate_prompt, build_prior_rewrite_prompt, build_seed_prompt
from tests.test_search_integration import mutation_prior


def test_candidate_prompt_contains_generation_context():
    prompt = build_candidate_prompt(
        {
            "operator": "mutation",
            "parents": ["close"],
            "prior_context": {"lineage_prior": {}},
            "constraints": {"factor_length_limit": 40},
        }
    )
    assert "Generate one mutated child factor" in prompt
    assert "Current parent factor:" in prompt
    assert "close" in prompt
    assert "Maximum expression length: 40" in prompt


def test_seed_prompt_contains_dsl_and_context():
    prompt = build_seed_prompt(
        {
            "seed_index": 0,
            "existing_seed_expressions": [],
            "allowed_expression_dsl": {"features": ["$close"]},
            "constraints": {"exactly_one_factor": True},
        }
    )
    assert "Generate exactly one initial formulaic alpha factor" in prompt
    assert "$close" in prompt
    assert '"factor"' in prompt


def test_prior_rewrite_prompt_contains_schema_and_evidence():
    prompt = build_prior_rewrite_prompt(
        {
            "target_prior_type": "mutation_lineage",
            "old_prior": mutation_prior().model_dump(mode="json"),
            "new_evidence": {"operator": "mutation"},
        },
        type(mutation_prior()),
    )
    assert "Rewrite the complete mutation prior" in prompt
    assert "Evolutionary evidence" in prompt
    assert "operator" in prompt
    assert "Output strict JSON matching this schema" in prompt

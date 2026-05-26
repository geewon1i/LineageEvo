from lineage_evo.prompts import build_candidate_prompt, build_prior_rewrite_prompt, build_seed_prompt
from lineage_evo.priors import MutationPrior
from tests.test_search_integration import mutation_prior


def test_candidate_prompt_contains_generation_context():
    prompt = build_candidate_prompt(
        {
            "operator": "mutation",
            "parents": ["close"],
            "prior_context": {
                "rendered_priors": {
                    "lineage_prior_text": "Mutation experience for this lineage:\n- Recent trend: improving",
                    "global_prior_text": "Global mutation experience across lineages:\n- Hint: stay diverse",
                },
                "mutation_control_state": {
                    "quality_trend": "improving",
                    "stagnation_state": "not_stagnant",
                    "mutation_strength": "moderate",
                },
                "fusion_decision": {
                    "local_weight": 0.7,
                    "global_weight": 0.3,
                    "reason": "recent improvement",
                    "instruction": "Prioritize lineage-specific experience.",
                },
            },
            "constraints": {"factor_prompt_length_limit": 40, "factor_length_limit": 50},
            "duplicate_feedback": [{"normalized_expression": "Rank($close)", "duplicate_of": "f1"}],
        }
    )
    assert "Generate one mutated child factor" in prompt
    assert "Current parent factor:" in prompt
    assert "close" in prompt
    assert "Previous duplicate candidate to avoid" in prompt
    assert "Rank($close)" in prompt
    assert "Maximum expression length: 40" in prompt
    assert "Mutation experience for this lineage" in prompt
    assert "Program-computed mutation search-control state" in prompt
    assert "Prior fusion / gating decision" in prompt
    assert '"factor": "new mutated factor expression"' in prompt
    assert "Do not output rationale" in prompt
    assert '"prior_usage"' not in prompt
    assert '"risk_notes"' not in prompt


def test_crossover_prompt_is_primary_parent_centered():
    prompt = build_candidate_prompt(
        {
            "operator": "crossover",
            "parents": ["Rank($open)", "TsMean($close, 5)"],
            "parent_ids": ["primary", "secondary"],
            "parent_metrics": [{"validation_icir": 0.4}, {"validation_icir": 0.1}],
            "prior_context": {
                "rendered_priors": {
                    "parent_lineage_prior_texts": ["primary experience", "secondary experience"],
                    "global_prior_text": "global crossover experience",
                },
                "fusion_decision": {
                    "local_weight": 0.7,
                    "global_weight": 0.3,
                    "reason": "primary lineage is improving",
                    "instruction": "Prioritize primary lineage with global checks.",
                },
            },
            "constraints": {"factor_prompt_length_limit": 40, "factor_length_limit": 50},
            "duplicate_feedback": [{"normalized_expression": "Rank($open)", "duplicate_of": "f2"}],
        }
    )
    assert "Primary parent factor:" in prompt
    assert "Secondary parent factor:" in prompt
    assert "primary-parent-centered" in prompt
    assert "similar in length and complexity" in prompt
    assert "Previous duplicate candidate to avoid" in prompt
    assert "Rank($open)" in prompt
    assert '"factor": "new crossover factor expression"' in prompt
    assert "Do not output rationale" in prompt
    assert "primary_subtree_replaced_or_modified" not in prompt
    assert "Use hints to:" in prompt


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
            "new_evidence": {"operator": "mutation", "child_metrics": {"validation_icir": 0.1}},
            "update_trigger": {"should_rewrite_prior": True, "trigger_reason": "significant_validation_improvement"},
        },
        MutationPrior,
    )
    assert "Rewrite the complete mutation semantic prior" in prompt
    assert "Evolutionary evidence" in prompt
    assert "operator" in prompt
    assert "Update trigger" in prompt
    assert "significant_validation_improvement" in prompt
    assert "Do not output quality_trend, stagnation_state, or mutation_strength" in prompt
    assert "hint" in prompt
    assert "Each evidence string must be no longer than 240 characters" in prompt
    assert "Keep at most 5 successful mutation patterns" in prompt
    assert "Output strict JSON with this schema" in prompt


def test_prior_rewrite_prompt_keeps_update_trigger_out_of_evidence():
    prompt = build_prior_rewrite_prompt(
        {
            "target_prior_type": "mutation_lineage",
            "old_prior": mutation_prior().model_dump(mode="json"),
            "new_evidence": {
                "operator": "mutation",
                "child_metrics": {"train_icir": 0.1, "validation_icir": 0.2},
                "delta_metrics": {"train_icir_delta": 0.01, "validation_icir_delta": 0.02},
            },
            "update_trigger": {"should_rewrite_prior": True, "trigger_reason": "significant_validation_improvement"},
        },
        MutationPrior,
    )
    assert prompt.count("Update trigger:") == 1
    assert prompt.count('"should_rewrite_prior"') == 1
    assert "child_metrics" in prompt
    assert "train_score" not in prompt
    assert "validation_score" not in prompt

import pytest
from pydantic import ValidationError

from lineage_evo.priors import Confidence, GlobalCrossoverPrior, GlobalMutationPrior, MutationPrior, MutationSemanticPrior


def valid_mutation_prior_dict():
    return {
        "successful_mutation_patterns": [
            {
                "pattern": "replace raw close with rank(close)",
                "evidence": "improved validation ICIR",
                "confidence": "medium",
                "support_count": 1,
                "last_updated_generation": 3,
            }
        ],
        "failed_mutation_patterns": [],
        "hint": "This lineage prefers compact ranked price refinements.",
        "bias_risk": "low",
    }


def test_mutation_prior_accepts_complete_json():
    prior = MutationPrior.model_validate(valid_mutation_prior_dict())
    assert prior.successful_mutation_patterns[0].confidence == Confidence.MEDIUM


def test_mutation_prior_rejects_unknown_fields():
    data = valid_mutation_prior_dict()
    data["free_form_reflection"] = "try something better"
    with pytest.raises(ValidationError):
        MutationPrior.model_validate(data)


def test_mutation_prior_rejects_deterministic_control_fields():
    data = valid_mutation_prior_dict()
    data["quality_trend"] = "improving"
    with pytest.raises(ValidationError):
        MutationPrior.model_validate(data)


def test_mutation_prior_rejects_patch_style_output():
    with pytest.raises(ValidationError):
        MutationPrior.model_validate({"add": [{"pattern": "rank(close)"}]})


def test_mutation_semantic_prior_accepts_semantic_json_only():
    prior = MutationSemanticPrior.model_validate(
        {
            "successful_mutation_patterns": valid_mutation_prior_dict()["successful_mutation_patterns"],
            "failed_mutation_patterns": [],
            "hint": "Use rank-like refinements cautiously.",
            "bias_risk": "low",
        }
    )
    assert prior.bias_risk == "low"


def test_mutation_semantic_prior_rejects_control_fields():
    data = {
        "successful_mutation_patterns": [],
        "failed_mutation_patterns": [],
        "hint": "Use rank-like refinements cautiously.",
        "bias_risk": "low",
        "mutation_strength": "exploratory",
    }
    with pytest.raises(ValidationError):
        MutationSemanticPrior.model_validate(data)


def test_global_prior_uses_hint_not_old_guidance_fields():
    GlobalMutationPrior.model_validate(
        {
            "global_successful_mutation_patterns": [],
            "global_failed_mutation_patterns": [],
            "common_invalid_patterns": [],
            "hint": "Prefer compact robust mutation changes.",
            "last_updated_generation": 0,
        }
    )
    with pytest.raises(ValidationError):
        GlobalMutationPrior.model_validate(
            {
                "global_successful_mutation_patterns": [],
                "global_failed_mutation_patterns": [],
                "common_invalid_patterns": [],
                "general_mutation_guidance": "old field",
                "last_updated_generation": 0,
            }
        )
    with pytest.raises(ValidationError):
        GlobalCrossoverPrior.model_validate(
            {
                "global_transferable_patterns": [],
                "global_harmful_patterns": [],
                "global_complementarity_patterns": [],
                "general_crossover_guidance": "old field",
                "last_updated_generation": 0,
            }
        )

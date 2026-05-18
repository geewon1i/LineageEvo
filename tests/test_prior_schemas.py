import pytest
from pydantic import ValidationError

from lineage_evo.priors import Confidence, MutationPrior


def valid_mutation_prior_dict():
    return {
        "quality_trend": "validation improving slowly",
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
        "mutation_strength": "moderate",
        "stagnation_state": "not_stagnant",
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


def test_mutation_prior_rejects_patch_style_output():
    with pytest.raises(ValidationError):
        MutationPrior.model_validate({"add": [{"pattern": "rank(close)"}]})


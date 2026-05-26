from lineage_evo.priors import (
    render_crossover_prior,
    render_global_crossover_prior,
    render_global_mutation_prior,
    render_mutation_prior,
)
from tests.test_search_integration import crossover_prior, global_crossover_prior, global_mutation_prior, mutation_prior


def test_prior_renderer_outputs_compact_experience_text_without_json():
    text = render_mutation_prior(mutation_prior("improving"))
    assert "Mutation experience for this lineage" in text
    assert "Hint: improving" in text
    assert "{" not in text
    assert "}" not in text


def test_prior_renderer_handles_empty_priors_with_fallback_text():
    assert "No reliable patterns yet" in render_crossover_prior(crossover_prior())
    assert "No reliable patterns yet" in render_global_mutation_prior(global_mutation_prior())
    assert "No reliable patterns yet" in render_global_crossover_prior(global_crossover_prior())
    assert "Hint:" in render_crossover_prior(crossover_prior())

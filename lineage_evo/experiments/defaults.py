"""Default priors and seed factors."""

from __future__ import annotations

from lineage_evo.factor import FactorExpression
from lineage_evo.priors import CrossoverPrior, GlobalCrossoverPrior, GlobalMutationPrior, MutationPrior


def default_mutation_prior() -> MutationPrior:
    return MutationPrior(
        quality_trend="insufficient lineage evidence",
        successful_mutation_patterns=[],
        failed_mutation_patterns=[],
        mutation_strength="moderate",
        stagnation_state="not_stagnant",
        bias_risk="low",
    )


def default_crossover_prior() -> CrossoverPrior:
    return CrossoverPrior(
        transferable_patterns=[],
        harmful_patterns=[],
        complementarity_profile="insufficient crossover evidence",
        heritable_structures=[],
        crossover_risk="low",
    )


def default_global_mutation_prior() -> GlobalMutationPrior:
    return GlobalMutationPrior(
        global_successful_mutation_patterns=[],
        global_failed_mutation_patterns=[],
        common_invalid_patterns=[],
        general_mutation_guidance="Prefer compact, executable factors and preserve validation robustness.",
        last_updated_generation=0,
    )


def default_global_crossover_prior() -> GlobalCrossoverPrior:
    return GlobalCrossoverPrior(
        global_transferable_patterns=[],
        global_harmful_patterns=[],
        global_complementarity_patterns=[],
        general_crossover_guidance="Combine complementary parent structures without making expressions too long.",
        last_updated_generation=0,
    )


def default_seed_factors() -> list[FactorExpression]:
    return [FactorExpression("$close"), FactorExpression("$open"), FactorExpression("$volume")]

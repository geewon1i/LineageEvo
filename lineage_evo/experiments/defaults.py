"""Default priors and seed factors."""

from __future__ import annotations

from lineage_evo.factor import FactorExpression
from lineage_evo.priors import CrossoverPrior, GlobalCrossoverPrior, GlobalMutationPrior, MutationPrior


def default_mutation_prior() -> MutationPrior:
    return MutationPrior(
        successful_mutation_patterns=[],
        failed_mutation_patterns=[],
        hint="No lineage-specific mutation experience yet; prefer compact, valid exploratory changes.",
        bias_risk="low",
    )


def default_crossover_prior() -> CrossoverPrior:
    return CrossoverPrior(
        transferable_patterns=[],
        harmful_patterns=[],
        complementarity_profile="insufficient crossover evidence",
        heritable_structures=[],
        hint="No lineage-specific crossover experience yet; preserve useful primary structure and import only clearly complementary subtrees.",
        crossover_risk="low",
    )


def default_global_mutation_prior() -> GlobalMutationPrior:
    return GlobalMutationPrior(
        global_successful_mutation_patterns=[],
        global_failed_mutation_patterns=[],
        common_invalid_patterns=[],
        hint="Prefer compact, executable mutation candidates and preserve validation robustness across lineages.",
        last_updated_generation=0,
    )


def default_global_crossover_prior() -> GlobalCrossoverPrior:
    return GlobalCrossoverPrior(
        global_transferable_patterns=[],
        global_harmful_patterns=[],
        global_complementarity_patterns=[],
        hint="Combine complementary parent structures without making expressions too long or losing the primary parent's backbone.",
        last_updated_generation=0,
    )


def default_seed_factors() -> list[FactorExpression]:
    return [FactorExpression("$close"), FactorExpression("$open"), FactorExpression("$volume")]

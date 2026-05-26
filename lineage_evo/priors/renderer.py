"""Render structured priors into compact prompt guidance."""

from __future__ import annotations

from typing import Iterable

from lineage_evo.priors.schemas import CrossoverPrior, GlobalCrossoverPrior, GlobalMutationPrior, MutationPrior


def render_mutation_prior(prior: MutationPrior) -> str:
    return "\n".join(
        [
            "Mutation experience for this lineage:",
            f"- Effective mutation patterns: {_patterns(prior.successful_mutation_patterns, 'support_count')}",
            f"- Failed mutation patterns: {_patterns(prior.failed_mutation_patterns, 'fail_count')}",
            f"- Hint: {prior.hint}",
            f"- Bias warning: {prior.bias_risk.value}",
        ]
    )


def render_crossover_prior(prior: CrossoverPrior) -> str:
    return "\n".join(
        [
            "Crossover experience for this lineage:",
            f"- Transferable patterns: {_patterns(prior.transferable_patterns, 'support_count')}",
            f"- Harmful patterns: {_patterns(prior.harmful_patterns, 'fail_count')}",
            f"- Complementarity profile: {prior.complementarity_profile}",
            f"- Heritable structures: {_structures(prior.heritable_structures)}",
            f"- Hint: {prior.hint}",
            f"- Crossover risk: {prior.crossover_risk.value}",
        ]
    )


def render_global_mutation_prior(prior: GlobalMutationPrior) -> str:
    return "\n".join(
        [
            "Global mutation experience across lineages:",
            f"- Successful mutation patterns: {_patterns(prior.global_successful_mutation_patterns, 'support_count')}",
            f"- Failed mutation patterns: {_patterns(prior.global_failed_mutation_patterns, 'fail_count')}",
            f"- Common invalid patterns: {_patterns(prior.common_invalid_patterns, 'fail_count')}",
            f"- Hint: {prior.hint}",
            f"- Last updated generation: {prior.last_updated_generation}",
        ]
    )


def render_global_crossover_prior(prior: GlobalCrossoverPrior) -> str:
    return "\n".join(
        [
            "Global crossover experience across lineages:",
            f"- Transferable patterns: {_patterns(prior.global_transferable_patterns, 'support_count')}",
            f"- Harmful patterns: {_patterns(prior.global_harmful_patterns, 'fail_count')}",
            f"- Complementarity patterns: {_patterns(prior.global_complementarity_patterns, 'support_count')}",
            f"- Hint: {prior.hint}",
            f"- Last updated generation: {prior.last_updated_generation}",
        ]
    )


def _patterns(items: Iterable[object], count_name: str) -> str:
    rendered: list[str] = []
    for item in items:
        pattern = getattr(item, "pattern", "")
        evidence = getattr(item, "evidence", "")
        confidence = getattr(getattr(item, "confidence", None), "value", getattr(item, "confidence", ""))
        count = getattr(item, count_name, 0)
        generation = getattr(item, "last_updated_generation", 0)
        rendered.append(f"{pattern} ({confidence}, {count_name}={count}, gen={generation}; evidence: {evidence})")
    return "; ".join(rendered) if rendered else "No reliable patterns yet."


def _structures(items: Iterable[object]) -> str:
    rendered: list[str] = []
    for item in items:
        structure = getattr(item, "structure", "")
        evidence = getattr(item, "evidence", "")
        confidence = getattr(getattr(item, "confidence", None), "value", getattr(item, "confidence", ""))
        generation = getattr(item, "last_updated_generation", 0)
        rendered.append(f"{structure} ({confidence}, gen={generation}; evidence: {evidence})")
    return "; ".join(rendered) if rendered else "No reliable heritable structures yet."

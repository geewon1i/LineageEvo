"""LLM-assisted complete-prior rewriting."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from lineage_evo.llm import LLMClient
from lineage_evo.prompts import PRIOR_REWRITE_SYSTEM_PROMPT, build_prior_rewrite_prompt
from lineage_evo.prior_rewrite.types import LLMRewriteResponse, PriorRewriteInput, PriorTarget
from lineage_evo.priors import CrossoverPrior, GlobalCrossoverPrior, GlobalMutationPrior, MutationPrior


class LLMPriorRewriter:
    """Ask an LLM for a complete rewritten prior JSON object."""

    def __init__(self, llm_client: LLMClient, reporter=None) -> None:
        self.llm_client = llm_client
        self.reporter = reporter

    def rewrite_mutation_prior(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        self._ensure_target(rewrite_input, PriorTarget.MUTATION_LINEAGE)
        return self._rewrite(rewrite_input, MutationPrior)

    def rewrite_crossover_prior(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        self._ensure_target(rewrite_input, PriorTarget.CROSSOVER_LINEAGE)
        return self._rewrite(rewrite_input, CrossoverPrior)

    def rewrite_global_mutation_prior(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        self._ensure_target(rewrite_input, PriorTarget.GLOBAL_MUTATION)
        return self._rewrite(rewrite_input, GlobalMutationPrior)

    def rewrite_global_crossover_prior(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        self._ensure_target(rewrite_input, PriorTarget.GLOBAL_CROSSOVER)
        return self._rewrite(rewrite_input, GlobalCrossoverPrior)

    def _rewrite(self, rewrite_input: PriorRewriteInput, schema_model: Type[BaseModel]) -> LLMRewriteResponse:
        payload = {
            "target_prior_type": rewrite_input.target_prior_type.value,
            "old_prior": rewrite_input.old_prior.model_dump(mode="json")
            if hasattr(rewrite_input.old_prior, "model_dump")
            else rewrite_input.old_prior,
            "new_evidence": rewrite_input.evidence_dict(),
            "expression_diff": rewrite_input.expression_diff.as_dict() if rewrite_input.expression_diff else None,
            "parent_ids": rewrite_input.parent_ids,
            "child_id": rewrite_input.child_id,
            "lineage_id": rewrite_input.lineage_id,
        }
        user_prompt = build_prior_rewrite_prompt(payload, schema_model)
        if self.reporter is not None:
            self.reporter.llm_input("PRIOR REWRITE", PRIOR_REWRITE_SYSTEM_PROMPT, user_prompt)
        response = self.llm_client.complete(system_prompt=PRIOR_REWRITE_SYSTEM_PROMPT, user_prompt=user_prompt)
        if self.reporter is not None:
            self.reporter.llm_output("PRIOR REWRITE", response.content)
        return LLMRewriteResponse(raw_content=response.content)

    @staticmethod
    def _ensure_target(rewrite_input: PriorRewriteInput, target: PriorTarget) -> None:
        if rewrite_input.target_prior_type != target:
            raise ValueError(f"expected target {target.value}, got {rewrite_input.target_prior_type.value}")

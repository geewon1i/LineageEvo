"""Mock prior rewriter for fully offline smoke runs."""

from __future__ import annotations

from lineage_evo.prior_rewrite.types import LLMRewriteResponse, PriorRewriteInput


class MockPriorRewriter:
    """Return the old prior as a valid complete prior JSON object."""

    def __init__(self) -> None:
        self.calls: list[PriorRewriteInput] = []

    def rewrite_mutation_prior(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        return self._response(rewrite_input)

    def rewrite_crossover_prior(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        return self._response(rewrite_input)

    def rewrite_global_mutation_prior(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        return self._response(rewrite_input)

    def rewrite_global_crossover_prior(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        return self._response(rewrite_input)

    def _response(self, rewrite_input: PriorRewriteInput) -> LLMRewriteResponse:
        self.calls.append(rewrite_input)
        return LLMRewriteResponse(raw_content=rewrite_input.old_prior.model_dump_json())

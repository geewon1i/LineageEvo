"""Mock prior rewriter for fully offline smoke runs."""

from __future__ import annotations

import json

from lineage_evo.prior_rewrite.types import LLMRewriteResponse, PriorRewriteInput, PriorTarget


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
        if rewrite_input.target_prior_type == PriorTarget.MUTATION_LINEAGE:
            old = rewrite_input.old_prior
            return LLMRewriteResponse(
                raw_content=json.dumps(
                    {
                        "successful_mutation_patterns": [
                            item.model_dump(mode="json") for item in old.successful_mutation_patterns
                        ],
                        "failed_mutation_patterns": [item.model_dump(mode="json") for item in old.failed_mutation_patterns],
                        "hint": old.hint,
                        "bias_risk": old.bias_risk.value if hasattr(old.bias_risk, "value") else old.bias_risk,
                    },
                    ensure_ascii=False,
                )
            )
        return LLMRewriteResponse(raw_content=rewrite_input.old_prior.model_dump_json())

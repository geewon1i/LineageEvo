"""Initial seed factor generation."""

from __future__ import annotations

from dataclasses import dataclass

from lineage_evo.candidate import CandidateGenerationResult, CandidateParseResult, parse_candidate_output
from lineage_evo.factor.dsl import DEFAULT_FACTOR_DSL, FactorDSL
from lineage_evo.llm import LLMClient
from lineage_evo.prompts import SEED_SYSTEM_PROMPT, build_seed_prompt


@dataclass(frozen=True)
class SeedRequest:
    seed_index: int
    existing_seed_expressions: list[str]
    constraints: dict[str, object]


class SeedPromptBuilder:
    def __init__(self, dsl: FactorDSL | None = None) -> None:
        self.dsl = dsl or DEFAULT_FACTOR_DSL

    def build(self, request: SeedRequest) -> tuple[str, str]:
        payload = {
            "seed_index": request.seed_index,
            "existing_seed_expressions": request.existing_seed_expressions,
            "allowed_expression_dsl": self.dsl.as_prompt_context(),
            "constraints": request.constraints,
        }
        return SEED_SYSTEM_PROMPT, build_seed_prompt(payload)


class LLMSeedGenerator:
    def __init__(self, llm_client: LLMClient, prompt_builder: SeedPromptBuilder | None = None, reporter=None) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or SeedPromptBuilder()
        self.reporter = reporter

    def generate(self, request: SeedRequest) -> CandidateGenerationResult:
        system_prompt, user_prompt = self.prompt_builder.build(request)
        if self.reporter is not None:
            self.reporter.llm_input("SEED", system_prompt, user_prompt)
        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        if self.reporter is not None:
            self.reporter.llm_output("SEED", response.content)
        return CandidateGenerationResult(parse_result=parse_candidate_output(response.content), raw_output=response.content)


class MockSeedGenerator:
    def __init__(self, outputs: list[str | CandidateGenerationResult]) -> None:
        self.outputs = list(outputs)
        self.requests: list[SeedRequest] = []

    def generate(self, request: SeedRequest) -> CandidateGenerationResult:
        self.requests.append(request)
        if not self.outputs:
            return CandidateGenerationResult(CandidateParseResult(False, failure_reason="empty output"), "")
        output = self.outputs.pop(0)
        if isinstance(output, CandidateGenerationResult):
            return output
        return CandidateGenerationResult(parse_candidate_output(output), output)


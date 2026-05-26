"""Candidate factor generation contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from lineage_evo.factor import FactorExpression
from lineage_evo.factor.dsl import DEFAULT_FACTOR_DSL, FactorDSL
from lineage_evo.llm import LLMClient
from lineage_evo.prompts import CANDIDATE_SYSTEM_PROMPT, build_candidate_prompt
from lineage_evo.prior_fusion import FusedPriorContext


@dataclass(frozen=True)
class CandidateRequest:
    operator: str
    parent_expressions: list[FactorExpression]
    fused_prior_context: FusedPriorContext
    constraints: dict[str, object]
    parent_ids: list[str] = field(default_factory=list)
    parent_metrics: list[dict[str, object]] = field(default_factory=list)
    recent_invalid_or_failed_patterns: list[str] = field(default_factory=list)
    duplicate_feedback: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateParseResult:
    is_success: bool
    factor: FactorExpression | None = None
    rationale: str = ""
    failure_reason: str | None = None


@dataclass(frozen=True)
class CandidateGenerationResult:
    parse_result: CandidateParseResult
    raw_output: str


class CandidateGenerator(Protocol):
    def generate(self, request: CandidateRequest) -> CandidateGenerationResult:
        ...


class CandidatePromptBuilder:
    def __init__(self, dsl: FactorDSL | None = None) -> None:
        self.dsl = dsl or DEFAULT_FACTOR_DSL

    def build(self, request: CandidateRequest) -> tuple[str, str]:
        user_payload = {
            "operator": request.operator,
            "parents": [expr.normalized for expr in request.parent_expressions],
            "prior_context": self._prompt_prior_context(request.fused_prior_context),
            "allowed_expression_dsl": self.dsl.as_prompt_context(),
            "constraints": request.constraints,
            "parent_ids": request.parent_ids,
            "parent_metrics": request.parent_metrics,
            "recent_invalid_or_failed_patterns": request.recent_invalid_or_failed_patterns,
            "duplicate_feedback": request.duplicate_feedback,
        }
        return CANDIDATE_SYSTEM_PROMPT, build_candidate_prompt(user_payload)

    @staticmethod
    def _prompt_prior_context(fused_context: FusedPriorContext) -> dict[str, object]:
        context = fused_context.prompt_context
        return {
            "mode": context.get("mode"),
            "operator": context.get("operator"),
            "rendered_priors": context.get("rendered_priors", {}),
            "fusion_decision": context.get("fusion_decision", {}),
            "mutation_control_state": context.get("mutation_control_state", {}),
            "prior_updates_enabled": fused_context.prior_updates_enabled,
        }


class LLMCandidateGenerator:
    def __init__(self, llm_client: LLMClient, prompt_builder: CandidatePromptBuilder | None = None, reporter=None) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or CandidatePromptBuilder()
        self.reporter = reporter

    def generate(self, request: CandidateRequest) -> CandidateGenerationResult:
        system_prompt, user_prompt = self.prompt_builder.build(request)
        if self.reporter is not None:
            self.reporter.llm_input("CANDIDATE", system_prompt, user_prompt)
        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        if self.reporter is not None:
            self.reporter.llm_output("CANDIDATE", response.content)
        return CandidateGenerationResult(
            parse_result=parse_candidate_output(response.content),
            raw_output=response.content,
        )


class MockCandidateGenerator:
    """Queue-backed generator for smoke runs and tests."""

    def __init__(self, outputs: list[str | CandidateGenerationResult]) -> None:
        self.outputs = list(outputs)
        self.requests: list[CandidateRequest] = []

    def generate(self, request: CandidateRequest) -> CandidateGenerationResult:
        self.requests.append(request)
        if not self.outputs:
            return CandidateGenerationResult(
                parse_result=CandidateParseResult(False, failure_reason="empty output"),
                raw_output="",
            )
        output = self.outputs.pop(0)
        if isinstance(output, CandidateGenerationResult):
            return output
        return CandidateGenerationResult(
            parse_result=parse_candidate_output(output),
            raw_output=output,
        )


def parse_candidate_output(raw_output: str) -> CandidateParseResult:
    # Only parse LLM output shape here; expression validity is checked later.
    if not raw_output or not raw_output.strip():
        return CandidateParseResult(False, failure_reason="empty output")
    candidate_json = _extract_json_object(_strip_markdown_fence(raw_output.strip()))
    try:
        payload = json.loads(candidate_json)
    except json.JSONDecodeError:
        return CandidateParseResult(False, failure_reason="non-json output")
    if not isinstance(payload, dict):
        return CandidateParseResult(False, failure_reason="output must be a JSON object")
    if "factors" in payload or isinstance(payload.get("factor"), list):
        return CandidateParseResult(False, failure_reason="multiple factors output")
    factor = payload.get("factor") or payload.get("expression")
    if not isinstance(factor, str) or not factor.strip():
        return CandidateParseResult(False, failure_reason="missing factor field")
    return CandidateParseResult(True, factor=FactorExpression(factor.strip()), rationale=str(payload.get("rationale", "")))


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text

"""Central place for all LLM prompt templates."""

from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel


SEED_SYSTEM_PROMPT = """You are an expert quantitative researcher and formulaic alpha factor designer.

Your task is to generate valid, interpretable, and diverse formulaic alpha factors for cross-sectional stock return prediction.

Principles:
1. Each factor should have a plausible market hypothesis such as momentum, reversal, volatility, liquidity, price-volume interaction, or trend persistence.
2. Use only the allowed variables, constants, and operators provided by the user.
3. Avoid overly complex, deeply nested, or over-engineered expressions.
4. Avoid near-duplicates of existing seed factors.
5. Do not use future information or test-set information.
6. Output strict JSON only. Do not output Markdown or text outside JSON.
"""


CANDIDATE_SYSTEM_PROMPT = """You are an expert quantitative researcher and LLM-based evolutionary factor mining operator.

Your task is to generate exactly one valid child factor expression according to the given evolutionary operator.

Principles:
1. For mutation, improve one parent factor with a targeted structural modification.
2. For crossover, combine useful and complementary structures from two parent factors.
3. Treat lineage and global priors as soft search priors, not hard constraints.
4. Use only the allowed variables, constants, and operators.
5. Avoid trivial rewrites, parent copies, unsupported syntax, and overly long expressions.
6. Output strict JSON only. Do not output Markdown or text outside JSON.
"""


PRIOR_REWRITE_SYSTEM_PROMPT = """You are a structured prior rewriting module for an LLM-based evolutionary factor mining system.

Your task is to update persistent search priors based on new evolutionary evidence.

You are NOT a free-form reflection agent.
You must NOT output open-ended advice.
You must NOT change validation results, backtest results, scores, or factor validity.

Principles:
1. Preserve useful old knowledge.
2. Incorporate new parent-child evidence.
3. Avoid overreacting to a single observation.
4. Compress and merge redundant patterns.
5. Respect the requested JSON schema exactly.
6. Output a complete updated prior state, not a patch.
7. Output strict JSON only. Do not output Markdown or text outside JSON.
"""


def build_seed_prompt(payload: dict[str, Any]) -> str:
    constraints = payload.get("constraints", {})
    return f"""Generate exactly one initial formulaic alpha factor.

This factor will become one root lineage in an evolutionary factor mining system.

Seed context:
- Seed index: {payload.get("seed_index")}
- Existing seed expressions to avoid: {_json(payload.get("existing_seed_expressions", []))}
- Market: {constraints.get("market", "csi500")}
- Universe: {constraints.get("stock_universe", constraints.get("market", "csi500"))}
- Frequency: daily
- Target: next-period stock return prediction

Allowed variables:
{_allowed_variables(payload.get("allowed_expression_dsl", {}))}

Allowed operators:
{_allowed_operators(payload.get("allowed_expression_dsl", {}))}

Expression constraints:
- Maximum expression length: {constraints.get("factor_length_limit", 40)}
- Use only the allowed variables, constants, and operators.
- Avoid invalid operations such as unsupported windows or unsupported functions.
- Prefer compact and interpretable expressions.

Generation requirements:
- Generate exactly one factor.
- Prefer a hypothesis different from existing seed factors.
- Do not generate code.
- Do not output Markdown.

Output strict JSON in this format:
{{
  "factor": "formulaic factor expression",
  "rationale": "why this factor may predict future returns",
  "factor_name": "short descriptive name",
  "hypothesis": "brief financial hypothesis",
  "expected_signal_type": "momentum|reversal|volatility|liquidity|price_volume|other",
  "risk_notes": "possible overfitting or instability risk"
}}
"""


def build_candidate_prompt(payload: dict[str, Any]) -> str:
    operator = payload.get("operator")
    if operator == "crossover":
        return _build_crossover_candidate_prompt(payload)
    return _build_mutation_candidate_prompt(payload)


def build_prior_rewrite_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    target = payload.get("target_prior_type")
    if target == "mutation_lineage":
        return _build_mutation_prior_prompt(payload, schema_model)
    if target == "crossover_lineage":
        return _build_crossover_prior_prompt(payload, schema_model)
    if target == "global_mutation":
        return _build_global_mutation_prior_prompt(payload, schema_model)
    if target == "global_crossover":
        return _build_global_crossover_prior_prompt(payload, schema_model)
    return _build_generic_prior_prompt(payload, schema_model)


def _build_mutation_candidate_prompt(payload: dict[str, Any]) -> str:
    context = payload.get("prior_context", {})
    parents = payload.get("parents", [])
    metrics = payload.get("parent_metrics", [])
    constraints = payload.get("constraints", {})
    return f"""Generate one mutated child factor.

Operator:
mutation

Current parent factor:
{parents[0] if parents else ""}

Parent factor metrics:
{_json(metrics[0] if metrics else {})}

Current lineage mutation prior:
{_json(context.get("lineage_prior", {}))}

Global mutation prior:
{_json(context.get("global_prior", {}))}

Recent invalid or failed patterns to avoid:
{_json(payload.get("recent_invalid_or_failed_patterns", []))}

Allowed variables:
{_allowed_variables(payload.get("allowed_expression_dsl", {}))}

Allowed operators:
{_allowed_operators(payload.get("allowed_expression_dsl", {}))}

Expression constraints:
- Maximum expression length: {constraints.get("factor_length_limit", 40)}
- Use only allowed variables, constants, and operators.
- Do not generate code.
- Generate exactly one formulaic factor expression.
- Avoid trivial rewrites that do not change factor meaning.

Output strict JSON in this format:
{{
  "operator": "mutation",
  "parent_factor": "{_first(payload.get("parent_ids", []))}",
  "factor": "new mutated factor expression",
  "rationale": "brief explanation of the mutation",
  "expected_effect": "why the mutation may improve IC or ICIR"
}}
"""


def _build_crossover_candidate_prompt(payload: dict[str, Any]) -> str:
    context = payload.get("prior_context", {})
    parents = payload.get("parents", [])
    metrics = payload.get("parent_metrics", [])
    constraints = payload.get("constraints", {})
    parent_ids = payload.get("parent_ids", [])
    return f"""Generate one crossover child factor.

Operator:
crossover

Parent factor 1:
{parents[0] if len(parents) > 0 else ""}

Parent factor 1 metrics:
{_json(metrics[0] if len(metrics) > 0 else {})}

Parent factor 1 crossover prior:
{_json(context.get("lineage_prior", {}))}

Parent factor 2:
{parents[1] if len(parents) > 1 else ""}

Parent factor 2 metrics:
{_json(metrics[1] if len(metrics) > 1 else {})}

Parent factor 2 crossover prior:
{_json(context.get("secondary_lineage_prior", {}))}

Global crossover prior:
{_json(context.get("global_prior", {}))}

Allowed variables:
{_allowed_variables(payload.get("allowed_expression_dsl", {}))}

Allowed operators:
{_allowed_operators(payload.get("allowed_expression_dsl", {}))}

Expression constraints:
- Maximum expression length: {constraints.get("factor_length_limit", 40)}
- Use only allowed variables, constants, and operators.
- Do not generate code.
- Generate exactly one formulaic factor expression.
- Do not simply concatenate both parent expressions.
- Do not copy one parent unchanged.

Output strict JSON in this format:
{{
  "operator": "crossover",
  "parent_factor_1": "{parent_ids[0] if len(parent_ids) > 0 else ""}",
  "parent_factor_2": "{parent_ids[1] if len(parent_ids) > 1 else ""}",
  "factor": "new crossover factor expression",
  "rationale": "brief explanation of how the two parents are combined",
  "expected_effect": "why this crossover may improve IC or ICIR"
}}
"""


def _build_mutation_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _prior_prompt(
        title="Rewrite the complete mutation prior for this lineage.",
        target="mutation_lineage_prior",
        payload=payload,
        rules=[
            "If validation ICIR improves, strengthen the relevant successful mutation pattern.",
            "If train improves but validation does not, do not treat it as a strong success; increase bias risk if appropriate.",
            "Preserve useful old prior information.",
            "Merge redundant patterns.",
            "Single new evidence should usually produce low or medium confidence, not high confidence.",
        ],
        schema_model=schema_model,
    )


def _build_crossover_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _prior_prompt(
        title="Rewrite the complete crossover prior for the primary lineage.",
        target="crossover_lineage_prior",
        payload=payload,
        rules=[
            "The child inherits only the primary lineage.",
            "The secondary parent is used only as crossover context.",
            "If validation ICIR improves over the primary parent, strengthen transferable or heritable structures.",
            "If the child is worse or unstable, record harmful patterns or crossover risk.",
            "Do not blindly copy all structures from the secondary parent.",
        ],
        schema_model=schema_model,
    )


def _build_global_mutation_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _prior_prompt(
        title="Rewrite the complete global mutation prior.",
        target="global_mutation_prior",
        payload=payload,
        rules=[
            "Capture only patterns that may generalize across lineages.",
            "Do not overfit global prior to one lineage.",
            "If evidence is weak, keep confidence low.",
            "If train improves but validation does not, record possible overfitting or failure.",
            "Keep the prior compact.",
        ],
        schema_model=schema_model,
    )


def _build_global_crossover_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _prior_prompt(
        title="Rewrite the complete global crossover prior.",
        target="global_crossover_prior",
        payload=payload,
        rules=[
            "Summarize cross-lineage transferable and complementarity patterns.",
            "Do not overfit to one crossover event.",
            "If validation ICIR improves, update useful complementarity or transferable patterns.",
            "If performance is poor, update harmful crossover patterns or risk guidance.",
            "Keep the prior compact.",
        ],
        schema_model=schema_model,
    )


def _build_generic_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _prior_prompt(
        title="Rewrite the complete structured prior.",
        target=str(payload.get("target_prior_type", "unknown")),
        payload=payload,
        rules=["Output a complete prior state matching the schema."],
        schema_model=schema_model,
    )


def _prior_prompt(title: str, target: str, payload: dict[str, Any], rules: list[str], schema_model: Type[BaseModel]) -> str:
    return f"""{title}

Target prior type:
{target}

Old prior:
{_json(payload.get("old_prior"))}

Evolutionary evidence:
{_json(payload.get("new_evidence"))}

Expression diff:
{_json(payload.get("expression_diff"))}

Parent ids:
{_json(payload.get("parent_ids", []))}

Child id:
{payload.get("child_id")}

Update rules:
{chr(10).join(f"- {rule}" for rule in rules)}
- Preserve useful old prior information.
- Merge redundant patterns.
- Keep at most the configured top-K pattern items when applicable.
- High confidence requires repeated support or strong validation improvement.

Output strict JSON matching this schema. Do not include extra fields:
{_json(schema_model.model_json_schema())}
"""


def _allowed_variables(dsl: dict[str, Any]) -> str:
    return "\n".join(f"- {item}" for item in dsl.get("features", []))


def _allowed_operators(dsl: dict[str, Any]) -> str:
    operators = dsl.get("operators", [])
    constants = {
        "rolling_constants": dsl.get("rolling_constants", []),
        "arithmetic_constants": dsl.get("arithmetic_constants", []),
    }
    return "\n".join(f"- {item}" for item in operators) + "\n\nAllowed constants:\n" + _json(constants)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _first(items: list[Any]) -> Any:
    return items[0] if items else ""

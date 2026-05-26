"""Central place for all LLM prompt templates."""

from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel


SEED_SYSTEM_PROMPT = """You are an expert quantitative researcher and formulaic alpha factor designer.

Your task is to generate valid, interpretable, and diverse formulaic alpha factors for cross-sectional stock return prediction.

You must follow these principles:

1. Financial rationale:
   Each factor should be motivated by a plausible market hypothesis, such as momentum, reversal, volatility, liquidity, price-volume interaction, or trend persistence.

2. Formula validity:
   Each factor must use only the allowed variables and allowed operators provided by the user.

3. Complexity control:
   Avoid overly complex, deeply nested, or over-engineered expressions.
   The expression length must not exceed the given threshold.

4. Diversity:
   Generated factors should not all follow the same idea.
   Try to cover different financial hypotheses and different structural patterns.

5. No data leakage:
   Do not use any future information or test-set information.

6. Output format:
   You must output a strict JSON object.
   Do not output Markdown.
   Do not include explanations outside the JSON.

Each factor should include:
- factor_name
- hypothesis
- expression
- expected_signal_type
- rationale
- risk_notes
"""


CANDIDATE_SYSTEM_PROMPT = """You are an expert quantitative researcher and an LLM-based evolutionary factor mining operator.

Your task is to generate exactly one valid child factor expression according to the given evolutionary operator.

Definitions of operators:

1. Mutation:
   Mutation modifies one parent factor by making a targeted structural change.
   It may replace an operator, change a time window, add or remove a sub-expression, simplify a structure, or introduce a new related signal.
   Mutation should improve the parent while preserving useful structure unless stronger exploration is requested by the program-provided search-control state.

2. Crossover:
   Crossover combines two parent factors into one coherent child factor.
   In this framework, crossover is primary-parent-centered:
   - Treat the primary parent as the main expression tree.
   - Generate the child mainly by replacing, modifying, or enriching one or a few subtrees of the primary parent using complementary structures from the secondary parent.
   - Do not simply concatenate the two parent expressions.
   - Do not copy one parent unchanged.
   - The child should preserve the useful backbone of the primary parent while incorporating useful subtree-level information from the secondary parent.

You will receive:
- parent factor(s)
- operator type: mutation or crossover
- rendered operator-specific lineage prior experience text
- rendered global operator prior experience text
- local-global prior fusion / gating decision
- program-computed search-control state, when relevant
- factor grammar and constraints
- evaluation summaries of parent factors

Important principles:

1. Operator-conditioned generation:
   Use different reasoning depending on the operator.
   For mutation, improve one parent factor by making a targeted structural modification.
   For crossover, use the primary parent as the main tree and replace or enrich subtrees using complementary structures from the secondary parent.

2. Lineage prior usage:
   The lineage prior is a soft search prior, not a hard constraint.
   It should guide the search direction, but you may still propose novel structures if they are valid and well motivated.
   Do not blindly follow lineage prior if it appears biased, stagnant, or too narrow.

3. Hint usage:
   The rendered prior may contain a hint field.
   The hint is a concise LLM-authored high-level experience note distilled from lineage evolution history.
   It may include financial intuition, search taste, structural preference, risk warning, or operator-specific reflection.
   The hint is not a concrete structural pattern to copy, not necessarily a new search direction, and not a hard rule.
   Use it as soft context when it is relevant to the current parent factor and operator.

4. Search-control state:
   quality_trend, stagnation_state, and mutation_strength are computed by deterministic code.
   Treat them as program-provided search-control guidance.
   Do not reinterpret or override mutation_strength.
   If mutation_strength is "exploratory", make a meaningfully larger structural change.
   If mutation_strength is "conservative", make a small targeted refinement.
   If mutation_strength is "moderate", balance refinement and exploration.

5. Global prior usage:
   The global operator prior provides cross-lineage experience.
   Use it to avoid lineage bias, overfitting, repeated failures, and overly narrow search.

6. Prior fusion / gating:
   You will receive a local-global gating decision.
   Use this decision to decide how strongly to rely on lineage prior versus global prior.
   If local prior weight is high, prioritize lineage-specific experience.
   If global prior weight is high, rely more on global operator guidance to avoid lineage bias or stagnation.

7. Validity:
   The generated factor must use only allowed variables and operators.
   The expression length must not exceed the threshold.
   The factor must be syntactically valid and executable.

8. Novelty and non-redundancy:
   Avoid generating a factor that is nearly identical to the parent(s) or recent factors.
   Avoid copying the parent expression with only trivial changes.

9. Output:
   Generate exactly one candidate factor.
   Output strict JSON only.
   Do not output Markdown.
   Do not include extra text outside JSON.
"""


PRIOR_REWRITE_SYSTEM_PROMPT = """You are a structured prior rewriting module for an LLM-based evolutionary factor mining system.

Your task is to rewrite persistent semantic search priors based on new evolutionary evidence.

You are NOT a free-form reflection agent.
You must NOT output open-ended advice.
You must NOT change validation results, backtest results, scores, factor validity, or deterministic search-control states.

You will receive:
- the old prior state,
- parent factor(s),
- child factor,
- operator type,
- deterministic expression diff,
- train and validation performance,
- score deltas,
- validity information,
- update trigger information,
- program-computed search-control state, when relevant,
- current generation.

Your task:
Rewrite the complete updated semantic prior state.

Definitions:
- Mutation semantic prior stores successful and failed mutation patterns, plus a hint.
- Crossover prior stores transferable patterns, harmful patterns, complementarity profile, heritable structures, plus a hint.
- Global prior stores cross-lineage semantic experience and a hint useful for reducing lineage bias and improving robustness.
- hint is a concise LLM-authored high-level experience note distilled from lineage evolution history.
- hint may include financial intuition, search taste, structural preference, risk warning, or operator-specific reflection.
- hint is not a concrete structural pattern, not necessarily a new search direction, and not a hard rule.
- quality_trend, stagnation_state, and mutation_strength are deterministic search-control states. They are computed by code and must not be rewritten by you.
- These priors are persistent search states, not temporary reflections.

Important principles:

1. Preserve useful old knowledge:
   Do not remove high-confidence old patterns unless the new evidence clearly contradicts them.

2. Incorporate new evidence only when meaningful:
   The deterministic update trigger has already decided whether this event is informative enough to rewrite the prior.
   Use the trigger reason as semantic context; do not infer or change the trigger decision.

3. Success / failure evidence:
   Do not treat every small metric change as success or failure.
   Strengthen successful patterns only when the update trigger indicates meaningful validation improvement.
   Record failed patterns only when the update trigger indicates meaningful validation degradation, or when there is clear instability / overfitting evidence.

4. Search-control states are read-only:
   Do not output or modify quality_trend, stagnation_state, or mutation_strength.
   These fields are controlled by deterministic code.
   You may use them as context to interpret evidence, but you must not rewrite them.

5. Hint:
   You may freely rewrite hint based on the old prior and the new evidence.
   The hint should summarize high-level lineage experience, search taste, risk warning, or operator-specific reflection.
   The hint should not merely restate one concrete pattern.
   The hint should not be a long free-form paragraph.
   Keep it concise, preferably one or two sentences.

6. Avoid overreacting:
   A single observation should not become high confidence unless the validation improvement is very strong and consistent with prior evidence.

7. Compress and merge:
   Merge redundant patterns.
   Remove stale, low-value, or low-confidence patterns when necessary.
   Keep the prior compact.
   Keep at most 5 items in each pattern list.
   Each evidence string must be concise and no longer than 240 characters.

8. Respect schema:
   You must output strict JSON matching the requested prior schema.
   Do not include extra fields.
   Do not output Markdown.
   Do not include explanations outside JSON.

9. The output is a persistent prior state:
   The updated prior will be used to condition future mutation or crossover prompts.
"""


def build_seed_prompt(payload: dict[str, Any]) -> str:
    constraints = payload.get("constraints", {})
    dsl = payload.get("allowed_expression_dsl", {})
    return _fill(
        """Generate exactly one initial formulaic alpha factor.

Task:
We need seed factors for an LLM-based evolutionary factor mining system. These seed factors will become the roots of different evolutionary lineages.

Dataset:
- Market: <<MARKET>>
- Frequency: daily
- Target: next-period stock return prediction
- Universe: <<UNIVERSE>>

Allowed variables:
<<ALLOWED_VARIABLES>>

Allowed operators:
<<ALLOWED_OPERATORS>>

Expression constraints:
- Maximum expression length: <<FACTOR_LENGTH_THRESHOLD>>
- Use only the allowed variables and operators.
- Avoid invalid operations such as division by zero, undefined windows, or unsupported functions.
- Prefer compact and interpretable expressions.

Existing factors to avoid duplicating:
<<EXISTING_FACTOR_LIST>>

Generation requirements:
- Generate exactly one factor.
- Factors should be diverse.
- Cover different hypotheses if possible:
  - momentum
  - reversal
  - volatility
  - liquidity
  - price-volume interaction
  - trend persistence
- Do not generate factors that are nearly identical to each other.
- Do not output code.
- Do not output Markdown.

Output strict JSON in the following format:

{
  "factor_name": "short descriptive name",
  "hypothesis": "brief financial hypothesis",
  "factor": "formulaic factor expression",
  "expected_signal_type": "momentum|reversal|volatility|liquidity|price_volume|other",
  "rationale": "why this factor may predict future returns",
  "risk_notes": "possible overfitting or instability risk"
}
""",
        {
            "MARKET": constraints.get("market", "csi500"),
            "UNIVERSE": constraints.get("stock_universe", constraints.get("market", "csi500")),
            "ALLOWED_VARIABLES": _allowed_variables(dsl),
            "ALLOWED_OPERATORS": _allowed_operators_with_constants(dsl),
            "FACTOR_LENGTH_THRESHOLD": constraints.get("factor_prompt_length_limit", 40),
            "EXISTING_FACTOR_LIST": _json(payload.get("existing_seed_expressions", [])),
        },
    )


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
    rendered = context.get("rendered_priors", {})
    decision = context.get("fusion_decision", {})
    control_state = _mutation_search_control_state(context)
    parents = payload.get("parents", [])
    metrics = payload.get("parent_metrics", [])
    constraints = payload.get("constraints", {})
    dsl = payload.get("allowed_expression_dsl", {})
    return _fill(
        """Generate one mutated child factor.

Operator:
mutation

Meaning of mutation:
Mutation modifies one parent factor through a targeted local or structural change. It should use lineage-specific experience as soft guidance, but it may still explore novel structures when the lineage is stagnant, biased, or no longer improving.

Current parent factor:
<<PARENT_FACTOR_EXPRESSION>>

Parent factor metrics:
<<PARENT_FACTOR_METRICS>>

Rendered lineage mutation experience:
<<RENDERED_LINEAGE_MUTATION_EXPERIENCE_TEXT>>

Note:
The rendered lineage mutation experience may include:
- concrete successful / failed mutation patterns;
- a hint summarizing high-level lineage experience or search taste.
Patterns provide structural evidence. The hint provides broader soft context and may include financial intuition, structural preference, risk warning, or operator-specific reflection.

Rendered global mutation prior:
<<RENDERED_GLOBAL_MUTATION_PRIOR_TEXT>>

Program-computed mutation search-control state:
{
  "quality_trend": "<<QUALITY_TREND>>",
  "stagnation_state": "<<STAGNATION_STATE>>",
  "mutation_strength": "<<MUTATION_STRENGTH>>"
}

Prior fusion / gating decision:
{
  "operator": "mutation",
  "lineage_prior_weight_lambda": <<LAMBDA_LINEAGE>>,
  "global_prior_weight": <<LAMBDA_GLOBAL>>,
  "gating_reason": "<<GATING_REASON>>",
  "instruction": "<<GATING_INSTRUCTION>>"
}

Recent invalid or failed patterns to avoid:
<<RECENT_INVALID_OR_FAILED_PATTERNS>>

Previous duplicate candidate to avoid:
<<DUPLICATE_FEEDBACK>>

If duplicate feedback is provided, do not regenerate an expression equivalent to the rejected normalized expression. Make a clearly different structural change.

Allowed variables:
<<ALLOWED_VARIABLES>>

Allowed operators:
<<ALLOWED_OPERATORS>>

Expression constraints:
- Maximum expression length: <<FACTOR_LENGTH_THRESHOLD>>
- Use only allowed variables and operators.
- Do not generate code.
- Generate exactly one formulaic factor expression.
- Avoid trivial rewrites that do not change the factor meaning.
- Avoid patterns listed in failed mutation patterns unless you have a clear reason to modify them safely.

Mutation guidance:
Use the lineage mutation experience to understand:
- which modifications have worked in this lineage,
- which modifications have failed in this lineage,
- which local structures may be worth preserving or avoiding.

Use the hint to:
- understand the high-level experience or taste of this lineage;
- inspire generation when it is relevant to the current parent;
- support free exploration beyond concrete patterns when mutation_strength is exploratory;
- identify possible risks or preferences that are not captured by specific patterns.

Use the program-computed search-control state to decide mutation intensity:
- If mutation_strength = "conservative", make a small targeted refinement.
- If mutation_strength = "moderate", make a meaningful but not disruptive change.
- If mutation_strength = "exploratory", make a stronger structural modification and avoid merely small local edits.
- If quality_trend is worsening or stagnation_state indicates stagnation, increase exploration according to the program-provided mutation_strength.

Use the global mutation prior to:
- avoid overfitting to this lineage,
- introduce generally effective mutation patterns if local lineage prior is biased,
- avoid globally known failure patterns.

Important:
- The prior is soft guidance, not a hard constraint.
- You may deviate from the prior if the deviation is valid, interpretable, and well motivated.
- Do not override the program-provided mutation_strength.
- If the lineage is improving and bias risk is low, prefer targeted refinement.
- If the lineage is stagnating, worsening, or biased, follow the exploratory guidance given by the program.
- If mutation_strength is exploratory, give more attention to hint and global prior while still satisfying expression constraints.

Output strict JSON in the following format:

{
  "factor": "new mutated factor expression"
}

Do not output rationale, analysis, prior_usage, risk notes, Markdown, or any extra fields.
""",
        {
            "PARENT_FACTOR_EXPRESSION": parents[0] if parents else "",
            "PARENT_FACTOR_METRICS": _json(metrics[0] if metrics else {}),
            "RENDERED_LINEAGE_MUTATION_EXPERIENCE_TEXT": rendered.get("lineage_prior_text", "Mutation lineage experience is unavailable."),
            "RENDERED_GLOBAL_MUTATION_PRIOR_TEXT": rendered.get("global_prior_text", "Global mutation experience is unavailable."),
            "QUALITY_TREND": control_state.get("quality_trend", "unknown"),
            "STAGNATION_STATE": control_state.get("stagnation_state", "not_stagnant"),
            "MUTATION_STRENGTH": control_state.get("mutation_strength", "moderate"),
            "LAMBDA_LINEAGE": _json_scalar(decision.get("local_weight", "unknown")),
            "LAMBDA_GLOBAL": _json_scalar(decision.get("global_weight", "unknown")),
            "GATING_REASON": decision.get("reason", ""),
            "GATING_INSTRUCTION": decision.get("instruction", ""),
            "RECENT_INVALID_OR_FAILED_PATTERNS": _json(payload.get("recent_invalid_or_failed_patterns", [])),
            "DUPLICATE_FEEDBACK": _json(payload.get("duplicate_feedback", [])),
            "ALLOWED_VARIABLES": _allowed_variables(dsl),
            "ALLOWED_OPERATORS": _allowed_operators_with_constants(dsl),
            "FACTOR_LENGTH_THRESHOLD": constraints.get("factor_prompt_length_limit", 40),
        },
    )


def _build_crossover_candidate_prompt(payload: dict[str, Any]) -> str:
    context = payload.get("prior_context", {})
    rendered = context.get("rendered_priors", {})
    decision = context.get("fusion_decision", {})
    parents = payload.get("parents", [])
    metrics = payload.get("parent_metrics", [])
    constraints = payload.get("constraints", {})
    parent_ids = payload.get("parent_ids", [])
    parent_prior_texts = rendered.get("parent_lineage_prior_texts", [])
    dsl = payload.get("allowed_expression_dsl", {})
    return _fill(
        """Generate one crossover child factor.

Operator:
crossover

Meaning of crossover:
Crossover combines two parent factors into one coherent child factor.
In this framework, crossover must be primary-parent-centered:
- Treat the primary parent as the main expression tree.
- Use the secondary parent only as a source of complementary subtree-level material.
- Generate the child by replacing, modifying, or enriching one or a few subtrees of the primary parent.
- The secondary subtree or pattern should be comparable in size to the primary subtree it replaces or enriches.
- Do not simply concatenate both parent expressions.
- Do not copy one parent unchanged.
- The child should mainly preserve the backbone of the primary parent while incorporating useful complementary structure from the secondary parent.

Primary parent factor:
<<PRIMARY_PARENT_FACTOR_EXPRESSION>>

Primary parent metrics:
<<PRIMARY_PARENT_METRICS>>

Rendered primary parent crossover experience:
<<RENDERED_PRIMARY_LINEAGE_CROSSOVER_EXPERIENCE_TEXT>>

Note:
The primary parent crossover experience may include concrete transferable / harmful patterns and a hint.
The hint is a high-level experience note about how this lineage may be used in crossover.

Secondary parent factor:
<<SECONDARY_PARENT_FACTOR_EXPRESSION>>

Secondary parent metrics:
<<SECONDARY_PARENT_METRICS>>

Rendered secondary parent crossover experience:
<<RENDERED_SECONDARY_LINEAGE_CROSSOVER_EXPERIENCE_TEXT>>

Note:
The secondary parent crossover experience may include concrete transferable / harmful patterns and a hint.
Use the hint to judge whether the secondary parent provides useful context or complementary information. Do not copy structures from the hint directly.

Rendered global crossover prior:
<<RENDERED_GLOBAL_CROSSOVER_PRIOR_TEXT>>

Previous duplicate candidate to avoid:
<<DUPLICATE_FEEDBACK>>

If duplicate feedback is provided, do not regenerate an expression equivalent to the rejected normalized expression. Keep the same primary-parent-centered crossover task, but replace or enrich a different subtree.

Prior fusion / gating decision:
{
  "operator": "crossover",
  "primary_parent_lineage_prior_weight": <<LAMBDA_PRIMARY_LINEAGE>>,
  "secondary_parent_lineage_prior_weight": <<LAMBDA_SECONDARY_LINEAGE>>,
  "global_prior_weight": <<LAMBDA_GLOBAL>>,
  "gating_reason": "<<GATING_REASON>>",
  "instruction": "<<GATING_INSTRUCTION>>"
}

Allowed variables:
<<ALLOWED_VARIABLES>>

Allowed operators:
<<ALLOWED_OPERATORS>>

Expression constraints:
- Maximum expression length: <<FACTOR_LENGTH_THRESHOLD>>
- Use only allowed variables and operators.
- Do not generate code.
- Generate exactly one formulaic factor expression.
- Do not simply concatenate both parent expressions.
- Do not copy one parent unchanged.
- The generated factor should be a primary-parent-centered subtree replacement or subtree enrichment.
- The inserted secondary subtree or motif should be similar in length and complexity to the primary subtree it replaces or enriches.

Crossover guidance:
Use the primary parent crossover experience to identify:
- transferable structures from the primary lineage,
- harmful structures from the primary lineage that should be replaced or avoided,
- heritable structures that should be preserved,
- crossover risks of the primary lineage.

Use the secondary parent crossover experience to identify:
- complementary subtrees or structural motifs that may replace or enrich part of the primary parent,
- harmful structures that should not be imported from the secondary parent,
- whether the secondary lineage is actually complementary to the primary lineage.

Use the global crossover prior to:
- avoid globally observed bad crossover patterns,
- prefer broadly effective structural combinations,
- reduce lineage bias.

Use hints to:
- understand high-level preferences, risks, or operator-specific reflections from each lineage;
- reason about whether the two lineages are complementary beyond concrete structural patterns;
- inspire subtree-level replacement or enrichment when the hint is relevant.

Important:
- The priors are soft guidance, not hard constraints.
- The primary parent must serve as the main tree.
- The secondary parent should only contribute selected subtree-level information.
- If the two lineages are redundant or risky, avoid copying whole parent structures.
- If a parent lineage has high crossover risk, rely more on global crossover guidance.
- You may propose a novel crossover structure if it is valid and well motivated.
- Do not copy hint text into the factor; use it only as high-level guidance.

Output strict JSON in the following format:

{
  "factor": "new crossover factor expression"
}

Do not output rationale, analysis, prior_usage, risk notes, Markdown, or any extra fields.
""",
        {
            "PRIMARY_PARENT_FACTOR_EXPRESSION": parents[0] if len(parents) > 0 else "",
            "PRIMARY_PARENT_METRICS": _json(metrics[0] if len(metrics) > 0 else {}),
            "RENDERED_PRIMARY_LINEAGE_CROSSOVER_EXPERIENCE_TEXT": parent_prior_texts[0]
            if len(parent_prior_texts) > 0
            else rendered.get("lineage_prior_text", "Crossover lineage experience is unavailable."),
            "SECONDARY_PARENT_FACTOR_EXPRESSION": parents[1] if len(parents) > 1 else "",
            "SECONDARY_PARENT_METRICS": _json(metrics[1] if len(metrics) > 1 else {}),
            "RENDERED_SECONDARY_LINEAGE_CROSSOVER_EXPERIENCE_TEXT": parent_prior_texts[1]
            if len(parent_prior_texts) > 1
            else rendered.get("secondary_lineage_prior_text", "Crossover lineage experience is unavailable."),
            "RENDERED_GLOBAL_CROSSOVER_PRIOR_TEXT": rendered.get("global_prior_text", "Global crossover experience is unavailable."),
            "DUPLICATE_FEEDBACK": _json(payload.get("duplicate_feedback", [])),
            "LAMBDA_PRIMARY_LINEAGE": _json_scalar(decision.get("local_weight", "unknown")),
            "LAMBDA_SECONDARY_LINEAGE": _json_scalar(decision.get("local_weight", "unknown")),
            "LAMBDA_GLOBAL": _json_scalar(decision.get("global_weight", "unknown")),
            "GATING_REASON": decision.get("reason", ""),
            "GATING_INSTRUCTION": decision.get("instruction", ""),
            "ALLOWED_VARIABLES": _allowed_variables(dsl),
            "ALLOWED_OPERATORS": _allowed_operators_with_constants(dsl),
            "FACTOR_LENGTH_THRESHOLD": constraints.get("factor_prompt_length_limit", 40),
        },
    )


def _build_mutation_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    old_prior = payload.get("old_prior") or {}
    return _fill(
        """Rewrite the complete mutation semantic prior for this lineage.

Target prior type:
mutation_lineage_semantic_prior

Field meanings:
- successful_mutation_patterns: mutation patterns that have repeatedly or significantly improved validation performance.
- failed_mutation_patterns: mutation patterns that caused significant validation degradation, invalidity, instability, or overfitting.
- hint: a concise LLM-authored high-level experience note for this mutation lineage. It may include financial intuition, search taste, structural preference, risk warning, or mutation-specific reflection.
- bias_risk: whether the lineage is over-relying on narrow patterns or showing train-validation inconsistency.

Read-only program-computed search-control state:
<<SEARCH_CONTROL_STATE>>

Important:
- The search-control state above is read-only.
- Do not output quality_trend, stagnation_state, or mutation_strength.
- These fields will be updated by deterministic code, not by the LLM prior rewriter.
- You may use them only to interpret the new evidence.

Old mutation semantic prior:
<<OLD_MUTATION_PRIOR_JSON>>

Evolutionary evidence:
<<EVOLUTIONARY_EVIDENCE>>

Update trigger:
<<UPDATE_TRIGGER>>

Update rules:
- If should_rewrite_prior is false, preserve most of the old prior and avoid adding strong new success/failure patterns.
- If trigger_reason indicates significant validation improvement, strengthen the relevant successful mutation pattern.
- If trigger_reason indicates significant validation degradation, record the relevant failed mutation pattern.
- If train improves but validation does not improve, do not treat it as a strong success; increase bias_risk if appropriate.
- You may rewrite hint based on old prior and new evidence.
- The hint should summarize high-level lineage experience or search taste, not merely repeat one concrete pattern.
- Preserve useful old prior information.
- Merge redundant patterns.
- Keep at most 5 successful mutation patterns.
- Keep at most 5 failed mutation patterns.
- Each evidence string must be no longer than 240 characters.
- Single new evidence should usually produce low or medium confidence, not high confidence.
- High confidence requires repeated support or strong validation improvement.

Output strict JSON with this schema:

<<SCHEMA_JSON>>
""",
        {
            "SEARCH_CONTROL_STATE": _json(payload.get("mutation_control_state") or _search_control_state_from_prior(old_prior)),
            "OLD_MUTATION_PRIOR_JSON": _json(_mutation_semantic_prior(old_prior)),
            "EVOLUTIONARY_EVIDENCE": _json(_evidence_with_diff(payload)),
            "UPDATE_TRIGGER": _json(payload.get("update_trigger")),
            "SCHEMA_JSON": _json(schema_model.model_json_schema()),
        },
    )


def _build_crossover_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _fill(
        """Rewrite the complete crossover prior for the primary lineage.

Target prior type:
crossover_lineage_prior

Meaning of crossover prior:
A crossover prior summarizes what this lineage can contribute during crossover. It should identify which structures are worth inheriting, which structures should not be transmitted, which other lineage types may be complementary, and how risky this lineage is as crossover material.

Important:
The child inherits only the primary lineage.
The secondary parent is used only as crossover context.
Do not assign the child to the secondary lineage.

Field meanings:
- transferable_patterns: structures from this lineage that are useful to preserve or transmit during crossover.
- harmful_patterns: structures from this lineage that should not be transmitted to offspring.
- complementarity_profile: what types of other lineages or structures this lineage may combine well with.
- heritable_structures: stable structures preserved across multiple descendants and associated with good performance.
- hint: a concise LLM-authored high-level experience note for this crossover lineage. It may include financial intuition, search taste, structural preference, risk warning, or crossover-specific reflection.
- crossover_risk: whether this lineage is reliable or risky as crossover material.

Old crossover prior of the primary lineage:
<<OLD_PRIOR_JSON>>

Evolutionary evidence:
<<EVOLUTIONARY_EVIDENCE>>

Update trigger:
<<UPDATE_TRIGGER>>

Update rules:
- If should_rewrite_prior is false, preserve most of the old prior and avoid adding strong new transferable or harmful patterns.
- If trigger_reason indicates significant validation improvement, strengthen transferable or heritable structures from the primary lineage.
- If trigger_reason indicates significant validation degradation, record harmful patterns or increase crossover risk.
- If the child benefits from replacing or enriching a primary-parent subtree using secondary-parent information, update the complementarity profile.
- Do not blindly copy all structures from the secondary parent.
- You may rewrite hint based on old prior and new evidence.
- The hint should summarize high-level lineage experience or search taste, not merely repeat one concrete pattern.
- Preserve useful old crossover prior information.
- Merge redundant patterns.
- Keep at most 5 transferable patterns.
- Keep at most 5 harmful patterns.
- Keep at most 5 heritable structures.
- Each evidence string must be no longer than 240 characters.
- Single new evidence should usually produce low or medium confidence, not high confidence.

Output strict JSON with this schema:

<<SCHEMA_JSON>>
""",
        {
            "OLD_PRIOR_JSON": _json(payload.get("old_prior")),
            "EVOLUTIONARY_EVIDENCE": _json(_evidence_with_diff(payload)),
            "UPDATE_TRIGGER": _json(payload.get("update_trigger")),
            "SCHEMA_JSON": _json(schema_model.model_json_schema()),
        },
    )


def _build_global_mutation_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _fill(
        """Rewrite the complete global mutation prior.

Target prior type:
global_mutation_prior

Field meanings:
- global_successful_mutation_patterns: mutation patterns that appear useful across multiple lineages.
- global_failed_mutation_patterns: mutation patterns that tend to fail or overfit across lineages.
- common_invalid_patterns: structures that frequently cause invalid or non-executable factors.
- hint: a concise LLM-authored high-level global mutation experience note.
- last_updated_generation: the latest generation reflected in this prior.

Old global mutation prior:
<<OLD_PRIOR_JSON>>

New evidence from one lineage:
<<EVOLUTIONARY_EVIDENCE>>

Update trigger:
<<UPDATE_TRIGGER>>

Update rules:
- Global prior should only capture patterns that may generalize across lineages.
- Do not overfit global prior to one lineage.
- If should_rewrite_prior is false, preserve most of the old prior.
- If trigger_reason indicates significant validation improvement, you may update a general successful mutation pattern, but keep confidence conservative unless supported by multiple lineages.
- If trigger_reason indicates significant validation degradation, record possible failure or overfitting.
- If train improves but validation does not, record possible overfitting or failure.
- You may rewrite hint based on old global prior and new evidence.
- The hint should summarize high-level global mutation experience, not merely repeat one concrete pattern.
- Preserve useful old global prior information.
- Keep at most 5 successful mutation patterns.
- Keep at most 5 failed mutation patterns.
- Each evidence string must be no longer than 240 characters.
- Keep the prior compact.

Output strict JSON with this schema:

<<SCHEMA_JSON>>
""",
        {
            "OLD_PRIOR_JSON": _json(payload.get("old_prior")),
            "EVOLUTIONARY_EVIDENCE": _json(_evidence_with_diff(payload)),
            "UPDATE_TRIGGER": _json(payload.get("update_trigger")),
            "SCHEMA_JSON": _json(schema_model.model_json_schema()),
        },
    )


def _build_global_crossover_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _fill(
        """Rewrite the complete global crossover prior.

Target prior type:
global_crossover_prior

Field meanings:
- global_transferable_patterns: structures that tend to transfer well across lineages.
- global_harmful_patterns: structures that tend to harm crossover offspring.
- global_complementarity_patterns: cross-lineage combinations that tend to work well.
- hint: a concise LLM-authored high-level global crossover experience note.
- last_updated_generation: the latest generation reflected in this prior.

Old global crossover prior:
<<OLD_PRIOR_JSON>>

New evidence:
<<EVOLUTIONARY_EVIDENCE>>

Update trigger:
<<UPDATE_TRIGGER>>

Update rules:
- Global crossover prior should summarize cross-lineage transferable patterns and complementarity patterns.
- Do not overfit to one crossover event.
- If should_rewrite_prior is false, preserve most of the old prior.
- If trigger_reason indicates significant validation improvement, update useful complementarity or transferable patterns.
- If trigger_reason indicates significant validation degradation, update harmful crossover patterns or risk guidance.
- You may rewrite hint based on old global prior and new evidence.
- The hint should summarize high-level global crossover experience, not merely repeat one concrete pattern.
- Preserve useful old prior information.
- Merge redundant patterns.
- Keep at most 5 transferable patterns.
- Keep at most 5 harmful patterns.
- Keep at most 5 complementarity patterns.
- Each evidence string must be no longer than 240 characters.
- Keep the prior compact.

Output strict JSON with this schema:

<<SCHEMA_JSON>>
""",
        {
            "OLD_PRIOR_JSON": _json(payload.get("old_prior")),
            "EVOLUTIONARY_EVIDENCE": _json(_evidence_with_diff(payload)),
            "UPDATE_TRIGGER": _json(payload.get("update_trigger")),
            "SCHEMA_JSON": _json(schema_model.model_json_schema()),
        },
    )


def _build_generic_prior_prompt(payload: dict[str, Any], schema_model: Type[BaseModel]) -> str:
    return _fill(
        """Rewrite the complete structured prior.

Target prior type:
<<TARGET_PRIOR_TYPE>>

Old prior:
<<OLD_PRIOR_JSON>>

Evolutionary evidence:
<<EVOLUTIONARY_EVIDENCE>>

Update trigger:
<<UPDATE_TRIGGER>>

Output constraints:
- Keep at most 5 items in each pattern list.
- Each evidence string must be no longer than 240 characters.
- Keep hint concise, preferably one or two sentences.

Output strict JSON with this schema:

<<SCHEMA_JSON>>
""",
        {
            "TARGET_PRIOR_TYPE": str(payload.get("target_prior_type", "unknown")),
            "OLD_PRIOR_JSON": _json(payload.get("old_prior")),
            "EVOLUTIONARY_EVIDENCE": _json(_evidence_with_diff(payload)),
            "UPDATE_TRIGGER": _json(payload.get("update_trigger")),
            "SCHEMA_JSON": _json(schema_model.model_json_schema()),
        },
    )


def _fill(template: str, values: dict[str, Any]) -> str:
    text = template
    for key, value in values.items():
        text = text.replace(f"<<{key}>>", str(value))
    return text


def _allowed_variables(dsl: dict[str, Any]) -> str:
    features = dsl.get("features", [])
    if not features and dsl.get("definition_text"):
        return str(dsl["definition_text"]).split("2. You can use int constants", 1)[0].strip()
    return "\n".join(f"- {item}" for item in features)


def _allowed_operators_with_constants(dsl: dict[str, Any]) -> str:
    if dsl.get("definition_text"):
        return str(dsl["definition_text"])
    constants = {
        "rolling_constants": dsl.get("rolling_constants", []),
        "arithmetic_constants": dsl.get("arithmetic_constants", []),
    }
    operators = "\n".join(f"- {item}" for item in dsl.get("operators", []))
    return operators + "\n\nAllowed constants:\n" + _json(constants)


def _mutation_search_control_state(context: dict[str, Any]) -> dict[str, Any]:
    return context.get("mutation_control_state") or {
        "quality_trend": "unknown",
        "stagnation_state": "not_stagnant",
        "mutation_strength": "moderate",
    }


def _search_control_state_from_prior(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality_trend": prior.get("quality_trend", "unknown"),
        "stagnation_state": prior.get("stagnation_state", "unknown"),
        "mutation_strength": prior.get("mutation_strength", "moderate"),
    }


def _mutation_semantic_prior(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "successful_mutation_patterns": prior.get("successful_mutation_patterns", []),
        "failed_mutation_patterns": prior.get("failed_mutation_patterns", []),
        "hint": prior.get("hint", ""),
        "bias_risk": prior.get("bias_risk", "low"),
    }


def _evidence_with_diff(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(payload.get("new_evidence") or {})
    if payload.get("expression_diff") is not None:
        evidence["expression_diff"] = payload.get("expression_diff")
    if payload.get("parent_ids"):
        evidence["parent_ids"] = payload.get("parent_ids")
    if payload.get("child_id") is not None:
        evidence["child_id"] = payload.get("child_id")
    return evidence


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _json_scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _first(items: list[Any]) -> Any:
    return items[0] if items else ""

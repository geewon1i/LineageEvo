# LineageEvo Architecture

LineageEvo is organized around a deterministic evolutionary search loop with two LLM-facing boundaries: candidate generation and structured prior rewriting.

## Main Flow

1. Generate initial seed factors.
2. Select an operator slot: mutation or crossover.
3. Select parent factor(s) from the active pool.
4. Apply the selected ablation mode, then fuse lineage/global priors with an explicit local-global weight.
5. Ask the candidate LLM for exactly one factor expression.
6. Validate the expression deterministically.
7. Evaluate train/validation IC and ICIR with Qlib; use IC as the main search metric.
8. Add valid children to the lineage DAG.
9. Rewrite lineage/global priors with a constrained LLM prior rewriter.
10. Finalize by selecting top validation factors, testing them, and optionally running Qlib backtest.

Invalid candidates stop at validation/evaluation logging. They are not added to the DAG and do not trigger prior rewriting in v1.

## Package Structure

- `lineage_evo.candidate`: candidate prompt requests, LLM candidate generator, mock candidate generator, JSON output parsing.
- `lineage_evo.factor`: factor expression wrapper, AlphaPROBE-style DSL validation, deterministic Qlib expression adaptation, expression diff.
- `lineage_evo.validation`: deterministic validation result types and validation boundaries.
- `lineage_evo.evaluation`: evaluator protocol, mock evaluator, Qlib evaluator for IC/ICIR.
- `lineage_evo.lineage`: factor nodes, evolutionary edges, lineage DAG, active pool pruning.
- `lineage_evo.operators`: IC-based parent selection and operator scheduling.
- `lineage_evo.priors`: strict Pydantic schemas plus renderers that turn structured priors into compact experience text.
- `lineage_evo.ablation`: experiment-mode selection for baselines such as lineage-only, global-only, shuffled prior, and raw ancestral trace.
- `lineage_evo.prior_fusion`: Method section 4.7 local-global gating that exposes lambda and 1-lambda to candidate prompts.
- `lineage_evo.prior_rewrite`: LLM prior rewriter, deterministic prior manager, fallback and pruning.
- `lineage_evo.recording`: JSONL/CSV recorders, console reporting, run directory creation.
- `lineage_evo.finalize`: absolute-validation-IC final factor selection, oriented test IC/ICIR, and optional Qlib backtest.
- `lineage_evo.experiments`: runner assembly for mock and Qlib-backed experiments.
- `lineage_evo.llm`: provider-neutral LLM protocol and OpenAI-compatible chat-completions client.

## LLM Boundaries

The candidate LLM receives compact natural-language experience text rendered from structured priors. It outputs strict JSON containing one factor expression. The prior rewriter LLM receives structured old-prior JSON plus evidence JSON and outputs a complete updated prior JSON state, not a free-form reflection or partial patch.

The following remain deterministic:

- expression parsing and validation;
- factor length, feature, operator, and constant checks;
- Qlib executability checks;
- IC and ICIR computation, with IC used for search decisions and ICIR kept as auxiliary stability output;
- DAG updates;
- final test and backtest;
- prior schema validation, pruning, confidence constraints, and fallback.

## Priors

Each lineage stores:

- a mutation prior for same-lineage mutation guidance;
- a crossover prior for crossover guidance involving that lineage.

The run also stores:

- a global mutation prior;
- a global crossover prior.

For valid mutation children, the engine updates the parent lineage mutation prior and the global mutation prior. For valid crossover children, the engine updates only the primary lineage crossover prior and the global crossover prior.

## Configuration and Reproducibility

Experiment settings are centralized in `configs/default.toml`. Local secrets and machine-specific paths belong in `.env`.

Every run writes a unique run directory with:

- merged config snapshot;
- raw LLM outputs;
- validation/evaluation status;
- accepted prior states;
- DAG events;
- final selected factors;
- test and backtest outputs.

Mock components remain part of the architecture so tests and offline debugging can run without API keys or network access.

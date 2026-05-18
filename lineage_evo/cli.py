"""Command line interface for LineageEvo."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from lineage_evo.candidate import CandidateRequest, LLMCandidateGenerator, MockCandidateGenerator
from lineage_evo.config import BacktestConfig, ExperimentConfig, LLMConfig, QlibConfig, SearchConfig, SelectionConfig
from lineage_evo.evaluation import QlibEvaluator
from lineage_evo.experiments import ExperimentRunner
from lineage_evo.experiments.defaults import default_mutation_prior
from lineage_evo.factor import FactorExpression, QlibExpressionValidator, diff_expressions
from lineage_evo.lineage import OperatorType
from lineage_evo.llm import OpenAICompatibleLLMClient
from lineage_evo.prior_fusion import FusedPriorContext
from lineage_evo.prior_fusion import FusionMode
from lineage_evo.prior_rewrite import LLMPriorRewriter, MockPriorRewriter, PriorRewriteInput, PriorTarget
from lineage_evo.recording import ConsoleReporter, RunDirectoryResolver
from lineage_evo.seed import LLMSeedGenerator, MockSeedGenerator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.toml"


def main() -> None:
    parser = argparse.ArgumentParser(prog="lineage-evo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-run", help="Run a fully mocked offline smoke experiment.")
    smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    smoke.add_argument("--target-valid", type=int, default=None)
    smoke.add_argument("--log-dir", type=Path, default=Path("runs") / "smoke")
    smoke.add_argument("--fusion-mode", choices=[mode.value for mode in FusionMode], default=FusionMode.OURS_FULL.value)
    smoke.add_argument("--llm", choices=["mock", "openai-compatible"], default=None)
    smoke.add_argument("--candidate-llm", choices=["mock", "openai-compatible"], default=None)
    smoke.add_argument("--prior-llm", choices=["mock", "openai-compatible"], default=None)
    smoke.add_argument("--seed-count", type=int, default=None)
    smoke.add_argument("--max-seed-generation-attempts", type=int, default=None)
    smoke.add_argument("--verbose", action="store_true")
    smoke.add_argument("--print-llm-io", action="store_true")
    smoke.add_argument("--no-color", action="store_true")

    qlib_smoke = subparsers.add_parser("qlib-smoke-run", help="Run smoke search with real Qlib evaluation.")
    qlib_smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    qlib_smoke.add_argument("--target-valid", type=int, default=None)
    qlib_smoke.add_argument("--log-dir", type=Path, default=Path("runs") / "qlib_smoke")
    qlib_smoke.add_argument("--provider-uri", default=None)
    qlib_smoke.add_argument("--market", default=None)
    qlib_smoke.add_argument("--train-start", default=None)
    qlib_smoke.add_argument("--train-end", default=None)
    qlib_smoke.add_argument("--valid-start", default=None)
    qlib_smoke.add_argument("--valid-end", default=None)
    qlib_smoke.add_argument("--test-start", default=None)
    qlib_smoke.add_argument("--test-end", default=None)
    qlib_smoke.add_argument("--benchmark", default=None)
    qlib_smoke.add_argument("--llm", choices=["mock", "openai-compatible"], default=None)
    qlib_smoke.add_argument("--candidate-llm", choices=["mock", "openai-compatible"], default=None)
    qlib_smoke.add_argument("--prior-llm", choices=["mock", "openai-compatible"], default=None)
    qlib_smoke.add_argument("--seed-count", type=int, default=None)
    qlib_smoke.add_argument("--max-seed-generation-attempts", type=int, default=None)
    qlib_smoke.add_argument("--final-top-k", type=int, default=None)
    qlib_smoke.add_argument("--backtest-topk", type=int, default=None)
    qlib_smoke.add_argument("--backtest-n-drop", type=int, default=None)
    qlib_smoke.add_argument("--account", type=float, default=None)
    qlib_smoke.add_argument("--skip-final-backtest", action="store_true")
    qlib_smoke.add_argument("--verbose", action="store_true")
    qlib_smoke.add_argument("--print-llm-io", action="store_true")
    qlib_smoke.add_argument("--no-color", action="store_true")

    dry = subparsers.add_parser("llm-dry-run", help="Send one minimal LLM request without search side effects.")
    dry.add_argument("--kind", choices=["candidate", "prior-rewrite"], required=True)
    dry.add_argument("--llm", choices=["mock", "openai-compatible"], default="mock")
    dry.add_argument("--print-llm-io", action="store_true")
    dry.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    if args.command == "smoke-run":
        reporter = _build_reporter(args)
        experiment_config = ExperimentConfig.from_toml(args.config)
        config = _search_config_from_args(args, experiment_config.search)
        # The offline smoke run stays mock by default even when default.toml
        # points real experiments at an OpenAI-compatible endpoint.
        candidate_provider = args.candidate_llm or args.llm or "mock"
        prior_provider = args.prior_llm or args.llm or "mock"
        run_id = _run_id("smoke", candidate_provider, prior_provider)
        run_dir = RunDirectoryResolver(args.log_dir).create(run_id=run_id, label="smoke")
        candidate_generator = _build_candidate_generator(candidate_provider, reporter=reporter, llm_settings=experiment_config.llm)
        seed_generator = _build_seed_generator(candidate_provider, reporter=reporter, llm_settings=experiment_config.llm)
        prior_rewriter = _build_prior_rewriter(prior_provider, reporter=reporter, llm_settings=experiment_config.llm)
        runner = ExperimentRunner(
            log_dir=run_dir,
            config=config,
            fusion_mode=FusionMode(args.fusion_mode),
            run_id=run_id,
            candidate_generator=candidate_generator,
            seed_generator=seed_generator,
            prior_rewriter=prior_rewriter,
            reporter=reporter,
            selection_config=experiment_config.selection,
            backtest_config=BacktestConfig(enabled=False),
            component_names={
                "candidate_generator": _component_name("candidate", candidate_provider),
                "seed_generator": _component_name("seed", candidate_provider),
                "prior_rewriter": _component_name("prior", prior_provider),
            },
        )
        engine = runner.run()
        print(f"run_id={engine.run_id}")
        print(f"valid_evaluated_count={engine.counters.valid_evaluated_count}")
        print(f"generated_count={engine.counters.generated_count}")
        print(f"log_dir={run_dir}")
    elif args.command == "qlib-smoke-run":
        reporter = _build_reporter(args)
        experiment_config = ExperimentConfig.from_toml(args.config)
        qlib_config = _qlib_config_from_args(args, experiment_config.qlib)
        config = _search_config_from_args(args, experiment_config.search)
        selection_config = _selection_config_from_args(args, experiment_config.selection)
        backtest_config = _backtest_config_from_args(args, experiment_config.backtest)
        candidate_provider = args.candidate_llm or args.llm or experiment_config.llm.get("candidate_provider", "mock")
        prior_provider = args.prior_llm or args.llm or experiment_config.llm.get("prior_provider", "mock")
        run_id = _run_id("qlib", candidate_provider, prior_provider)
        run_dir = RunDirectoryResolver(args.log_dir).create(run_id=run_id, label="qlib_smoke")
        candidate_generator = _build_candidate_generator(candidate_provider, reporter=reporter, llm_settings=experiment_config.llm)
        seed_generator = _build_seed_generator(candidate_provider, reporter=reporter, llm_settings=experiment_config.llm)
        prior_rewriter = _build_prior_rewriter(prior_provider, reporter=reporter, llm_settings=experiment_config.llm)
        runner = ExperimentRunner(
            log_dir=run_dir,
            config=config,
            run_id=run_id,
            candidate_generator=candidate_generator,
            seed_generator=seed_generator,
            prior_rewriter=prior_rewriter,
            evaluator=QlibEvaluator(qlib_config),
            validator=QlibExpressionValidator(qlib_config, max_length=config.factor_length_limit),
            reporter=reporter,
            qlib_config=qlib_config,
            selection_config=selection_config,
            backtest_config=backtest_config,
            component_names={
                "candidate_generator": _component_name("candidate", candidate_provider),
                "seed_generator": _component_name("seed", candidate_provider),
                "prior_rewriter": _component_name("prior", prior_provider),
                "evaluator": "QlibEvaluator",
            },
            extra_config={
                "merged_experiment_config": experiment_config.snapshot(),
                "qlib_config": qlib_config.__dict__,
                "console": {
                    "verbose": args.verbose or args.print_llm_io,
                    "print_llm_io": args.print_llm_io,
                    "color": not args.no_color,
                },
            },
        )
        engine = runner.run()
        print(f"run_id={engine.run_id}")
        print(f"valid_evaluated_count={engine.counters.valid_evaluated_count}")
        print(f"generated_count={engine.counters.generated_count}")
        print(f"market={qlib_config.market}")
        print(f"log_dir={run_dir}")
    elif args.command == "llm-dry-run":
        reporter = ConsoleReporter(enabled=args.print_llm_io, print_llm_io=args.print_llm_io, use_color=not args.no_color)
        if args.kind == "candidate":
            result = _build_candidate_generator(args.llm, reporter=reporter).generate(
                CandidateRequest(
                    operator="mutation",
                    parent_expressions=[FactorExpression("$close")],
                    fused_prior_context=FusedPriorContext(
                        mode=FusionMode.OURS_FULL,
                        operator="mutation",
                        lineage_id="dry_lineage",
                        prompt_context={"lineage_prior": {}, "global_prior": {}},
                    ),
                    constraints={"factor_length_limit": 40, "exactly_one_factor": True},
                )
            )
            print(result.raw_output)
        else:
            parent = FactorExpression("$close")
            child = FactorExpression("Rank($close)")
            rewrite_input = PriorRewriteInput(
                run_id="dry_run",
                generation=1,
                operator=OperatorType.MUTATION,
                target_prior_type=PriorTarget.MUTATION_LINEAGE,
                old_prior=default_mutation_prior(),
                parent_factors=[parent],
                child_factor=child,
                expression_diff=diff_expressions(parent, child),
                train_score=None,
                validation_score=None,
                delta_train_score=None,
                delta_validation_score=None,
                validity_info={"is_valid": True, "reasons": []},
            )
            raw = _build_prior_rewriter(args.llm, reporter=reporter).rewrite_mutation_prior(rewrite_input)
            print(raw.raw_content)


def _build_candidate_generator(provider: str, reporter=None, llm_settings: dict | None = None):
    if provider == "mock":
        return MockCandidateGenerator(ExperimentRunner._mock_candidate_outputs())
    return LLMCandidateGenerator(_openai_client(llm_settings), reporter=reporter)


def _build_seed_generator(provider: str, reporter=None, llm_settings: dict | None = None):
    if provider == "mock":
        return MockSeedGenerator(ExperimentRunner._mock_seed_outputs())
    return LLMSeedGenerator(_openai_client(llm_settings), reporter=reporter)


def _build_prior_rewriter(provider: str, reporter=None, llm_settings: dict | None = None):
    if provider == "mock":
        return MockPriorRewriter()
    return LLMPriorRewriter(_openai_client(llm_settings), reporter=reporter)


def _openai_client(llm_settings: dict | None = None) -> OpenAICompatibleLLMClient:
    base = LLMConfig.from_env()
    settings = llm_settings or {}
    return OpenAICompatibleLLMClient(
        LLMConfig(
            base_url=settings.get("base_url", base.base_url),
            api_key=base.api_key,
            model=settings.get("model", base.model),
            timeout_seconds=settings.get("timeout_seconds", base.timeout_seconds),
            max_retry=settings.get("max_retry", base.max_retry),
            retry_wait_seconds=settings.get("retry_wait_seconds", base.retry_wait_seconds),
            max_tokens=settings.get("max_tokens", base.max_tokens),
            temperature=settings.get("temperature", base.temperature),
        )
    )


def _component_name(kind: str, provider: str) -> str:
    if kind == "candidate":
        return "MockCandidateGenerator" if provider == "mock" else "LLMCandidateGenerator(OpenAICompatibleLLMClient)"
    if kind == "seed":
        return "MockSeedGenerator" if provider == "mock" else "LLMSeedGenerator(OpenAICompatibleLLMClient)"
    return "MockPriorRewriter" if provider == "mock" else "LLMPriorRewriter(OpenAICompatibleLLMClient)"


def _build_reporter(args) -> ConsoleReporter:
    return ConsoleReporter(
        enabled=bool(args.verbose or args.print_llm_io),
        print_llm_io=bool(args.print_llm_io),
        use_color=not bool(args.no_color),
    )


def _search_config_from_args(args, base: SearchConfig) -> SearchConfig:
    return SearchConfig(
        active_pool_size=base.active_pool_size,
        factor_length_limit=base.factor_length_limit,
        target_valid_evaluations=args.target_valid if args.target_valid is not None else base.target_valid_evaluations,
        valid_per_generation=base.valid_per_generation,
        mutation_per_generation=base.mutation_per_generation,
        crossover_per_generation=base.crossover_per_generation,
        max_attempts_per_operator_slot=base.max_attempts_per_operator_slot,
        max_runtime_seconds=base.max_runtime_seconds,
        use_local=base.use_local,
        seed_count=args.seed_count if args.seed_count is not None else base.seed_count,
        max_seed_generation_attempts=args.max_seed_generation_attempts
        if args.max_seed_generation_attempts is not None
        else base.max_seed_generation_attempts,
        parent_train_icir_weight=base.parent_train_icir_weight,
        parent_validation_icir_weight=base.parent_validation_icir_weight,
        parent_gap_penalty_weight=base.parent_gap_penalty_weight,
        parent_weight_epsilon=base.parent_weight_epsilon,
    )


def _selection_config_from_args(args, base: SelectionConfig) -> SelectionConfig:
    return SelectionConfig(
        final_top_k=args.final_top_k if args.final_top_k is not None else base.final_top_k,
        selection_metric=base.selection_metric,
    )


def _backtest_config_from_args(args, base: BacktestConfig) -> BacktestConfig:
    return BacktestConfig(
        enabled=False if args.skip_final_backtest else base.enabled,
        account=args.account if args.account is not None else base.account,
        topk=args.backtest_topk if args.backtest_topk is not None else base.topk,
        n_drop=args.backtest_n_drop if args.backtest_n_drop is not None else base.n_drop,
        risk_degree=base.risk_degree,
    )


def _qlib_config_from_args(args, base: QlibConfig) -> QlibConfig:
    return QlibConfig(
        provider_uri=args.provider_uri or base.provider_uri,
        region=base.region,
        market=args.market or base.market,
        train_start=args.train_start or base.train_start,
        train_end=args.train_end or base.train_end,
        valid_start=args.valid_start or base.valid_start,
        valid_end=args.valid_end or base.valid_end,
        test_start=args.test_start or base.test_start,
        test_end=args.test_end or base.test_end,
        label_expression=base.label_expression,
        ic_method=base.ic_method,
        benchmark=args.benchmark or base.benchmark,
    )


def _run_id(prefix: str, candidate_provider: str, prior_provider: str) -> str:
    candidate = "llm" if candidate_provider != "mock" else "mock"
    prior = "llmprior" if prior_provider != "mock" else "mockprior"
    return f"{prefix}_{candidate}_{prior}_{int(time.time())}"


if __name__ == "__main__":
    main()

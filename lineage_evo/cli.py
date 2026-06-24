"""Command line interface for LineageEvo."""

from __future__ import annotations

import argparse
import csv
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
from lineage_evo.ablation import AblationMode
from lineage_evo.prior_fusion import FusedPriorContext
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
    mode_choices = [mode.value for mode in AblationMode]
    smoke.add_argument("--ablation-mode", choices=mode_choices, default=None)
    smoke.add_argument("--fusion-mode", dest="fusion_mode", choices=mode_choices, default=None, help="Deprecated alias for --ablation-mode.")
    smoke.add_argument("--llm", choices=["mock", "openai-compatible"], default=None)
    smoke.add_argument("--candidate-llm", choices=["mock", "openai-compatible"], default=None)
    smoke.add_argument("--prior-llm", choices=["mock", "openai-compatible"], default=None)
    smoke.add_argument("--seed-count", type=int, default=None)
    smoke.add_argument("--train-only", action="store_true", help="Use train IC for search decisions and final selection.")
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
    qlib_smoke.add_argument("--train-only", action="store_true", help="Merge valid into train and use train IC for decisions.")
    qlib_smoke.add_argument("--max-seed-generation-attempts", type=int, default=None)
    qlib_smoke.add_argument("--final-top-k", type=int, default=None)
    qlib_smoke.add_argument("--backtest-topk", type=int, default=None)
    qlib_smoke.add_argument("--backtest-n-drop", type=int, default=None)
    qlib_smoke.add_argument("--account", type=float, default=None)
    qlib_smoke.add_argument("--skip-final-backtest", action="store_true")
    qlib_smoke.add_argument("--ablation-mode", choices=mode_choices, default=None)
    qlib_smoke.add_argument("--fusion-mode", dest="fusion_mode", choices=mode_choices, default=None, help="Deprecated alias for --ablation-mode.")
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
            ablation_mode=_ablation_mode_from_args(args),
            run_id=run_id,
            candidate_generator=candidate_generator,
            seed_generator=seed_generator,
            prior_rewriter=prior_rewriter,
            reporter=reporter,
            selection_config=experiment_config.selection,
            backtest_config=BacktestConfig(enabled=False),
            prior_update_config=experiment_config.prior_update,
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
        config = _search_config_from_args(args, experiment_config.search)
        qlib_config = _qlib_config_from_args(args, experiment_config.qlib, train_only=config.train_only)
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
            ablation_mode=_ablation_mode_from_args(args),
            reporter=reporter,
            qlib_config=qlib_config,
            selection_config=selection_config,
            backtest_config=backtest_config,
            prior_update_config=experiment_config.prior_update,
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
        _print_final_outputs(run_dir)
    elif args.command == "llm-dry-run":
        reporter = ConsoleReporter(enabled=args.print_llm_io, print_llm_io=args.print_llm_io, use_color=not args.no_color)
        if args.kind == "candidate":
            result = _build_candidate_generator(args.llm, reporter=reporter).generate(
                CandidateRequest(
                    operator="mutation",
                    parent_expressions=[FactorExpression("$close")],
                    fused_prior_context=FusedPriorContext(
                        mode=AblationMode.OURS_FULL,
                        operator="mutation",
                        lineage_id="dry_lineage",
                        prompt_context={
                            "rendered_priors": {
                                "lineage_prior_text": "Mutation experience for this lineage:\n- Recent trend: dry run.",
                                "global_prior_text": "Global mutation experience across lineages:\n- Hint: dry run.",
                            },
                            "fusion_decision": {
                                "local_weight": 0.65,
                                "global_weight": 0.35,
                                "reason": "dry run",
                                "instruction": "Balance lineage-specific guidance with global operator guidance.",
                            },
                        },
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


def _ablation_mode_from_args(args) -> AblationMode:
    value = getattr(args, "ablation_mode", None) or getattr(args, "fusion_mode", None) or AblationMode.OURS_FULL.value
    return AblationMode(value)


def _search_config_from_args(args, base: SearchConfig) -> SearchConfig:
    return SearchConfig(
        active_pool_size=base.active_pool_size,
        train_only=bool(getattr(args, "train_only", False) or base.train_only),
        elite_archive_size=base.elite_archive_size,
        max_active_lineage_ratio=base.max_active_lineage_ratio,
        min_active_lineages_before_cap=base.min_active_lineages_before_cap,
        lineage_concentration_weight=base.lineage_concentration_weight,
        factor_prompt_length_limit=base.factor_prompt_length_limit,
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
        parent_train_ic_weight=base.parent_train_ic_weight,
        parent_validation_ic_weight=base.parent_validation_ic_weight,
        parent_gap_penalty_weight=base.parent_gap_penalty_weight,
        parent_weight_epsilon=base.parent_weight_epsilon,
    )


def _selection_config_from_args(args, base: SelectionConfig) -> SelectionConfig:
    return SelectionConfig(
        final_top_k=args.final_top_k if args.final_top_k is not None else base.final_top_k,
        selection_metric="train_ic" if getattr(args, "train_only", False) else base.selection_metric,
    )


def _backtest_config_from_args(args, base: BacktestConfig) -> BacktestConfig:
    return BacktestConfig(
        enabled=False if args.skip_final_backtest else base.enabled,
        account=args.account if args.account is not None else base.account,
        topk=args.backtest_topk if args.backtest_topk is not None else base.topk,
        n_drop=args.backtest_n_drop if args.backtest_n_drop is not None else base.n_drop,
        risk_degree=base.risk_degree,
    )


def _qlib_config_from_args(args, base: QlibConfig, *, train_only: bool = False) -> QlibConfig:
    valid_end = args.valid_end or base.valid_end
    train_end = valid_end if train_only or getattr(args, "train_only", False) else (args.train_end or base.train_end)
    return QlibConfig(
        provider_uri=args.provider_uri or base.provider_uri,
        region=base.region,
        market=args.market or base.market,
        train_start=args.train_start or base.train_start,
        train_end=train_end,
        valid_start=args.valid_start or base.valid_start,
        valid_end=valid_end,
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


def _print_final_outputs(run_dir: Path) -> None:
    selected = _read_csv_rows(run_dir / "selected_factors.csv")
    test_rows = _read_csv_rows(run_dir / "test_ic_results.csv")
    composite_test_rows = _read_csv_rows(run_dir / "composite_test_ic_results.csv")
    backtest_rows = _read_csv_rows(run_dir / "backtest_summary.csv")
    daily_rows = _read_csv_rows(run_dir / "backtest_daily_report.csv")

    if selected:
        print("selected_factors:")
        for row in selected:
            print(
                "  "
                f"rank={row.get('selection_rank')} "
                f"factor_id={row.get('factor_id')} "
                f"raw_validation_ic={_fmt_float(row.get('raw_validation_ic') or row.get('validation_ic'))} "
                f"selection_score={_fmt_float(row.get('selection_score'))} "
                f"orientation={row.get('orientation')} "
                f"train_ic={_fmt_float(row.get('train_ic'))} "
                f"train_icir={_fmt_float(row.get('train_icir'))} "
                f"expr={_shorten(row.get('expression', ''))}"
            )

    if composite_test_rows:
        print("final_composite_test_results:")
        for row in composite_test_rows:
            print(
                "  "
                f"signal={row.get('signal_name')} "
                f"selected_count={row.get('selected_count')} "
                f"test_ic={_fmt_float(row.get('test_ic'))} "
                f"test_icir={_fmt_float(row.get('test_icir'))} "
                f"status={row.get('status')}"
            )

    if test_rows:
        print("single_factor_oriented_test_results:")
        for row in test_rows:
            print(
                "  "
                f"rank={row.get('selection_rank')} "
                f"factor_id={row.get('factor_id')} "
                f"test_ic={_fmt_float(row.get('test_ic'))} "
                f"test_icir={_fmt_float(row.get('test_icir'))} "
                f"status={row.get('status')}"
            )

    if backtest_rows:
        summary = backtest_rows[0]
        print("backtest_summary:")
        print(
            "  "
            f"status={summary.get('status')} "
            f"annualized_return={_fmt_float(summary.get('annualized_return'))} "
            f"information_ratio={_fmt_float(summary.get('information_ratio'))} "
            f"max_drawdown={_fmt_float(summary.get('max_drawdown'))} "
            f"benchmark={summary.get('benchmark')}"
        )

    if daily_rows and daily_rows[0].get("account") and daily_rows[-1].get("account"):
        first_account = _to_float(daily_rows[0].get("account"))
        final_account = _to_float(daily_rows[-1].get("account"))
        if first_account and final_account:
            account_return = final_account / first_account - 1.0
            benchmark_return = _compound_return(row.get("bench") for row in daily_rows)
            print(
                "backtest_account:"
                f" final_account={final_account:.2f}"
                f" account_return={account_return:.4f}"
                f" benchmark_return={benchmark_return:.4f}"
            )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fmt_float(value: str | None) -> str:
    parsed = _to_float(value)
    return "NA" if parsed is None else f"{parsed:.6f}"


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _compound_return(values) -> float:
    total = 1.0
    seen = False
    for value in values:
        parsed = _to_float(value)
        if parsed is None:
            continue
        seen = True
        total *= 1.0 + parsed
    return total - 1.0 if seen else 0.0


def _shorten(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    main()

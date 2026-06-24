"""Experiment runner for offline-first smoke runs."""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, replace
from pathlib import Path

from lineage_evo.candidate import CandidateGenerator, MockCandidateGenerator
from lineage_evo.config import BacktestConfig, PriorUpdateConfig, QlibConfig, SearchConfig, SelectionConfig
from lineage_evo.evaluation import Evaluator, MockEvaluator
from lineage_evo.experiments.defaults import (
    default_crossover_prior,
    default_global_crossover_prior,
    default_global_mutation_prior,
    default_mutation_prior,
)
from lineage_evo.lineage import LineageDAG
from lineage_evo.factor import QlibExpressionNormalizer, QlibExpressionValidator
from lineage_evo.finalize import Finalizer
from lineage_evo.ablation import AblationMode
from lineage_evo.prior_rewrite import MockPriorRewriter, MutationStrengthController, PriorManager, PriorUpdateTrigger
from lineage_evo.recording import SearchRecorder
from lineage_evo.search.engine import PriorStores, SearchEngine
from lineage_evo.seed import MockSeedGenerator, SeedRequest


class ExperimentRunner:
    """Assemble one smoke run; mock components remain the default."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        config: SearchConfig | None = None,
        ablation_mode: AblationMode = AblationMode.OURS_FULL,
        fusion_mode: AblationMode | None = None,
        run_id: str | None = None,
        candidate_generator: CandidateGenerator | None = None,
        prior_rewriter=None,
        evaluator: Evaluator | None = None,
        validator=None,
        seed_generator=None,
        qlib_config: QlibConfig | None = None,
        selection_config: SelectionConfig | None = None,
        backtest_config: BacktestConfig | None = None,
        prior_update_config: PriorUpdateConfig | None = None,
        component_names: dict[str, str] | None = None,
        extra_config: dict | None = None,
        reporter=None,
    ) -> None:
        self.config = config or SearchConfig(target_valid_evaluations=10)
        self.ablation_mode = fusion_mode or ablation_mode
        self.run_id = run_id or f"mock_{int(time.time())}"
        self.recorder = SearchRecorder(log_dir)
        self.candidate_generator = candidate_generator
        self.prior_rewriter = prior_rewriter
        self.evaluator = evaluator
        self.validator = validator
        self.seed_generator = seed_generator
        self.selection_config = selection_config or SelectionConfig()
        if self.config.train_only:
            self.selection_config = replace(self.selection_config, selection_metric="train_ic")
        self.backtest_config = backtest_config or BacktestConfig(enabled=False)
        self.prior_update_config = replace(prior_update_config or PriorUpdateConfig(), train_only=self.config.train_only)
        if self.config.train_only and qlib_config is not None:
            qlib_config = replace(qlib_config, train_end=qlib_config.valid_end)
        self.qlib_config = qlib_config
        self.component_names = component_names or {}
        self.extra_config = extra_config or {}
        self.reporter = reporter

    def run(self) -> SearchEngine:
        evaluator = self.evaluator or MockEvaluator()
        validator = self.validator or QlibExpressionValidator(max_length=self.config.factor_length_limit, execute_check=False)
        dag = LineageDAG(
            active_pool_size=self.config.active_pool_size,
            train_only=self.config.train_only,
            elite_archive_size=self.config.elite_archive_size,
            max_active_lineage_ratio=self.config.max_active_lineage_ratio,
            min_active_lineages_before_cap=self.config.min_active_lineages_before_cap,
        )
        self._generate_seed_nodes(dag, evaluator, validator)     #生成初始因子

        prior_stores = PriorStores(
            mutation_by_lineage={node.lineage_id: default_mutation_prior() for node in dag.nodes.values()},
            crossover_by_lineage={node.lineage_id: default_crossover_prior() for node in dag.nodes.values()},
            global_mutation=default_global_mutation_prior(),
            global_crossover=default_global_crossover_prior(),
        )
        candidate_generator = self.candidate_generator or MockCandidateGenerator(self._mock_candidate_outputs())
        prior_rewriter = self.prior_rewriter or MockPriorRewriter()
        prior_manager = PriorManager(
            logger=self.recorder,
            strength_controller=MutationStrengthController(self.prior_update_config),
        )
        engine = SearchEngine(
            run_id=self.run_id,
            dag=dag,
            validator=validator,
            evaluator=evaluator,
            prior_stores=prior_stores,
            prior_rewriter=prior_rewriter,
            prior_manager=prior_manager,
            config=self.config,
            candidate_generator=candidate_generator,
            prior_update_trigger=PriorUpdateTrigger(self.prior_update_config),
            ablation_mode=self.ablation_mode,
            recorder=self.recorder,
            reporter=self.reporter,
        )
        self.recorder.write_config_snapshot(
            {
                "run_id": self.run_id,
                "run_dir": str(self.recorder.log_dir),
                "search_config": asdict(self.config),
                "selection_config": asdict(self.selection_config),
                "backtest_config": asdict(self.backtest_config),
                "prior_update_config": asdict(self.prior_update_config),
                "ablation_mode": self.ablation_mode.value,
                "components": {
                    "prior_rewriter": self.component_names.get("prior_rewriter", "MockPriorRewriter"),
                    "candidate_generator": self.component_names.get("candidate_generator", "MockCandidateGenerator"),
                    "seed_generator": self.component_names.get("seed_generator", "MockSeedGenerator"),
                    "evaluator": self.component_names.get("evaluator", "MockEvaluator"),
                },
                **self.extra_config,
            }
        )
        engine.run_until_target(self.config.target_valid_evaluations)
        finalizer = Finalizer(
            qlib_config=self.qlib_config,
            selection_config=self.selection_config,
            backtest_config=self.backtest_config,
            recorder=self.recorder,
        )
        final_result = finalizer.run(engine.dag)
        if final_result.warnings:
            self.recorder.log_search({"run_id": self.run_id, "event": "finalization_warnings", "warnings": final_result.warnings})
        return engine

    def _generate_seed_nodes(self, dag: LineageDAG, evaluator: Evaluator, validator) -> None:
        seed_generator = self.seed_generator or MockSeedGenerator(self._mock_seed_outputs())
        normalizer = QlibExpressionNormalizer()
        normalized_seed_index: dict[str, str] = {}
        valid_count = 0
        attempts = 0
        while valid_count < self.config.seed_count and attempts < self.config.max_seed_generation_attempts:
            attempts += 1
            if self.reporter is not None:
                self.reporter.seed_attempt(attempt=attempts)
            generated = seed_generator.generate(
                SeedRequest(
                    seed_index=valid_count,
                    existing_seed_expressions=[dag.nodes[node_id].expression.raw for node_id in dag.active_ids],
                    constraints={
                        "factor_prompt_length_limit": self.config.factor_prompt_length_limit,
                        "factor_length_limit": self.config.factor_length_limit,
                        "exactly_one_factor": True,
                        "target_seed_count": self.config.seed_count,
                        "market": self.qlib_config.market if self.qlib_config is not None else "csi500",
                        "stock_universe": self.qlib_config.market if self.qlib_config is not None else "csi500",
                    },
                )
            )
            if not generated.parse_result.is_success:
                self._log_seed_candidate(attempts, generated.raw_output, "generation_failure", generated.parse_result.failure_reason)
                if self.reporter is not None:
                    self.reporter.failure(status="seed_generation_failure", reason=generated.parse_result.failure_reason, raw_output=generated.raw_output)
                continue
            expression = generated.parse_result.factor
            try:
                normalized = normalizer.normalize(expression)
            except Exception:
                normalized = None
            if normalized is not None and normalized in normalized_seed_index:
                self._log_seed_candidate(
                    attempts,
                    generated.raw_output,
                    "duplicate_candidate",
                    "duplicate normalized expression",
                    duplicate_of=normalized_seed_index[normalized],
                    normalized_expression=normalized,
                )
                if self.reporter is not None:
                    self.reporter.failure(status="seed_duplicate_candidate", reason="duplicate normalized expression", raw_output=generated.raw_output)
                continue
            validation = validator.validate(expression)
            if not validation.is_valid:
                reason = "; ".join(validation.reasons)
                self._log_seed_candidate(attempts, generated.raw_output, "validation_failure", reason)
                if self.reporter is not None:
                    self.reporter.failure(status="seed_validation_failure", reason=reason, raw_output=generated.raw_output)
                continue
            try:
                evaluation = evaluator.evaluate(expression)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                self._log_seed_candidate(attempts, generated.raw_output, "evaluation_failure", reason)
                if self.reporter is not None:
                    self.reporter.failure(status="seed_evaluation_failure", reason=reason, raw_output=generated.raw_output)
                continue
            node = dag.add_seed(expression, evaluation)
            if normalized is not None:
                normalized_seed_index.setdefault(normalized, node.factor_id)
            valid_count += 1
            self._log_seed_candidate(attempts, generated.raw_output, "valid", None, child_id=node.factor_id)
            if self.reporter is not None:
                self.reporter.valid_factor(expression=expression.raw, evaluation=evaluation, child_id=node.factor_id)
        if valid_count < self.config.seed_count:
            raise RuntimeError(
                f"only generated {valid_count} valid seed factors after {attempts} attempts; "
                f"required {self.config.seed_count}"
            )

    def _log_seed_candidate(
        self,
        attempt: int,
        raw_output: str,
        status: str,
        failure_reason: str | None,
        child_id: str | None = None,
        duplicate_of: str | None = None,
        normalized_expression: str | None = None,
    ) -> None:
        self.recorder.log_candidate(
            {
                "run_id": self.run_id,
                "generation": 0,
                "operator": "seed",
                "parent_ids": [],
                "child_id": child_id,
                "status": status,
                "failure_reason": failure_reason,
                "raw_output": raw_output,
                "duplicate_of": duplicate_of,
                "normalized_expression": normalized_expression,
                "attempt": attempt,
            }
        )

    @staticmethod
    def _mock_seed_outputs() -> list[str]:
        templates = [
            '{"factor": "$close", "rationale": "seed"}',
            '{"factor": "$open", "rationale": "seed"}',
            '{"factor": "$volume", "rationale": "seed"}',
            '{"factor": "$vwap", "rationale": "seed"}',
            '{"factor": "Rank($close)", "rationale": "seed"}',
            '{"factor": "TsMean($close, 5)", "rationale": "seed"}',
            '{"factor": "TsMean($open, 5)", "rationale": "seed"}',
            '{"factor": "TsStd($close, 10)", "rationale": "seed"}',
            '{"factor": "TsPctChange($close, 5)", "rationale": "seed"}',
            '{"factor": "TsCorr($close, $volume, 20)", "rationale": "seed"}',
        ]
        return list(itertools.islice(itertools.cycle(templates), 500))

    @staticmethod
    def _mock_candidate_outputs() -> list[str]:
        templates = [
            '{"factor": "Rank($close)", "rationale": "mock mutation"}',
            '{"factor": "TsMean($open, 5)", "rationale": "mock crossover"}',
            '{"factor": "TsMean($volume, 5)", "rationale": "mock mutation"}',
            '{"factor": "Div(Sub($open, $close), Add(Sub($high, $low), 0.001))", "rationale": "mock crossover"}',
            '{"factor": "TsCorr($close, $volume, 20)", "rationale": "mock mutation"}',
            '{"factor": "TsDelta($close, 5)", "rationale": "mock mutation"}',
            '{"factor": "TsStd($vwap, 10)", "rationale": "mock crossover"}',
            '{"factor": "TsSum($volume, 10)", "rationale": "mock mutation"}',
            '{"factor": "TsRank($close, 10)", "rationale": "mock crossover"}',
            '{"factor": "Abs(TsDelta($open, 5))", "rationale": "mock mutation"}',
        ]
        return list(itertools.islice(itertools.cycle(templates), 500))

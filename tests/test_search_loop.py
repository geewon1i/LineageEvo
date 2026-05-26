from lineage_evo.candidate import MockCandidateGenerator
from lineage_evo.config import SearchConfig
from lineage_evo.evaluation import EvaluationResult, MockEvaluator
from lineage_evo.factor import FactorExpression
from lineage_evo.lineage import LineageDAG
from lineage_evo.prior_rewrite import MockPriorRewriter, PriorManager
from lineage_evo.recording import SearchRecorder
from lineage_evo.search.engine import PriorStores, SearchEngine
from lineage_evo.validation import Validator
from tests.test_search_integration import crossover_prior, global_crossover_prior, global_mutation_prior, mutation_prior


class RaisingEvaluator:
    def evaluate(self, expression):
        raise RuntimeError("qlib data is empty")


def test_generation_failure_does_not_validate_or_rewrite():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    prior_rewriter = MockPriorRewriter()
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=MockEvaluator(),
        prior_stores=PriorStores(
            mutation_by_lineage={seed.lineage_id: mutation_prior()},
            crossover_by_lineage={seed.lineage_id: crossover_prior()},
            global_mutation=global_mutation_prior(),
            global_crossover=global_crossover_prior(),
        ),
        prior_rewriter=prior_rewriter,
        prior_manager=PriorManager(),
        config=SearchConfig(mutation_per_generation=1, crossover_per_generation=0, max_attempts_per_operator_slot=1),
        candidate_generator=MockCandidateGenerator(["not json"]),
    )

    result = engine.run_generation(1)

    assert result["valid_mutation"] == 0
    assert engine.counters.generated_count == 1
    assert engine.counters.generation_failure_count == 1
    assert engine.counters.validation_failure_count == 0
    assert engine.counters.evaluation_failure_count == 0
    assert len(prior_rewriter.calls) == 0


def test_evaluation_failure_does_not_enter_dag_or_rewrite():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    prior_rewriter = MockPriorRewriter()
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=RaisingEvaluator(),
        prior_stores=PriorStores(
            mutation_by_lineage={seed.lineage_id: mutation_prior()},
            crossover_by_lineage={seed.lineage_id: crossover_prior()},
            global_mutation=global_mutation_prior(),
            global_crossover=global_crossover_prior(),
        ),
        prior_rewriter=prior_rewriter,
        prior_manager=PriorManager(),
        config=SearchConfig(mutation_per_generation=1, crossover_per_generation=0, max_attempts_per_operator_slot=1),
        candidate_generator=MockCandidateGenerator(['{"factor": "rank(close)", "rationale": "test"}']),
    )

    result = engine.run_generation(1)

    assert result["valid_mutation"] == 0
    assert engine.counters.generated_count == 1
    assert engine.counters.validation_failure_count == 0
    assert engine.counters.evaluation_failure_count == 1
    assert len(dag.nodes) == 1
    assert len(prior_rewriter.calls) == 0


def test_duplicate_candidate_does_not_enter_dag_or_rewrite():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    prior_rewriter = MockPriorRewriter()
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=MockEvaluator(),
        prior_stores=PriorStores(
            mutation_by_lineage={seed.lineage_id: mutation_prior()},
            crossover_by_lineage={seed.lineage_id: crossover_prior()},
            global_mutation=global_mutation_prior(),
            global_crossover=global_crossover_prior(),
        ),
        prior_rewriter=prior_rewriter,
        prior_manager=PriorManager(),
        config=SearchConfig(mutation_per_generation=1, crossover_per_generation=0, max_attempts_per_operator_slot=1),
        candidate_generator=MockCandidateGenerator(['{"factor": "$close", "rationale": "duplicate"}']),
    )

    result = engine.run_generation(1)

    assert result["valid_mutation"] == 0
    assert engine.counters.generated_count == 1
    assert engine.counters.duplicate_candidate_count == 1
    assert engine.counters.validation_failure_count == 0
    assert engine.counters.evaluation_failure_count == 0
    assert len(dag.nodes) == 1
    assert len(prior_rewriter.calls) == 0


def test_duplicate_retry_reuses_parent_and_adds_feedback():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    generator = MockCandidateGenerator(
        [
            '{"factor": "$close", "rationale": "duplicate"}',
            '{"factor": "rank(close)", "rationale": "different"}',
        ]
    )
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=MockEvaluator(),
        prior_stores=PriorStores(
            mutation_by_lineage={seed.lineage_id: mutation_prior()},
            crossover_by_lineage={seed.lineage_id: crossover_prior()},
            global_mutation=global_mutation_prior(),
            global_crossover=global_crossover_prior(),
        ),
        prior_rewriter=MockPriorRewriter(),
        prior_manager=PriorManager(),
        config=SearchConfig(mutation_per_generation=1, crossover_per_generation=0, max_attempts_per_operator_slot=2),
        candidate_generator=generator,
    )

    result = engine.run_generation(1)

    assert result["valid_mutation"] == 1
    assert engine.counters.generated_count == 2
    assert engine.counters.duplicate_candidate_count == 1
    assert len(generator.requests) == 2
    assert generator.requests[1].parent_ids == generator.requests[0].parent_ids
    assert generator.requests[1].duplicate_feedback == [
        {
            "raw_factor": "$close",
            "normalized_expression": "$close",
            "duplicate_of": seed.factor_id,
        }
    ]


def test_run_until_target_stops_on_runtime_limit(tmp_path):
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=MockEvaluator(),
        prior_stores=PriorStores(
            mutation_by_lineage={seed.lineage_id: mutation_prior()},
            crossover_by_lineage={seed.lineage_id: crossover_prior()},
            global_mutation=global_mutation_prior(),
            global_crossover=global_crossover_prior(),
        ),
        prior_rewriter=MockPriorRewriter(),
        prior_manager=PriorManager(),
        config=SearchConfig(max_runtime_seconds=1),
        candidate_generator=MockCandidateGenerator(['{"factor": "rank(close)", "rationale": "test"}']),
        recorder=SearchRecorder(tmp_path),
    )
    engine._runtime_exceeded = lambda start_time: True

    engine.run_until_target(10)

    assert engine.stop_reason == "max_runtime_seconds"
    assert engine.counters.generated_count == 0
    assert "max_runtime_seconds" in (tmp_path / "summary_results.csv").read_text(encoding="utf-8")

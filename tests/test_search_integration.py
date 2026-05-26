import json

from lineage_evo.evaluation import EvaluationResult, MockEvaluator
from lineage_evo.factor import FactorExpression
from lineage_evo.lineage import LineageDAG
from lineage_evo.llm import MockLLMClient
from lineage_evo.prior_rewrite import LLMPriorRewriter, MockPriorRewriter, PriorManager
from lineage_evo.priors import CrossoverPrior, GlobalCrossoverPrior, GlobalMutationPrior, MutationPrior
from lineage_evo.recording import SearchRecorder
from lineage_evo.search.engine import PriorStores, SearchEngine
from lineage_evo.validation import Validator


class FixedEvaluator:
    def __init__(self, result: EvaluationResult):
        self.result = result

    def evaluate(self, expression):
        return self.result


class FailingGlobalMutationRewriter(MockPriorRewriter):
    def rewrite_global_mutation_prior(self, rewrite_input):
        self.calls.append(rewrite_input)
        raise TimeoutError("The read operation timed out")


def mutation_prior(label="flat"):
    return MutationPrior(
        successful_mutation_patterns=[],
        failed_mutation_patterns=[],
        hint=label,
        bias_risk="low",
    )


def crossover_prior():
    return CrossoverPrior(
        transferable_patterns=[],
        harmful_patterns=[],
        complementarity_profile="unknown",
        heritable_structures=[],
        hint="crossover hint",
        crossover_risk="low",
    )


def global_mutation_prior():
    return GlobalMutationPrior(
        global_successful_mutation_patterns=[],
        global_failed_mutation_patterns=[],
        common_invalid_patterns=[],
        hint="stay diverse",
        last_updated_generation=0,
    )


def global_crossover_prior():
    return GlobalCrossoverPrior(
        global_transferable_patterns=[],
        global_harmful_patterns=[],
        global_complementarity_patterns=[],
        hint="combine complementary structures",
        last_updated_generation=0,
    )


def test_mutation_candidate_updates_lineage_and_global_priors():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    updated_lineage = {
        "successful_mutation_patterns": [
            {
                "pattern": "rank transform",
                "evidence": "validation strength improved",
                "confidence": "medium",
                "support_count": 1,
                "last_updated_generation": 1,
            }
        ],
        "failed_mutation_patterns": [],
        "hint": "rank transforms may help this lineage",
        "bias_risk": "low",
    }
    updated_global = global_mutation_prior()
    updated_global.last_updated_generation = 1
    client = MockLLMClient([json.dumps(updated_lineage), updated_global.model_dump_json()])
    stores = PriorStores(
        mutation_by_lineage={seed.lineage_id: mutation_prior()},
        crossover_by_lineage={seed.lineage_id: crossover_prior()},
        global_mutation=global_mutation_prior(),
        global_crossover=global_crossover_prior(),
    )
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=MockEvaluator(),
        prior_stores=stores,
        prior_rewriter=LLMPriorRewriter(client),
        prior_manager=PriorManager(),
    )

    child = engine.accept_mutation_candidate(seed.factor_id, FactorExpression("rank(close)"), generation=1)

    assert child is not None
    assert child.lineage_id == seed.lineage_id
    assert stores.mutation_by_lineage[seed.lineage_id].successful_mutation_patterns[0].pattern == "rank transform"
    assert stores.mutation_by_lineage[seed.lineage_id].hint == "rank transforms may help this lineage"
    assert stores.global_mutation.last_updated_generation == 1


def test_minor_fluctuation_valid_child_skips_prior_rewrite():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    prior_rewriter = MockPriorRewriter()
    stores = PriorStores(
        mutation_by_lineage={seed.lineage_id: mutation_prior()},
        crossover_by_lineage={seed.lineage_id: crossover_prior()},
        global_mutation=global_mutation_prior(),
        global_crossover=global_crossover_prior(),
    )
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=FixedEvaluator(EvaluationResult(0, 0.1, 0, 0.105)),
        prior_stores=stores,
        prior_rewriter=prior_rewriter,
        prior_manager=PriorManager(),
    )

    child = engine.accept_mutation_candidate(seed.factor_id, FactorExpression("rank(close)"), generation=1)

    assert child is not None
    assert len(prior_rewriter.calls) == 0


def test_validation_strength_improvement_triggers_prior_rewrite():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, -0.1))
    prior_rewriter = MockPriorRewriter()
    stores = PriorStores(
        mutation_by_lineage={seed.lineage_id: mutation_prior()},
        crossover_by_lineage={seed.lineage_id: crossover_prior()},
        global_mutation=global_mutation_prior(),
        global_crossover=global_crossover_prior(),
    )
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=FixedEvaluator(EvaluationResult(0, 0.1, 0, -0.13)),
        prior_stores=stores,
        prior_rewriter=prior_rewriter,
        prior_manager=PriorManager(),
    )

    child = engine.accept_mutation_candidate(seed.factor_id, FactorExpression("rank(close)"), generation=1)

    assert child is not None
    assert len(prior_rewriter.calls) == 2
    assert prior_rewriter.calls[0].update_trigger["trigger_reason"] == "significant_validation_improvement"


def test_prior_rewrite_timeout_falls_back_and_keeps_valid_child(tmp_path):
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, -0.1))
    prior_rewriter = FailingGlobalMutationRewriter()
    old_global = global_mutation_prior()
    stores = PriorStores(
        mutation_by_lineage={seed.lineage_id: mutation_prior()},
        crossover_by_lineage={seed.lineage_id: crossover_prior()},
        global_mutation=old_global,
        global_crossover=global_crossover_prior(),
    )
    recorder = SearchRecorder(tmp_path)
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=FixedEvaluator(EvaluationResult(0, 0.1, 0, -0.13)),
        prior_stores=stores,
        prior_rewriter=prior_rewriter,
        prior_manager=PriorManager(logger=recorder),
        recorder=recorder,
    )

    child = engine.accept_mutation_candidate(seed.factor_id, FactorExpression("rank(close)"), generation=1)

    assert child is not None
    assert stores.global_mutation == old_global
    log_text = (tmp_path / "prior_rewrite_log.jsonl").read_text(encoding="utf-8")
    assert '"target_prior_type": "global_mutation"' in log_text
    assert '"fallback_used": true' in log_text
    assert "TimeoutError" in log_text


def test_invalid_candidate_does_not_enter_dag_or_rewrite_priors():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    prior_rewriter = MockPriorRewriter()
    stores = PriorStores(
        mutation_by_lineage={seed.lineage_id: mutation_prior()},
        crossover_by_lineage={seed.lineage_id: crossover_prior()},
        global_mutation=global_mutation_prior(),
        global_crossover=global_crossover_prior(),
    )
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close"}, {"rank"}),
        evaluator=MockEvaluator(),
        prior_stores=stores,
        prior_rewriter=prior_rewriter,
        prior_manager=PriorManager(),
    )

    child = engine.accept_mutation_candidate(seed.factor_id, FactorExpression("rank(foo)"), generation=2)

    assert child is None
    assert len(dag.nodes) == 1
    assert stores.global_mutation.common_invalid_patterns == []
    assert len(prior_rewriter.calls) == 0
    assert engine.counters.validation_failure_count == 1


def test_crossover_candidate_uses_higher_validation_parent_as_primary_lineage():
    dag = LineageDAG()
    weak = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    strong = dag.add_seed(FactorExpression("open"), EvaluationResult(0, 0.2, 0, 0.9))
    updated_cross = crossover_prior()
    updated_cross.complementarity_profile = "ranked price inputs transfer"
    updated_global = global_crossover_prior()
    updated_global.last_updated_generation = 3
    client = MockLLMClient([updated_cross.model_dump_json(), updated_global.model_dump_json()])
    stores = PriorStores(
        mutation_by_lineage={weak.lineage_id: mutation_prior(), strong.lineage_id: mutation_prior()},
        crossover_by_lineage={weak.lineage_id: crossover_prior(), strong.lineage_id: crossover_prior()},
        global_mutation=global_mutation_prior(),
        global_crossover=global_crossover_prior(),
    )
    engine = SearchEngine(
        run_id="run",
        dag=dag,
        validator=Validator({"close", "open"}, {"rank"}),
        evaluator=MockEvaluator(),
        prior_stores=stores,
        prior_rewriter=LLMPriorRewriter(client),
        prior_manager=PriorManager(),
    )

    child = engine.accept_crossover_candidate(weak.factor_id, strong.factor_id, FactorExpression("rank(open)"), generation=3)

    assert child is not None
    assert child.lineage_id == strong.lineage_id
    primary_edge = [edge for edge in dag.edges if edge.child_id == child.factor_id and edge.role == "primary"][0]
    assert "open" in primary_edge.expression_diff.unchanged_tokens
    assert "close" not in primary_edge.expression_diff.removed_tokens
    assert stores.crossover_by_lineage[strong.lineage_id].complementarity_profile == "ranked price inputs transfer"
    assert stores.crossover_by_lineage[weak.lineage_id].complementarity_profile == "unknown"
    assert stores.global_crossover.last_updated_generation == 3

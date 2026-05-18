from lineage_evo.evaluation import EvaluationResult, MockEvaluator
from lineage_evo.factor import FactorExpression
from lineage_evo.lineage import LineageDAG
from lineage_evo.llm import MockLLMClient
from lineage_evo.prior_rewrite import LLMPriorRewriter, MockPriorRewriter, PriorManager
from lineage_evo.priors import CrossoverPrior, GlobalCrossoverPrior, GlobalMutationPrior, MutationPrior
from lineage_evo.search.engine import PriorStores, SearchEngine
from lineage_evo.validation import Validator


def mutation_prior(label="flat"):
    return MutationPrior(
        quality_trend=label,
        successful_mutation_patterns=[],
        failed_mutation_patterns=[],
        mutation_strength="moderate",
        stagnation_state="not_stagnant",
        bias_risk="low",
    )


def crossover_prior():
    return CrossoverPrior(
        transferable_patterns=[],
        harmful_patterns=[],
        complementarity_profile="unknown",
        heritable_structures=[],
        crossover_risk="low",
    )


def global_mutation_prior():
    return GlobalMutationPrior(
        global_successful_mutation_patterns=[],
        global_failed_mutation_patterns=[],
        common_invalid_patterns=[],
        general_mutation_guidance="stay diverse",
        last_updated_generation=0,
    )


def global_crossover_prior():
    return GlobalCrossoverPrior(
        global_transferable_patterns=[],
        global_harmful_patterns=[],
        global_complementarity_patterns=[],
        general_crossover_guidance="combine complementary structures",
        last_updated_generation=0,
    )


def test_mutation_candidate_updates_lineage_and_global_priors():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    updated_lineage = mutation_prior("updated lineage")
    updated_global = global_mutation_prior()
    updated_global.last_updated_generation = 1
    client = MockLLMClient([updated_lineage.model_dump_json(), updated_global.model_dump_json()])
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
    assert stores.mutation_by_lineage[seed.lineage_id].quality_trend == "updated lineage"
    assert stores.global_mutation.last_updated_generation == 1


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
    assert stores.crossover_by_lineage[strong.lineage_id].complementarity_profile == "ranked price inputs transfer"
    assert stores.crossover_by_lineage[weak.lineage_id].complementarity_profile == "unknown"
    assert stores.global_crossover.last_updated_generation == 3

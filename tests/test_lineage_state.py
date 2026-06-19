from lineage_evo.evaluation import EvaluationResult
from lineage_evo.factor import FactorExpression, diff_expressions
from lineage_evo.lineage import LineageDAG


def test_lineage_base_state_updates_after_mutation_child():
    dag = LineageDAG()
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0.0, 0.1, 0.01, 0.1))
    child_expr = FactorExpression("rank(close)")
    child = dag.add_mutation_child(
        seed.factor_id,
        child_expr,
        EvaluationResult(0.0, 0.2, 0.05, 0.5),
        generation=1,
        expression_diff=diff_expressions(seed.expression, child_expr),
    )

    state = dag.lineage_state(seed.lineage_id)

    assert child.lineage_id == seed.lineage_id
    assert state.lineage_id == seed.lineage_id
    assert state.representative_factor_id == seed.factor_id
    assert state.best_factor_id == child.factor_id
    assert state.best_validation_ic == 0.05
    assert state.best_validation_icir == 0.5
    assert state.age == 1
    assert state.size == 2
    assert state.active_size == 2


def test_lineage_base_state_active_size_updates_after_pruning():
    dag = LineageDAG(active_pool_size=1)
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0.0, 0.1, 0.05, 0.5))
    child_expr = FactorExpression("rank(close)")
    dag.add_mutation_child(
        seed.factor_id,
        child_expr,
        EvaluationResult(0.0, 0.2, -0.01, -0.1),
        generation=1,
        expression_diff=diff_expressions(seed.expression, child_expr),
    )

    state = dag.lineage_state(seed.lineage_id)

    assert state.size == 2
    assert state.active_size == 1
    assert state.best_factor_id == seed.factor_id


def test_crossover_child_updates_only_primary_lineage_state():
    dag = LineageDAG()
    weak = dag.add_seed(FactorExpression("close"), EvaluationResult(0.0, 0.1, 0.01, 0.1))
    strong = dag.add_seed(FactorExpression("open"), EvaluationResult(0.0, 0.2, 0.09, 0.9))
    child_expr = FactorExpression("rank(open)")
    child = dag.add_crossover_child(
        weak.factor_id,
        strong.factor_id,
        child_expr,
        EvaluationResult(0.0, 0.3, 0.04, 0.4),
        generation=1,
        expression_diff=diff_expressions(strong.expression, child_expr),
    )

    strong_state = dag.lineage_state(strong.lineage_id)
    weak_state = dag.lineage_state(weak.lineage_id)

    assert child.lineage_id == strong.lineage_id
    assert strong_state.size == 2
    assert weak_state.size == 1


def test_elite_archive_keeps_top_validation_ic_independent_of_active_pruning():
    dag = LineageDAG(active_pool_size=2, elite_archive_size=3, max_active_lineage_ratio=0.5)
    seed = dag.add_seed(FactorExpression("close"), EvaluationResult(0.0, 0.1, 0.09, 0.9))
    child_expr = FactorExpression("rank(close)")
    child = dag.add_mutation_child(
        seed.factor_id,
        child_expr,
        EvaluationResult(0.0, 0.2, 0.08, 0.8),
        generation=1,
        expression_diff=diff_expressions(seed.expression, child_expr),
    )
    other = dag.add_seed(FactorExpression("open"), EvaluationResult(0.0, 0.1, 0.01, 0.1))

    assert child.factor_id not in dag.active_ids
    assert dag.nodes[child.factor_id].is_active is False
    assert child.factor_id in dag.elite_ids
    assert dag.elite_ids[:3] == [seed.factor_id, child.factor_id, other.factor_id]


def test_active_by_lineage_returns_current_active_buckets():
    dag = LineageDAG()
    first = dag.add_seed(FactorExpression("close"), EvaluationResult(0.0, 0.1, 0.09, 0.9))
    second = dag.add_seed(FactorExpression("open"), EvaluationResult(0.0, 0.1, 0.08, 0.8))

    buckets = dag.active_by_lineage()

    assert buckets == {
        first.lineage_id: [first.factor_id],
        second.lineage_id: [second.factor_id],
    }

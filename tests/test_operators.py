import random
import pytest

from lineage_evo.config import SearchConfig
from lineage_evo.evaluation import EvaluationResult
from lineage_evo.factor import FactorExpression
from lineage_evo.lineage import LineageDAG, OperatorType
from lineage_evo.operators import OperatorSchedule, ParentSelector


def test_operator_schedule_defaults_to_three_mutation_two_crossover():
    slots = OperatorSchedule(SearchConfig()).slots_for_generation()
    assert [slot.operator for slot in slots].count(OperatorType.MUTATION) == 3
    assert [slot.operator for slot in slots].count(OperatorType.CROSSOVER) == 2


def test_crossover_parent_selection_uses_distinct_factors_and_prefers_lineages():
    dag = LineageDAG()
    first = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 0.1, 0, 0.1))
    second = dag.add_seed(FactorExpression("open"), EvaluationResult(0, 0.2, 0, 0.9))
    parent_a, parent_b = ParentSelector(rng=random.Random(1)).select_crossover_parents(dag)
    assert parent_a.factor_id != parent_b.factor_id
    assert {parent_a.lineage_id, parent_b.lineage_id} == {first.lineage_id, second.lineage_id}


def test_mutation_parent_selection_uses_lineage_first_penalty():
    dag = LineageDAG()
    dominant = dag.add_seed(FactorExpression("close"), EvaluationResult(0.10, 1.0, 0.10, 1.0))
    child_expr = FactorExpression("rank(close)")
    dag.add_mutation_child(
        dominant.factor_id,
        child_expr,
        EvaluationResult(0.09, 0.9, 0.09, 0.9),
        generation=1,
        expression_diff=None,
    )
    balanced = dag.add_seed(FactorExpression("open"), EvaluationResult(0.10, 1.0, 0.10, 1.0))
    selector = ParentSelector(SearchConfig(lineage_concentration_weight=0.20), rng=random.Random(1))

    dominant_score = selector._lineage_score(dominant.lineage_id, dag.active_by_lineage()[dominant.lineage_id], dag)
    balanced_score = selector._lineage_score(balanced.lineage_id, dag.active_by_lineage()[balanced.lineage_id], dag)

    assert balanced_score > dominant_score


def test_parent_score_uses_combined_ic_and_gap_penalty():
    dag = LineageDAG()
    stable = dag.add_seed(FactorExpression("close"), EvaluationResult(0.10, 1.0, 0.10, 1.0))
    overfit = dag.add_seed(FactorExpression("open"), EvaluationResult(0.20, 2.0, -0.10, -1.0))
    selector = ParentSelector(SearchConfig(parent_gap_penalty_weight=0.2), rng=random.Random(1))

    assert selector.parent_score(stable) == pytest.approx(0.10)
    assert selector.parent_score(overfit) == pytest.approx(0.15)
    assert selector.parent_score(overfit) > selector.parent_score(stable)


def test_parent_score_penalizes_large_train_validation_ic_gap():
    dag = LineageDAG()
    stable = dag.add_seed(FactorExpression("close"), EvaluationResult(0.10, 1.0, 0.10, 1.0))
    overfit = dag.add_seed(FactorExpression("open"), EvaluationResult(0.30, 3.0, 0.02, 0.2))
    selector = ParentSelector(SearchConfig(parent_gap_penalty_weight=1.0), rng=random.Random(1))

    assert selector.parent_score(stable) == pytest.approx(0.10)
    assert selector.parent_score(overfit) < selector.parent_score(stable)
    assert selector._score_weighted_choice([stable, overfit]).factor_id == stable.factor_id


def test_train_only_parent_score_uses_train_ic_only():
    dag = LineageDAG()
    train_strong = dag.add_seed(FactorExpression("close"), EvaluationResult(0.20, 2.0, 0.01, 0.1))
    valid_strong = dag.add_seed(FactorExpression("open"), EvaluationResult(0.05, 0.5, 0.30, 3.0))
    selector = ParentSelector(SearchConfig(train_only=True), rng=random.Random(1))

    assert selector.parent_score(train_strong) == pytest.approx(0.20)
    assert selector.parent_score(valid_strong) == pytest.approx(0.05)

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


def test_parent_score_uses_combined_icir_and_gap_penalty():
    dag = LineageDAG()
    stable = dag.add_seed(FactorExpression("close"), EvaluationResult(0, 1.0, 0, 1.0))
    overfit = dag.add_seed(FactorExpression("open"), EvaluationResult(0, 2.0, 0, -1.0))
    selector = ParentSelector(SearchConfig(parent_gap_penalty_weight=0.2), rng=random.Random(1))

    assert selector.parent_score(stable) == 1.0
    assert selector.parent_score(overfit) == pytest.approx(0.5)
    assert selector._score_weighted_choice([stable, overfit]).factor_id == stable.factor_id

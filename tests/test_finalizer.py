import csv

from lineage_evo.config import BacktestConfig, SelectionConfig
from lineage_evo.evaluation import EvaluationResult
from lineage_evo.factor import FactorExpression
from lineage_evo.finalize import Finalizer
from lineage_evo.lineage import LineageDAG
from lineage_evo.recording import SearchRecorder


def test_finalizer_selects_top_validation_icir_and_writes_selected_csv(tmp_path):
    dag = LineageDAG()
    low = dag.add_seed(FactorExpression("$close"), EvaluationResult(0.0, 0.1, 0.0, -0.2))
    high = dag.add_seed(FactorExpression("Rank($close)"), EvaluationResult(0.0, 0.2, 0.0, 0.8))
    mid = dag.add_seed(FactorExpression("TsMean($close, 5)"), EvaluationResult(0.0, 0.3, 0.0, 0.1))

    finalizer = Finalizer(
        qlib_config=None,
        selection_config=SelectionConfig(final_top_k=2),
        backtest_config=BacktestConfig(enabled=False),
        recorder=SearchRecorder(tmp_path),
    )

    result = finalizer.run(dag)

    rows = list(csv.DictReader((tmp_path / "selected_factors.csv").open(encoding="utf-8")))
    assert result.selected_count == 2
    assert rows[0]["factor_id"] == high.factor_id
    assert rows[1]["factor_id"] == mid.factor_id
    assert low.factor_id not in {row["factor_id"] for row in rows}

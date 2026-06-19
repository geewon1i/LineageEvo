import csv
import sys
import types

import pandas as pd

from lineage_evo.config import BacktestConfig, QlibConfig, SelectionConfig
from lineage_evo.evaluation import EvaluationResult
from lineage_evo.factor import FactorExpression
from lineage_evo.finalize import Finalizer
from lineage_evo.lineage import LineageDAG
from lineage_evo.recording import SearchRecorder


def test_finalizer_selects_top_absolute_validation_ic_and_writes_orientation(tmp_path):
    dag = LineageDAG()
    negative_strong = dag.add_seed(FactorExpression("$close"), EvaluationResult(0.0, 0.1, -0.09, -0.9))
    high = dag.add_seed(FactorExpression("Rank($close)"), EvaluationResult(0.0, 0.2, 0.08, 0.8))
    mid = dag.add_seed(FactorExpression("TsMean($close, 5)"), EvaluationResult(0.0, 0.3, 0.01, 0.1))

    finalizer = Finalizer(
        qlib_config=None,
        selection_config=SelectionConfig(final_top_k=2),
        backtest_config=BacktestConfig(enabled=False),
        recorder=SearchRecorder(tmp_path),
    )

    result = finalizer.run(dag)

    rows = list(csv.DictReader((tmp_path / "selected_factors.csv").open(encoding="utf-8")))
    assert result.selected_count == 2
    assert rows[0]["factor_id"] == negative_strong.factor_id
    assert rows[0]["selection_source"] == "elite_archive"
    assert rows[0]["orientation"] == "-1"
    assert rows[0]["selection_score"] == "0.09"
    assert rows[1]["factor_id"] == high.factor_id
    assert mid.factor_id not in {row["factor_id"] for row in rows}


def test_finalizer_skips_duplicate_normalized_expressions(tmp_path):
    dag = LineageDAG()
    best = dag.add_seed(FactorExpression("$close"), EvaluationResult(0.0, 0.1, 0.09, 0.9))
    duplicate = dag.add_seed(FactorExpression("close"), EvaluationResult(0.0, 0.2, 0.08, 0.8))
    distinct = dag.add_seed(FactorExpression("Rank($open)"), EvaluationResult(0.0, 0.3, 0.07, 0.7))

    finalizer = Finalizer(
        qlib_config=None,
        selection_config=SelectionConfig(final_top_k=2),
        backtest_config=BacktestConfig(enabled=False),
        recorder=SearchRecorder(tmp_path),
    )

    selected = finalizer.select(dag)

    assert [node.factor_id for node in selected] == [best.factor_id, distinct.factor_id]
    assert duplicate.factor_id not in {node.factor_id for node in selected}


def test_finalizer_prefers_elite_archive_over_active_pool(tmp_path):
    dag = LineageDAG(active_pool_size=1, elite_archive_size=3)
    strong = dag.add_seed(FactorExpression("$close"), EvaluationResult(0.0, 0.1, 0.09, 0.9))
    weak = dag.add_seed(FactorExpression("$open"), EvaluationResult(0.0, 0.2, 0.01, 0.1))
    assert weak.factor_id in dag.active_ids
    assert strong.factor_id not in dag.active_ids
    assert strong.factor_id in dag.elite_ids

    finalizer = Finalizer(
        qlib_config=None,
        selection_config=SelectionConfig(final_top_k=1),
        backtest_config=BacktestConfig(enabled=False),
        recorder=SearchRecorder(tmp_path),
    )

    selected = finalizer.select(dag)

    assert [node.factor_id for node in selected] == [strong.factor_id]


def test_test_ic_rows_use_validation_orientation(tmp_path):
    dag = LineageDAG()
    node = dag.add_seed(FactorExpression("$close"), EvaluationResult(0.0, 0.1, -0.09, -0.9))
    finalizer = Finalizer(
        qlib_config=QlibConfig(),
        selection_config=SelectionConfig(final_top_k=1),
        backtest_config=BacktestConfig(enabled=False),
        recorder=SearchRecorder(tmp_path),
    )
    index = pd.MultiIndex.from_product(
        [["2020-01-01", "2020-01-02"], ["a", "b"]],
        names=["datetime", "instrument"],
    )
    raw_negative = pd.DataFrame(
        {"factor": [2.0, 1.0, 4.0, 3.0], "label": [1.0, 2.0, 3.0, 4.0]},
        index=index,
    )
    finalizer._load_factor_label = lambda _expr: raw_negative.copy()

    rows = finalizer._test_ic_rows([node])

    assert rows[0]["status"] == "ok"
    assert rows[0]["test_ic"] > 0


def test_composite_signal_uses_validation_orientation(tmp_path, monkeypatch):
    dag = LineageDAG()
    node = dag.add_seed(FactorExpression("$close"), EvaluationResult(0.0, 0.1, -0.09, -0.9))
    finalizer = Finalizer(
        qlib_config=QlibConfig(),
        selection_config=SelectionConfig(final_top_k=1),
        backtest_config=BacktestConfig(enabled=False),
        recorder=SearchRecorder(tmp_path),
    )
    finalizer._ensure_qlib = lambda: None
    index = pd.MultiIndex.from_product(
        [["2020-01-01"], ["a", "b"]],
        names=["datetime", "instrument"],
    )

    class FakeD:
        @staticmethod
        def instruments(_market):
            return ["a", "b"]

        @staticmethod
        def features(_instruments, fields, start_time=None, end_time=None):
            return pd.DataFrame({fields[0]: [2.0, 1.0]}, index=index)

    qlib_module = types.ModuleType("qlib")
    qlib_data_module = types.ModuleType("qlib.data")
    qlib_data_module.D = FakeD
    monkeypatch.setitem(sys.modules, "qlib", qlib_module)
    monkeypatch.setitem(sys.modules, "qlib.data", qlib_data_module)

    signal = finalizer._composite_signal([node])

    assert signal.loc[("2020-01-01", "b")] > signal.loc[("2020-01-01", "a")]


def test_composite_test_ic_rows_use_oriented_equal_weight_signal(tmp_path):
    dag = LineageDAG()
    node = dag.add_seed(FactorExpression("$close"), EvaluationResult(0.0, 0.1, -0.09, -0.9))
    finalizer = Finalizer(
        qlib_config=QlibConfig(),
        selection_config=SelectionConfig(final_top_k=1),
        backtest_config=BacktestConfig(enabled=False),
        recorder=SearchRecorder(tmp_path),
    )
    index = pd.MultiIndex.from_product(
        [["2020-01-01", "2020-01-02"], ["a", "b"]],
        names=["datetime", "instrument"],
    )
    signal = pd.Series([-2.0, -1.0, -4.0, -3.0], index=index, name="score")
    label = pd.DataFrame({"label": [1.0, 2.0, 3.0, 4.0]}, index=index)
    finalizer._composite_signal = lambda _selected: signal
    finalizer._load_label = lambda: label

    rows = finalizer._composite_test_ic_rows([node])

    assert rows[0]["status"] == "ok"
    assert rows[0]["signal_name"] == "oriented_equal_weight_top_k"
    assert rows[0]["test_ic"] > 0

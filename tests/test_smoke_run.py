import json

from lineage_evo.config import SearchConfig
from lineage_evo.experiments import ExperimentRunner


def test_fully_mocked_smoke_run_writes_required_outputs(tmp_path):
    runner = ExperimentRunner(log_dir=tmp_path, config=SearchConfig(target_valid_evaluations=5))
    engine = runner.run()

    assert engine.counters.valid_evaluated_count >= 5
    assert len([node for node in engine.dag.nodes.values() if node.generation == 0]) == 10
    required = [
        "summary_results.csv",
        "final_factor_pool.csv",
        "search_log.jsonl",
        "candidate_log.jsonl",
        "prior_rewrite_log.jsonl",
        "dag_events.jsonl",
        "config_snapshot.json",
    ]
    for name in required:
        assert (tmp_path / name).exists()

    candidate_line = (tmp_path / "candidate_log.jsonl").read_text(encoding="utf-8").splitlines()[0]
    candidate = json.loads(candidate_line)
    assert candidate["status"] == "valid"
    assert candidate["operator"] == "seed"
    assert "raw_output" in candidate

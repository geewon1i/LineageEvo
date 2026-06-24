import csv
import json

from lineage_evo.config import SearchConfig
from lineage_evo.experiments import ExperimentRunner
from lineage_evo.seed import MockSeedGenerator


def test_fully_mocked_smoke_run_writes_required_outputs(tmp_path):
    runner = ExperimentRunner(log_dir=tmp_path, config=SearchConfig(target_valid_evaluations=5))
    engine = runner.run()

    assert engine.counters.valid_evaluated_count >= 5
    assert len([node for node in engine.dag.nodes.values() if node.generation == 0]) == 10
    required = [
        "summary_results.csv",
        "final_factor_pool.csv",
        "elite_archive.csv",
        "lineage_forest.svg",
        "lineage_forest.png",
        "lineage_forest_factor_mapping.csv",
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

    elite_rows = list(csv.DictReader((tmp_path / "elite_archive.csv").open(encoding="utf-8")))
    elite_scores = [float(row["decision_score"]) for row in elite_rows]
    assert [int(row["elite_rank"]) for row in elite_rows] == list(range(1, len(elite_rows) + 1))
    assert elite_scores == sorted(elite_scores, reverse=True)

    summary = next(csv.DictReader((tmp_path / "summary_results.csv").open(encoding="utf-8")))
    assert int(summary["elite_archive_count"]) == len(elite_rows)

    mapping_rows = list(csv.DictReader((tmp_path / "lineage_forest_factor_mapping.csv").open(encoding="utf-8")))
    assert mapping_rows[0]["display_id"] == "F001"
    assert mapping_rows[0]["lineage_label"] == "L001"
    assert (tmp_path / "lineage_forest.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert (tmp_path / "lineage_forest.png").stat().st_size > 0
    assert not (tmp_path / "lineage_forest.pdf").exists()


def test_seed_generation_logs_and_skips_duplicate_normalized_expression(tmp_path):
    runner = ExperimentRunner(
        log_dir=tmp_path,
        config=SearchConfig(seed_count=2, target_valid_evaluations=0, max_seed_generation_attempts=3),
        seed_generator=MockSeedGenerator(
            [
                '{"factor": "$close", "rationale": "seed"}',
                '{"factor": "close", "rationale": "duplicate"}',
                '{"factor": "$open", "rationale": "seed"}',
            ]
        ),
    )

    engine = runner.run()

    seed_nodes = [node for node in engine.dag.nodes.values() if node.generation == 0]
    records = [json.loads(line) for line in (tmp_path / "candidate_log.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(seed_nodes) == 2
    assert any(record["status"] == "duplicate_candidate" for record in records)

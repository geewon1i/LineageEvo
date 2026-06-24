"""Search run recording."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from lineage_evo.lineage import LineageDAG
from lineage_evo.recording.lineage_forest import write_lineage_forest_artifacts


class SearchRecorder:
    """Write JSONL/CSV artifacts required by smoke runs and experiments."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.search_log_path = self.log_dir / "search_log.jsonl"
        self.candidate_log_path = self.log_dir / "candidate_log.jsonl"
        self.prior_rewrite_log_path = self.log_dir / "prior_rewrite_log.jsonl"
        self.dag_events_path = self.log_dir / "dag_events.jsonl"
        self.summary_path = self.log_dir / "summary_results.csv"
        self.final_pool_path = self.log_dir / "final_factor_pool.csv"
        self.elite_archive_path = self.log_dir / "elite_archive.csv"
        self.lineage_forest_svg_path = self.log_dir / "lineage_forest.svg"
        self.lineage_forest_png_path = self.log_dir / "lineage_forest.png"
        self.lineage_forest_mapping_path = self.log_dir / "lineage_forest_factor_mapping.csv"
        self.config_path = self.log_dir / "config_snapshot.json"

    def log_search(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.search_log_path, record)

    def log_candidate(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.candidate_log_path, record)

    def log_dag_event(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.dag_events_path, record)

    def log_rewrite(self, rewrite_input, result) -> None:
        old_prior = rewrite_input.old_prior.model_dump(mode="json") if hasattr(rewrite_input.old_prior, "model_dump") else rewrite_input.old_prior
        accepted = result.updated_prior.model_dump(mode="json") if hasattr(result.updated_prior, "model_dump") else result.updated_prior
        evidence = rewrite_input.evidence_dict(compact=True)
        evidence.pop("lineage_summary", None)
        self._append_jsonl(
            self.prior_rewrite_log_path,
            {
                "run_id": rewrite_input.run_id,
                "generation": rewrite_input.generation,
                "operator": rewrite_input.operator.value,
                "target_prior_type": rewrite_input.target_prior_type.value,
                "lineage_id": rewrite_input.lineage_id,
                "parent_ids": rewrite_input.parent_ids,
                "child_id": rewrite_input.child_id,
                "old_prior": old_prior,
                "new_evidence": evidence,
                "update_trigger": rewrite_input.update_trigger,
                "expression_diff": rewrite_input.expression_diff.as_dict() if rewrite_input.expression_diff else None,
                "raw_llm_output": result.raw_llm_output,
                "accepted_updated_prior": accepted,
                "schema_valid": result.schema_valid,
                "fallback_used": result.fallback_used,
                "warnings": result.warnings,
                "fields_changed": result.fields_changed,
                "removed_patterns": result.removed_patterns,
                "deterministic_updates": result.deterministic_updates,
            },
        )

    def write_config_snapshot(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_csv(self.summary_path, [summary])

    def write_final_factor_pool(self, dag: LineageDAG) -> None:
        self._write_csv(self.final_pool_path, self._factor_rows(dag, dag.active_ids, rank_field="active_rank"))

    def write_elite_archive(self, dag: LineageDAG) -> None:
        ranked_ids = sorted(dag.elite_ids, key=lambda factor_id: self._decision_score(dag, factor_id), reverse=True)
        self._write_csv(self.elite_archive_path, self._factor_rows(dag, ranked_ids, rank_field="elite_rank"))

    def write_lineage_forest(self, dag: LineageDAG) -> dict[str, Path]:
        return write_lineage_forest_artifacts(dag, self.log_dir)

    @staticmethod
    def _factor_rows(dag: LineageDAG, factor_ids: list[str], *, rank_field: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, factor_id in enumerate(factor_ids, start=1):
            node = dag.nodes[factor_id]
            rows.append(
                {
                    rank_field: rank,
                    "factor_id": node.factor_id,
                    "lineage_id": node.lineage_id,
                    "generation": node.generation,
                    "is_active": node.is_active,
                    "expression": node.expression.raw,
                    "train_ic": node.evaluation.train_ic if node.evaluation else None,
                    "train_icir": node.evaluation.train_icir if node.evaluation else None,
                    "validation_ic": node.evaluation.validation_ic if node.evaluation else None,
                    "validation_icir": node.evaluation.validation_icir if node.evaluation else None,
                    "decision_metric": "train_ic" if dag.train_only else "validation_ic",
                    "decision_score": SearchRecorder._decision_score(dag, factor_id),
                }
            )
        return rows

    @staticmethod
    def _decision_score(dag: LineageDAG, factor_id: str) -> float | None:
        node = dag.nodes[factor_id]
        if node.evaluation is None:
            return None
        decision_ic = node.evaluation.train_ic if dag.train_only else node.evaluation.validation_ic
        return abs(decision_ic)

    def write_selected_factors(self, rows: list[dict[str, Any]]) -> None:
        self._write_csv(self.log_dir / "selected_factors.csv", rows)

    def write_test_ic_results(self, rows: list[dict[str, Any]]) -> None:
        self._write_csv(self.log_dir / "test_ic_results.csv", rows)

    def write_composite_test_ic_results(self, rows: list[dict[str, Any]]) -> None:
        self._write_csv(self.log_dir / "composite_test_ic_results.csv", rows)

    def write_backtest_summary(self, rows: list[dict[str, Any]]) -> None:
        self._write_csv(self.log_dir / "backtest_summary.csv", rows)

    def write_backtest_daily_report(self, rows: list[dict[str, Any]]) -> None:
        self._write_csv(self.log_dir / "backtest_daily_report.csv", rows)

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

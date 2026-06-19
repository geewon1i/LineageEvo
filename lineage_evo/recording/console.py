"""Optional colorful console reporting for interactive runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lineage_evo.evaluation import EvaluationResult, ScoreDelta


@dataclass
class ConsoleReporter:
    enabled: bool = False
    print_llm_io: bool = False
    use_color: bool = True

    def llm_input(self, label: str, system_prompt: str, user_prompt: str) -> None:
        if not self.enabled or not self.print_llm_io:
            return
        self._section(f"{label} LLM INPUT", "cyan")
        self._block("SYSTEM PROMPT", system_prompt)
        self._block("USER PROMPT", user_prompt)

    def llm_output(self, label: str, raw_output: str) -> None:
        if not self.enabled or not self.print_llm_io:
            return
        self._section(f"{label} LLM OUTPUT", "magenta")
        print(raw_output)

    def attempt(self, *, generation: int, operator: str, parents: list[str]) -> None:
        if not self.enabled:
            return
        self._section(f"GEN {generation} | {operator.upper()} ATTEMPT", "blue")
        print(f"parents: {parents}")

    def seed_attempt(self, *, attempt: int) -> None:
        if not self.enabled:
            return
        self._section(f"SEED ATTEMPT {attempt}", "blue")

    def failure(self, *, status: str, reason: str | None, raw_output: str | None = None) -> None:
        if not self.enabled:
            return
        self._section(status.upper(), "red")
        if reason:
            print(f"reason: {reason}")
        if raw_output and not self.print_llm_io:
            print(f"raw_output: {raw_output}")

    def valid_factor(
        self,
        *,
        expression: str,
        evaluation: EvaluationResult,
        delta: ScoreDelta | None = None,
        child_id: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._section("VALID FACTOR", "green")
        if child_id:
            print(f"child_id: {child_id}")
        print(f"factor: {expression}")
        print(
            "metrics: "
            f"train_ic={evaluation.train_ic:.6f}, "
            f"train_icir={evaluation.train_icir:.6f}, "
            f"valid_ic={evaluation.validation_ic:.6f}, "
            f"valid_icir={evaluation.validation_icir:.6f}"
        )
        if delta is not None:
            print(
                "delta: "
                f"train_ic_delta={delta.train_ic_delta:.6f}, "
                f"valid_ic_delta={delta.validation_ic_delta:.6f}, "
                f"train_icir_delta={delta.train_icir_delta:.6f}, "
                f"valid_icir_delta={delta.validation_icir_delta:.6f}"
            )

    def generation_summary(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._section(f"GEN {record.get('generation')} SUMMARY", "yellow")
        print(
            f"valid_mutation={record.get('valid_mutation', 0)}, "
            f"valid_crossover={record.get('valid_crossover', 0)}, "
            f"generated_count={record.get('generated_count', 0)}, "
            f"valid_evaluated_count={record.get('valid_evaluated_count', 0)}"
        )

    def _section(self, title: str, color: str) -> None:
        text = f"\n========== {title} =========="
        print(self._color(text, color, bright=True))

    def _block(self, title: str, content: str) -> None:
        print(self._color(f"-- {title} --", "yellow", bright=True))
        print(content)

    def _color(self, text: str, color: str, *, bright: bool = False) -> str:
        if not self.use_color:
            return text
        colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35, "cyan": 36}
        code = colors.get(color, 37)
        prefix = f"\033[{1 if bright else 0};{code}m"
        return f"{prefix}{text}\033[0m"

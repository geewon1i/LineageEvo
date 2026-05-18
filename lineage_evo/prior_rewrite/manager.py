"""Deterministic gatekeeper for LLM-rewritten priors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from lineage_evo.config import PriorRewriteConfig
from lineage_evo.prior_rewrite.types import LLMRewriteResponse, PriorRewriteInput, PriorRewriteResult, PriorTarget
from lineage_evo.priors.schemas import (
    CommonInvalidPattern,
    Confidence,
    CrossoverPrior,
    FailedPatternEvidence,
    GlobalCrossoverPrior,
    GlobalMutationPrior,
    MutationPrior,
    PatternEvidence,
    RiskLevel,
    StructureEvidence,
)


@dataclass(frozen=True)
class PriorManagerConfig(PriorRewriteConfig):
    pass


class PriorManager:
    """Validate, constrain, prune, and log complete prior rewrites."""

    def __init__(self, config: PriorManagerConfig | None = None, logger: Any | None = None) -> None:
        self.config = config or PriorManagerConfig()
        self.logger = logger

    def accept_rewrite(
        self,
        rewrite_input: PriorRewriteInput,
        llm_response: LLMRewriteResponse,
    ) -> PriorRewriteResult:
        schema_model = self._schema_for_target(rewrite_input.target_prior_type)
        warnings: list[str] = []
        try:
            parsed_json = json.loads(llm_response.raw_content)
            candidate = schema_model.model_validate(parsed_json)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            result = self._fallback_result(rewrite_input, llm_response, f"invalid prior rewrite: {exc}")
            self._log(rewrite_input, result)
            return result

        old_prior = rewrite_input.old_prior
        before_dump = old_prior.model_dump(mode="json") if hasattr(old_prior, "model_dump") else old_prior

        candidate, removed_by_constraints, constraint_warnings = self._apply_constraints(candidate, rewrite_input)
        warnings.extend(constraint_warnings)
        after_dump = candidate.model_dump(mode="json")

        fields_changed = [
            key for key, value in after_dump.items() if not isinstance(before_dump, dict) or before_dump.get(key) != value
        ]
        removed_patterns = self._removed_patterns(before_dump, after_dump) + removed_by_constraints
        result = PriorRewriteResult(
            updated_prior=candidate,
            update_summary="accepted complete prior rewrite",
            fields_changed=sorted(set(fields_changed)),
            removed_patterns=sorted(set(removed_patterns)),
            warnings=warnings,
            fallback_used=False,
            schema_valid=True,
            raw_llm_output=llm_response.raw_content,
        )
        self._log(rewrite_input, result)
        return result

    def _apply_constraints(self, prior: BaseModel, rewrite_input: PriorRewriteInput) -> tuple[BaseModel, list[str], list[str]]:
        data = prior.model_dump(mode="json")
        removed: list[str] = []
        warnings: list[str] = []
        list_fields = self._pattern_list_fields(data)

        for field_name in list_fields:
            pruned, field_removed, field_warnings = self._sanitize_pattern_list(
                data[field_name],
                current_generation=rewrite_input.generation,
                delta_validation=rewrite_input.delta_validation_score,
                delta_train=rewrite_input.delta_train_score,
            )
            data[field_name] = pruned
            removed.extend([f"{field_name}:{pattern}" for pattern in field_removed])
            warnings.extend(field_warnings)

        if (
            isinstance(prior, MutationPrior)
            and rewrite_input.delta_train_score is not None
            and rewrite_input.delta_validation_score is not None
            and rewrite_input.delta_train_score > 0
            and rewrite_input.delta_validation_score <= 0
        ):
            if data.get("bias_risk") == RiskLevel.LOW.value:
                data["bias_risk"] = RiskLevel.MEDIUM.value
                warnings.append("train improved without validation improvement; raised bias_risk to medium")
            data["successful_mutation_patterns"] = self._downgrade_high_confidence(
                data["successful_mutation_patterns"],
                "train-only improvement cannot support high-confidence success",
                warnings,
            )

        schema_model = type(prior)
        return schema_model.model_validate(data), removed, warnings

    def _sanitize_pattern_list(
        self,
        items: list[dict[str, Any]],
        *,
        current_generation: int,
        delta_validation: float | None,
        delta_train: float | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        warnings: list[str] = []
        merged: dict[str, dict[str, Any]] = {}
        removed: list[str] = []

        for item in items:
            text_key = "pattern" if "pattern" in item else "structure"
            key = self._compact_text(str(item[text_key]))
            item[text_key] = key
            item["evidence"] = self._compact_text(str(item.get("evidence", "")))
            if item.get("last_updated_generation", 0) > current_generation:
                item["last_updated_generation"] = current_generation
                warnings.append(f"clamped future generation for {key}")

            if item.get("confidence") == Confidence.HIGH.value and not self._high_confidence_allowed(item, delta_validation):
                item["confidence"] = Confidence.MEDIUM.value
                warnings.append(f"downgraded unsupported high confidence for {key}")

            if key in merged:
                merged[key] = self._merge_item(merged[key], item)
                removed.append(key)
            else:
                merged[key] = item

        ranked = sorted(merged.values(), key=self._pattern_rank, reverse=True)
        for item in ranked[self.config.top_k_patterns :]:
            removed.append(str(item.get("pattern") or item.get("structure")))
        return ranked[: self.config.top_k_patterns], removed, warnings

    def _fallback_result(
        self,
        rewrite_input: PriorRewriteInput,
        llm_response: LLMRewriteResponse,
        warning: str,
    ) -> PriorRewriteResult:
        return PriorRewriteResult(
            updated_prior=rewrite_input.old_prior,
            update_summary="fallback to old prior",
            fields_changed=[],
            removed_patterns=[],
            warnings=[warning],
            fallback_used=True,
            schema_valid=False,
            raw_llm_output=llm_response.raw_content,
        )

    def _log(self, rewrite_input: PriorRewriteInput, result: PriorRewriteResult) -> None:
        if self.logger is None:
            return
        self.logger.log_rewrite(rewrite_input, result)

    @staticmethod
    def _schema_for_target(target: PriorTarget) -> Type[BaseModel]:
        return {
            PriorTarget.MUTATION_LINEAGE: MutationPrior,
            PriorTarget.CROSSOVER_LINEAGE: CrossoverPrior,
            PriorTarget.GLOBAL_MUTATION: GlobalMutationPrior,
            PriorTarget.GLOBAL_CROSSOVER: GlobalCrossoverPrior,
        }[target]

    @staticmethod
    def _pattern_list_fields(data: dict[str, Any]) -> list[str]:
        return [key for key, value in data.items() if isinstance(value, list)]

    def _compact_text(self, text: str) -> str:
        text = " ".join(text.split())
        if len(text) <= self.config.max_text_length:
            return text
        return text[: self.config.max_text_length].rstrip()

    def _high_confidence_allowed(self, item: dict[str, Any], delta_validation: float | None) -> bool:
        support_count = int(item.get("support_count", item.get("fail_count", 0)))
        if support_count >= 2:
            return True
        return delta_validation is not None and delta_validation >= self.config.strong_validation_delta

    @staticmethod
    def _merge_item(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        merged = dict(old)
        for count_key in ("support_count", "fail_count"):
            if count_key in new or count_key in old:
                merged[count_key] = max(int(old.get(count_key, 0)), int(new.get(count_key, 0)))
        merged["last_updated_generation"] = max(
            int(old.get("last_updated_generation", 0)),
            int(new.get("last_updated_generation", 0)),
        )
        if len(str(new.get("evidence", ""))) > len(str(old.get("evidence", ""))):
            merged["evidence"] = new["evidence"]
        confidence_rank = {Confidence.LOW.value: 0, Confidence.MEDIUM.value: 1, Confidence.HIGH.value: 2}
        if confidence_rank.get(new.get("confidence"), 0) > confidence_rank.get(old.get("confidence"), 0):
            merged["confidence"] = new["confidence"]
        return merged

    @staticmethod
    def _pattern_rank(item: dict[str, Any]) -> tuple[int, int, int]:
        confidence_rank = {Confidence.LOW.value: 0, Confidence.MEDIUM.value: 1, Confidence.HIGH.value: 2}
        count = int(item.get("support_count", item.get("fail_count", 0)))
        generation = int(item.get("last_updated_generation", 0))
        return confidence_rank.get(item.get("confidence"), 0), count, generation

    @staticmethod
    def _downgrade_high_confidence(
        items: list[dict[str, Any]],
        message: str,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        for item in items:
            if item.get("confidence") == Confidence.HIGH.value:
                item["confidence"] = Confidence.MEDIUM.value
                warnings.append(f"{message}: {item.get('pattern')}")
        return items

    @staticmethod
    def _removed_patterns(before: Any, after: Any) -> list[str]:
        if not isinstance(before, dict) or not isinstance(after, dict):
            return []
        before_patterns = PriorManager._collect_pattern_names(before)
        after_patterns = PriorManager._collect_pattern_names(after)
        return sorted(before_patterns - after_patterns)

    @staticmethod
    def _collect_pattern_names(data: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for value in data.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict):
                    name = item.get("pattern") or item.get("structure")
                    if name:
                        names.add(str(name))
        return names

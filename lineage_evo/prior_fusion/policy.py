"""Ablation-aware prior fusion before candidate prompt construction."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FusionMode(StrEnum):
    OURS_FULL = "ours_full"
    LINEAGE_ONLY = "lineage_only"
    GLOBAL_ONLY = "global_only"
    UNIFIED_LINEAGE_STATE = "unified_lineage_state"
    RAW_ANCESTRAL_TRACE = "raw_ancestral_trace"
    SHUFFLED_LINEAGE_PRIOR = "shuffled_lineage_prior"
    REEVO_STYLE_GLOBAL_REFLECTION = "reevo_style_global_reflection"
    NO_PRIOR_UPDATE = "no_prior_update"


@dataclass(frozen=True)
class PriorFusionInput:
    mode: FusionMode
    operator: str
    lineage_id: str | None
    lineage_prior: Any | None
    global_prior: Any | None
    all_lineage_priors: dict[str, Any] = field(default_factory=dict)
    raw_ancestral_trace: list[dict[str, Any]] = field(default_factory=list)
    global_reflection: str | None = None
    rng: random.Random | None = None


@dataclass(frozen=True)
class FusedPriorContext:
    mode: FusionMode
    operator: str
    lineage_id: str | None
    prompt_context: dict[str, Any]
    prior_updates_enabled: bool = True


class PriorFusionPolicy:
    """决定哪些 prior 进入候选生成 prompt，用于论文消融。"""

    def fuse(self, fusion_input: PriorFusionInput) -> FusedPriorContext:
        mode = fusion_input.mode
        context: dict[str, Any] = {"mode": mode.value, "operator": fusion_input.operator}
        updates_enabled = mode != FusionMode.NO_PRIOR_UPDATE

        if mode == FusionMode.OURS_FULL:
            # 完整方法：同时使用 lineage prior 和 global prior。
            context["lineage_prior"] = self._dump(fusion_input.lineage_prior)
            context["global_prior"] = self._dump(fusion_input.global_prior)
        elif mode == FusionMode.LINEAGE_ONLY:
            context["lineage_prior"] = self._dump(fusion_input.lineage_prior)
        elif mode == FusionMode.GLOBAL_ONLY:
            context["global_prior"] = self._dump(fusion_input.global_prior)
        elif mode == FusionMode.UNIFIED_LINEAGE_STATE:
            context["unified_lineage_state"] = self._dump(fusion_input.lineage_prior)
        elif mode == FusionMode.RAW_ANCESTRAL_TRACE:
            context["raw_ancestral_trace"] = fusion_input.raw_ancestral_trace
        elif mode == FusionMode.SHUFFLED_LINEAGE_PRIOR:
            # 打乱 lineage prior 只影响 prompt，不改变 DAG 中真实 lineage。
            context["lineage_prior"] = self._dump(self._choose_shuffled_prior(fusion_input))
            context["shuffled_from_lineage"] = context["lineage_prior"].get("_lineage_id") if isinstance(context["lineage_prior"], dict) else None
            context["global_prior"] = self._dump(fusion_input.global_prior)
        elif mode == FusionMode.REEVO_STYLE_GLOBAL_REFLECTION:
            context["global_reflection"] = fusion_input.global_reflection or ""
        elif mode == FusionMode.NO_PRIOR_UPDATE:
            context["lineage_prior"] = self._dump(fusion_input.lineage_prior)
            context["global_prior"] = self._dump(fusion_input.global_prior)
            context["prior_updates_enabled"] = False
        else:
            raise ValueError(f"unsupported fusion mode: {mode}")

        return FusedPriorContext(
            mode=mode,
            operator=fusion_input.operator,
            lineage_id=fusion_input.lineage_id,
            prompt_context=context,
            prior_updates_enabled=updates_enabled,
        )

    @staticmethod
    def _dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    def _choose_shuffled_prior(self, fusion_input: PriorFusionInput) -> Any:
        candidates = [
            (lineage_id, prior)
            for lineage_id, prior in fusion_input.all_lineage_priors.items()
            if lineage_id != fusion_input.lineage_id
        ]
        if not candidates:
            return fusion_input.lineage_prior
        rng = fusion_input.rng or random.Random(0)
        lineage_id, prior = rng.choice(candidates)
        dumped = self._dump(prior)
        if isinstance(dumped, dict):
            return {**dumped, "_lineage_id": lineage_id}
        return dumped

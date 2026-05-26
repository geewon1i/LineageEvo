"""Ablation policies decide which prior sources are available to a run."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AblationMode(StrEnum):
    OURS_FULL = "ours_full"
    LINEAGE_ONLY = "lineage_only"
    GLOBAL_ONLY = "global_only"
    UNIFIED_LINEAGE_STATE = "unified_lineage_state"
    RAW_ANCESTRAL_TRACE = "raw_ancestral_trace"
    SHUFFLED_LINEAGE_PRIOR = "shuffled_lineage_prior"
    REEVO_STYLE_GLOBAL_REFLECTION = "reevo_style_global_reflection"
    NO_PRIOR_UPDATE = "no_prior_update"


@dataclass(frozen=True)
class AblationInput:
    mode: AblationMode
    operator: str
    lineage_id: str | None
    lineage_prior: Any | None
    global_prior: Any | None
    secondary_lineage_id: str | None = None
    secondary_lineage_prior: Any | None = None
    all_lineage_priors: dict[str, Any] = field(default_factory=dict)
    raw_ancestral_trace: list[dict[str, Any]] = field(default_factory=list)
    global_reflection: str | None = None
    rng: random.Random | None = None


@dataclass(frozen=True)
class AblationContext:
    mode: AblationMode
    operator: str
    lineage_id: str | None
    lineage_prior: Any | None
    global_prior: Any | None
    secondary_lineage_id: str | None = None
    secondary_lineage_prior: Any | None = None
    raw_ancestral_trace: list[dict[str, Any]] = field(default_factory=list)
    global_reflection: str | None = None
    shuffled_from_lineage: str | None = None
    prior_updates_enabled: bool = True


class AblationPolicy:
    """Select prior sources for ablation baselines.

    This module does not compute local/global trust weights. That is handled by
    lineage_evo.prior_fusion, matching Method section 4.7.
    """

    def apply(self, ablation_input: AblationInput) -> AblationContext:
        mode = ablation_input.mode
        updates_enabled = mode != AblationMode.NO_PRIOR_UPDATE
        lineage_prior = ablation_input.lineage_prior
        global_prior = ablation_input.global_prior
        secondary_prior = ablation_input.secondary_lineage_prior
        shuffled_from: str | None = None

        if mode == AblationMode.GLOBAL_ONLY:
            lineage_prior = None
            secondary_prior = None
        elif mode == AblationMode.LINEAGE_ONLY:
            global_prior = None
        elif mode == AblationMode.SHUFFLED_LINEAGE_PRIOR:
            shuffled_from, lineage_prior = self._choose_shuffled_prior(ablation_input)
        elif mode == AblationMode.RAW_ANCESTRAL_TRACE:
            lineage_prior = None
            secondary_prior = None
            global_prior = None
        elif mode == AblationMode.REEVO_STYLE_GLOBAL_REFLECTION:
            lineage_prior = None
            secondary_prior = None
            global_prior = None
        elif mode in {
            AblationMode.OURS_FULL,
            AblationMode.UNIFIED_LINEAGE_STATE,
            AblationMode.NO_PRIOR_UPDATE,
        }:
            pass
        else:
            raise ValueError(f"unsupported ablation mode: {mode}")

        return AblationContext(
            mode=mode,
            operator=ablation_input.operator,
            lineage_id=ablation_input.lineage_id,
            lineage_prior=lineage_prior,
            global_prior=global_prior,
            secondary_lineage_id=ablation_input.secondary_lineage_id,
            secondary_lineage_prior=secondary_prior,
            raw_ancestral_trace=ablation_input.raw_ancestral_trace,
            global_reflection=ablation_input.global_reflection,
            shuffled_from_lineage=shuffled_from,
            prior_updates_enabled=updates_enabled,
        )

    def fuse(self, ablation_input: AblationInput) -> AblationContext:
        return self.apply(ablation_input)

    def _choose_shuffled_prior(self, ablation_input: AblationInput) -> tuple[str | None, Any]:
        candidates = [
            (lineage_id, prior)
            for lineage_id, prior in ablation_input.all_lineage_priors.items()
            if lineage_id != ablation_input.lineage_id
        ]
        if not candidates:
            return None, ablation_input.lineage_prior
        rng = ablation_input.rng or random.Random(0)
        lineage_id, prior = rng.choice(candidates)
        return lineage_id, prior

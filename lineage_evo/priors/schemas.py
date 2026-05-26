"""Strict schemas for persistent structured priors."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MutationStrength(StrEnum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    EXPLORATORY = "exploratory"


class StagnationState(StrEnum):
    NOT_STAGNANT = "not_stagnant"
    MILD_STAGNATION = "mild_stagnation"
    SEVERE_STAGNATION = "severe_stagnation"


class PatternEvidence(StrictModel):
    pattern: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence: Confidence
    support_count: NonNegativeInt = 0
    last_updated_generation: NonNegativeInt


class FailedPatternEvidence(StrictModel):
    pattern: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence: Confidence
    fail_count: NonNegativeInt = 0
    last_updated_generation: NonNegativeInt


class StructureEvidence(StrictModel):
    structure: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence: Confidence
    last_updated_generation: NonNegativeInt


class CommonInvalidPattern(StrictModel):
    pattern: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence: Confidence
    fail_count: NonNegativeInt = 0
    last_updated_generation: NonNegativeInt


class MutationPrior(StrictModel):
    successful_mutation_patterns: list[PatternEvidence]
    failed_mutation_patterns: list[FailedPatternEvidence]
    hint: str
    bias_risk: RiskLevel


class MutationSemanticPrior(MutationPrior):
    pass


class CrossoverPrior(StrictModel):
    transferable_patterns: list[PatternEvidence]
    harmful_patterns: list[FailedPatternEvidence]
    complementarity_profile: str
    heritable_structures: list[StructureEvidence]
    hint: str
    crossover_risk: RiskLevel


class GlobalMutationPrior(StrictModel):
    global_successful_mutation_patterns: list[PatternEvidence]
    global_failed_mutation_patterns: list[FailedPatternEvidence]
    common_invalid_patterns: list[CommonInvalidPattern]
    hint: str
    last_updated_generation: NonNegativeInt


class GlobalCrossoverPrior(StrictModel):
    global_transferable_patterns: list[PatternEvidence]
    global_harmful_patterns: list[FailedPatternEvidence]
    global_complementarity_patterns: list[PatternEvidence]
    hint: str
    last_updated_generation: NonNegativeInt

"""Evaluator interfaces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from lineage_evo.factor import FactorExpression


@dataclass(frozen=True)
class EvaluationResult:
    train_ic: float
    train_icir: float
    validation_ic: float
    validation_icir: float

    def as_dict(self) -> dict[str, float]:
        return {
            "train_ic": self.train_ic,
            "train_icir": self.train_icir,
            "validation_ic": self.validation_ic,
            "validation_icir": self.validation_icir,
        }


@dataclass(frozen=True)
class ScoreDelta:
    train_icir_delta: float
    validation_icir_delta: float

    def as_dict(self) -> dict[str, float]:
        return {
            "train_icir_delta": self.train_icir_delta,
            "validation_icir_delta": self.validation_icir_delta,
        }

    @classmethod
    def from_results(cls, parent: EvaluationResult, child: EvaluationResult) -> "ScoreDelta":
        return cls(
            train_icir_delta=child.train_icir - parent.train_icir,
            validation_icir_delta=child.validation_icir - parent.validation_icir,
        )


class Evaluator(Protocol):
    def evaluate(self, expression: FactorExpression) -> EvaluationResult:
        ...


class MockEvaluator:
    """Deterministic evaluator for tests and smoke runs."""

    def evaluate(self, expression: FactorExpression) -> EvaluationResult:
        digest = hashlib.sha256(expression.normalized.encode("utf-8")).hexdigest()
        base = int(digest[:8], 16) / 0xFFFFFFFF
        train = (base - 0.5) / 5
        validation = ((int(digest[8:16], 16) / 0xFFFFFFFF) - 0.5) / 5
        return EvaluationResult(
            train_ic=train,
            train_icir=train * 10,
            validation_ic=validation,
            validation_icir=validation * 10,
        )


"""Deterministic validation for factor expressions."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from lineage_evo.factor import FactorExpression


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"is_valid": self.is_valid, "reasons": self.reasons}


class Validator:
    """Validate syntax, variables, operators, and length without using an LLM."""

    def __init__(
        self,
        allowed_variables: set[str],
        allowed_functions: set[str],
        max_length: int = 40,
    ) -> None:
        self.allowed_variables = allowed_variables
        self.allowed_functions = allowed_functions
        self.max_length = max_length

    def validate(self, expression: FactorExpression) -> ValidationResult:
        reasons: list[str] = []
        if expression.length > self.max_length:
            reasons.append(f"factor length {expression.length} exceeds {self.max_length}")

        try:
            tree = ast.parse(expression.raw, mode="eval")
        except SyntaxError as exc:
            return ValidationResult(False, reasons + [f"syntax error: {exc.msg}"])

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in self.allowed_variables and node.id not in self.allowed_functions:
                    reasons.append(f"unknown name: {node.id}")
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_functions:
                    reasons.append("unknown function call")
            elif isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.ListComp)):
                reasons.append(f"unsupported syntax: {type(node).__name__}")

        return ValidationResult(not reasons, reasons)


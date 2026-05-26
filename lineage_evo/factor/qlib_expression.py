"""Qlib adapter for AlphaPROBE-style factor expressions."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from datetime import timedelta

from lineage_evo.config import QlibConfig
from lineage_evo.factor.dsl import DEFAULT_FACTOR_DSL
from lineage_evo.factor.expression import FactorExpression
from lineage_evo.qlib_warnings import suppress_qlib_all_nan_slice_warning
from lineage_evo.validation import ValidationResult


class QlibExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class QlibOperatorRegistry:
    features: frozenset[str] = frozenset(DEFAULT_FACTOR_DSL.features)
    rolling_constants: frozenset[int] = frozenset(DEFAULT_FACTOR_DSL.rolling_constants)
    arithmetic_constants: frozenset[float] = frozenset(DEFAULT_FACTOR_DSL.arithmetic_constants)


class QlibExpressionNormalizer:
    """Convert AlphaPROBE-style expressions to Qlib executable expressions."""

    def __init__(self, registry: QlibOperatorRegistry | None = None) -> None:
        self.registry = registry or QlibOperatorRegistry()

    def normalize(self, expression: FactorExpression | str) -> str:
        raw = expression.raw if isinstance(expression, FactorExpression) else expression
        tree = ast.parse(self._preprocess(raw), mode="eval")
        return self._convert(tree.body)

    def _preprocess(self, raw: str) -> str:
        return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"\1", raw.strip())

    def _convert(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            feature = f"${node.id.lower()}"
            if feature not in self.registry.features:
                raise QlibExpressionError(f"unknown feature: {node.id}")
            return feature
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return self._constant(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
                return self._constant(-node.operand.value)
            return f"(-{self._convert(node.operand)})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [self._convert(arg) for arg in node.args]
            return self._call(node.func.id, args)
        raise QlibExpressionError(f"unsupported syntax: {type(node).__name__}")

    def _constant(self, value: int | float) -> str:
        if value in self.registry.rolling_constants or float(value) in self.registry.arithmetic_constants:
            return str(value)
        raise QlibExpressionError(f"constant not allowed: {value}")

    def _call(self, func_name: str, args: list[str]) -> str:
        func = func_name.lower()
        if func in {"abs", "log", "sign", "rank", "slog1p"}:
            self._expect(func_name, args, 1)
            return self._unary(func, args[0])
        if func in {"add", "sub", "mul", "div", "pow", "greater", "less", "getgreater", "getless"}:
            self._expect(func_name, args, 2)
            return self._binary(func, args[0], args[1])
        if func == "ref":
            self._expect(func_name, args, 2)
            self._require_rolling_constant(args[1], func_name)
            return f"Ref({args[0]}, {args[1]})"
        if func.startswith("ts"):
            return self._timeseries(func, func_name, args)
        raise QlibExpressionError(f"unknown operator: {func_name}")

    @staticmethod
    def _expect(func_name: str, args: list[str], count: int) -> None:
        if len(args) != count:
            raise QlibExpressionError(f"{func_name} expects {count} arguments")

    def _require_rolling_constant(self, value: str, func_name: str) -> None:
        try:
            parsed = int(float(value))
        except ValueError as exc:
            raise QlibExpressionError(f"{func_name} window must be a constant") from exc
        if parsed not in self.registry.rolling_constants:
            raise QlibExpressionError(f"{func_name} window constant not allowed: {value}")

    @staticmethod
    def _unary(func: str, x: str) -> str:
        if func == "abs":
            return f"Abs({x})"
        if func == "log":
            return f"Log({x})"
        if func == "sign":
            return f"Sign({x})"
        if func == "rank":
            return f"Rank({x}, 5)"
        if func == "slog1p":
            return f"Sign({x}) * Log(Abs({x}) + 1)"
        raise QlibExpressionError(f"unknown unary operator: {func}")

    @staticmethod
    def _binary(func: str, x: str, y: str) -> str:
        if func == "add":
            return f"({x} + {y})"
        if func == "sub":
            return f"({x} - {y})"
        if func == "mul":
            return f"({x} * {y})"
        if func == "div":
            return f"({x} / {y})"
        if func == "pow":
            return f"Power({x}, {y})"
        if func == "greater":
            return f"Gt({x}, {y})"
        if func == "less":
            return f"Lt({x}, {y})"
        if func == "getgreater":
            return f"Greater({x}, {y})"
        if func == "getless":
            return f"Less({x}, {y})"
        raise QlibExpressionError(f"unknown binary operator: {func}")

    def _timeseries(self, func: str, func_name: str, args: list[str]) -> str:
        three_arg = {"tscov": "Cov", "tscorr": "Corr"}
        two_arg = {
            "tsmean": "Mean",
            "tssum": "Sum",
            "tsstd": "Std",
            "tsmin": "Min",
            "tsmax": "Max",
            "tsvar": "Var",
            "tsskew": "Skew",
            "tskurt": "Kurt",
            "tsmed": "Med",
            "tsmad": "Mad",
            "tsrank": "Rank",
            "tsdelta": "Delta",
            "tsema": "EMA",
            "tswma": "WMA",
            "tsir": "Mean",
        }
        if func in three_arg:
            self._expect(func_name, args, 3)
            self._require_rolling_constant(args[2], func_name)
            return f"{three_arg[func]}({args[0]}, {args[1]}, {args[2]})"
        self._expect(func_name, args, 2)
        self._require_rolling_constant(args[1], func_name)
        x, d = args
        if func == "tsminmaxdiff":
            return f"(Max({x}, {d}) - Min({x}, {d}))"
        if func == "tsmaxdiff":
            return f"({x} - Max({x}, {d}))"
        if func == "tsmindiff":
            return f"({x} - Min({x}, {d}))"
        if func == "tsratio":
            return f"({x} / Ref({x}, {d}))"
        if func == "tspctchange":
            return f"({x} / Ref({x}, {d}) - 1)"
        if func == "tsir":
            return f"(Mean({x}, {d}) / Std({x}, {d}))"
        if func in two_arg:
            return f"{two_arg[func]}({x}, {d})"
        raise QlibExpressionError(f"unknown operator: {func_name}")


class QlibExpressionValidator:
    """Static checks plus an optional small-window Qlib execution check."""

    def __init__(
        self,
        config: QlibConfig | None = None,
        max_length: int = 50,
        execute_check: bool = True,
        execution_check_window_days: int = 120,
    ) -> None:
        self.config = config or QlibConfig.from_env()
        self.max_length = max_length
        self.execute_check = execute_check
        self.execution_check_window_days = execution_check_window_days
        self.normalizer = QlibExpressionNormalizer()
        self._qlib_initialized = False
        self._validation_cache: dict[str, ValidationResult] = {}

    def validate(self, expression: FactorExpression) -> ValidationResult:
        reasons: list[str] = []
        if expression.length > self.max_length:
            reasons.append(f"factor length {expression.length} exceeds {self.max_length}")
        try:
            qlib_expr = self.normalizer.normalize(expression)
        except (SyntaxError, QlibExpressionError) as exc:
            return ValidationResult(False, reasons + [str(exc)])
        if reasons or not self.execute_check:
            return ValidationResult(not reasons, reasons)
        return self._validate_execution(qlib_expr)

    def _validate_execution(self, qlib_expr: str) -> ValidationResult:
        if qlib_expr in self._validation_cache:
            return self._validation_cache[qlib_expr]
        try:
            self._ensure_qlib()
            from qlib.data import D

            start_time, end_time = self._execution_check_window()
            with suppress_qlib_all_nan_slice_warning():
                data = D.features(
                    D.instruments(self.config.market),
                    [qlib_expr],
                    start_time=start_time,
                    end_time=end_time,
                )
            series = data.iloc[:, 0].dropna()
        except Exception as exc:
            result = ValidationResult(False, [f"qlib execution error: {exc}"])
            self._validation_cache[qlib_expr] = result
            return result
        if series.empty:
            result = ValidationResult(False, ["qlib execution produced empty/all-NaN factor"])
            self._validation_cache[qlib_expr] = result
            return result
        if not all(math.isfinite(float(value)) for value in series.head(100)):
            result = ValidationResult(False, ["qlib execution produced NaN/Inf factor"])
            self._validation_cache[qlib_expr] = result
            return result
        if series.nunique(dropna=True) <= 1:
            result = ValidationResult(False, ["qlib execution produced constant factor"])
            self._validation_cache[qlib_expr] = result
            return result
        result = ValidationResult(True, [])
        self._validation_cache[qlib_expr] = result
        return result

    def _execution_check_window(self) -> tuple[str, str]:
        import pandas as pd

        start = pd.Timestamp(self.config.train_start)
        train_end = pd.Timestamp(self.config.train_end)
        end = min(start + timedelta(days=self.execution_check_window_days), train_end)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _ensure_qlib(self) -> None:
        if self._qlib_initialized:
            return
        import qlib
        from qlib.constant import REG_CN

        region = REG_CN if self.config.region.lower() == "cn" else self.config.region
        qlib.init(provider_uri=self.config.provider_uri, region=region)
        self._qlib_initialized = True

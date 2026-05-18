"""Shared AlphaPROBE-style factor DSL contract."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FactorDSL:
    """One expression contract shared by prompts and deterministic validation."""

    features: tuple[str, ...] = ("$open", "$high", "$low", "$close", "$vwap", "$volume")
    rolling_constants: tuple[int, ...] = (1, 3, 5, 10, 20, 30, 60)
    arithmetic_constants: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.0, 1.0, 2.0)
    operators: tuple[str, ...] = (
        "Abs(x): absolute value",
        "Log(x): natural logarithm",
        "SLog1p(x): signed log(1 + abs(x))",
        "Sign(x): sign of x",
        "Rank(x): cross-sectional rank",
        "Add(x, y): x + y",
        "Sub(x, y): x - y",
        "Mul(x, y): x * y",
        "Div(x, y): x / y",
        "Pow(x, y): x ** y, y must be a constant",
        "Greater(x, y): 1 if x > y else 0",
        "Less(x, y): 1 if x < y else 0",
        "GetGreater(x, y): max-like pairwise selection",
        "GetLess(x, y): min-like pairwise selection",
        "Ref(x, d): value of x d days ago",
        "TsMean(x, d): rolling mean",
        "TsSum(x, d): rolling sum",
        "TsStd(x, d): rolling standard deviation",
        "TsMin(x, d): rolling minimum",
        "TsMax(x, d): rolling maximum",
        "TsMinMaxDiff(x, d): TsMax(x,d) - TsMin(x,d)",
        "TsMaxDiff(x, d): x - TsMax(x,d)",
        "TsMinDiff(x, d): x - TsMin(x,d)",
        "TsIr(x, d): rolling information ratio",
        "TsVar(x, d): rolling variance",
        "TsSkew(x, d): rolling skewness",
        "TsKurt(x, d): rolling kurtosis",
        "TsMed(x, d): rolling median",
        "TsMad(x, d): rolling median absolute deviation",
        "TsRank(x, d): time-series rank",
        "TsDelta(x, d): x - Ref(x,d)",
        "TsRatio(x, d): x / Ref(x,d)",
        "TsPctChange(x, d): x / Ref(x,d) - 1",
        "TsWMA(x, d): weighted moving average",
        "TsEMA(x, d): exponential moving average",
        "TsCov(x, y, d): rolling covariance",
        "TsCorr(x, y, d): rolling correlation",
    )
    examples: tuple[str, ...] = (
        "Div(Sub($open, $close), Add(Sub($high, $low), 0.001))",
        "TsMean($close, 5)",
        "TsCorr($close, $volume, 20)",
    )
    notes: tuple[str, ...] = field(
        default=(
            "Use only the listed features, constants, and operators.",
            "Return exactly one AlphaPROBE-style expression.",
            "Do not output Python code, assignments, explanations, or multiple factors.",
        )
    )

    @property
    def all_constants(self) -> tuple[int | float, ...]:
        return self.rolling_constants + self.arithmetic_constants

    def as_prompt_context(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "rolling_constants": list(self.rolling_constants),
            "arithmetic_constants": list(self.arithmetic_constants),
            "operators": list(self.operators),
            "examples": list(self.examples),
            "rules": list(self.notes),
        }


DEFAULT_FACTOR_DSL = FactorDSL()


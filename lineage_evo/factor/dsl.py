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
        "Abs(x): Absolute value of x",
        "Log(x): Natural logarithm of x",
        "SLog1p(x): Signed log transform: sign(input) times log of (1 plus the absolute value)",
        "Sign(x): Sign of x: 1 if x > 0, -1 if x < 0, 0 if x = 0",
        "Rank(x): Cross-sectional rank of x",
        "Add(x,y): x + y",
        "Sub(x, y): x - y",
        "Mul(x, y): x * y",
        "Div(x, y): x / y",
        "Pow(x, y): x raised to the power of y (x ** y) y must be a constant",
        "Greater(x, y): 1 if x > y, else 0",
        "Less(x, y): 1 if x < y, else 0",
        "GetGreater(x, y): x if x > y, else y",
        "GetLess(x, y): x if x < y, else y",
        "Ref(x, d): Value of x d days ago",
        "TsMean(x, d): Rolling mean of x over the past d days",
        "TsSum(x, d): Rolling sum of x over the past d days",
        "TsStd(x, d): Rolling standard deviation of x over the past d days",
        "TsMin(x, d): Rolling minimum of x over the past d days",
        "TsMax(x, d): Rolling maximum of x over the past d days",
        "TsMinMaxDiff(x, d): Difference between TsMax(x, d) and TsMin(x, d)",
        "TsMaxDiff(x, d): Difference between current x and TsMax(x, d)",
        "TsMinDiff(x, d): Difference between current x and TsMin(x, d)",
        "TsIr(x, d): Rolling Information ratio over past d days",
        "TsVar(x, d): Rolling variance of x over the past d days",
        "TsSkew(x, d): Rolling skewness of x over the past d days",
        "TsKurt(x, d): Rolling kurtosis of x over the past d days",
        "TsMed(x, d): Rolling median of x over the past d days",
        "TsMad(x, d): Rolling median absolute deviation over the past d days",
        "TsRank(x, d): Time-series rank of x over the past d days",
        "TsDelta(x, d): Today's value of x minus the value of x d days ago",
        "TsRatio(x, d): Today's value of x divided by the value of x d days ago",
        "TsPctChange(x, d): Percentage change in x over the past d days",
        "TsWMA(x, d): Weighted moving average over the past d days with linearly decaying weights.",
        "TsEMA(x, d): Exponential moving average of x with span d",
        "TsCov(x, y, d): Time-series covariance of x and y for the past d days",
        "TsCorr(x, y, d): Time-series correlation of x and y for the past d days",
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

    @property
    def definition_text(self) -> str:
        feature_text = "\n".join(
            [
                "$open, $high, $low, $close: Opening, daily highest, daily lowest, and closing prices.",
                "$vwap: Daily average price, weighted by the volume of trades at each price.",
                "$volume: Trading number of shares.",
            ]
        )
        operator_text = "\n".join(self.operators)
        return f"""The available features, constants and operators are listed below.
1. You can use the following features:
{feature_text}
2. You can use int constants eg: 1, 3, 5, 10, 20, 30, 60 during rolling(time-series)
calculations, and float: 0.0001, 0.001, 0.01, 0.0, 1.0, 2.0 during arithmetic calculations.
Other constants are not allowed.
3. The following operators are available:
(BEGIN OF FEATURES AND OPERATORS DEFINITIONS)
{operator_text}
(END OF FEATURES AND OPERATORS DEFINITIONS)
Examples of valid alpha expressions:
{chr(10).join(self.examples)}"""

    def as_prompt_context(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "rolling_constants": list(self.rolling_constants),
            "arithmetic_constants": list(self.arithmetic_constants),
            "operators": list(self.operators),
            "examples": list(self.examples),
            "rules": list(self.notes),
            "definition_text": self.definition_text,
        }


DEFAULT_FACTOR_DSL = FactorDSL()

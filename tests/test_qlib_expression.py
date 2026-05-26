import pytest

from lineage_evo.config import QlibConfig
from lineage_evo.factor import FactorExpression, QlibExpressionNormalizer, QlibExpressionValidator


def test_qlib_expression_normalizes_common_shortcuts():
    normalizer = QlibExpressionNormalizer()
    assert normalizer.normalize(FactorExpression("$close")) == "$close"
    assert normalizer.normalize(FactorExpression("Rank($close)")) == "Rank($close, 5)"
    assert normalizer.normalize(FactorExpression("TsMean($close, 5)")) == "Mean($close, 5)"
    assert normalizer.normalize(FactorExpression("$vwap")) == "$vwap"
    assert normalizer.normalize(FactorExpression("Div(Sub($open, $close), Add(Sub($high, $low), 0.001))")) == "(($open - $close) / (($high - $low) + 0.001))"
    assert normalizer.normalize(FactorExpression("TsCorr($close, $volume, 20)")) == "Corr($close, $volume, 20)"
    assert normalizer.normalize(FactorExpression("SLog1p($close)")) == "Sign($close) * Log(Abs($close) + 1)"
    assert normalizer.normalize(FactorExpression("TsPctChange($close, 5)")) == "($close / Ref($close, 5) - 1)"
    assert normalizer.normalize(FactorExpression("TsWMA($close, 10)")) == "WMA($close, 10)"


def test_qlib_expression_rejects_unknown_variable_or_operator():
    normalizer = QlibExpressionNormalizer()
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("Rank($foo)"))
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("Mystery($close)"))
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("TsMean($close, 7)"))
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("Add($close, 0.1)"))


def test_qlib_validator_execution_check_uses_warmup_window():
    validator = QlibExpressionValidator(
        QlibConfig(
            train_start="2015-01-01",
            train_end="2020-12-31",
        ),
        execution_check_window_days=120,
    )

    assert validator._execution_check_window() == ("2015-01-01", "2015-05-01")

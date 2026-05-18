import pytest

from lineage_evo.factor import FactorExpression, QlibExpressionNormalizer


def test_qlib_expression_normalizes_common_shortcuts():
    normalizer = QlibExpressionNormalizer()
    assert normalizer.normalize(FactorExpression("$close")) == "$close"
    assert normalizer.normalize(FactorExpression("Rank($close)")) == "Rank($close, 5)"
    assert normalizer.normalize(FactorExpression("TsMean($close, 5)")) == "Mean($close, 5)"
    assert normalizer.normalize(FactorExpression("$vwap")) == "$vwap"
    assert normalizer.normalize(FactorExpression("Div(Sub($open, $close), Add(Sub($high, $low), 0.001))")) == "(($open - $close) / (($high - $low) + 0.001))"
    assert normalizer.normalize(FactorExpression("TsCorr($close, $volume, 20)")) == "Corr($close, $volume, 20)"


def test_qlib_expression_rejects_unknown_variable_or_operator():
    normalizer = QlibExpressionNormalizer()
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("Rank($foo)"))
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("Mystery($close)"))
    with pytest.raises(ValueError):
        normalizer.normalize(FactorExpression("TsMean($close, 7)"))

import pytest

from lineage_evo.config import QlibConfig
from lineage_evo.evaluation import QlibEvaluator
from lineage_evo.factor import FactorExpression, QlibExpressionValidator


def qlib_available() -> bool:
    try:
        import qlib  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not qlib_available(), reason="qlib is not installed")
def test_qlib_validator_and_evaluator_on_local_data():
    config = QlibConfig(
        train_start="2018-01-02",
        train_end="2018-01-10",
        valid_start="2018-01-11",
        valid_end="2018-01-20",
    )
    validator = QlibExpressionValidator(config)
    expression = FactorExpression("Rank($close)")

    result = validator.validate(expression)
    evaluation = QlibEvaluator(config).evaluate(expression)

    assert result.is_valid is True
    assert isinstance(evaluation.train_ic, float)
    assert isinstance(evaluation.validation_icir, float)

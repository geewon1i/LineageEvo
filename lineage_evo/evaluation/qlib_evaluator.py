"""Qlib-backed factor evaluator."""

from __future__ import annotations

import math

from lineage_evo.config import QlibConfig
from lineage_evo.evaluation.evaluator import EvaluationResult
from lineage_evo.factor import FactorExpression, QlibExpressionNormalizer
from lineage_evo.qlib_warnings import suppress_qlib_all_nan_slice_warning


class QlibEvaluator:
    """Evaluate one factor with Qlib data and cross-sectional IC/ICIR."""

    def __init__(self, config: QlibConfig | None = None) -> None:
        self.config = config or QlibConfig.from_env()
        self.normalizer = QlibExpressionNormalizer()
        self._qlib_initialized = False
        self._evaluation_cache: dict[str, EvaluationResult] = {}

    def evaluate(self, expression: FactorExpression) -> EvaluationResult:
        qlib_expr = self.normalizer.normalize(expression)
        if qlib_expr in self._evaluation_cache:
            return self._evaluation_cache[qlib_expr]
        data = self._load_factor_and_label(qlib_expr)
        train = data.loc[
            (slice(None), slice(self.config.train_start, self.config.train_end)),
            :,
        ]
        valid = data.loc[
            (slice(None), slice(self.config.valid_start, self.config.valid_end)),
            :,
        ]
        train_ic, train_icir = self._ic_metrics(train)
        valid_ic, valid_icir = self._ic_metrics(valid)
        result = EvaluationResult(
            train_ic=train_ic,
            train_icir=train_icir,
            validation_ic=valid_ic,
            validation_icir=valid_icir,
        )
        self._evaluation_cache[qlib_expr] = result
        return result

    def _load_factor_and_label(self, qlib_expr: str):
        self._ensure_qlib()
        from qlib.data import D

        fields = [qlib_expr, self.config.label_expression]
        start = min(self.config.train_start, self.config.valid_start)
        end = max(self.config.train_end, self.config.valid_end)
        with suppress_qlib_all_nan_slice_warning():
            data = D.features(D.instruments(self.config.market), fields, start_time=start, end_time=end)
        data = data.rename(columns={qlib_expr: "factor", self.config.label_expression: "label"})
        data = data.sort_index().replace([float("inf"), float("-inf")], float("nan")).dropna()
        if data.empty:
            raise ValueError("Qlib evaluation produced empty factor/label data")
        return data

    def _ic_metrics(self, data) -> tuple[float, float]:
        if data.empty:
            raise ValueError("empty train/validation split")
        method = self.config.ic_method
        daily_ic = data.groupby(level="datetime").apply(lambda frame: frame["factor"].corr(frame["label"], method=method))
        daily_ic = daily_ic.dropna()
        if daily_ic.empty:
            raise ValueError("IC series is empty")
        mean_ic = float(daily_ic.mean())
        std_ic = float(daily_ic.std(ddof=1))
        if not math.isfinite(mean_ic):
            raise ValueError("IC mean is not finite")
        icir = 0.0 if std_ic == 0 or not math.isfinite(std_ic) else mean_ic / std_ic
        return mean_ic, float(icir)

    def _ensure_qlib(self) -> None:
        if self._qlib_initialized:
            return
        import qlib
        from qlib.constant import REG_CN

        region = REG_CN if self.config.region.lower() == "cn" else self.config.region
        qlib.init(provider_uri=self.config.provider_uri, region=region)
        self._qlib_initialized = True

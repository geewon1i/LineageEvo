"""Final factor selection, test IC evaluation, and Qlib backtest."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from lineage_evo.config import BacktestConfig, QlibConfig, SelectionConfig
from lineage_evo.factor import QlibExpressionNormalizer
from lineage_evo.lineage import FactorNode, LineageDAG
from lineage_evo.recording import SearchRecorder
from lineage_evo.qlib_warnings import suppress_qlib_all_nan_slice_warning


@dataclass(frozen=True)
class FinalizationResult:
    selected_count: int
    backtest_ran: bool
    warnings: list[str]


class Finalizer:
    def __init__(
        self,
        *,
        qlib_config: QlibConfig | None,
        selection_config: SelectionConfig,
        backtest_config: BacktestConfig,
        recorder: SearchRecorder,
    ) -> None:
        self.qlib_config = qlib_config
        self.selection_config = selection_config
        self.backtest_config = backtest_config
        self.recorder = recorder
        self.normalizer = QlibExpressionNormalizer()
        self._qlib_initialized = False

    def run(self, dag: LineageDAG) -> FinalizationResult:
        warnings: list[str] = []
        selected = self.select(dag)
        if len(selected) < self.selection_config.final_top_k:
            warnings.append(f"selected {len(selected)} factors; requested {self.selection_config.final_top_k}")
        self.recorder.write_selected_factors(self._selected_rows(selected))

        backtest_ran = False
        if self.qlib_config is not None:
            test_rows = self._test_ic_rows(selected)
            self.recorder.write_test_ic_results(test_rows)
            self.recorder.write_composite_test_ic_results(self._composite_test_ic_rows(selected))
            if self.backtest_config.enabled and selected:
                try:
                    self._run_backtest(selected)
                    backtest_ran = True
                except Exception as exc:
                    warnings.append(f"backtest failed: {type(exc).__name__}: {exc}")
                    self.recorder.write_backtest_summary([{"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}])
        return FinalizationResult(selected_count=len(selected), backtest_ran=backtest_ran, warnings=warnings)

    def select(self, dag: LineageDAG) -> list[FactorNode]:
        active = [dag.nodes[node_id] for node_id in dag.active_ids if dag.nodes[node_id].is_active and dag.nodes[node_id].evaluation is not None]
        ranked = sorted(active, key=lambda node: abs(node.evaluation.validation_icir), reverse=True)
        selected: list[FactorNode] = []
        seen_normalized: set[str] = set()
        for node in ranked:
            key = self._normalized_selection_key(node)
            if key in seen_normalized:
                continue
            seen_normalized.add(key)
            selected.append(node)
            if len(selected) >= self.selection_config.final_top_k:
                break
        return selected

    def _selected_rows(self, selected: list[FactorNode]) -> list[dict[str, Any]]:
        rows = []
        for rank, node in enumerate(selected, start=1):
            evaluation = node.evaluation
            rows.append(
                {
                    "selection_rank": rank,
                    "factor_id": node.factor_id,
                    "lineage_id": node.lineage_id,
                    "generation": node.generation,
                    "expression": node.expression.raw,
                    "normalized_expression": self._normalized_selection_key(node),
                    "train_ic": evaluation.train_ic,
                    "train_icir": evaluation.train_icir,
                    "raw_validation_ic": evaluation.validation_ic,
                    "raw_validation_icir": evaluation.validation_icir,
                    "selection_score": abs(evaluation.validation_icir),
                    "orientation": self._orientation(node),
                }
            )
        return rows

    def _normalized_selection_key(self, node: FactorNode) -> str:
        try:
            return self.normalizer.normalize(node.expression)
        except Exception:
            return f"raw:{node.expression.raw}"

    def _composite_test_ic_rows(self, selected: list[FactorNode]) -> list[dict[str, Any]]:
        if not selected:
            return []
        try:
            signal = self._composite_signal(selected)
            label = self._load_label()
            data = signal.to_frame("factor").join(label, how="inner").dropna()
            if data.empty:
                raise ValueError("composite test factor/label data is empty")
            test_ic, test_icir = self._ic_metrics(data)
            return [
                {
                    "signal_name": "oriented_equal_weight_top_k",
                    "selected_count": len(selected),
                    "factor_ids": json.dumps([node.factor_id for node in selected], ensure_ascii=False),
                    "test_ic": test_ic,
                    "test_icir": test_icir,
                    "status": "ok",
                    "failure_reason": None,
                }
            ]
        except Exception as exc:
            return [
                {
                    "signal_name": "oriented_equal_weight_top_k",
                    "selected_count": len(selected),
                    "factor_ids": json.dumps([node.factor_id for node in selected], ensure_ascii=False),
                    "test_ic": None,
                    "test_icir": None,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            ]

    def _test_ic_rows(self, selected: list[FactorNode]) -> list[dict[str, Any]]:
        rows = []
        for rank, node in enumerate(selected, start=1):
            try:
                factor = self.normalizer.normalize(node.expression)
                data = self._load_factor_label(factor)
                data["factor"] = data["factor"] * self._orientation(node)
                test_ic, test_icir = self._ic_metrics(data)
                rows.append(
                    {
                        "selection_rank": rank,
                        "factor_id": node.factor_id,
                        "expression": node.expression.raw,
                        "test_ic": test_ic,
                        "test_icir": test_icir,
                        "status": "ok",
                        "failure_reason": None,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "selection_rank": rank,
                        "factor_id": node.factor_id,
                        "expression": node.expression.raw,
                        "test_ic": None,
                        "test_icir": None,
                        "status": "failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return rows

    def _run_backtest(self, selected: list[FactorNode]) -> None:
        signal = self._composite_signal(selected)
        from qlib.backtest import backtest
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy import TopkDropoutStrategy

        strategy = TopkDropoutStrategy(
            signal=signal,
            topk=self.backtest_config.topk,
            n_drop=self.backtest_config.n_drop,
            risk_degree=self.backtest_config.risk_degree,
        )
        executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
        portfolio_metric, _indicator_metric = backtest(
            start_time=self.qlib_config.test_start,
            end_time=self.qlib_config.test_end,
            strategy=strategy,
            executor=executor,
            benchmark=self.qlib_config.benchmark,
            account=self.backtest_config.account,
        )
        report = self._extract_report(portfolio_metric)
        daily_rows = self._daily_report_rows(report)
        self.recorder.write_backtest_daily_report(daily_rows)

        if "return" in report:
            analysis = risk_analysis(report["return"], freq="day")
            summary = self._flatten_risk_analysis(analysis)
        else:
            summary = {}
        summary.update({"status": "ok", "selected_count": len(selected), "benchmark": self.qlib_config.benchmark})
        if "max_drawdown" not in summary and "return" in report:
            summary["max_drawdown"] = self._max_drawdown(report["return"])
        self.recorder.write_backtest_summary([summary])

    def _composite_signal(self, selected: list[FactorNode]):
        self._ensure_qlib()
        import pandas as pd
        from qlib.data import D

        fields = list(dict.fromkeys(self.normalizer.normalize(node.expression) for node in selected))
        with suppress_qlib_all_nan_slice_warning():
            data = D.features(D.instruments(self.qlib_config.market), fields, start_time=self.qlib_config.test_start, end_time=self.qlib_config.test_end)
        data = data.replace([float("inf"), float("-inf")], float("nan"))
        oriented = pd.DataFrame(index=data.index)
        for node in selected:
            qlib_expr = self.normalizer.normalize(node.expression)
            oriented[node.factor_id] = data[qlib_expr] * self._orientation(node)

        def zscore(frame):
            std = frame.std(ddof=0)
            return (frame - frame.mean()) / std.replace(0, float("nan"))

        standardized = oriented.groupby(level="datetime", group_keys=False).apply(zscore)
        signal = standardized.mean(axis=1).dropna()
        if signal.empty:
            raise ValueError("composite signal is empty")
        return pd.Series(signal, name="score")

    def _load_factor_label(self, qlib_expr: str):
        self._ensure_qlib()
        from qlib.data import D

        fields = [qlib_expr, self.qlib_config.label_expression]
        with suppress_qlib_all_nan_slice_warning():
            data = D.features(D.instruments(self.qlib_config.market), fields, start_time=self.qlib_config.test_start, end_time=self.qlib_config.test_end)
        data = data.rename(columns={qlib_expr: "factor", self.qlib_config.label_expression: "label"})
        data = data.sort_index().replace([float("inf"), float("-inf")], float("nan")).dropna()
        if data.empty:
            raise ValueError("test factor/label data is empty")
        return data

    def _load_label(self):
        self._ensure_qlib()
        from qlib.data import D

        with suppress_qlib_all_nan_slice_warning():
            data = D.features(
                D.instruments(self.qlib_config.market),
                [self.qlib_config.label_expression],
                start_time=self.qlib_config.test_start,
                end_time=self.qlib_config.test_end,
            )
        data = data.rename(columns={self.qlib_config.label_expression: "label"})
        data = data.sort_index().replace([float("inf"), float("-inf")], float("nan")).dropna()
        if data.empty:
            raise ValueError("test label data is empty")
        return data

    def _ic_metrics(self, data) -> tuple[float, float]:
        daily_ic = data.groupby(level="datetime").apply(lambda frame: frame["factor"].corr(frame["label"], method=self.qlib_config.ic_method))
        daily_ic = daily_ic.dropna()
        if daily_ic.empty:
            raise ValueError("test IC series is empty")
        mean_ic = float(daily_ic.mean())
        std_ic = float(daily_ic.std(ddof=1))
        if not math.isfinite(mean_ic):
            raise ValueError("test IC mean is not finite")
        icir = 0.0 if std_ic == 0 or not math.isfinite(std_ic) else mean_ic / std_ic
        return mean_ic, float(icir)

    def _ensure_qlib(self) -> None:
        if self._qlib_initialized:
            return
        import qlib
        from qlib.constant import REG_CN

        region = REG_CN if self.qlib_config.region.lower() == "cn" else self.qlib_config.region
        qlib.init(provider_uri=self.qlib_config.provider_uri, region=region)
        self._qlib_initialized = True

    @staticmethod
    def _extract_report(portfolio_metric):
        if isinstance(portfolio_metric, tuple):
            return portfolio_metric[0]
        if isinstance(portfolio_metric, dict) and "1day" in portfolio_metric:
            return portfolio_metric["1day"][0] if isinstance(portfolio_metric["1day"], tuple) else portfolio_metric["1day"]
        return portfolio_metric

    @staticmethod
    def _daily_report_rows(report) -> list[dict[str, Any]]:
        if hasattr(report, "reset_index"):
            frame = report.reset_index()
            return frame.to_dict(orient="records")
        return []

    @staticmethod
    def _flatten_risk_analysis(analysis) -> dict[str, Any]:
        if hasattr(analysis, "to_dict"):
            raw = analysis.to_dict()
        else:
            raw = dict(analysis)
        flat: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    flat[str(inner_key)] = inner_value
            else:
                flat[str(key)] = value
        return flat

    @staticmethod
    def _max_drawdown(returns) -> float:
        wealth = (1 + returns.fillna(0)).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        return float(drawdown.min())

    @staticmethod
    def _orientation(node: FactorNode) -> int:
        if node.evaluation is not None and node.evaluation.validation_icir < 0:
            return -1
        return 1

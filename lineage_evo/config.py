"""Default experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib

from lineage_evo.env import load_dotenv


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class SearchConfig:
    active_pool_size: int = 50
    factor_prompt_length_limit: int = 40
    factor_length_limit: int = 50
    target_valid_evaluations: int = 200
    valid_per_generation: int = 5
    mutation_per_generation: int = 3
    crossover_per_generation: int = 2
    max_attempts_per_operator_slot: int = 5
    mutation_rank_weight: float = 0.8
    mutation_lineage_balance_weight: float = 0.2
    max_runtime_seconds: int = 28800
    use_local: bool = True
    seed_count: int = 10
    max_seed_generation_attempts: int = 50
    parent_train_icir_weight: float = 0.7
    parent_validation_icir_weight: float = 0.3
    parent_gap_penalty_weight: float = 0.2
    parent_weight_epsilon: float = 1e-6

    @classmethod
    def from_env(cls) -> "SearchConfig":
        load_dotenv()
        return cls(
            max_runtime_seconds=_env_int("LINEAGEEVO_FACTOR_MINING_TIMEOUT", 28800),
            use_local=_env_bool("LINEAGEEVO_USE_LOCAL", True),
        )


@dataclass(frozen=True)
class PriorRewriteConfig:
    top_k_patterns: int = 5
    max_text_length: int = 240
    strong_validation_delta: float = 0.05


@dataclass(frozen=True)
class PriorUpdateConfig:
    improvement_abs_floor: float = 0.02
    improvement_ratio: float = 0.20
    degradation_abs_floor: float = 0.05
    degradation_ratio: float = 0.30
    trend_window: int = 5
    trend_alpha: float = 1.0 / 3.0
    trend_epsilon: float = 0.01


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai-compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 60.0
    max_retry: int = 5
    retry_wait_seconds: float = 5.0
    max_tokens: int = 4000
    temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()
        return cls(
            base_url=os.getenv("LINEAGEEVO_LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("LINEAGEEVO_LLM_API_KEY"),
            model=os.getenv("LINEAGEEVO_LLM_MODEL"),
            timeout_seconds=_env_float("LINEAGEEVO_LLM_TIMEOUT_SECONDS", 60.0),
            max_retry=_env_int("LINEAGEEVO_MAX_RETRY", 5),
            retry_wait_seconds=_env_float("LINEAGEEVO_RETRY_WAIT_SECONDS", 5.0),
            max_tokens=_env_int("LINEAGEEVO_LLM_MAX_TOKENS", 4000),
            temperature=_env_float("LINEAGEEVO_LLM_TEMPERATURE", 0.7),
        )

    def validate_for_request(self) -> None:
        missing = []
        if not self.api_key:
            missing.append("LINEAGEEVO_LLM_API_KEY")
        if not self.model:
            missing.append("LINEAGEEVO_LLM_MODEL")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"missing LLM configuration: {joined}")


@dataclass(frozen=True)
class QlibConfig:
    provider_uri: str = "C:/Users/94875/.qlib/qlib_data/cn_data"
    region: str = "cn"
    market: str = "csi500"
    train_start: str = "2015-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2022-04-30"
    test_start: str = "2022-05-01"
    test_end: str = "2026-04-30"
    label_expression: str = "Ref($close, -2) / Ref($close, -1) - 1"
    ic_method: str = "spearman"
    benchmark: str = "SH000905"

    @classmethod
    def from_env(cls) -> "QlibConfig":
        load_dotenv()
        return cls(
            provider_uri=os.getenv("LINEAGEEVO_QLIB_PROVIDER_URI", cls.provider_uri),
            region=os.getenv("LINEAGEEVO_QLIB_REGION", cls.region),
            market=os.getenv("LINEAGEEVO_QLIB_MARKET", cls.market),
        )


@dataclass(frozen=True)
class SelectionConfig:
    final_top_k: int = 5
    selection_metric: str = "validation_icir"


@dataclass(frozen=True)
class BacktestConfig:
    enabled: bool = True
    account: float = 100000000.0
    topk: int = 50
    n_drop: int = 5
    risk_degree: float = 0.95


@dataclass(frozen=True)
class ExperimentConfig:
    search: SearchConfig
    qlib: QlibConfig
    selection: SelectionConfig
    backtest: BacktestConfig
    prior_update: PriorUpdateConfig
    llm: dict
    raw: dict

    @classmethod
    def from_toml(cls, path: str | Path = "configs/default.toml") -> "ExperimentConfig":
        load_dotenv()
        config_path = Path(path)
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        dataset = data.get("dataset", {})
        time_split = data.get("time_split", {})
        evaluation = data.get("evaluation", {})
        search = data.get("search", {})
        selection = data.get("selection", {})
        backtest = data.get("backtest", {})
        prior_update = data.get("prior_update", {})
        llm = data.get("llm", {})

        qlib = QlibConfig(
            provider_uri=os.getenv("LINEAGEEVO_QLIB_PROVIDER_URI", dataset.get("provider_uri", QlibConfig.provider_uri)),
            region=dataset.get("region", QlibConfig.region),
            market=dataset.get("market", QlibConfig.market),
            benchmark=dataset.get("benchmark", QlibConfig.benchmark),
            train_start=time_split.get("train_start", QlibConfig.train_start),
            train_end=time_split.get("train_end", QlibConfig.train_end),
            valid_start=time_split.get("valid_start", QlibConfig.valid_start),
            valid_end=time_split.get("valid_end", QlibConfig.valid_end),
            test_start=time_split.get("test_start", QlibConfig.test_start),
            test_end=time_split.get("test_end", QlibConfig.test_end),
            label_expression=evaluation.get("label_expression", QlibConfig.label_expression),
            ic_method=evaluation.get("ic_method", QlibConfig.ic_method),
        )
        return cls(
            search=SearchConfig(
                target_valid_evaluations=search.get("target_valid_evaluations", SearchConfig.target_valid_evaluations),
                active_pool_size=search.get("active_pool_size", SearchConfig.active_pool_size),
                factor_prompt_length_limit=search.get("factor_prompt_length_limit", SearchConfig.factor_prompt_length_limit),
                factor_length_limit=search.get("factor_length_limit", SearchConfig.factor_length_limit),
                mutation_per_generation=search.get("mutation_per_generation", SearchConfig.mutation_per_generation),
                crossover_per_generation=search.get("crossover_per_generation", SearchConfig.crossover_per_generation),
                max_attempts_per_operator_slot=search.get("max_attempts_per_operator_slot", SearchConfig.max_attempts_per_operator_slot),
                seed_count=search.get("seed_count", SearchConfig.seed_count),
                max_seed_generation_attempts=search.get("max_seed_generation_attempts", SearchConfig.max_seed_generation_attempts),
                parent_train_icir_weight=search.get("parent_train_icir_weight", SearchConfig.parent_train_icir_weight),
                parent_validation_icir_weight=search.get("parent_validation_icir_weight", SearchConfig.parent_validation_icir_weight),
                parent_gap_penalty_weight=search.get("parent_gap_penalty_weight", SearchConfig.parent_gap_penalty_weight),
            ),
            qlib=qlib,
            selection=SelectionConfig(
                final_top_k=selection.get("final_top_k", SelectionConfig.final_top_k),
                selection_metric=selection.get("selection_metric", SelectionConfig.selection_metric),
            ),
            backtest=BacktestConfig(
                enabled=backtest.get("enabled", BacktestConfig.enabled),
                account=backtest.get("account", BacktestConfig.account),
                topk=backtest.get("topk", BacktestConfig.topk),
                n_drop=backtest.get("n_drop", BacktestConfig.n_drop),
                risk_degree=backtest.get("risk_degree", BacktestConfig.risk_degree),
            ),
            prior_update=PriorUpdateConfig(
                improvement_abs_floor=prior_update.get(
                    "improvement_abs_floor",
                    prior_update.get("improvement_threshold", PriorUpdateConfig.improvement_abs_floor),
                ),
                improvement_ratio=prior_update.get("improvement_ratio", PriorUpdateConfig.improvement_ratio),
                degradation_abs_floor=prior_update.get(
                    "degradation_abs_floor",
                    prior_update.get("degradation_threshold", PriorUpdateConfig.degradation_abs_floor),
                ),
                degradation_ratio=prior_update.get("degradation_ratio", PriorUpdateConfig.degradation_ratio),
                trend_window=prior_update.get("trend_window", PriorUpdateConfig.trend_window),
                trend_alpha=prior_update.get("trend_alpha", PriorUpdateConfig.trend_alpha),
                trend_epsilon=prior_update.get("trend_epsilon", PriorUpdateConfig.trend_epsilon),
            ),
            llm=llm,
            raw=data,
        )

    def snapshot(self) -> dict:
        return {
            "search_config": self.search.__dict__,
            "qlib_config": self.qlib.__dict__,
            "selection_config": self.selection.__dict__,
            "backtest_config": self.backtest.__dict__,
            "prior_update_config": self.prior_update.__dict__,
            "llm": self.llm,
            "raw_config": self.raw,
        }

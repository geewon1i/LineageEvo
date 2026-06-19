import os

from lineage_evo.config import ExperimentConfig, QlibConfig
from lineage_evo.env import load_dotenv


def test_dotenv_loader_reads_values_without_overriding_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LINEAGEEVO_QLIB_MARKET=csi300\nLINEAGEEVO_KEEP=from_file\n", encoding="utf-8")
    monkeypatch.setenv("LINEAGEEVO_QLIB_MARKET", "csi500")

    load_dotenv(env_file)

    assert os.environ["LINEAGEEVO_QLIB_MARKET"] == "csi500"
    assert os.environ["LINEAGEEVO_KEEP"] == "from_file"


def test_qlib_config_defaults_to_csi500():
    assert QlibConfig().market == "csi500"


def test_default_toml_config_loads():
    config = ExperimentConfig.from_toml("configs/default.toml")

    assert config.qlib.market == "csi500"
    assert config.qlib.train_start == "2015-01-01"
    assert config.qlib.valid_start == "2021-01-01"
    assert config.qlib.test_start == "2022-05-01"
    assert config.search.factor_prompt_length_limit == 40
    assert config.search.factor_length_limit == 50
    assert config.search.elite_archive_size == 20
    assert config.search.max_active_lineage_ratio == 0.40
    assert config.search.min_active_lineages_before_cap == 2
    assert config.search.lineage_concentration_weight == 0.20
    assert config.selection.final_top_k == 1
    assert config.backtest.topk == 50
    assert config.search.max_runtime_seconds == 28800
    assert config.selection.selection_metric == "validation_ic"
    assert config.search.parent_train_ic_weight == 0.7
    assert config.search.parent_validation_ic_weight == 0.3
    assert config.prior_update.improvement_abs_floor == 0.003
    assert config.prior_update.improvement_ratio == 0.30
    assert config.prior_update.degradation_abs_floor == 0.008
    assert config.prior_update.degradation_ratio == 0.45
    assert config.llm["candidate_provider"] == "openai-compatible"
    assert config.llm["prior_provider"] == "openai-compatible"
    assert config.llm["max_tokens"] == 8000

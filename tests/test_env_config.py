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
    assert config.qlib.test_start == "2019-01-01"
    assert config.selection.final_top_k == 5
    assert config.backtest.topk == 50
    assert config.llm["candidate_provider"] == "openai-compatible"
    assert config.llm["prior_provider"] == "openai-compatible"

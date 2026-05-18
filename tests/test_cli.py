import pytest

from lineage_evo import cli


def test_cli_dry_run_mock_candidate(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["lineage-evo", "llm-dry-run", "--kind", "candidate", "--llm", "mock"])
    cli.main()
    captured = capsys.readouterr()
    assert "Rank($close)" in captured.out


def test_cli_openai_compatible_missing_config_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LINEAGEEVO_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LINEAGEEVO_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LINEAGEEVO_LLM_API_KEY"):
        cli._build_candidate_generator("openai-compatible")


def test_qlib_smoke_real_candidate_missing_config_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LINEAGEEVO_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LINEAGEEVO_LLM_MODEL", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "lineage-evo",
            "qlib-smoke-run",
            "--log-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="LINEAGEEVO_LLM_API_KEY"):
        cli.main()

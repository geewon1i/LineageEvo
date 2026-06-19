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


def test_ablation_mode_argument_aliases_are_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "lineage-evo",
            "smoke-run",
            "--ablation-mode",
            "lineage_only",
            "--target-valid",
            "1",
            "--log-dir",
            str(tmp_path),
        ],
    )
    cli.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "lineage-evo",
            "smoke-run",
            "--fusion-mode",
            "global_only",
            "--target-valid",
            "1",
            "--log-dir",
            str(tmp_path),
        ],
    )
    cli.main()


def test_print_final_outputs_includes_test_and_backtest_summary(tmp_path, capsys):
    (tmp_path / "selected_factors.csv").write_text(
        "selection_rank,factor_id,expression,train_ic,train_icir,raw_validation_ic,raw_validation_icir,selection_score,orientation\n"
        "1,f1,Rank($close),0.01,0.1,-0.02,-0.2,0.02,-1\n",
        encoding="utf-8",
    )
    (tmp_path / "test_ic_results.csv").write_text(
        "selection_rank,factor_id,expression,test_ic,test_icir,status,failure_reason\n"
        "1,f1,Rank($close),0.01,0.2,ok,\n",
        encoding="utf-8",
    )
    (tmp_path / "composite_test_ic_results.csv").write_text(
        "signal_name,selected_count,factor_ids,test_ic,test_icir,status,failure_reason\n"
        'oriented_equal_weight_top_k,1,"[""f1""]",0.03,0.4,ok,\n',
        encoding="utf-8",
    )
    (tmp_path / "backtest_summary.csv").write_text(
        "annualized_return,information_ratio,max_drawdown,status,benchmark\n"
        "0.12,0.8,-0.2,ok,SH000905\n",
        encoding="utf-8",
    )
    (tmp_path / "backtest_daily_report.csv").write_text(
        "datetime,account,bench\n"
        "2019-01-02,100.0,0.01\n"
        "2019-01-03,110.0,0.02\n",
        encoding="utf-8",
    )

    cli._print_final_outputs(tmp_path)

    captured = capsys.readouterr().out
    assert "selected_factors:" in captured
    assert "orientation=-1" in captured
    assert "final_composite_test_results:" in captured
    assert "test_icir=0.400000" in captured
    assert "single_factor_oriented_test_results:" in captured
    assert "backtest_summary:" in captured
    assert "account_return=0.1000" in captured

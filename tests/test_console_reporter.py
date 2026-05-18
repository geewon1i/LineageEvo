from lineage_evo.evaluation import EvaluationResult
from lineage_evo.recording import ConsoleReporter


def test_console_reporter_prints_metrics_without_color(capsys):
    reporter = ConsoleReporter(enabled=True, use_color=False)

    reporter.valid_factor(
        expression="Rank($close)",
        evaluation=EvaluationResult(train_ic=0.01, train_icir=0.5, validation_ic=0.02, validation_icir=0.7),
        child_id="f1",
    )

    out = capsys.readouterr().out
    assert "VALID FACTOR" in out
    assert "Rank($close)" in out
    assert "valid_icir=0.700000" in out
    assert "\033[" not in out


def test_console_reporter_can_print_full_llm_io(capsys):
    reporter = ConsoleReporter(enabled=True, print_llm_io=True, use_color=False)

    reporter.llm_input("CANDIDATE", "system", "user")
    reporter.llm_output("CANDIDATE", '{"factor": "$close"}')

    out = capsys.readouterr().out
    assert "CANDIDATE LLM INPUT" in out
    assert "SYSTEM PROMPT" in out
    assert "user" in out
    assert '{"factor": "$close"}' in out


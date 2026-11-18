import evals.run_evals as run_evals
from evals.run_evals import main


def test_eval_gate_passes_at_the_recorded_baseline(capsys):
    assert main(["--min-pass", "17"]) == 0
    assert "overall" in capsys.readouterr().out


def test_eval_gate_fails_when_the_score_is_below_the_threshold(capsys):
    assert main(["--min-pass", "99"]) == 1
    assert "FAILED" in capsys.readouterr().out


class CountingModel:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return "[]"


def test_llm_eval_refuses_to_start_when_planned_calls_exceed_the_cap(monkeypatch, capsys):
    model = CountingModel()
    monkeypatch.setattr(run_evals, "llm_from_env", lambda role=None: model)
    assert main(["--extractor", "llm", "--max-llm-calls", "5"]) == 2
    out = capsys.readouterr().out
    assert "calls planned" in out and "Refusing" in out
    assert model.calls == 0  # nothing was spent


def test_llm_eval_runs_within_the_cap_and_reports_the_call_count(monkeypatch, capsys):
    model = CountingModel()
    monkeypatch.setattr(run_evals, "llm_from_env", lambda role=None: model)
    main(["--extractor", "llm", "--max-llm-calls", "100"])
    out = capsys.readouterr().out
    assert "model calls: 36" in out
    assert model.calls == 36

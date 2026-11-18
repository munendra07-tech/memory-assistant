from evals.run_evals import main


def test_eval_gate_passes_at_the_recorded_baseline(capsys):
    assert main(["--min-pass", "17"]) == 0
    assert "overall" in capsys.readouterr().out


def test_eval_gate_fails_when_the_score_is_below_the_threshold(capsys):
    assert main(["--min-pass", "99"]) == 1
    assert "FAILED" in capsys.readouterr().out

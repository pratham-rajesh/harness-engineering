"""Tests for ml-harness: tripwire, accept/reject policy, final-once, resume, proposers. All offline.

Scoring a candidate runs a real (small) cross-validation in a child process, so the few tests that need real scores
use the cheap Ridge/mean models. Everything involving an LLM uses the scripted fake model.
"""
import json
import sys
from pathlib import Path

import pytest

ML = Path(__file__).resolve().parent.parent / "ml-harness"
sys.path.insert(0, str(ML))
import evaluate  # noqa: E402
import lab as lab_module  # noqa: E402
import proposers  # noqa: E402
from conftest import call, say  # noqa: E402
from lab import FinalAlreadyRun, Lab  # noqa: E402

RIDGE = ("from sklearn.linear_model import Ridge\nfrom sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\n\n\ndef build_model():\n"
         "    return make_pipeline(StandardScaler(), Ridge(alpha={a}))\n")
OK = "from sklearn.linear_model import Ridge\n\n\ndef build_model():\n    return Ridge()\n"


# ------------------------------------------------------------------ static tripwire
@pytest.mark.parametrize("source,fragment", [
    ("def build_model():\n    open('x')\n", "open"),
    ("def build_model():\n    return eval('1')\n", "eval"),
    ("def build_model():\n    return __import__('os')\n", "__import__"),
    ("import os\ndef build_model():\n    pass\n", "os"),
    ("import subprocess\ndef build_model():\n    pass\n", "subprocess"),
    ("from sklearn.datasets import load_diabetes\ndef build_model():\n    pass\n", "datasets"),
    ("import sklearn.datasets\ndef build_model():\n    pass\n", "datasets"),
    ("import numpy as np\ndef build_model():\n    np.load('x.npy')\n", "load"),
    ("def build_model():\n    return ().__class__.__bases__\n", "__class__"),
    ("def build_model():\n    print('hi')\n", "print"),
    ("def helper():\n    pass\n", "build_model"),
    ("def build_model(:\n", "syntax"),
    ("x = 1\n" * 20_000 + "def build_model():\n    pass\n", "too long"),
])
def test_static_check_rejects(source, fragment):
    problems = evaluate.static_check(source)
    assert problems and any(fragment in p for p in problems), problems


def test_static_check_accepts_plain_sklearn_candidates():
    assert evaluate.static_check(OK) == []
    assert evaluate.static_check(Path(ML / "baseline_candidate.py").read_text()) == []


def test_every_random_template_passes_the_tripwire():
    for seed in range(3):
        for i in range(1, 60):
            hypothesis, source = proposers.random_propose(i, seed)
            assert evaluate.static_check(source) == [], hypothesis


def test_random_proposer_is_deterministic_and_seed_dependent():
    assert proposers.random_propose(3, 0) == proposers.random_propose(3, 0)
    assert {proposers.random_propose(i, 0)[1] for i in range(1, 12)} != {proposers.random_propose(i, 1)[1] for i in range(1, 12)}


# ------------------------------------------------------------------ evaluator isolation
def test_invalid_candidate_never_reaches_a_child_process(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("import os\ndef build_model():\n    pass\n")
    assert evaluate.score(f)["status"] == "invalid"


def test_runtime_failure_is_reported_as_crash(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("def build_model():\n    raise ValueError('boom')\n")
    result = evaluate.score(f)
    assert result["status"] == "crash" and "boom" in result["error"]


def test_timeout_is_reported(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("def build_model():\n    while True:\n        pass\n")
    assert evaluate.score(f, timeout=3)["status"] == "timeout"


def test_baseline_scores_as_documented(tmp_path):
    result = evaluate.score(ML / "baseline_candidate.py")
    assert result["status"] == "ok" and 77 < result["rmse"] < 79


# ------------------------------------------------------------------ lab policy (scores faked for speed)
@pytest.fixture
def fake_scores(monkeypatch):
    """Make evaluate.score return scripted RMSEs; `plan` maps a marker in the source to a result."""
    plan = {}

    def fake(path, timeout=0):
        src = Path(path).read_text()
        if "baseline" in Path(path).name or "DummyRegressor" in src:
            return {"status": "ok", "rmse": 78.0, "rmse_std": 3.0, "fit_seconds": 0.1}
        for marker, result in plan.items():
            if marker in src:
                return result if isinstance(result, dict) else {"status": "ok", "rmse": result, "rmse_std": 1.0, "fit_seconds": 0.1}
        raise AssertionError("unplanned candidate")
    monkeypatch.setattr(lab_module.evaluate, "score", fake)
    return plan


def cand(tag):
    return f"# {tag}\n" + OK


def test_baseline_is_attempt_zero_and_first_best(tmp_path, fake_scores):
    lab = Lab(tmp_path)
    entry = lab.init_baseline()
    assert entry["iter"] == 0 and entry["status"] == "accepted" and lab.best_rmse == 78.0
    assert lab.n_attempts() == 0 and lab.best_path.exists()


def test_accept_requires_a_margin_and_updates_best_file(tmp_path, fake_scores):
    fake_scores.update({"# a": 60.0, "# b": 59.9, "# c": 57.0})
    lab = Lab(tmp_path, min_rel_improvement=0.005)
    assert lab.try_candidate(cand("a"), "a")["status"] == "accepted"
    assert lab.try_candidate(cand("b"), "b")["status"] == "rejected"          # 59.9 is not 0.5% better than 60.0
    assert "# a" in lab.best_source()
    assert lab.try_candidate(cand("c"), "c")["status"] == "accepted"
    assert lab.best_rmse == 57.0 and "# c" in lab.best_source()
    assert (tmp_path / "attempts" / "iter_02.py").exists()                    # rejected attempts are kept too


def test_worse_candidate_is_rejected(tmp_path, fake_scores):
    fake_scores["# w"] = 90.0
    lab = Lab(tmp_path)
    assert lab.try_candidate(cand("w"), "w")["status"] == "rejected" and lab.best_rmse == 78.0


def test_suspiciously_good_result_is_rejected_as_possible_leak(tmp_path, fake_scores):
    fake_scores["# leak"] = 5.0
    lab = Lab(tmp_path)
    entry = lab.try_candidate(cand("leak"), "perfect model")
    assert entry["status"] == "suspicious" and lab.best_rmse == 78.0 and "leak" in entry["error"]


def test_duplicate_source_is_not_rescored(tmp_path, fake_scores):
    fake_scores["# d"] = 60.0
    lab = Lab(tmp_path)
    lab.try_candidate(cand("d"), "first")
    del fake_scores["# d"]                                                    # a second score() call would now blow up
    assert lab.try_candidate(cand("d"), "again")["status"] == "duplicate"


@pytest.mark.parametrize("status", ["invalid", "crash", "timeout"])
def test_failed_evaluations_are_logged_not_accepted(tmp_path, fake_scores, status):
    fake_scores["# f"] = {"status": status, "error": "it broke"}
    lab = Lab(tmp_path)
    entry = lab.try_candidate(cand("f"), "f")
    assert entry["status"] == status and entry["rmse"] is None and lab.best_rmse == 78.0


def test_resumes_from_an_existing_ledger(tmp_path, fake_scores):
    fake_scores["# r"] = 60.0
    Lab(tmp_path).try_candidate(cand("r"), "r")
    again = Lab(tmp_path)
    assert again.n_attempts() == 1 and again.best_rmse == 60.0 and "# r" in again.best_source()
    assert again.init_baseline()["iter"] == 0                                  # does not re-score or duplicate the baseline
    assert len(again.entries) == 2


def test_random_run_resumes_instead_of_restarting(tmp_path, fake_scores, monkeypatch):
    monkeypatch.setattr(proposers, "random_propose", lambda i, seed: (f"h{i}", cand(f"r{i}")))
    for i in range(1, 6):
        fake_scores[f"# r{i}"] = 70.0 - i
    lab = Lab(tmp_path)
    proposers.random_run(lab, 2, seed=0, log=lambda *_: None)
    proposers.random_run(Lab(tmp_path), 4, seed=0, log=lambda *_: None)
    assert Lab(tmp_path).n_attempts() == 4


def test_final_runs_once_and_a_crash_does_not_burn_it(tmp_path, fake_scores, monkeypatch):
    lab = Lab(tmp_path)
    lab.init_baseline()
    monkeypatch.setattr(lab_module.evaluate, "final_test", lambda *a, **k: {"status": "crash", "error": "x"})
    assert lab.final()["status"] == "crash" and not lab.final_path.exists()
    monkeypatch.setattr(lab_module.evaluate, "final_test",
                        lambda *a, **k: {"status": "ok", "best_test_rmse": 70.0, "baseline_test_rmse": 78.0, "n_test_rows": 89})
    assert lab.final()["best_test_rmse"] == 70.0
    with pytest.raises(FinalAlreadyRun):
        lab.final()
    with pytest.raises(FinalAlreadyRun):
        Lab(tmp_path).final()                                                  # also across processes / restarts


def test_report_writes_table_and_svg(tmp_path, fake_scores):
    fake_scores.update({"# good": 60.0, "# worse": 90.0, "# broken": {"status": "crash", "error": "boom | pipe"}})
    lab = Lab(tmp_path)
    for tag in ("good", "worse", "broken"):
        lab.try_candidate(cand(tag), f"idea {tag}")
    out = lab.report()
    text = out.read_text()
    assert "| 1 | accepted | 60.00" in text and "crash" in text and "not evaluated yet" in text
    svg = (tmp_path / "progress.svg").read_text()
    assert svg.startswith("<svg") and "<polyline" in svg


# ------------------------------------------------------------------ real end-to-end (one real child process)
def test_real_ridge_beats_the_mean_baseline_end_to_end(tmp_path):
    lab = Lab(tmp_path)
    entry = lab.try_candidate(RIDGE.format(a=1.0), "linear model with scaling")
    assert entry["status"] == "accepted" and entry["rmse"] < 60 and lab.baseline_rmse > 77
    result = lab.final()
    assert result["status"] == "ok" and result["best_test_rmse"] < result["baseline_test_rmse"]


# ------------------------------------------------------------------ llm proposer on the scripted fake model
def test_llm_proposer_submits_through_run_experiment_only(tmp_path, fake_scores, script):
    fake_scores["# llm1"] = 62.0
    fake = script(call("run_experiment", hypothesis="try ridge", source=cand("llm1")),
                  say("tried ridge, it helped"))
    lab = Lab(tmp_path)
    proposers.llm_run(lab, iterations=1, log=lambda *_: None)
    assert lab.n_attempts() == 1 and lab.entries[1]["proposer"] == "llm" and lab.best_rmse == 62.0
    names = [t["function"]["name"] for t in fake.requests[0]["tools"]]
    assert names == ["read_file", "list_dir", "run_experiment"]               # no write_file / run_python / spawn_subagent
    assert "Current best dev RMSE: 78.00" in fake.requests[0]["messages"][1]["content"]
    assert "run_experiment" in fake.requests[1]["messages"][-1]["content"] or "status=accepted" in fake.requests[1]["messages"][-1]["content"]


def test_llm_cannot_call_forbidden_tools(tmp_path, fake_scores, script):
    script(call("write_file", path="evaluate.py", content="# pwned"), say("done"), say("done"))
    lab = Lab(tmp_path)
    proposers.llm_run(lab, iterations=1, max_sessions=1, log=lambda *_: None)
    assert lab.n_attempts() == 0
    assert not (tmp_path / "workspace" / "evaluate.py").exists()


def test_llm_session_cap_stops_a_model_that_never_experiments(tmp_path, fake_scores, script):
    script(say("I have no ideas"), say("still none"))
    lab = Lab(tmp_path)
    proposers.llm_run(lab, iterations=5, max_sessions=2, log=lambda *_: None)
    assert lab.n_attempts() == 0                                              # terminated, did not loop forever


def test_llm_run_experiment_quota_is_enforced(tmp_path, fake_scores, script):
    fake_scores.update({"# q1": 70.0, "# q2": 69.0, "# q3": 68.0})
    script(call("run_experiment", "c1", hypothesis="1", source=cand("q1")),
           call("run_experiment", "c2", hypothesis="2", source=cand("q2")),
           call("run_experiment", "c3", hypothesis="3", source=cand("q3")),
           say("done"))
    lab = Lab(tmp_path)
    proposers.llm_run(lab, iterations=3, max_sessions=1, log=lambda *_: None)
    assert lab.n_attempts() == 2                                              # 2 per session; the 3rd got a QUOTA message


# ------------------------------------------------------------------ CLI
def test_cli_try_and_status_and_final_refusal(tmp_path, fake_scores, capsys):
    import run
    fake_scores["# cli"] = 61.0
    f = tmp_path / "cand.py"
    f.write_text(cand("cli"))
    d = str(tmp_path / "runs")
    assert run.main(["try", str(f), "-m", "cli idea", "--run-dir", d]) == 0
    assert "accepted" in capsys.readouterr().out
    assert run.main(["status", "--run-dir", d]) == 0
    assert "best dev RMSE 61.00" in capsys.readouterr().out

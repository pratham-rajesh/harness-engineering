"""proposers.py -- the two ways a candidate can be proposed. Neither can touch the evaluator.

  random_propose(lab, iteration, seed)   no LLM. Seeded random search over templated scikit-learn pipelines.
                                         This is what produces the numbers in the README.
  llm_run(lab, iterations, ...)          an LLM agent, built on harness/step_10_subagents.py. It gets read_file,
                                         list_dir and ONE extra tool, run_experiment(hypothesis, source), whose
                                         implementation is Lab.try_candidate. It has no write_file / run_python, so
                                         the only thing it can do to the world is submit a candidate to the lab.
"""
from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "harness"


# =========================================================================== (a) random search, no LLM
def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return float(f"{math.exp(rng.uniform(math.log(lo), math.log(hi))):.4g}")


def _scaled(model_line: str, imports: str) -> str:
    return (f"{imports}\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\n\n"
            f"def build_model():\n    return make_pipeline(StandardScaler(), {model_line})\n")


def _plain(model_line: str, imports: str) -> str:
    return f"{imports}\n\n\ndef build_model():\n    return {model_line}\n"


def _ridge(rng):
    a = _log_uniform(rng, 1e-3, 1e2)
    return f"StandardScaler + Ridge(alpha={a})", _scaled(f"Ridge(alpha={a})", "from sklearn.linear_model import Ridge")


def _lasso(rng):
    a = _log_uniform(rng, 1e-3, 1.0)
    return f"StandardScaler + Lasso(alpha={a})", _scaled(f"Lasso(alpha={a}, max_iter=20000)", "from sklearn.linear_model import Lasso")


def _elastic(rng):
    a, l1 = _log_uniform(rng, 1e-3, 1.0), round(rng.uniform(0.05, 0.95), 2)
    return (f"StandardScaler + ElasticNet(alpha={a}, l1_ratio={l1})",
            _scaled(f"ElasticNet(alpha={a}, l1_ratio={l1}, max_iter=20000)", "from sklearn.linear_model import ElasticNet"))


def _bayes(rng):
    return "StandardScaler + BayesianRidge", _scaled("BayesianRidge()", "from sklearn.linear_model import BayesianRidge")


def _poly(rng):
    a, inter = _log_uniform(rng, 1.0, 1e3), rng.choice([True, False])
    src = ("from sklearn.linear_model import Ridge\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import PolynomialFeatures, StandardScaler\n\n\ndef build_model():\n"
           f"    return make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, interaction_only={inter}, include_bias=False),\n"
           f"                         StandardScaler(), Ridge(alpha={a}))\n")
    return f"degree-2 polynomial features (interaction_only={inter}) + Ridge(alpha={a})", src


def _forest(rng):
    n, depth = rng.choice([100, 200, 300]), rng.choice([3, 4, 6, 8, None])
    leaf, feats = rng.choice([1, 3, 5, 10, 20]), rng.choice([0.3, 0.5, 1.0])
    line = f"RandomForestRegressor(n_estimators={n}, max_depth={depth}, min_samples_leaf={leaf}, max_features={feats}, random_state=0, n_jobs=1)"
    return (f"RandomForest(n={n}, depth={depth}, leaf={leaf}, max_features={feats})",
            _plain(line, "from sklearn.ensemble import RandomForestRegressor"))


def _gbr(rng):
    n, lr, depth = rng.choice([50, 100, 200]), _log_uniform(rng, 0.01, 0.2), rng.choice([1, 2, 3])
    sub = rng.choice([0.5, 0.7, 1.0])
    line = f"GradientBoostingRegressor(n_estimators={n}, learning_rate={lr}, max_depth={depth}, subsample={sub}, random_state=0)"
    return (f"GradientBoosting(n={n}, lr={lr}, depth={depth}, subsample={sub})",
            _plain(line, "from sklearn.ensemble import GradientBoostingRegressor"))


def _hist(rng):
    it, lr, depth, l2 = rng.choice([50, 100, 200]), _log_uniform(rng, 0.01, 0.2), rng.choice([2, 3, None]), _log_uniform(rng, 0.1, 30)
    line = f"HistGradientBoostingRegressor(max_iter={it}, learning_rate={lr}, max_depth={depth}, l2_regularization={l2}, min_samples_leaf=10, random_state=0)"
    return (f"HistGradientBoosting(iter={it}, lr={lr}, depth={depth}, l2={l2})",
            _plain(line, "from sklearn.ensemble import HistGradientBoostingRegressor"))


def _svr(rng):
    c, eps, gamma = _log_uniform(rng, 1, 300), _log_uniform(rng, 1, 30), _log_uniform(rng, 1e-3, 0.1)
    return (f"StandardScaler + SVR(rbf, C={c}, epsilon={eps}, gamma={gamma})",
            _scaled(f"SVR(C={c}, epsilon={eps}, gamma={gamma})", "from sklearn.svm import SVR"))


def _stack(rng):
    a = _log_uniform(rng, 0.1, 30)
    src = ("from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor\n"
           "from sklearn.linear_model import Ridge\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import SVR\n\n\ndef build_model():\n"
           "    base = [('ridge', make_pipeline(StandardScaler(), Ridge(alpha=%s))),\n"
           "            ('svr', make_pipeline(StandardScaler(), SVR(C=50, epsilon=10, gamma=0.01))),\n"
           "            ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.7, random_state=0))]\n"
           "    return StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0), cv=3)\n") % a
    return f"stack Ridge(alpha={a}) + SVR + GradientBoosting with a Ridge meta-model", src


FAMILIES = [_ridge, _ridge, _lasso, _elastic, _bayes, _poly, _forest, _gbr, _hist, _svr, _stack]   # ridge twice: cheap, usually strong


def random_propose(iteration: int, seed: int) -> tuple[str, str]:
    """(hypothesis, source) for attempt number `iteration`. Pure function of (seed, iteration): resumable, reproducible."""
    rng = random.Random(f"{seed}:{iteration}")
    return rng.choice(FAMILIES)(rng)


def random_run(lab, iterations: int, seed: int, log=print) -> None:
    lab.init_baseline()
    while lab.n_attempts() < iterations:
        n = lab.n_attempts() + 1
        hypothesis, source = random_propose(n, seed)
        entry = lab.try_candidate(source, hypothesis, proposer=f"random(seed={seed})")
        log(_fmt(entry, lab))


def _fmt(entry: dict, lab) -> str:
    score = f"{entry['rmse']:.2f}" if entry.get("rmse") is not None else "  -  "
    return f"iter {entry['iter']:>2}  {entry['status']:<10} rmse={score}  best={lab.best_rmse:.2f}  {entry['hypothesis'][:70]}"


# =========================================================================== (b) LLM agent on top of step_10
def load_step10():
    """Import harness/step_10_subagents.py as a module. Fresh copy per call so SANDBOX/DISPATCH edits stay local."""
    sys.path.insert(0, str(HARNESS))                       # step_10 does `import llm`
    spec = importlib.util.spec_from_file_location("step_10_for_ml", HARNESS / "step_10_subagents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPERIMENT_SCHEMA = {"type": "function", "function": {
    "name": "run_experiment",
    "description": ("Submit ONE candidate to the trusted evaluator. `source` is the full text of a Python file defining "
                    "build_model() that returns an unfitted scikit-learn regressor. Returns status, dev RMSE and the "
                    "current best. Allowed imports: sklearn, numpy, scipy."),
    "parameters": {"type": "object", "properties": {"hypothesis": {"type": "string", "description": "one sentence: what you change and why"},
                                                    "source": {"type": "string"}},
                   "required": ["hypothesis", "source"]}}}


def prepare_workspace(lab, program_text: str) -> Path:
    """The agent's read-only view: the rules, the best candidate and the ledger. It cannot write here (no write tool)."""
    ws = lab.dir / "workspace"
    ws.mkdir(exist_ok=True)
    (ws / "program.md").write_text(program_text, encoding="utf-8")
    (ws / "best_candidate.py").write_text(lab.best_source(), encoding="utf-8")
    (ws / "ledger_tail.txt").write_text(lab.ledger_tail(30), encoding="utf-8")
    return ws


def llm_run(lab, iterations: int, max_sessions: int | None = None, log=print) -> None:
    """Drive step_10's agent_loop until `iterations` experiments were run (or sessions run out)."""
    step = load_step10()
    lab.init_baseline()
    program_text = (HERE / "program.md").read_text(encoding="utf-8")
    max_sessions = max_sessions or iterations * 2
    sessions = 0
    while lab.n_attempts() < iterations and sessions < max_sessions:
        sessions += 1
        step.SANDBOX = prepare_workspace(lab, program_text).resolve()
        remaining = iterations - lab.n_attempts()

        def run_experiment(hypothesis: str, source: str) -> str:
            entry = lab.try_candidate(source, hypothesis, proposer="llm")
            log(_fmt(entry, lab))
            score = f"dev RMSE {entry['rmse']:.2f}" if entry.get("rmse") is not None else f"error: {entry.get('error')}"
            return (f"status={entry['status']}; {score}; best so far {lab.best_rmse:.2f} "
                    f"(baseline {lab.baseline_rmse:.2f}); experiments left this run: {iterations - lab.n_attempts()}")

        step.DISPATCH["run_experiment"] = run_experiment
        tools = [t for t in step.TOOLS if t["function"]["name"] in ("read_file", "list_dir")] + [EXPERIMENT_SCHEMA]
        allowed = {"read_file", "list_dir", "run_experiment"}          # write_file / run_python / spawn_subagent are denied
        budget = step.Budget(max_model_calls=8, max_tool_calls=10, max_seconds=600,
                             per_tool={"run_experiment": min(2, remaining)})
        messages = [{"role": "system", "content": program_text},
                    {"role": "user", "content": (
                        f"Current best dev RMSE: {lab.best_rmse:.2f} (baseline {lab.baseline_rmse:.2f}).\n\n"
                        f"Best candidate so far:\n```python\n{lab.best_source()}\n```\n\n"
                        f"Recent attempts:\n{lab.ledger_tail(10)}\n\n"
                        "Propose and run your next experiment now with run_experiment.")}]
        reply = step.agent_loop(messages, budget, tools=tools, allowed=allowed)
        log(f"  [session {sessions}] {budget.summary()} :: {reply[:100]!r}")
        if reply.startswith("(model unavailable"):
            log("  stopping: the model backend is unavailable")
            break

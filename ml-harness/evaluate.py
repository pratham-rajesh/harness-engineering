"""evaluate.py -- the TRUSTED half of the autoresearch loop.

The agent (an LLM, a random-search script, or you in an editor) may only produce a
*candidate*: a Python file defining `build_model()` that returns an unfitted
scikit-learn regressor. Everything that decides whether the candidate is any good
lives here and is never editable by the agent:

  * the data and the dev/test split          (fixed seed; test rows stay locked)
  * the metric and the CV protocol            (RMSE, repeated 5-fold on the dev set)
  * the static tripwire on candidate source   (no file/IO/eval/dataset access)
  * process isolation + wall-clock timeout    (candidate code runs in a child process)

Task: scikit-learn's bundled diabetes dataset (442 rows, regress disease progression).
Lower RMSE is better. A mean-predictor scores ~77; good linear models ~54.

CLI (what a coding assistant can call directly):
    python ml-harness/evaluate.py check  candidate.py     # static check only
    python ml-harness/evaluate.py score  candidate.py     # repeated-CV RMSE on the dev set
The test set is only reachable through lab.Lab.final(), once per run.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT_SEED = 42
TEST_FRACTION = 0.2
CV_SPLITS, CV_REPEATS, CV_SEED = 5, 3, 0
DEFAULT_TIMEOUT_S = 120
MAX_SOURCE_CHARS = 20_000

ALLOWED_IMPORT_ROOTS = {"sklearn", "numpy", "scipy", "math", "typing", "functools", "itertools", "warnings", "collections"}
BANNED_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview", "print"}
# attribute names that read files, reach the dataset, or spawn processes
BANNED_ATTRS = {"load", "loadtxt", "genfromtxt", "fromfile", "memmap", "save", "savez", "savetxt", "tofile",
                "datasets", "io", "system", "popen", "load_diabetes", "fetch_openml", "ctypeslib", "f2py"}


# --------------------------------------------------------------------------- static tripwire
def static_check(source: str) -> list[str]:
    """Return a list of violations (empty = acceptable). A tripwire, not a security sandbox."""
    problems: list[str] = []
    if len(source) > MAX_SOURCE_CHARS:
        return [f"source too long ({len(source)} > {MAX_SOURCE_CHARS} chars)"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    has_build = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                problems += _check_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append("relative imports are not allowed")
            problems += _check_module(node.module or "")
            for alias in node.names:
                if alias.name in BANNED_ATTRS:
                    problems.append(f"importing '{alias.name}' is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            problems.append(f"call to '{node.func.id}' is not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BANNED_ATTRS:
                problems.append(f"attribute '.{node.attr}' is not allowed (line {node.lineno})")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            problems.append(f"name '{node.id}' is not allowed (line {node.lineno})")
        elif isinstance(node, ast.FunctionDef) and node.name == "build_model" and node.col_offset == 0:
            has_build = True
    if not has_build:
        problems.append("candidate must define a top-level function build_model()")
    return sorted(set(problems))


def _check_module(name: str) -> list[str]:
    root = name.split(".")[0]
    out = []
    if root not in ALLOWED_IMPORT_ROOTS:
        out.append(f"import of '{name}' is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORT_ROOTS))})")
    if "datasets" in name.split(".") or name.startswith("scipy.io"):
        out.append(f"import of '{name}' is not allowed")
    return out


# --------------------------------------------------------------------------- data and protocol
def load_split():
    """Dev rows (visible to CV) and test rows (locked until final)."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=TEST_FRACTION, random_state=SPLIT_SEED)  # Xdev, Xtest, ydev, ytest


def _load_candidate_model(path: str):
    source = Path(path).read_text(encoding="utf-8")
    problems = static_check(source)                                 # defence in depth: re-check in the child
    if problems:
        raise ValueError("static check failed: " + "; ".join(problems))
    namespace: dict = {"__name__": "candidate"}
    exec(compile(source, path, "exec"), namespace)                  # noqa: S102 -- runs in an isolated child process
    model = namespace["build_model"]()
    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        raise TypeError("build_model() must return an estimator with fit() and predict()")
    return model


def _child_score(path: str) -> dict:
    import warnings
    import numpy as np
    from sklearn.model_selection import RepeatedKFold, cross_val_score
    warnings.filterwarnings("ignore")
    Xdev, _Xtest, ydev, _ytest = load_split()                       # test rows are never touched here
    model = _load_candidate_model(path)
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=CV_SEED)
    t0 = time.perf_counter()
    scores = -cross_val_score(model, Xdev, ydev, cv=cv, scoring="neg_root_mean_squared_error", error_score="raise")
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite CV score")
    return {"rmse": float(scores.mean()), "rmse_std": float(scores.std()), "n_folds": int(len(scores)),
            "fit_seconds": round(time.perf_counter() - t0, 2)}


def _child_final(best_path: str, baseline_path: str) -> dict:
    import warnings
    import numpy as np
    warnings.filterwarnings("ignore")
    Xdev, Xtest, ydev, ytest = load_split()
    out = {}
    for label, path in (("best", best_path), ("baseline", baseline_path)):
        model = _load_candidate_model(path)
        model.fit(Xdev, ydev)
        pred = model.predict(Xtest)
        out[f"{label}_test_rmse"] = float(np.sqrt(np.mean((pred - ytest) ** 2)))
    out["n_test_rows"] = int(len(ytest))
    return out


# --------------------------------------------------------------------------- parent side
def _run_child(args: list[str], timeout: int) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONHASHSEED": "0"})
    with tempfile.TemporaryDirectory() as scratch:                  # cwd is an empty temp dir
        try:
            done = subprocess.run([sys.executable, "-I", str(HERE / "evaluate.py"), *args], cwd=scratch, env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"exceeded {timeout}s"}
    lines = [ln for ln in done.stdout.strip().splitlines() if ln.startswith("{")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr.strip().splitlines() or ["no output"])[-1]
        return {"status": "crash", "error": tail[:300]}
    return {"status": "ok", **json.loads(lines[-1])}


def score(path: str | Path, timeout: int = DEFAULT_TIMEOUT_S) -> dict:
    """Static check + isolated repeated-CV. Returns {"status": ok|invalid|crash|timeout, ...}."""
    problems = static_check(Path(path).read_text(encoding="utf-8"))
    if problems:
        return {"status": "invalid", "error": "; ".join(problems)[:500]}
    return _run_child(["--child-score", str(Path(path).resolve())], timeout)


def final_test(best_path: str | Path, baseline_path: str | Path, timeout: int = DEFAULT_TIMEOUT_S) -> dict:
    return _run_child(["--child-final", str(Path(best_path).resolve()), str(Path(baseline_path).resolve())], timeout)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--child-score"]:
        print(json.dumps(_child_score(argv[1])))
    elif argv[:1] == ["--child-final"]:
        print(json.dumps(_child_final(argv[1], argv[2])))
    elif argv[:1] == ["check"] and len(argv) == 2:
        problems = static_check(Path(argv[1]).read_text(encoding="utf-8"))
        print("OK" if not problems else "VIOLATIONS:\n  " + "\n  ".join(problems))
        sys.exit(1 if problems else 0)
    elif argv[:1] == ["score"] and len(argv) == 2:
        print(json.dumps(score(argv[1]), indent=2))
    else:
        sys.exit(__doc__)

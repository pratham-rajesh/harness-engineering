# Part C — `ml-harness/`: an autoresearch loop on top of the harness

The idea (Karpathy's *autoresearch*): let an agent run experiments in a loop, but split the world in two.
The **agent proposes**; a **trusted evaluator decides**. The agent can never edit the thing that grades it.

## Files

| file | role | who may change it |
|---|---|---|
| `evaluate.py` | data, dev/test split, metric, CV protocol, static tripwire, process isolation | trusted (not the agent) |
| `lab.py` | ledger, accept/reject policy, final-once rule, `RESULTS.md` + `progress.svg` | trusted |
| `proposers.py` | (a) seeded random search, no LLM; (b) LLM agent built on `harness/step_10_subagents.py` | the proposer layer |
| `run.py` | CLI: `run`, `try`, `status`, `final`, `report` | — |
| `program.md` | the agent's rules, in plain language | human |
| `baseline_candidate.py` | attempt 0: predict the training mean | human |
| `runs/random-seed0/` | **a real run**: ledger, every candidate, `RESULTS.md`, `progress.svg`, `final.json` | output |
| `../.claude/skills/autoresearch/SKILL.md` | lets Claude Code / Codex drive the same loop by hand | human |

## The task
scikit-learn's bundled diabetes dataset (442 rows, 10 features), regress disease progression. Metric: **RMSE, lower is better**.
Rows are split once, with a fixed seed, into 353 dev rows and 89 test rows. Candidates are scored by repeated 5-fold CV
(3 repeats = 15 fits) **on the dev rows only**. The test rows are reachable only through `Lab.final()`.

A candidate is a Python file defining `build_model()` that returns an unfitted scikit-learn estimator.

## How the loop is kept honest

1. **The agent cannot grade itself.** `evaluate.py`/`lab.py` are outside its tool set. The LLM proposer's only
   world-changing tool is `run_experiment(hypothesis, source)`; `write_file`, `run_python` and `spawn_subagent` are
   removed from its schema *and* denied in `run_tool` (`allowed=`, the step 10 mechanism).
2. **Static tripwire** (`static_check`): AST check rejecting `open/eval/exec/__import__`, dunder attributes, `np.load`,
   `sklearn.datasets`, `os/subprocess`, and anything outside `sklearn/numpy/scipy` + a few stdlib modules.
   This is a tripwire for honest mistakes and lazy cheating, **not** a security sandbox: a determined adversary
   with arbitrary code execution is out of scope.
3. **Process isolation**: each candidate is fitted in a child `python -I` process in an empty temp directory,
   with `*KEY*/*TOKEN*/*SECRET*` env vars stripped and a wall-clock timeout.
4. **Accept margin**: a candidate becomes the new best only if it beats the best by >= 0.5% — otherwise CV noise would ratchet.
5. **"Too good to be true" rule**: dev RMSE below half the baseline is recorded as `suspicious` and rejected (a leak, not progress).
6. **Duplicates** are detected by source hash and not re-scored.
7. **One look at the test set**: `final()` writes `final.json`; a second call raises `FinalAlreadyRun`.
   A crashed final does not write the file, so it does not burn the single look.
8. **Everything is logged** to an append-only `ledger.jsonl`, including failures, and a run resumes from it.

## Results that really happened

Run: `python ml-harness/run.py run --proposer random --iterations 15 --seed 0`, then `final`, then `report`.
No LLM was involved; the proposer is a seeded random search over 11 templated scikit-learn families
(output committed in `runs/random-seed0/`).

| | dev RMSE (CV, 353 rows) | test RMSE (89 rows, looked at once) |
|---|---:|---:|
| mean-predictor baseline | 78.06 | 73.22 |
| best candidate (attempt 4: stack of Ridge + SVR + GradientBoosting, Ridge meta-model) | 55.00 | 52.06 |

Honest reading:
- Going from "predict the mean" to "any sensible regularised linear model" is the whole story: ElasticNet (attempt 1,
  55.47) already captured almost all of the gain.
- Attempt 4 improved on attempt 1 by 0.47 RMSE, **less than the fold-to-fold std (about 3)**. Ridge, Lasso and BayesianRidge
  (attempts 5, 13, 14, 15: 54.87-54.93) are statistically tied with the "winner"; they were rejected only because of the
  0.5% margin rule and ordering. Do not read "stacking is best" out of this run.
- Tree ensembles and SVR did worse than linear models here, as is typical for this small, near-linear dataset.
- The 29.5% dev improvement is the minimum over 15 tries on shared folds, hence optimistic; the test column is the unbiased one.
  The test baseline (73.22) differs from the dev baseline (78.06) simply because the two splits have different spread.
- One run, one seed. These numbers carry no claim about generality.

## LLM proposer — status: NOT run against a real model
`python ml-harness/run.py run --proposer llm --iterations 15` drives `step_10`'s `agent_loop` once per session
(fresh messages: `program.md` as the stable system prompt, then current best source and ledger tail), at most two
experiments per session, with a small `Budget`. It is covered by offline tests that use a scripted fake model
(`tests/test_ml_harness.py`): it submits through `run_experiment`, cannot call forbidden tools, respects the
per-session experiment quota, and stops when the model never experiments. **It has not been run with a live model**,
because no API key was available while building this. To run it yourself: put a free OpenRouter key in `.env`
(see `.env.example`) or start Ollama, run the command above, then `final` and `report` on `runs/llm/`.
Do not compare it with the random run until it has actually happened.

## Driving it by hand (Claude Code / Codex)
`/.claude/skills/autoresearch/SKILL.md` documents the manual loop: `run.py status`, write a candidate, `evaluate.py check`,
`run.py try -m "hypothesis"`, repeat, then `run.py final` once and `run.py report`.

## Reproduce
```
.venv/Scripts/python ml-harness/evaluate.py score ml-harness/baseline_candidate.py    # 78.06
.venv/Scripts/python ml-harness/run.py run --proposer random --iterations 15 --seed 0 --run-dir /tmp/rerun
.venv/Scripts/python -m pytest tests/test_ml_harness.py -q
```
Scores are deterministic for a fixed library version (fixed split and CV seeds; `random_state=0` in the models), but
may differ slightly across scikit-learn versions.

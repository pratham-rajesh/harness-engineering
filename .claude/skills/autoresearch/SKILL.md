---
name: autoresearch
description: Run the autoresearch loop on the diabetes regression task in ml-harness/. Use when asked to improve the model score, run experiments, or "do autoresearch". You propose candidates; the trusted evaluator scores them.
---

# Autoresearch (manual driver)

You are the proposer. `ml-harness/evaluate.py` and `ml-harness/lab.py` are the trusted referee: **do not edit them,
do not read the dataset yourself, and never look at the test split.** Read `ml-harness/program.md` for the full rules.

## Loop
1. `python ml-harness/run.py status --run-dir ml-harness/runs/manual` to see the ledger (creates the baseline on first use).
2. Read `ml-harness/runs/manual/best_candidate.py`. Form ONE hypothesis and write a candidate file in a scratch location
   outside `ml-harness/` (a Python file defining `build_model()` that returns an unfitted scikit-learn regressor).
3. `python ml-harness/evaluate.py check <file>` (fast static check), then
   `python ml-harness/run.py try <file> -m "one-sentence hypothesis" --run-dir ml-harness/runs/manual`.
4. Read the status: `accepted` becomes the new best; `rejected`/`invalid`/`crash`/`timeout`/`duplicate` are logged
   and teach you something; `suspicious` means "too good to be true", treat it as a leak and fix your approach.
5. Repeat 10-20 times, one change at a time. Then run `final` ONCE (`python ml-harness/run.py final --run-dir ...`)
   and `report`. A second `final` is refused by design.

## Honesty rules
- Quote only numbers printed by these commands. The dev RMSE is optimistic after many tries; the final test RMSE is the one to report.
- If a run did not happen, say so; never fill in expected results.

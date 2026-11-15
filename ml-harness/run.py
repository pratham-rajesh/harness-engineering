"""run.py -- command line for the autoresearch loop.

Automatic proposers:
    python ml-harness/run.py run --proposer random --iterations 15 --seed 0      # no LLM, real numbers
    python ml-harness/run.py run --proposer llm    --iterations 15               # needs OPENROUTER_API_KEY or Ollama

Manual (a coding assistant such as Claude Code / Codex drives these; see .claude/skills/autoresearch/SKILL.md):
    python ml-harness/run.py try candidate.py -m "why this should help" --run-dir ml-harness/runs/manual
    python ml-harness/run.py status --run-dir ...
    python ml-harness/run.py final  --run-dir ...      # the test set; works ONCE per run dir
    python ml-harness/run.py report --run-dir ...      # RESULTS.md + progress.svg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lab import FinalAlreadyRun, Lab  # noqa: E402
import proposers  # noqa: E402


def default_dir(args) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.cmd == "run":
        return HERE / "runs" / (f"random-seed{args.seed}" if args.proposer == "random" else "llm")
    return HERE / "runs" / "manual"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an automatic proposer")
    r.add_argument("--proposer", choices=["random", "llm"], required=True)
    r.add_argument("--iterations", type=int, default=15, help="total attempts in the ledger, baseline excluded")
    r.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("try", help="evaluate one candidate file")
    t.add_argument("file")
    t.add_argument("-m", "--hypothesis", required=True)
    t.add_argument("--proposer-name", default="manual")
    sub.add_parser("status", help="show the ledger")
    sub.add_parser("final", help="evaluate the best candidate on the held-out test set (once)")
    sub.add_parser("report", help="write RESULTS.md and progress.svg")
    for s in (r, t, *[sub.choices[c] for c in ("status", "final", "report")]):
        s.add_argument("--run-dir", help="default: ml-harness/runs/<proposer or manual>")
    args = p.parse_args(argv)
    if args.cmd != "run":
        args.seed = 0
    lab = Lab(default_dir(args))

    if args.cmd == "run":
        if args.proposer == "random":
            proposers.random_run(lab, args.iterations, args.seed)
        else:
            proposers.llm_run(lab, args.iterations)
        print(f"best dev RMSE {lab.best_rmse:.2f} (baseline {lab.baseline_rmse:.2f}); report: {lab.report()}")
    elif args.cmd == "try":
        entry = lab.try_candidate(Path(args.file).read_text(encoding="utf-8"), args.hypothesis, args.proposer_name)
        print(proposers._fmt(entry, lab) + (f"\n  error: {entry['error']}" if entry.get("error") else ""))
    elif args.cmd == "status":
        lab.init_baseline()
        print(lab.ledger_tail(len(lab.entries)))
        print(f"best dev RMSE {lab.best_rmse:.2f}; baseline {lab.baseline_rmse:.2f}; "
              f"final test evaluated: {lab.final_path.exists()}")
    elif args.cmd == "final":
        try:
            print(lab.final())
        except FinalAlreadyRun as exc:
            print(f"REFUSED: {exc}")
            return 2
    elif args.cmd == "report":
        print(lab.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())

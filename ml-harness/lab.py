"""lab.py -- the experiment ledger and the accept/reject policy of the autoresearch loop.

One `Lab` owns one run directory:

    <run_dir>/ledger.jsonl        one JSON line per attempt (append-only; the source of truth)
    <run_dir>/baseline.py         the starting candidate (attempt 0)
    <run_dir>/best_candidate.py   the best accepted candidate so far
    <run_dir>/attempts/iter_NN.py every candidate that was ever tried, accepted or not
    <run_dir>/final.json          written ONCE by final(); its existence blocks a second test-set look
    <run_dir>/RESULTS.md, progress.svg   written by report()

Policy for a candidate (decided here, never by the proposer):

    invalid     failed the static tripwire
    crash       raised while fitting / scoring
    timeout     exceeded the wall-clock limit
    duplicate   identical source was already scored (not re-run)
    suspicious  RMSE < SUSPICIOUS_FRACTION x baseline -> "too good to be true", treated as a possible
                leak and REJECTED so that it can never become the best
    accepted    dev RMSE < best * (1 - min_rel_improvement)   (a margin, so CV noise does not ratchet)
    rejected    valid but not better by the margin

Resumable: constructing a Lab on an existing directory re-reads the ledger and carries on.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402

SUSPICIOUS_FRACTION = 0.5
MAX_SAVED_SOURCE_CHARS = 50_000
BASELINE_SOURCE = HERE / "baseline_candidate.py"


def source_hash(source: str) -> str:
    return hashlib.sha256(source.strip().encode("utf-8")).hexdigest()[:16]


class FinalAlreadyRun(RuntimeError):
    """The held-out test set may be evaluated once per run directory."""


class Lab:
    def __init__(self, run_dir, min_rel_improvement: float = 0.005, timeout: int = evaluate.DEFAULT_TIMEOUT_S):
        self.dir = Path(run_dir).resolve()
        self.min_rel = min_rel_improvement
        self.timeout = timeout
        (self.dir / "attempts").mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.dir / "ledger.jsonl"
        self.entries: list[dict] = []
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))

    # ------------------------------------------------------------------ state
    @property
    def baseline_rmse(self):
        return self.entries[0]["rmse"] if self.entries else None

    @property
    def best_rmse(self):
        accepted = [e["rmse"] for e in self.entries if e["status"] == "accepted"]
        return min(accepted) if accepted else None

    @property
    def best_path(self) -> Path:
        return self.dir / "best_candidate.py"

    def best_source(self) -> str:
        return self.best_path.read_text(encoding="utf-8")

    def n_attempts(self) -> int:
        """Attempts made by proposers (the baseline is not an attempt)."""
        return max(0, len(self.entries) - 1)

    def seen(self, source: str) -> bool:
        h = source_hash(source)
        return any(e.get("source_sha") == h for e in self.entries)

    def ledger_tail(self, n: int = 8) -> str:
        rows = self.entries[-n:]
        lines = []
        for e in rows:
            score = f"rmse={e['rmse']:.2f}" if e.get("rmse") is not None else (e.get("error") or "")[:80]
            lines.append(f"iter {e['iter']:>2} [{e['status']}] {score} :: {e['hypothesis'][:110]}")
        return "\n".join(lines) or "(no attempts yet)"

    # ------------------------------------------------------------------ writing
    def _append(self, entry: dict) -> dict:
        self.entries.append(entry)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def _save_source(self, iteration: int, source: str) -> str:
        rel = f"attempts/iter_{iteration:02d}.py"
        (self.dir / rel).write_text(source[:MAX_SAVED_SOURCE_CHARS], encoding="utf-8")
        return rel

    def init_baseline(self) -> dict:
        """Score the starting candidate once; it is attempt 0 and the first 'best'."""
        if self.entries:
            return self.entries[0]
        source = BASELINE_SOURCE.read_text(encoding="utf-8")
        (self.dir / "baseline.py").write_text(source, encoding="utf-8")
        result = evaluate.score(self.dir / "baseline.py", self.timeout)
        if result["status"] != "ok":
            raise RuntimeError(f"baseline failed to score: {result}")
        self.best_path.write_text(source, encoding="utf-8")
        return self._append({
            "iter": 0, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "proposer": "baseline",
            "hypothesis": "mean predictor (no features)", "status": "accepted",
            "rmse": result["rmse"], "rmse_std": result["rmse_std"], "fit_seconds": result["fit_seconds"],
            "error": None, "file": self._save_source(0, source), "source_sha": source_hash(source)})

    def try_candidate(self, source: str, hypothesis: str, proposer: str = "manual") -> dict:
        """Evaluate one candidate and apply the accept/reject policy. Always appends exactly one ledger row."""
        if not self.entries:
            self.init_baseline()
        iteration = len(self.entries)
        entry = {"iter": iteration, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "proposer": proposer,
                 "hypothesis": (hypothesis or "(none given)").strip()[:300], "status": None, "rmse": None,
                 "rmse_std": None, "fit_seconds": None, "error": None, "source_sha": source_hash(source),
                 "file": self._save_source(iteration, source)}
        if self.seen(source):
            entry.update(status="duplicate", error="identical source was already scored")
            return self._append(entry)
        result = evaluate.score(self.dir / entry["file"], self.timeout)
        if result["status"] != "ok":
            entry.update(status=result["status"], error=result.get("error"))
            return self._append(entry)
        entry.update(rmse=result["rmse"], rmse_std=result["rmse_std"], fit_seconds=result["fit_seconds"])
        if entry["rmse"] < SUSPICIOUS_FRACTION * self.baseline_rmse:
            entry.update(status="suspicious", error="RMSE under half the baseline's: possible leak, rejected")
        elif entry["rmse"] < self.best_rmse * (1 - self.min_rel):
            entry["status"] = "accepted"
            self.best_path.write_text(source, encoding="utf-8")
        else:
            entry["status"] = "rejected"
        return self._append(entry)

    # ------------------------------------------------------------------ final test (once)
    @property
    def final_path(self) -> Path:
        return self.dir / "final.json"

    def final(self) -> dict:
        """Look at the held-out test set. Allowed once per run directory."""
        if self.final_path.exists():
            raise FinalAlreadyRun(f"{self.final_path} exists: the test set was already evaluated for this run")
        if not self.entries:
            raise RuntimeError("nothing to evaluate: run init_baseline()/try_candidate() first")
        result = evaluate.final_test(self.best_path, self.dir / "baseline.py", self.timeout)
        if result["status"] != "ok":
            return result                                   # nothing written, so a crash does not burn the one look
        result["dev_rmse_best"] = self.best_rmse
        result["dev_rmse_baseline"] = self.baseline_rmse
        result["best_iter"] = max(e["iter"] for e in self.entries if e["status"] == "accepted")
        result["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def final_result(self):
        return json.loads(self.final_path.read_text(encoding="utf-8")) if self.final_path.exists() else None

    # ------------------------------------------------------------------ report
    def report(self) -> Path:
        if not self.entries:
            raise RuntimeError("empty ledger")
        counts: dict = {}
        for e in self.entries[1:]:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        base, best = self.baseline_rmse, self.best_rmse
        proposers = sorted({e["proposer"] for e in self.entries[1:]}) or ["(none)"]
        lines = [f"# Results: `{self.dir.name}`", "",
                 f"- Task: diabetes regression, RMSE (lower is better). Dev = 80% of rows, test = 20% (locked until `final`).",
                 f"- Proposer(s): {', '.join(proposers)}; attempts: {self.n_attempts()} "
                 f"({', '.join(f'{v} {k}' for k, v in sorted(counts.items())) or 'none'})",
                 f"- Baseline dev RMSE (repeated 5-fold x3 CV): **{base:.2f}**",
                 f"- Best dev RMSE: **{best:.2f}** ({100 * (base - best) / base:.1f}% below the baseline)"]
        final = self.final_result()
        if final:
            lines += [f"- Held-out TEST RMSE (looked at once): baseline **{final['baseline_test_rmse']:.2f}**, "
                      f"best candidate **{final['best_test_rmse']:.2f}** on {final['n_test_rows']} rows"]
        else:
            lines += ["- Held-out test RMSE: not evaluated yet (`python ml-harness/run.py final --run-dir ...`)"]
        lines += ["", "![progress](progress.svg)", "",
                  "| iter | status | dev RMSE | std | hypothesis |", "|---:|---|---:|---:|---|"]
        for e in self.entries:
            rmse = f"{e['rmse']:.2f}" if e.get("rmse") is not None else "-"
            std = f"{e['rmse_std']:.2f}" if e.get("rmse_std") is not None else "-"
            hyp = escape(e["hypothesis"].replace("|", "/").replace("\n", " "))
            if e.get("error") and e["status"] not in ("accepted", "rejected"):
                hyp += f" _(error: {escape(e['error'].replace('|', '/'))[:120]})_"
            lines.append(f"| {e['iter']} | {e['status']} | {rmse} | {std} | {hyp} |")
        lines += ["", "> The best dev RMSE is the minimum over many tries on the same CV folds, so it is optimistic. "
                      "The held-out test number is the honest one."]
        out = self.dir / "RESULTS.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.dir / "progress.svg").write_text(self._svg(), encoding="utf-8")
        return out

    def _svg(self) -> str:
        w, h, left, right, top, bottom = 720, 360, 56, 16, 24, 40
        pts = [(e["iter"], e["rmse"], e["status"]) for e in self.entries if e.get("rmse") is not None]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_max = max(max(xs), 1)
        y_lo, y_hi = min(ys) - 2, max(ys) + 2

        def px(x):
            return left + (w - left - right) * x / x_max

        def py(y):
            return top + (h - top - bottom) * (y_hi - y) / (y_hi - y_lo)

        best, step = float("inf"), []
        for it, rmse, status in pts:
            if status == "accepted" and rmse < best:
                best = rmse
                step.append((it, best))
        step.append((x_max, best))
        path, prev = [], None
        for it, val in step:
            path.append(f"{px(it):.1f},{py(prev if prev is not None else val):.1f}")
            path.append(f"{px(it):.1f},{py(val):.1f}")
            prev = val
        colour = {"accepted": "#1a7f37", "rejected": "#8c959f", "suspicious": "#cf222e"}
        dots = "".join(f'<circle cx="{px(i):.1f}" cy="{py(r):.1f}" r="4" fill="{colour.get(s, "#8c959f")}"><title>iter {i}: {r:.2f} ({s})</title></circle>'
                       for i, r, s in pts)
        ticks = "".join(f'<text x="{left - 6}" y="{py(v) + 4:.1f}" text-anchor="end" font-size="11">{v:.0f}</text>'
                        f'<line x1="{left}" x2="{w - right}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="#d0d7de" stroke-width="0.5"/>'
                        for v in range(int(y_lo) + 1, int(y_hi) + 1, max(1, int((y_hi - y_lo) / 6))))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">'
                f'<rect width="{w}" height="{h}" fill="white"/>{ticks}'
                f'<polyline points="{" ".join(path)}" fill="none" stroke="#1a7f37" stroke-width="2"/>{dots}'
                f'<text x="{w / 2}" y="{h - 8}" text-anchor="middle" font-size="12">attempt number (0 = baseline)</text>'
                f'<text x="14" y="{h / 2}" font-size="12" transform="rotate(-90 14 {h / 2})" text-anchor="middle">dev RMSE (lower is better)</text>'
                f'<text x="{w - right}" y="14" text-anchor="end" font-size="11">green = accepted, grey = rejected, red = suspicious; line = best so far</text>'
                f'</svg>')


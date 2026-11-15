# Recording guide — three videos

The videos have to be recorded by you; nothing here has been recorded. Each section lists what to show, what to say, and the exact commands.
Put the three YouTube links into the `TODO` placeholders in `README.md`. Aim for 6–10 minutes each. Unlisted is fine unless the assignment says public.

**Before any recording**: `git status` clean enough to show, `.env` present with a **free** OpenRouter key (never show it on screen — blur or use a
separate terminal profile), `.venv` activated, font size large, terminal at ~100 columns.

Be honest on camera about the same things the README is honest about: Part A and the LLM proposer in Part C have only been tested with a scripted
fake model until *you* run them live; say so if you have not.

---

## Video 1 — `harness/` (Part A, ~8–10 min)

**Smoke test first (off camera):** `python harness/llm.py` must print a short reply. If a free model rate-limits, set `LLM_FALLBACK_MODELS` (see `.env.example`).

| time | show | say |
|---|---|---|
| 0:00 | `harness/NOTES.md` table | "A model is stateless; everything else is the harness. Ten steps, each adds one idea." |
| 0:45 | `harness/llm.py` (retries, fallbacks, `.env`) | "The only file that does HTTP, so every step file contains only its own lesson." |
| 1:30 | `step_01_loop.py` → run it; tell it your name, then ask for it | "Memory is me re-sending the history." |
| 2:30 | `step_02_tools.py` → "read note.txt" | "The model asks, the harness executes. Note this read_file is deliberately unsafe." |
| 3:15 | `step_03_errors.py` → ask for a missing file | "Errors are strings the model can read, not crashes." |
| 4:00 | `step_04_sandbox.py` → "read ../../.env" → `DENIED` | "Resolved paths, deny-list, scrubbed env: the child never sees the API key." |
| 5:00 | `step_05_approval.py` → write a file, answer `n` then `y` | "Approval is bound to a hash of the exact call and re-checked at execution time." |
| 5:45 | `step_06_quotas.py` → ask for 10 writes | "Quota errors are strings too; the harness, not the model, decides enough is enough." |
| 6:30 | `step_07_caching.py` → `[cache]` line | "Byte-stable prefix; volatile text goes last; the tracker shouts if it changes." |
| 7:15 | `step_08_skills.py` → "review note.txt" → `load_skill` | "The prompt carries an index; bodies load on demand." |
| 8:00 | `step_09_compaction.py` (lower `COMPACT_AT_CHARS` for the demo, revert after) | "Summarise the old part; the kept tail must start on a user message so no tool result is orphaned." |
| 8:45 | `step_10_subagents.py` → "summarise every file in the sandbox" | "Fresh context, read-only tools, own budget; only the answer comes back." |
| 9:30 | `python -m pytest -q` | "95 offline tests with a scripted model; that is how I know the guards work without paying for tokens." |

Commands: `.venv/Scripts/python harness/step_NN_*.py` · `HARNESS_AUTO_APPROVE=1` only to skip prompts in step 10 if you must (say that out loud).

## Video 2 — `dsh-plugins/` (Part B, ~8–10 min)

**Off camera first:** follow "Run it yourself" in `dsh-plugins/NOTES.md`. None of that has been verified by the author; if DSH rejects the patch, fix it
(path with spaces / `file:///` URL / missing `--profile`) and update `NOTES.md` with what actually worked.

| time | show | say |
|---|---|---|
| 0:00 | DSH web UI, Plugins page | "DSH is a Cordis harness; everything is a plugin: `name`, `inject`, `apply(ctx)`." |
| 1:00 | `second-brain/src/index.js` then `vault.js` | "Thin wiring, pure logic. Raw tool definitions so nothing imports DSH and tests run under `node --test`." |
| 2:30 | `npm test --prefix dsh-plugins` | "35 tests, including a title like `../../x` that cannot escape the vault." |
| 3:00 | live: ask it to save a note linking `[[Project Falcon]]`, then read `Project Falcon` | show `brain_save`, then the backlink in `brain_read`. |
| 4:30 | new session: ask a question only the vault can answer | `brain_search`; "memory that outlives the context window". |
| 5:30 | `tool-ledger/src/index.js` | "pre-execute returns a deny decision, never throws; reserve before `await next()` because calls run in parallel; refund if someone else denies." |
| 7:00 | live: trigger the quota (limit 10 in the generated patch) → `QUOTA:` message; call `tool_ledger`; `cat .tool-ledger/audit.jsonl` | "Same idea as Part A's step 6, as a plugin; secrets are redacted in the audit." |
| 8:30 | `installed-plugins.txt` + Creator mode | demo at least two community plugins you actually installed; show Creator mode with a prompt from `NOTES.md` and say whether the result matched the hand-written plugin. |

## Video 3 — `ml-harness/` (Part C, ~8–10 min)

| time | show | say |
|---|---|---|
| 0:00 | `ml-harness/NOTES.md` | "The agent proposes; a trusted evaluator it cannot edit decides." |
| 1:00 | `evaluate.py`: split, CV, `static_check`, `_run_child` | "Data, metric and test set live here. Tripwire plus child process plus timeout. A tripwire, not a sandbox." |
| 2:30 | `lab.py`: statuses, margin, `suspicious`, `final()` | "0.5% margin so noise doesn't ratchet; results under half the baseline are treated as leaks; the test set can be looked at once." |
| 3:30 | `python ml-harness/evaluate.py check` on a file containing `open(...)` → violation | show a rejected cheat. |
| 4:15 | `python ml-harness/run.py run --proposer random --iterations 8 --seed 1 --run-dir <scratch dir>` | live loop; "no LLM, seeded random search over sklearn pipelines". |
| 5:30 | `ml-harness/runs/random-seed0/RESULTS.md` + `progress.svg` | 78.06 → 55.00 dev, 73.22 → 52.06 test; "the winner beat ElasticNet by less than one fold std; don't read 'stacking is best' out of it". |
| 6:45 | `python ml-harness/run.py run --proposer llm --iterations 6 --run-dir ml-harness/runs/llm` **(needs your key)** | "Same loop, now step_10's `agent_loop` proposes; its only tool that changes anything is `run_experiment`." Then `final` and `report`. If it fails, show the failure and say so. |
| 8:30 | `.claude/skills/autoresearch/SKILL.md` | optional: drive the loop from Claude Code with `/autoresearch`; show `run.py try`. |
| 9:15 | `final` twice → second is `REFUSED` | "One look at the test set, enforced in code." |

After recording the LLM run, add its real numbers to `ml-harness/NOTES.md` and the README table; do not leave the `TODO`.

---

## Before you submit
- [ ] three links replaced in `README.md`
- [ ] `.venv/Scripts/python -m pytest -q` and `npm test --prefix dsh-plugins` green
- [ ] `git status`: no `.env`, no `.venv`, no `.second-brain/`, no `dsh-plugins/cordis.local.yml`
- [ ] `HANDOFF.md` deleted
- [ ] README table updated if you ran the LLM proposer or DSH live

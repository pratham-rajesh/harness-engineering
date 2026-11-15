# Part A — `harness/`: building an agent harness from scratch

A model is a stateless function from messages to a message. Everything else — memory, tools, safety, cost control —
is the **harness**. This folder builds one in ten small, cumulative steps. Each `step_NN_*.py` runs on its own; each
adds one idea to the previous one, and `# ---- NEW:` comments mark what changed. `llm.py` is the only file that does HTTP.

## Status — read this first
- **Tested offline, never run against a live model.** There was no API key while building this. The 95 tests in
  `tests/test_harness_steps.py` use a scripted fake model (`tests/conftest.py`) so every behaviour below is asserted
  without a network. Run `python harness/llm.py` with a key to verify the transport yourself.
- One test is skipped on Windows accounts that cannot create symlinks (the symlink-escape check).

## Running it
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt     # Windows; use .venv/bin on Linux/macOS
cp .env.example .env     # then put a free OpenRouter key in OPENROUTER_API_KEY   (or point LLM_BASE_URL at Ollama)
.venv/Scripts/python harness/llm.py                  # smoke test: one short model call
.venv/Scripts/python harness/step_10_subagents.py    # the capstone, interactive
.venv/Scripts/python -m pytest -q                    # offline, no key needed
```
Backends: OpenRouter with `:free` models (default `openrouter/free`), or a local OpenAI-compatible server such as
Ollama (`LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=<your model>`). `llm.py` retries 429/5xx and empty replies
with backoff, then rotates through `LLM_FALLBACK_MODELS`.

## The ladder

| step | idea added | what to demo |
|---|---|---|
| 01 `step_01_loop.py` | the conversation loop; "memory" is re-sending history; a failed model call must not poison history | tell it your name, then ask for it |
| 02 `step_02_tools.py` | tool calls: the model *asks*, the harness *executes*; max-turn cap. Deliberately unsafe `read_file` | "read note.txt" |
| 03 `step_03_errors.py` | errors become strings the model can read and fix; tool output truncated | ask for a file that does not exist |
| 04 `step_04_sandbox.py` | `write_file`/`run_python` inside a fence: resolved paths, deny-list, timeout, scrubbed env (no API key in the child) | "read ../../.env" -> `DENIED` |
| 05 `step_05_approval.py` | human approval bound to a hash of the exact call, re-verified at execution time | write a file; answer `n`, then `y` |
| 06 `step_06_quotas.py` | hard budgets: model calls, tool calls, per-tool, wall clock; quota errors are strings, not exceptions | ask it to write 10 files |
| 07 `step_07_caching.py` | byte-stable prompt prefix, cache breakpoint, `PrefixTracker`; volatile text goes last | watch `prefix=... changes=0` |
| 08 `step_08_skills.py` | lazy skills: the prompt holds an index, bodies load on demand | "review note.txt" -> `load_skill(code-review)` |
| 09 `step_09_compaction.py` | summarise old turns when history is big, never orphan a tool result, never touch the prefix | long chat; watch `[context compacted]` |
| 10 `step_10_subagents.py` | read-only sub-agent with fresh context, narrower tools, own budget; only its answer returns | "summarise every file in the sandbox" |

## Design decisions worth saying out loud
- **Rules live in code, not in the prompt.** A prompt is a request; the sandbox, the deny-list and `allowed=` are guarantees.
  `allowed` is enforced in `run_tool`, not only by hiding the schema, so a hallucinated call to a forbidden tool is denied.
- **Errors are data.** A tool failure becomes a `ERROR:` / `DENIED:` / `QUOTA:` string; the model adapts. Only a harness
  bug should raise.
- **Approval binds to a digest of the exact call** and is re-checked at the moment of execution (TOCTOU defence); the
  fail-fast path check runs before bothering the human.
- **Quotas are charged only when a call will really run**, so a denied call does not eat budget.
- **Prefix stability is a bug class**: `PrefixTracker` shouts if the system prompt or tool list changes between calls.
- **Compaction keeps the tail starting on a user message** (providers reject orphaned `tool` messages) and leaves `messages[0]` alone.
- **Sub-agents save the parent's context**, not money: the parent pays for one short result instead of N file dumps.

## Known limits (honest)
- The sandbox is path-based: it is not a container. `run_python` runs real Python with the user's privileges, which is
  why it needs approval. Do not run untrusted prompts with `HARNESS_AUTO_APPROVE=1`.
- Free models vary in tool-calling quality; expect some malformed calls (that is exactly what step 03 is for).
- Token accounting relies on the provider's `usage`; free routes often report zero cached tokens.
- `step_05`–`step_10` were generated from `step_04` by a throw-away script; the files in the repo are the source of truth.

## Part C reuses this
`ml-harness/proposers.py` imports `step_10_subagents.py` and calls its `agent_loop`/`run_tool`/`Budget` with a narrower
tool set, so the autoresearch agent inherits the quotas and the deny-by-default `allowed=` mechanism.

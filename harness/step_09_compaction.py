"""step_09 -- context compaction.

A long session eventually overflows the context window (and gets slow and
expensive long before). New: when the history passes a size threshold, the OLD
part is replaced by a model-written summary and the recent tail is kept verbatim.

Two details that matter:
  * the cut must never split an assistant tool_call from its tool results
    (providers reject orphaned tool messages) -- the kept tail always starts on a
    user message;
  * messages[0] (the cached prefix) is left untouched.

    python harness/step_09_compaction.py
"""
import hashlib
import json
import os
from datetime import date
import subprocess
import sys
import time
from pathlib import Path

import llm

SANDBOX = (Path(__file__).resolve().parent / "sandbox").resolve()
DENY_NAMES = {".env", "id_rsa", "id_ed25519", ".git", "credentials"}
SYSTEM = ("You are a concise coding assistant working inside a sandbox directory. "
          "Writing files and running code needs human approval; if denied, accept it. "
          "All paths are relative to it. Use tools; do not guess file contents.")
MAX_TOOL_OUTPUT = 4000
RUN_TIMEOUT_S = 10
COMPACT_AT_CHARS = 12000
KEEP_LAST = 6
SKILLS_DIR = Path(__file__).resolve().parent / "skills"
SENSITIVE = {"write_file", "run_python"}
ALWAYS_ALLOWED: set = set()                                      # tools the human answered "a" to


# ---- NEW: the fence ----
def safe_path(rel: str) -> Path:
    """Resolve `rel` against the sandbox or raise PermissionError."""
    target = (SANDBOX / rel).resolve()                           # resolves symlinks and ..
    if target != SANDBOX and SANDBOX not in target.parents:
        raise PermissionError(f"'{rel}' escapes the sandbox")
    if any(part.lower() in DENY_NAMES for part in target.relative_to(SANDBOX).parts):
        raise PermissionError(f"'{rel}' is on the deny list")
    return target


def read_file(path: str) -> str:
    return safe_path(path).read_text(encoding="utf-8")


def list_dir(path: str = ".") -> str:
    return "\n".join(sorted(p.name for p in safe_path(path).iterdir()))


def write_file(path: str, content: str) -> str:
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def run_python(path: str) -> str:
    script = safe_path(path)
    if script.suffix != ".py":
        raise PermissionError("only .py files can be run")
    clean_env = {k: v for k, v in os.environ.items()
                 if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    try:
        done = subprocess.run([sys.executable, "-I", str(script)], cwd=SANDBOX, env=clean_env,
                              capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {RUN_TIMEOUT_S}s"
    return f"exit={done.returncode}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"


# ---- NEW: skills ----
def load_skills() -> dict:
    skills = {}
    for f in sorted(SKILLS_DIR.glob("*.md")):
        raw = f.read_text(encoding="utf-8")
        if not raw.startswith("---"):
            continue
        _, front, body = raw.split("---", 2)
        meta = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in front.strip().splitlines() if ":" in ln)}
        if "name" in meta:
            skills[meta["name"]] = {"description": meta.get("description", ""), "body": body.strip()}
    return skills


SKILLS = load_skills()


def load_skill(name: str) -> str:
    if name not in SKILLS:
        raise KeyError(f"no skill '{name}'. Available: {', '.join(SKILLS)}")
    return SKILLS[name]["body"]


def schema(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    schema("read_file", "Read a UTF-8 text file in the sandbox.", {"path": {"type": "string"}}, ["path"]),
    schema("list_dir", "List a sandbox directory.", {"path": {"type": "string"}}, []),
    schema("write_file", "Create or overwrite a file in the sandbox.",
           {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    schema("run_python", "Run a .py file from the sandbox and return its output.", {"path": {"type": "string"}}, ["path"]),
    schema("load_skill", "Load the full instructions of a named skill.", {"name": {"type": "string"}}, ["name"]),
]
DISPATCH = {"read_file": read_file, "list_dir": list_dir, "write_file": write_file, "run_python": run_python, "load_skill": load_skill}


# ---- NEW: approval bound to a digest of the exact call ----
def digest(name: str, args: dict) -> str:
    canonical = json.dumps({"tool": name, "args": args}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def ask_human(name: str, args: dict) -> bool:
    if os.environ.get("HARNESS_AUTO_APPROVE") == "1" or name in ALWAYS_ALLOWED:
        return True
    try:
        answer = input(f"  approve {name}({json.dumps(args)[:300]})? [y/a/n] ").strip().lower()
    except EOFError:
        return False                                             # no human present = no
    if answer == "a":
        ALWAYS_ALLOWED.add(name)
    return answer in ("y", "a")


# ---- NEW: budgets ----
class Budget:
    """Hard ceilings for one user request."""

    def __init__(self, max_model_calls=12, max_tool_calls=20, per_tool=None, max_seconds=180):
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.per_tool_limit = per_tool if per_tool is not None else {"write_file": 5, "run_python": 5}
        self.deadline = time.monotonic() + max_seconds
        self.model_calls = 0
        self.tool_calls = 0
        self.per_tool_used: dict = {}

    def model_call(self):
        """Reserve one model call, or return the reason we must stop."""
        if time.monotonic() > self.deadline:
            return "wall-clock budget exhausted"
        if self.model_calls >= self.max_model_calls:
            return f"model-call budget ({self.max_model_calls}) exhausted"
        self.model_calls += 1
        return None

    def charge(self, tool: str):
        """Reserve one tool call, or return a message for the model."""
        if self.tool_calls >= self.max_tool_calls:
            return f"QUOTA: total tool-call limit ({self.max_tool_calls}) reached. Answer with what you have."
        limit = self.per_tool_limit.get(tool)
        if limit is not None and self.per_tool_used.get(tool, 0) >= limit:
            return f"QUOTA: '{tool}' limit ({limit}) reached. Do not call it again; answer with what you have."
        self.tool_calls += 1
        self.per_tool_used[tool] = self.per_tool_used.get(tool, 0) + 1
        return None

    def summary(self) -> str:
        return (f"model_calls={self.model_calls}/{self.max_model_calls} "
                f"tool_calls={self.tool_calls}/{self.max_tool_calls} per_tool={self.per_tool_used}")


# ---- NEW: a byte-stable prefix ----
def system_text() -> str:
    index = "\n".join(f"- {n}: {s['description']}" for n, s in sorted(SKILLS.items()))   # sorted = stable bytes
    return f"{SYSTEM}\n\nSkills (call load_skill(name) to read one BEFORE using it):\n{index}"


def system_message() -> dict:
    text = system_text()
    if llm.supports_cache_control():                             # explicit breakpoint where the provider honours it
        return {"role": "system", "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}
    return {"role": "system", "content": text}


class PrefixTracker:
    """Detects accidental cache-busting and reports cache hits."""

    def __init__(self):
        self.digest = None
        self.changes = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0

    def check(self, messages: list) -> None:
        payload = json.dumps([messages[0], TOOLS], sort_keys=True)
        d = hashlib.sha256(payload.encode()).hexdigest()[:12]
        if self.digest is not None and d != self.digest:
            self.changes += 1
            print(f"  [cache] WARNING prefix changed {self.digest} -> {d}; the provider cache will miss")
        self.digest = d

    def record(self, usage: dict) -> None:
        self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
        self.cached_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0

    def summary(self) -> str:
        pct = 100 * self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0
        return (f"prefix={self.digest} changes={self.changes} "
                f"prompt_tokens={self.prompt_tokens} cached={self.cached_tokens} ({pct:.0f}%)")


# ---- NEW: compaction ----
def history_size(messages: list) -> int:
    return sum(len(json.dumps(m)) for m in messages)


def safe_cut(messages: list, keep_last: int) -> int:
    """Index where the summarised part ends; messages[cut:] is kept verbatim.

    The kept tail must start on a user message so no tool result is orphaned.
    Search backwards (keep more) and give up (return 1 = nothing to summarise)
    if the only user message is the first one.
    """
    cut = max(1, len(messages) - keep_last)
    while cut > 1 and messages[cut]["role"] != "user":
        cut -= 1
    return cut


def compact(messages: list) -> list:
    cut = safe_cut(messages, KEEP_LAST)
    old = messages[1:cut]
    if len(old) < 2:
        return messages
    transcript = "\n".join(f"{m['role']}: {(m.get('content') or '[tool call]')[:600]}" for m in old)
    ask = [{"role": "system", "content": "Summarise this conversation for your own later use. Keep the user's goals, "
                                         "decisions, file names, facts learned and open TODOs. Under 200 words."},
           {"role": "user", "content": transcript}]
    summary = llm.text_of(llm.chat(ask))
    tail = list(messages[cut:])
    tail[0] = {"role": "user", "content": f"[summary of the earlier conversation]\n{summary}\n\n[current request]\n{tail[0]['content']}"}
    return [messages[0]] + tail                                  # prefix untouched, no consecutive user messages


def maybe_compact(messages: list) -> None:
    if history_size(messages) <= COMPACT_AT_CHARS:
        return
    before = history_size(messages)
    try:
        messages[:] = compact(messages)
    except llm.LLMError as exc:
        print(f"  [compaction skipped: {exc}]")
        return
    print(f"  [context compacted: {before} -> {history_size(messages)} chars]")


def run_tool(name: str, raw_args: str, budget: Budget) -> str:
    if name not in DISPATCH:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(DISPATCH)}"
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            return "ERROR: tool arguments must be a JSON object"
    except json.JSONDecodeError as exc:
        return f"ERROR: arguments were not valid JSON ({exc})"
    if name in SENSITIVE:
        try:                                                     # fail fast BEFORE bothering the human
            if "path" in args:
                safe_path(args["path"])
        except PermissionError as exc:
            return f"DENIED: {exc}"
        approved = digest(name, args)
        if not ask_human(name, args):
            return "DENIED: the human did not approve this call"
        if digest(name, args) != approved:                       # re-verify at the moment of execution
            return "DENIED: arguments changed after approval"
    over_quota = budget.charge(name)                             # NEW: charged only when it will really run
    if over_quota:
        return over_quota
    try:
        out = str(DISPATCH[name](**args))
    except PermissionError as exc:
        return f"DENIED: {exc}"                                  # distinct prefix: model learns it is a policy wall
    except TypeError as exc:
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + f"\n...[truncated {len(out) - MAX_TOOL_OUTPUT} chars]"
    return out


def agent_loop(messages: list, budget: Budget, tracker: PrefixTracker) -> str:
    while True:
        stop = budget.model_call()                               # NEW: every model call is metered
        if stop:
            return f"(stopped: {stop})"
        maybe_compact(messages)                                  # NEW
        tracker.check(messages)
        try:
            reply = llm.chat(messages, TOOLS)
        except llm.LLMError as exc:
            return f"(model unavailable: {exc})"
        tracker.record(reply["usage"])                           # NEW
        msg = reply["message"]
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content") or ""
        for call in calls:
            name, raw = call["function"]["name"], call["function"]["arguments"]
            result = run_tool(name, raw, budget)
            print(f"  [tool] {name}({raw}) -> {result[:70]!r}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})


def main() -> None:
    SANDBOX.mkdir(exist_ok=True)
    messages = [system_message()]
    tracker = PrefixTracker()
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            break
        if user.lower() in ("exit", "quit"):
            break
        if user:
            # volatile facts ride at the END of the conversation, never in the prefix
            messages.append({"role": "user", "content": f"{user}\n\n[context: today is {date.today()}]"})
            budget = Budget()
            print(f"bot> {agent_loop(messages, budget, tracker)}")
            print(f"     [{budget.summary()}]")
            print(f"     [{tracker.summary()}]")


if __name__ == "__main__":
    main()

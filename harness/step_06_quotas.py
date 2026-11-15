"""step_06 -- budgets and quotas.

New: hard ceilings on model calls, total tool calls, per-tool calls and wall-clock
time, per user request. A model stuck in a loop (or a prompt injection that says
"keep calling write_file") burns money and time; the harness, not the model,
decides when enough is enough. Exceeding a quota returns a STRING to the model
("answer with what you have"), it never raises.

    python harness/step_06_quotas.py
"""
import hashlib
import json
import os
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
]
DISPATCH = {"read_file": read_file, "list_dir": list_dir, "write_file": write_file, "run_python": run_python}


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


def agent_loop(messages: list, budget: Budget) -> str:
    while True:
        stop = budget.model_call()                               # NEW: every model call is metered
        if stop:
            return f"(stopped: {stop})"
        try:
            msg = llm.chat(messages, TOOLS)["message"]
        except llm.LLMError as exc:
            return f"(model unavailable: {exc})"
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
    messages = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            break
        if user.lower() in ("exit", "quit"):
            break
        if user:
            messages.append({"role": "user", "content": user})
            budget = Budget()                                    # fresh budget per request
            print(f"bot> {agent_loop(messages, budget)}")
            print(f"     [{budget.summary()}]")


if __name__ == "__main__":
    main()

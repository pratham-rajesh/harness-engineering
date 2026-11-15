"""step_05 -- human approval, verified at execution time.

New: write_file and run_python are SENSITIVE, so a human must say yes first.
The subtle part is what "yes" is bound to. We hash the exact call the human saw
(tool name + canonical args). Right before executing we re-hash and compare, and
the sandbox check runs again inside the tool, so a call (or a symlink) that
changed between "approved" and "executed" is refused. Approval of one thing is
never approval of another.

  y = allow once    a = always allow this tool this session    n = deny
  HARNESS_AUTO_APPROVE=1 skips the prompt (CI and demos only)

    python harness/step_05_approval.py
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import llm

SANDBOX = (Path(__file__).resolve().parent / "sandbox").resolve()
DENY_NAMES = {".env", "id_rsa", "id_ed25519", ".git", "credentials"}
SYSTEM = ("You are a concise coding assistant working inside a sandbox directory. "
          "Writing files and running code needs human approval; if denied, accept it. "
          "All paths are relative to it. Use tools; do not guess file contents.")
MAX_TURNS = 10
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


def run_tool(name: str, raw_args: str) -> str:
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


def agent_loop(messages: list) -> str:
    for _ in range(MAX_TURNS):
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
            result = run_tool(name, raw)
            print(f"  [tool] {name}({raw}) -> {result[:70]!r}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return "(stopped: too many tool rounds)"


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
            print(f"bot> {agent_loop(messages)}")


if __name__ == "__main__":
    main()

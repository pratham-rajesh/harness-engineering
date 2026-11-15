"""step_04 -- the sandbox fence.

New: the model can now WRITE files and RUN Python, so the harness must decide
what it may touch. Rules live in code, not in the prompt -- a prompt is a request,
a guard is a guarantee.

  * every path is resolved (symlinks and `..` included) and must stay inside sandbox/
  * a deny-list of sensitive file names, even inside the sandbox
  * run_python executes a sandbox script with a timeout, no shell, and a scrubbed
    environment (the child never sees your API key)

    python harness/step_04_sandbox.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import llm

SANDBOX = (Path(__file__).resolve().parent / "sandbox").resolve()
DENY_NAMES = {".env", "id_rsa", "id_ed25519", ".git", "credentials"}
SYSTEM = ("You are a concise coding assistant working inside a sandbox directory. "
          "All paths are relative to it. Use tools; do not guess file contents.")
MAX_TURNS = 10
MAX_TOOL_OUTPUT = 4000
RUN_TIMEOUT_S = 10


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


def run_tool(name: str, raw_args: str) -> str:
    if name not in DISPATCH:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(DISPATCH)}"
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            return "ERROR: tool arguments must be a JSON object"
    except json.JSONDecodeError as exc:
        return f"ERROR: arguments were not valid JSON ({exc})"
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

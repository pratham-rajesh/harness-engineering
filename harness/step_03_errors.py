"""step_03 -- errors are strings, not crashes.

New: every tool failure (unknown tool, malformed JSON arguments, missing file,
a bug in our own tool) becomes a *tool result string*. The model reads it and
adapts -- usually by fixing its call. An exception would kill the whole session.
Also: tool output is truncated so one huge file cannot flood the context.

    python harness/step_03_errors.py
"""
import json
import os

import llm

SYSTEM = "You are a concise coding assistant. Use tools to look at files instead of guessing."
MAX_TURNS = 8
MAX_TOOL_OUTPUT = 4000


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def list_dir(path: str = ".") -> str:
    return "\n".join(sorted(os.listdir(path)))


def schema(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    schema("read_file", "Read a UTF-8 text file.", {"path": {"type": "string"}}, ["path"]),
    schema("list_dir", "List the entries of a directory.", {"path": {"type": "string"}}, []),
]
DISPATCH = {"read_file": read_file, "list_dir": list_dir}


# ---- NEW: one choke point that can never raise ----
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
    except TypeError as exc:                                     # wrong / missing argument names
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:                                     # FileNotFoundError, UnicodeDecodeError, ...
        return f"ERROR: {type(exc).__name__}: {exc}"
    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + f"\n...[truncated {len(out) - MAX_TOOL_OUTPUT} chars]"
    return out


def agent_loop(messages: list) -> str:
    for _ in range(MAX_TURNS):
        try:
            msg = llm.chat(messages, TOOLS)["message"]
        except llm.LLMError as exc:                              # NEW: model outage is also not a crash
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

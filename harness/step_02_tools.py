"""step_02 -- tool dispatch.

New: the model can ask us to run a function. It never runs anything itself;
it returns a `tool_calls` message, WE execute, append a `tool` message with the
result, and call the model again. That inner loop is what makes it an agent.

Deliberately UNSAFE at this stage: read_file will read any path on disk.
step_04 fixes that.

    python harness/step_02_tools.py
"""
import json
import os

import llm

SYSTEM = "You are a concise coding assistant. Use tools to look at files instead of guessing."
MAX_TURNS = 8


# ---- NEW: tools are plain functions + a JSON schema the model can read ----
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


def agent_loop(messages: list) -> str:
    """call model -> run requested tools -> feed results back -> repeat."""
    for _ in range(MAX_TURNS):
        reply = llm.chat(messages, TOOLS)
        msg = reply["message"]
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:                                            # plain text = we are done
            return msg.get("content") or ""
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            print(f"  [tool] {name}({args})")
            result = DISPATCH[name](**args)                      # step_03 makes this safe
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

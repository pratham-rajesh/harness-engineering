"""step_01 -- the conversation loop.

The whole idea: a model has no memory. "Memory" is us re-sending the full
message list every turn. A harness starts as a `while True` around that list.

    python harness/step_01_loop.py
"""
import llm

SYSTEM = "You are a concise assistant."


def main() -> None:
    messages = [{"role": "system", "content": SYSTEM}]          # the ENTIRE memory
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            break
        if user.lower() in ("exit", "quit"):
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        try:
            reply = llm.chat(messages)
        except llm.LLMError as exc:
            messages.pop()                                       # keep history consistent
            print(f"[model unavailable: {exc}]")
            continue
        messages.append(reply["message"])
        print(f"bot> {llm.text_of(reply)}")
        print(f"     [{len(messages)} messages in history, usage={reply['usage']}]")


if __name__ == "__main__":
    main()

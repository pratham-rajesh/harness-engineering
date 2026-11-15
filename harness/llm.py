"""llm.py -- the one place that talks to a model.

Every step_NN.py imports this and calls `llm.chat(messages, tools)`.
Keeping the transport here means each step file contains only the *harness*
idea it is teaching, not HTTP plumbing.

Works with any OpenAI-compatible /chat/completions endpoint:

  * OpenRouter (default). Models whose id ends in ":free" cost nothing, and
    "openrouter/free" auto-routes to whichever free model is available.
  * Ollama / llama.cpp / LM Studio: set LLM_BASE_URL=http://localhost:11434/v1

Environment (or a .env file in the repo root):
  OPENROUTER_API_KEY | LLM_API_KEY   key (optional for localhost endpoints)
  LLM_BASE_URL                       default https://openrouter.ai/api/v1
  LLM_MODEL                          default openrouter/free
  LLM_FALLBACK_MODELS                comma list tried when a model keeps failing

Smoke test:  python harness/llm.py
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the model could not be reached after all retries."""


def load_env(path: Path | None = None) -> None:
    """Minimal .env reader (no dependency). Real environment variables win."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def settings() -> dict:
    load_env()
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    local = "localhost" in base_url or "127.0.0.1" in base_url
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ("local" if local else "")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    fallbacks = [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
    return {"base_url": base_url, "api_key": api_key, "model": model, "fallbacks": fallbacks}


def supports_cache_control() -> bool:
    """Explicit prompt-cache breakpoints are an OpenRouter/Anthropic-style extension."""
    return "openrouter.ai" in settings()["base_url"]


def clean_message(msg: dict) -> dict:
    """Keep only the fields the API accepts back.

    Free models often attach extra keys (reasoning, refusal, annotations...).
    Echoing those into the next request makes some providers return a 400.
    """
    out = {"role": msg.get("role", "assistant"), "content": msg.get("content")}
    if msg.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"},
            }
            for c in msg["tool_calls"]
        ]
    return out


def _post(cfg: dict, model: str, messages: list, tools: list | None, timeout: int) -> requests.Response:
    body: dict = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    return requests.post(f"{cfg['base_url']}/chat/completions", headers=headers, json=body, timeout=timeout)


def chat(messages: list, tools: list | None = None, *, model: str | None = None,
         retries: int = 4, timeout: int = 120) -> dict:
    """One model call. Returns {"message": {...}, "usage": {...}, "model": str}.

    Retries 429/5xx/timeouts with exponential backoff, treats an empty reply
    (no text and no tool call -- common on free tiers) as retryable, and rotates
    through LLM_FALLBACK_MODELS before giving up.
    """
    cfg = settings()
    if not cfg["api_key"]:
        raise LLMError("No API key. Put OPENROUTER_API_KEY in .env (free key: https://openrouter.ai/keys) "
                       "or point LLM_BASE_URL at a local Ollama server.")
    candidates = [model or cfg["model"]] + [m for m in cfg["fallbacks"] if m != (model or cfg["model"])]
    last_error = "unknown error"
    for candidate in candidates:
        for attempt in range(retries):
            try:
                resp = _post(cfg, candidate, messages, tools, timeout)
            except requests.RequestException as exc:
                last_error = f"{candidate}: network error {type(exc).__name__}"
                resp = None
            if resp is not None and resp.ok:
                data = resp.json()
                choices = data.get("choices") or []
                msg = choices[0].get("message") if choices else None
                if msg and (msg.get("content") or msg.get("tool_calls")):
                    return {"message": clean_message(msg), "usage": data.get("usage") or {},
                            "model": data.get("model", candidate)}
                last_error = f"{candidate}: empty response"
            elif resp is not None:
                last_error = f"{candidate}: HTTP {resp.status_code} {resp.text[:200]}"
                if resp.status_code not in RETRYABLE_STATUS:
                    break  # 400/401/402/404 will not fix themselves; try the next model
            time.sleep(min(2 ** attempt, 20) + random.random())
    raise LLMError(f"model call failed after retries -- last error: {last_error}")


def text_of(reply: dict) -> str:
    return reply["message"].get("content") or ""


if __name__ == "__main__":
    cfg = settings()
    print(f"endpoint={cfg['base_url']}  model={cfg['model']}")
    try:
        r = chat([{"role": "user", "content": "Reply with exactly: pong"}])
    except LLMError as exc:
        sys.exit(f"FAILED: {exc}")
    print(f"model said: {text_of(r)!r}   (served by {r['model']}, usage={r['usage']})")

"""Shared fixtures: a scripted fake model and a loader for the step_NN.py files."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent / "harness"
sys.path.insert(0, str(HARNESS))
import llm  # noqa: E402


def load_step(filename: str):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), HARNESS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def say(text: str) -> dict:
    return {"role": "assistant", "content": text}


def call(name: str, call_id: str = "c1", **args) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


class ScriptedLLM:
    """Stand-in for llm.chat: returns canned assistant messages in order and records every request."""

    def __init__(self, *replies, usage=None):
        self.replies = list(replies)
        self.requests = []
        self.usage = usage or {"prompt_tokens": 100, "completion_tokens": 10}

    def __call__(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "tools": tools})
        if not self.replies:
            raise AssertionError("model was called more times than the test scripted")
        reply = self.replies.pop(0)
        reply = reply(messages) if callable(reply) else reply
        return {"message": reply, "usage": self.usage, "model": "fake"}


@pytest.fixture
def script(monkeypatch):
    def install(*replies, **kw):
        fake = ScriptedLLM(*replies, **kw)
        monkeypatch.setattr(llm, "chat", fake)
        return fake
    return install


@pytest.fixture
def sandboxed(tmp_path):
    """Returns a function that points a step module's SANDBOX at a temp dir."""
    def point(module):
        root = (tmp_path / "sandbox").resolve()
        root.mkdir()
        module.SANDBOX = root
        return root
    return point

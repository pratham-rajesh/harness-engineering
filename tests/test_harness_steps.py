"""Offline tests for harness/step_01..10 using a scripted fake model (no API key, no network)."""
import json
import os
import random
import sys

import pytest

import llm
from conftest import call, load_step, say


def feed_input(monkeypatch, *lines):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr("builtins.input", fake_input)


# ------------------------------------------------------------------ step 01
def test_step01_history_is_the_memory(script, monkeypatch, capsys):
    s1 = load_step("step_01_loop.py")
    fake = script(say("hello"), say("you said hi"))
    feed_input(monkeypatch, "hi", "what did I say?", "exit")
    s1.main()
    second_request = fake.requests[1]["messages"]
    assert [m["role"] for m in second_request] == ["system", "user", "assistant", "user"]
    assert second_request[2]["content"] == "hello"


def test_step01_model_outage_does_not_poison_history(monkeypatch, capsys):
    s1 = load_step("step_01_loop.py")

    def boom(*a, **k):
        raise llm.LLMError("down")
    monkeypatch.setattr(llm, "chat", boom)
    feed_input(monkeypatch, "hi", "exit")
    s1.main()
    assert "model unavailable" in capsys.readouterr().out


# ------------------------------------------------------------------ step 02
def test_step02_tool_round_trip(script, tmp_path):
    s2 = load_step("step_02_tools.py")
    f = tmp_path / "a.txt"
    f.write_text("secret-42", encoding="utf-8")
    fake = script(call("read_file", path=str(f)), say("it says secret-42"))
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "read it"}]
    assert s2.agent_loop(messages, ) == "it says secret-42"
    tool_msg = [m for m in messages if m["role"] == "tool"][0]
    assert tool_msg["content"] == "secret-42" and tool_msg["tool_call_id"] == "c1"
    assert len(fake.requests) == 2


def test_step02_gives_up_after_max_turns(script):
    s2 = load_step("step_02_tools.py")
    script(*[call("list_dir", path=".") for _ in range(s2.MAX_TURNS)])
    out = s2.agent_loop([{"role": "user", "content": "loop forever"}])
    assert "too many tool rounds" in out


# ------------------------------------------------------------------ step 03
@pytest.mark.parametrize("name,raw,expect", [
    ("nope", "{}", "unknown tool"),
    ("read_file", "{not json", "not valid JSON"),
    ("read_file", "[1,2]", "must be a JSON object"),
    ("read_file", "{}", "bad arguments"),
    ("read_file", json.dumps({"path": "/definitely/missing.txt"}), "FileNotFoundError"),
])
def test_step03_errors_become_strings(name, raw, expect):
    s3 = load_step("step_03_errors.py")
    assert expect in s3.run_tool(name, raw)


def test_step03_truncates_huge_output(tmp_path):
    s3 = load_step("step_03_errors.py")
    big = tmp_path / "big.txt"
    big.write_text("x" * 20000, encoding="utf-8")
    out = s3.run_tool("read_file", json.dumps({"path": str(big)}))
    assert len(out) < s3.MAX_TOOL_OUTPUT + 100 and "truncated" in out


def test_step03_model_recovers_from_error(script):
    s3 = load_step("step_03_errors.py")
    script(call("read_file", path="/missing.txt"), say("that file does not exist"))
    messages = [{"role": "user", "content": "read /missing.txt"}]
    assert s3.agent_loop(messages) == "that file does not exist"
    assert messages[-2]["content"].startswith("ERROR: FileNotFoundError")


# ------------------------------------------------------------------ step 04
@pytest.fixture
def s4(sandboxed):
    mod = load_step("step_04_sandbox.py")
    sandboxed(mod)
    return mod


@pytest.mark.parametrize("bad", ["../escape.txt", "..\\escape.txt", "a/../../escape.txt", ".env", "sub/id_rsa"])
def test_step04_blocks_escapes_and_denied_names(s4, bad):
    with pytest.raises(PermissionError):
        s4.safe_path(bad)


def test_step04_blocks_absolute_path_outside(s4, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    assert s4.run_tool("read_file", json.dumps({"path": str(outside)})).startswith("DENIED")


def test_step04_blocks_symlink_escape(s4, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("top secret")
    link = s4.SANDBOX / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this machine")
    assert s4.run_tool("read_file", json.dumps({"path": "link.txt"})).startswith("DENIED")


def test_step04_write_then_run(s4):
    assert "wrote" in s4.run_tool("write_file", json.dumps({"path": "hi.py", "content": "print('hello from sandbox')"}))
    out = s4.run_tool("run_python", json.dumps({"path": "hi.py"}))
    assert "exit=0" in out and "hello from sandbox" in out


def test_step04_child_process_never_sees_secrets(s4, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    (s4.SANDBOX / "env.py").write_text("import os; print('LEAK' if 'OPENROUTER_API_KEY' in os.environ else 'clean')")
    assert "clean" in s4.run_tool("run_python", json.dumps({"path": "env.py"}))


def test_step04_timeout_and_non_py(s4, monkeypatch):
    monkeypatch.setattr(s4, "RUN_TIMEOUT_S", 1)
    (s4.SANDBOX / "slow.py").write_text("import time; time.sleep(5)")
    assert "timed out" in s4.run_tool("run_python", json.dumps({"path": "slow.py"}))
    (s4.SANDBOX / "x.sh").write_text("echo hi")
    assert s4.run_tool("run_python", json.dumps({"path": "x.sh"})).startswith("DENIED")


# ------------------------------------------------------------------ step 05
@pytest.fixture
def s5(sandboxed, monkeypatch):
    mod = load_step("step_05_approval.py")
    sandboxed(mod)
    monkeypatch.delenv("HARNESS_AUTO_APPROVE", raising=False)
    mod.ALWAYS_ALLOWED.clear()
    return mod


WRITE = json.dumps({"path": "a.txt", "content": "hi"})


def test_step05_no_human_means_no(s5, monkeypatch):
    feed_input(monkeypatch)                                       # immediate EOF
    assert s5.run_tool("write_file", WRITE).startswith("DENIED")
    assert not (s5.SANDBOX / "a.txt").exists()


def test_step05_yes_allows_once_and_n_denies(s5, monkeypatch):
    feed_input(monkeypatch, "y", "n")
    assert "wrote" in s5.run_tool("write_file", WRITE)
    assert s5.run_tool("write_file", WRITE).startswith("DENIED")


def test_step05_always_remembers_per_tool(s5, monkeypatch):
    feed_input(monkeypatch, "a")                                  # only one answer available
    assert "wrote" in s5.run_tool("write_file", WRITE)
    assert "wrote" in s5.run_tool("write_file", WRITE)           # no prompt needed the second time


def test_step05_reads_never_prompt(s5, monkeypatch):
    (s5.SANDBOX / "r.txt").write_text("data")
    feed_input(monkeypatch)
    assert s5.run_tool("read_file", json.dumps({"path": "r.txt"})) == "data"


def test_step05_arguments_changed_after_approval_is_refused(s5, monkeypatch):
    def sneaky(name, args):
        args["content"] = "something the human never saw"
        return True
    monkeypatch.setattr(s5, "ask_human", sneaky)
    assert "arguments changed after approval" in s5.run_tool("write_file", WRITE)
    assert not (s5.SANDBOX / "a.txt").exists()


def test_step05_escaping_path_is_denied_before_asking(s5, monkeypatch):
    asked = []
    monkeypatch.setattr(s5, "ask_human", lambda n, a: asked.append(n) or True)
    out = s5.run_tool("write_file", json.dumps({"path": "../x.txt", "content": "z"}))
    assert out.startswith("DENIED") and not asked


def test_step05_digest_is_order_independent_but_content_sensitive(s5):
    assert s5.digest("t", {"a": 1, "b": 2}) == s5.digest("t", {"b": 2, "a": 1})
    assert s5.digest("t", {"a": 1}) != s5.digest("t", {"a": 2})


# ------------------------------------------------------------------ step 06
@pytest.fixture
def s6(sandboxed, monkeypatch):
    mod = load_step("step_06_quotas.py")
    sandboxed(mod)
    monkeypatch.setenv("HARNESS_AUTO_APPROVE", "1")
    return mod


def test_step06_per_tool_quota_returns_string(s6):
    b = s6.Budget(per_tool={"write_file": 2})
    results = [s6.run_tool("write_file", json.dumps({"path": f"{i}.txt", "content": "x"}), b) for i in range(3)]
    assert "wrote" in results[0] and "wrote" in results[1] and results[2].startswith("QUOTA")
    assert not (s6.SANDBOX / "2.txt").exists()


def test_step06_total_tool_quota(s6):
    b = s6.Budget(max_tool_calls=2)
    out = [s6.run_tool("list_dir", "{}", b) for _ in range(3)]
    assert out[2].startswith("QUOTA: total")


def test_step06_loop_stops_at_model_call_cap(s6, script):
    script(*[call("list_dir", call_id=f"c{i}") for i in range(10)])
    b = s6.Budget(max_model_calls=3)
    out = s6.agent_loop([{"role": "system", "content": "s"}, {"role": "user", "content": "go"}], b)
    assert "budget" in out and b.model_calls == 3


def test_step06_wall_clock_deadline(s6):
    b = s6.Budget(max_seconds=-1)
    assert "wall-clock" in b.model_call()


def test_step06_denied_calls_do_not_burn_quota(s6, monkeypatch):
    feed_input(monkeypatch)
    monkeypatch.delenv("HARNESS_AUTO_APPROVE")
    b = s6.Budget()
    assert s6.run_tool("write_file", WRITE, b).startswith("DENIED")
    assert b.tool_calls == 0


# ------------------------------------------------------------------ step 07
@pytest.fixture
def s7(sandboxed):
    mod = load_step("step_07_caching.py")
    sandboxed(mod)
    return mod


def test_step07_prefix_stable_across_turns_and_volatile_text_is_last(s7, script, monkeypatch):
    fake = script(say("one"), say("two"), usage={"prompt_tokens": 200, "prompt_tokens_details": {"cached_tokens": 150}})
    feed_input(monkeypatch, "first", "second", "exit")
    s7.main()
    prefixes = [json.dumps(r["messages"][0], sort_keys=True) for r in fake.requests]
    assert prefixes[0] == prefixes[1]
    assert "today is" not in prefixes[0]
    assert "today is" in fake.requests[1]["messages"][-1]["content"]


def test_step07_tracker_flags_a_cache_buster(s7, capsys):
    t = s7.PrefixTracker()
    t.check([{"role": "system", "content": "a"}])
    t.check([{"role": "system", "content": "a"}])
    assert t.changes == 0
    t.check([{"role": "system", "content": "a (now with a timestamp 12:01)"}])
    assert t.changes == 1 and "WARNING" in capsys.readouterr().out


def test_step07_cached_token_accounting(s7):
    t = s7.PrefixTracker()
    t.record({"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 800}})
    t.record({"prompt_tokens": 1000})
    assert t.cached_tokens == 800 and t.prompt_tokens == 2000 and "(40%)" in t.summary()


def test_step07_cache_control_only_where_supported(s7, monkeypatch):
    monkeypatch.setattr(llm, "supports_cache_control", lambda: True)
    assert s7.system_message()["content"][0]["cache_control"] == {"type": "ephemeral"}
    monkeypatch.setattr(llm, "supports_cache_control", lambda: False)
    assert isinstance(s7.system_message()["content"], str)


# ------------------------------------------------------------------ step 08
@pytest.fixture
def s8(sandboxed):
    mod = load_step("step_08_skills.py")
    sandboxed(mod)
    return mod


def test_step08_index_in_prompt_but_bodies_are_not(s8):
    assert {"code-review", "commit-message", "write-tests"} <= set(s8.SKILLS)
    prompt = s8.system_text()
    for name, skill in s8.SKILLS.items():
        assert name in prompt and skill["description"] in prompt
        assert skill["body"] not in prompt


def test_step08_index_is_deterministic(s8):
    assert s8.system_text() == s8.system_text()


def test_step08_load_skill_and_unknown(s8):
    b = s8.Budget()
    assert "Format:" in s8.run_tool("load_skill", json.dumps({"name": "commit-message"}), b)
    assert "no skill 'nope'" in s8.run_tool("load_skill", json.dumps({"name": "nope"}), b)


def test_step08_skill_files_have_front_matter(s8, tmp_path):
    (tmp_path / "bad.md").write_text("no front matter here")
    s8.SKILLS_DIR = tmp_path
    assert s8.load_skills() == {}


# ------------------------------------------------------------------ step 09
@pytest.fixture
def s9(sandboxed):
    mod = load_step("step_09_compaction.py")
    sandboxed(mod)
    return mod


def _history(n_pairs, tool_chain=False):
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"question {i} " + "x" * 50})
        if tool_chain:
            msgs.append(call("read_file", call_id=f"t{i}", path="a"))
            msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "data"})
        msgs.append(say(f"answer {i}"))
    return msgs


def assert_no_orphan_tool_messages(msgs):
    announced = set()
    for m in msgs:
        for c in m.get("tool_calls") or []:
            announced.add(c["id"])
        if m["role"] == "tool":
            assert m["tool_call_id"] in announced, "orphaned tool result"


@pytest.mark.parametrize("seed", range(40))
def test_step09_cut_never_orphans_tool_results(s9, seed):
    rng = random.Random(seed)
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(rng.randint(2, 12)):
        msgs.append({"role": "user", "content": f"u{i}"})
        for j in range(rng.randint(0, 3)):
            msgs.append(call("read_file", call_id=f"{i}-{j}", path="a"))
            msgs.append({"role": "tool", "tool_call_id": f"{i}-{j}", "content": "d"})
        msgs.append(say(f"a{i}"))
    cut = s9.safe_cut(msgs, rng.randint(1, 8))
    assert cut == 1 or msgs[cut]["role"] == "user"
    assert_no_orphan_tool_messages(msgs[cut:])


def test_step09_compaction_summarises_old_keeps_tail_and_prefix(s9, script):
    msgs = _history(12, tool_chain=True)
    fake = script(say("SUMMARY: user asked 12 questions"))
    original_prefix, last = msgs[0], msgs[-1]
    new = s9.compact(msgs)
    assert new[0] == original_prefix and new[-1] == last
    assert len(new) < len(msgs)
    assert new[1]["role"] == "user" and "SUMMARY" in new[1]["content"]
    assert all(not (a["role"] == "user" and b["role"] == "user") for a, b in zip(new, new[1:]))
    assert_no_orphan_tool_messages(new)
    assert len(fake.requests) == 1


def test_step09_small_history_is_untouched(s9, script):
    fake = script()
    msgs = _history(2)
    s9.maybe_compact(msgs)
    assert len(msgs) == 5 and not fake.requests


def test_step09_compaction_failure_keeps_history(s9, monkeypatch, capsys):
    msgs = _history(120)
    n = len(msgs)

    def boom(*a, **k):
        raise llm.LLMError("down")
    monkeypatch.setattr(llm, "chat", boom)
    s9.maybe_compact(msgs)
    assert len(msgs) == n and "skipped" in capsys.readouterr().out


def test_step09_triggers_inside_the_loop(s9, script):
    msgs = _history(120)
    script(say("SUMMARY"), say("final answer"))
    msgs.append({"role": "user", "content": "latest"})
    out = s9.agent_loop(msgs, s9.Budget(), s9.PrefixTracker())
    assert out == "final answer" and msgs[0]["content"] == "SYS" and len(msgs) < 20


# ------------------------------------------------------------------ step 10
@pytest.fixture
def s10(sandboxed, monkeypatch):
    mod = load_step("step_10_subagents.py")
    root = sandboxed(mod)
    (root / "note.txt").write_text("the harness is the loop", encoding="utf-8")
    monkeypatch.setenv("HARNESS_AUTO_APPROVE", "1")
    return mod


def test_step10_subagent_is_isolated_and_returns_only_its_answer(s10, script):
    fake = script(
        call("spawn_subagent", task="read note.txt and summarise"),    # parent
        call("read_file", call_id="s1", path="note.txt"),              # sub-agent
        say("it says the harness is the loop"),                        # sub-agent final
        say("The note says the harness is the loop."),                 # parent final
    )
    parent = [{"role": "system", "content": "P"}, {"role": "user", "content": "use a subagent"}]
    out = s10.agent_loop(parent, s10.Budget(), s10.PrefixTracker())
    assert out == "The note says the harness is the loop."
    assert [m["role"] for m in parent] == ["system", "user", "assistant", "tool", "assistant"]
    assert parent[3]["content"] == "it says the harness is the loop"
    sub_first = fake.requests[1]["messages"]
    assert sub_first[0]["content"] == s10.SUBAGENT_SYSTEM and len(sub_first) == 2     # fresh context
    assert "use a subagent" not in json.dumps(sub_first)


def test_step10_subagent_tools_are_read_only_and_cannot_recurse(s10, script):
    fake = script(call("spawn_subagent", task="t"), say("done"), say("ok"))
    s10.agent_loop([{"role": "system", "content": "P"}, {"role": "user", "content": "go"}], s10.Budget())
    names = {t["function"]["name"] for t in fake.requests[1]["tools"]}
    assert names == set(s10.READ_ONLY)
    assert "spawn_subagent" not in names and "write_file" not in names


def test_step10_forbidden_tool_is_denied_even_if_model_hallucinates_it(s10, script):
    script(call("spawn_subagent", task="t"), call("write_file", call_id="x", path="pwn.txt", content="z"),
           say("tried"), say("fin"))
    parent = [{"role": "system", "content": "P"}, {"role": "user", "content": "go"}]
    s10.agent_loop(parent, s10.Budget())
    assert not (s10.SANDBOX / "pwn.txt").exists()


def test_step10_subagent_has_its_own_budget(s10, script):
    script(call("spawn_subagent", task="t"), *[call("list_dir", call_id=f"l{i}") for i in range(10)], say("parent done"))
    parent = [{"role": "system", "content": "P"}, {"role": "user", "content": "go"}]
    s10.agent_loop(parent, s10.Budget())
    assert "budget" in parent[3]["content"]


def test_step10_all_earlier_capabilities_still_present(s10, monkeypatch):
    for attr in ("safe_path", "digest", "Budget", "PrefixTracker", "load_skill", "compact", "run_tool"):
        assert hasattr(s10, attr)
    b = s10.Budget()
    assert s10.run_tool("read_file", json.dumps({"path": "../x"}), b).startswith("DENIED")
    assert "unknown tool" in s10.run_tool("zzz", "{}", b)


# ------------------------------------------------------------------ llm.py transport
class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body, self.ok = status, body or {}, status < 400
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def ok_body(text="hi", model="m"):
    return {"model": model, "choices": [{"message": {"role": "assistant", "content": text, "reasoning": "junk"}}], "usage": {}}


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.delenv("LLM_FALLBACK_MODELS", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def install(*responses):
        queue = list(responses)

        def post(cfg, model, messages, tools, timeout):
            calls.append(model)
            return queue.pop(0)
        monkeypatch.setattr(llm, "_post", post)
        return calls
    return install


def test_llm_retries_429_then_succeeds(transport):
    calls = transport(FakeResp(429), FakeResp(200, ok_body("finally")))
    assert llm.text_of(llm.chat([{"role": "user", "content": "x"}])) == "finally" and len(calls) == 2


def test_llm_treats_empty_reply_as_retryable_and_strips_extra_fields(transport):
    empty = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    transport(FakeResp(200, empty), FakeResp(200, ok_body("real")))
    reply = llm.chat([{"role": "user", "content": "x"}])
    assert reply["message"] == {"role": "assistant", "content": "real"}


def test_llm_rotates_to_fallback_model(transport, monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "backup:free")
    monkeypatch.setenv("LLM_MODEL", "primary:free")
    calls = transport(FakeResp(401), FakeResp(200, ok_body("from backup")))
    assert llm.text_of(llm.chat([{"role": "user", "content": "x"}])) == "from backup"
    assert calls == ["primary:free", "backup:free"]


def test_llm_gives_up_with_clear_error(transport):
    transport(*[FakeResp(500) for _ in range(4)])
    with pytest.raises(llm.LLMError, match="HTTP 500"):
        llm.chat([{"role": "user", "content": "x"}])


def test_llm_missing_key_message(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "load_env", lambda path=None: None)
    with pytest.raises(llm.LLMError, match="openrouter.ai/keys"):
        llm.chat([{"role": "user", "content": "x"}])


def test_llm_local_endpoint_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(llm, "load_env", lambda path=None: None)
    assert llm.settings()["api_key"] == "local" and not llm.supports_cache_control()

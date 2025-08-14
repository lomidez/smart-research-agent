import asyncio
import pytest
import agent

def test_render_template_and_missing_prompt(tmp_path, monkeypatch):
    out = agent.render_template("synthesis_user.txt", prompt="P", context_block="CTX", unused="X")
    assert "Prompt: P" in out
    assert "Context:" in out
    missing = agent.load_prompt("nope_does_not_exist.txt")
    assert missing.startswith("[MISSING PROMPT:")

def test_log_span_sync_ok_and_err(caplog):
    calls = {"n": 0}

    @agent.log_span("unit.sync.ok")
    def ok():
        calls["n"] += 1
        return 42

    @agent.log_span("unit.sync.err")
    def bad():
        raise ValueError("boom")

    assert ok() == 42
    with pytest.raises(ValueError):
        bad()

def test_log_span_async_ok_and_err():
    results = {"ok": None}

    @agent.log_span("unit.async.ok")
    async def aok():
        results["ok"] = True
        return "done"

    @agent.log_span("unit.async.err")
    async def abad():
        raise RuntimeError("nope")

    assert asyncio.run(aok()) == "done"
    with pytest.raises(RuntimeError):
        asyncio.run(abad())

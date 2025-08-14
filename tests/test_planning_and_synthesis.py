import json
import agent

def test_plan_steps_with_llm_fallback(monkeypatch):
    monkeypatch.setattr(agent, "chat_completion", lambda *a, **k: "NOT JSON")
    j = agent.plan_steps_with_llm("alignment techniques")
    assert "web_queries" in j and "arxiv_queries" in j
    assert len(j["web_queries"]) <= 2
    assert len(j["arxiv_queries"]) <= 2

def test_synthesize_with_citations_builds_context(monkeypatch):
    captured = {}
    def fake_chat(messages, temperature=0.2, max_tokens=700):
        captured["messages"] = messages
        return "SYN"
    monkeypatch.setattr(agent, "chat_completion", fake_chat)
    ev = [
        {"chunk_text":"C1", "source_id": 10, "ordinal": 1},
        {"chunk_text":"C2", "source_id": 10, "ordinal": 1},
        {"chunk_text":"C3", "source_id": 11, "ordinal": 2},
    ]
    src = {
        10: {"title":"T10", "url":"u10", "authors":"A10", "published_at":"2025-01-01"},
        11: {"title":"T11", "url":"u11", "authors":"A11", "published_at":"2025-01-02"},
    }
    out = agent.synthesize_with_citations("PROMPT", ev, src)
    assert out == "SYN"
    m = captured["messages"]
    user_msg = [m for m in m if m["role"]=="user"][0]["content"]
    assert "[1] T10" in user_msg and "[2] T11" in user_msg
    assert "C1" in user_msg and "C3" in user_msg

def test_build_references_ordering():
    ord_map = {5:2, 3:1}
    src_meta = {
        3: {"title":"A", "url":"uA", "authors":"AuA", "published_at":"2020-05-01"},
        5: {"title":"B", "url":"uB", "authors":"AuB", "published_at":"2019-01-01"},
    }
    refs = agent.build_references(ord_map, src_meta)
    i1 = refs.find("[1] A")
    i2 = refs.find("[2] B")
    assert i1 != -1 and i2 != -1 and i1 < i2

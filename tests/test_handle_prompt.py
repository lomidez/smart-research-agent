import agent

def test_handle_prompt_local_gate_path(monkeypatch, fake_embed, fake_chat):
    """
    Force the local gate to pass by monkeypatching local_matching_source_ids,
    and stub retrieve/synthesis to avoid networks.
    """
    agent.init_db()
    # Seed a source, so get_sources_by_ids has something real
    with agent.db() as conn:
        sid, _ = agent.store_source(
            conn,
            url="https://example.com/local",
            raw_html="<html><title>Local</title></html>",
            text="AAA " * 500,
            source_type="web",
            hinted_title="Local Title",
            hinted_authors="Alice",
            hinted_date="2025-08-12",
        )

    monkeypatch.setattr(agent, "local_matching_source_ids", lambda *a, **k: {1, 2, 3, 4})

    def _retr(prompt, k=12):
        rows = [
            {"chunk_text":"X1", "source_id": sid, "ordinal": 1},
            {"chunk_text":"X2", "source_id": sid, "ordinal": 1},
        ]
        ordmap = {sid: 1}
        return rows, ordmap
    monkeypatch.setattr(agent, "retrieve_evidence", _retr)

    monkeypatch.setattr(agent, "synthesize_with_citations", lambda *a, **k: "ANSWER_BODY\n")

    out = agent.handle_prompt("test local", max_fetch=3, k_retrieve=2, min_local_sources=4)
    assert "ANSWER_BODY" in out
    assert "## References" in out
    assert ("Local Title" in out) or ("Local — Alice" in out)

def test_handle_prompt_full_web_flow(monkeypatch, fake_embed, fake_chat):
    """
    Force the local gate to fail, plan -> web/arxiv -> fetch -> store -> synthesize.
    All external calls are stubbed to run fast and deterministically.
    """
    agent.init_db()

    monkeypatch.setattr(agent, "local_matching_source_ids", lambda *a, **k: set())

    # 2) planner: fake_chat already returns a JSON with queries; still allow replace if needed
    #    (covered by fake_chat in conftest)

    # 3) web/arxiv search results
    def _w(q, num=3):
        return [
            {"title": f"W-{q}-1", "url": "https://w1.test/a", "snippet":"...", "source_type":"web"},
            {"title": f"W-{q}-2", "url": "https://w1.test/b?utm_source=x", "snippet":"...", "source_type":"web"},
        ]
    def _a(q, max_results=3):
        return [
            {"title": f"A-{q}-1", "url": "https://arxiv.org/abs/9999", "snippet":"...", "authors":"Au", "published":"2025-08-01", "source_type":"arxiv"},
        ]
    monkeypatch.setattr(agent, "web_search_serpapi", _w)
    monkeypatch.setattr(agent, "arxiv_search", _a)

    # 4) fetcher returns HTML or PDF text
    def _fetch(url):
        if url.endswith(".pdf"):
            return ("", "PDF CONTENT " + ("x"*600))
        return ("<html><title>T</title></html>", "HTML CONTENT " + ("y"*600))
    monkeypatch.setattr(agent, "fetch_url", _fetch)

    # 5) indexer as no-op (we don't need Chroma in this flow test)
    calls = {"count": 0}
    def _index(sid, text, vstore):
        calls["count"] += 1
        return True
    monkeypatch.setattr(agent, "ensure_index_and_embed", _index)

    # 6) retrieval + synthesis stubs (avoid vector store completely)
    def _retr(prompt, k=12):
        with agent.db() as conn:
            cur = conn.execute("SELECT id FROM sources ORDER BY id LIMIT 2")
            ids = [r[0] for r in cur.fetchall()]
        rows = []
        ordn = 1
        ordmap = {}
        for sid in ids:
            rows.append({"chunk_text":"DUM", "source_id": sid, "ordinal": ordn})
            ordmap[sid] = ordn
            ordn += 1
        return rows, ordmap

    monkeypatch.setattr(agent, "retrieve_evidence", _retr)
    monkeypatch.setattr(agent, "synthesize_with_citations", lambda *a, **k: "FINAL_BODY\n")

    out = agent.handle_prompt("test web", max_fetch=5, k_retrieve=4, min_local_sources=4)
    assert "FINAL_BODY" in out
    assert "## References" in out
    assert "http" in out or "arxiv" in out
    assert calls["count"] >= 2

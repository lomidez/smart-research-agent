import io
import types
import agent
import numpy as np
import pytest

def test_fetch_url_html_fallback(monkeypatch):
    monkeypatch.setattr(agent.trafilatura, "extract", lambda *a, **k: "")
    class Resp:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        content = b"<html><head><title>T</title></head><body>Hi <b>there</b></body></html>"
        def raise_for_status(self): pass
    class Client:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url): return Resp()
    monkeypatch.setattr(agent.httpx, "Client", Client)
    html, text = agent.fetch_url("https://x.test/page")
    assert "Hi there" in text
    assert "<title" in html.lower()

def test_fetch_url_pdf_success(monkeypatch):
    pdf_bytes = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                 b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
                 b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj\n"
                 b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 72 120 Td (Hello PDF) Tj ET\nendstream endobj\n"
                 b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000062 00000 n \n0000000117 00000 n \n0000000222 00000 n \n"
                 b"trailer<</Root 1 0 R/Size 5>>\nstartxref\n300\n%%EOF")
    class Resp:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        content = pdf_bytes
        def raise_for_status(self): pass
    class Client:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url): return Resp()
    monkeypatch.setattr(agent.httpx, "Client", Client)
    html, text = agent.fetch_url("https://x.test/file.pdf")
    assert html == ""  # PDF branch returns ("", text)
    assert isinstance(text, str)

def test_fetch_url_pdf_parse_error(monkeypatch):
    # Invalid PDF: should raise FetchError
    class Resp:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        content = b"not a real pdf"
        def raise_for_status(self): pass
    class Client:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url): return Resp()
    monkeypatch.setattr(agent.httpx, "Client", Client)
    with pytest.raises(agent.FetchError):
        agent.fetch_url("https://x.test/bad.pdf")

def test_ensure_index_and_embed_dupe_skip(monkeypatch):
    def fake_embed(texts):
        return [np.array([0.1, 0.1, 0.1], dtype="float32")]
    monkeypatch.setattr(agent, "embed_texts", fake_embed)

    class FakeV:
        def query(self, query_embeddings, n_results, include):
            return {
                "metadatas": [[{"source_id": 999}]],
                "distances": [[0.01]],  # sim ~ 0.99
            }
        def add(self, **k): raise AssertionError("should not add on dupe")
    v = FakeV()
    assert agent.ensure_index_and_embed(123, "x"*6000, v) is False

def test_local_matching_source_ids_threshold(monkeypatch):
    def fake_embed(texts):
        return [np.array([0.1, 0.2, 0.3], dtype="float32")]
    monkeypatch.setattr(agent, "embed_texts", fake_embed)

    class FakeV:
        def query(self, query_embeddings, n_results, include):
            return {
                "metadatas": [[{"source_id": 1}, {"source_id": 2}, {"source_id": 3}]],
                "distances": [[0.3, 0.6, 0.2]],  # sim=0.7,0.4,0.8 => keep 1 & 3
            }
    sids = agent.local_matching_source_ids("q", FakeV(), min_sim=0.5, pool_size=10, debug=False)
    assert sids == {1, 3}

def test_ensure_index_and_embed_skips_empty(monkeypatch):
    class DummyVStore:
        def query(self, *a, **k):
            return {"metadatas": [[]], "distances": [[]]}
        def add(self, *a, **k):
            raise AssertionError("should not add when there are no chunks")

    monkeypatch.setattr(agent, "embed_texts", lambda texts: [np.zeros(3, dtype="float32")])

    ok = agent.ensure_index_and_embed(42, "too short to make a 300+ char chunk.", DummyVStore())
    assert ok is True 

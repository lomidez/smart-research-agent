import sqlite3
import agent

def test_init_db_creates_tables():
    agent.init_db()
    with agent.db() as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sources'")
        assert cur.fetchone() is not None

def test_store_source_insert_update_reuse():
    agent.init_db()
    with agent.db() as conn:
        sid1, act1 = agent.store_source(
            conn,
            url="https://example.com/a",
            raw_html="<html><title>One</title></html>",
            text="hello world " * 30,  # non-empty
            source_type="web",
            hinted_title="H One",
        )
        assert act1 == "inserted"
        assert isinstance(sid1, int)

        sid2, act2 = agent.store_source(
            conn,
            url="https://example.com/a?utm_source=x",
            raw_html="<html><title>One</title></html>",
            text="hello world " * 30,
            source_type="web",
        )
        assert sid2 == sid1
        assert act2 == "reused"

        sid3, act3 = agent.store_source(
            conn,
            url="https://example.com/a",
            raw_html="<html><title>One v2</title></html>",
            text="hello world UPDATED",
            source_type="web",
        )

def test_precheck_source_and_get_sources_by_ids():
    agent.init_db()
    with agent.db() as conn:
        sid, _ = agent.store_source(
            conn,
            url="https://example.com/x",
            raw_html="<html><title>X</title></html>",
            text="content x " * 50,
            source_type="web",
            hinted_title="Title X",
            hinted_authors="Author X",
            hinted_date="2025-08-10",
        )
        pre = agent.precheck_source(conn, "https://example.com/x?utm_medium=z")
        assert pre["exists"] and pre["has_text"] and pre["source_id"] == sid

        meta = agent.get_sources_by_ids(conn, [sid])
        assert sid in meta
        assert meta[sid]["title"] in ("X", "Title X")
        assert meta[sid]["url"].startswith("https://")

def test_ensure_index_and_retrieve(fake_embed, monkeypatch):
    class FakeVStore:
        def __init__(self):
            self.docs = []
            self.metas = []
            self.ids = []

        def add(self, ids, embeddings, documents, metadatas):
            self.ids.extend(ids)
            self.docs.extend(documents)
            self.metas.extend(metadatas)

        def query(self, query_embeddings, n_results, include):
            k = min(n_results, len(self.docs))
            return {
                "documents": [self.docs[:k]],
                "metadatas": [self.metas[:k]],
                "distances": [[0.05] * k],
            }

        def get(self, where):
            sids = {m["source_id"] for m in self.metas}
            if where.get("source_id") in sids:
                return {"ids": ["X"]}
            return {"ids": []}

    v = FakeVStore()

    agent.init_db()
    with agent.db() as conn:
        sid, _ = agent.store_source(
            conn,
            url="https://example.com/z",
            raw_html="<html><title>Z</title></html>",
            text=("lorem " * 1000),
            source_type="web",
        )
        ok = agent.ensure_index_and_embed(sid, ("lorem " * 1000), v)
        assert ok is True
        docs, ordmap = agent.retrieve_evidence("what is lorem?", k=5)
        def _gv():
            return v
        monkeypatch.setattr(agent, "get_vstore", _gv)

        docs, ordmap = agent.retrieve_evidence("what is lorem?", k=5)
        assert len(docs) > 0
        assert len(ordmap) > 0
        assert all("chunk_text" in d and "source_id" in d and "ordinal" in d for d in docs)

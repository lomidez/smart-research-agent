import os
import json
import shutil
import tempfile
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def temp_env(tmp_path, monkeypatch):
    """
    Isolate each test with a temp DB, Chroma path, logs, and prompts folder.
    Also sets a fake OPENAI_API_KEY.
    """
    base = tmp_path

    db_path = base / "research.db"
    chroma_path = base / ".chroma"
    log_dir = base / "logs"
    prompts_dir = base / "prompts"

    for p in [chroma_path, log_dir, prompts_dir]:
        p.mkdir(parents=True, exist_ok=True)

    (prompts_dir / "planner_system.txt").write_text("You are a planner.")
    (prompts_dir / "planner_user.txt").write_text('Plan web/arxiv for "{{prompt}}".')
    (prompts_dir / "synthesis_system.txt").write_text("You are a synthesizer.")
    (prompts_dir / "synthesis_user.txt").write_text("Prompt: {{prompt}}\n\nContext:\n{{context_block}}")

    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("CHROMA_PATH", str(chroma_path))
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    monkeypatch.setenv("PROMPTS_DIR", str(prompts_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")  # fake

    import sys, importlib
    if "agent" in sys.modules:
        importlib.reload(sys.modules["agent"])

    yield

@pytest.fixture
def fake_embed(monkeypatch):
    """Monkeypatch embed_texts to return tiny deterministic vectors."""
    import numpy as np
    import agent

    def _fake_embed(texts):
        out = []
        for t in texts:
            n = max(1, len(t))
            v = np.array([min(1.0, n/1000.0), (n % 7) / 7.0, 0.5], dtype="float32")
            out.append(v)
        return out

    monkeypatch.setattr(agent, "embed_texts", _fake_embed)
    return _fake_embed

@pytest.fixture
def fake_chat(monkeypatch):
    """Monkeypatch chat_completion to return predictable JSON/text."""
    import agent

    def _fake_chat(messages, temperature=0.2, max_tokens=700):
        sys = next((m["content"] for m in messages if m["role"]=="system"), "")
        usr = next((m["content"] for m in messages if m["role"]=="user"), "")
        if "planner" in sys.lower() or "planner" in usr.lower():
            return json.dumps({
                "web_queries": ["example topic 2025", "example risks 2024"],
                "arxiv_queries": ["example arxiv 2025"]
            })
        return "SYNTHESIZED_ANSWER"

    monkeypatch.setattr(agent, "chat_completion", _fake_chat)
    return _fake_chat

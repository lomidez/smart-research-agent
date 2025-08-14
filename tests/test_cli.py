import sys
import io
import contextlib
import agent

def test_cli_main_with_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "DB_PATH", str(tmp_path / "research.db"), raising=False)

    monkeypatch.setattr(agent, "handle_prompt", lambda *a, **k: "CLI_OUT")

    # Fake argv for argparse
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xyz")
    argv = ["prog", "--prompt", "what is up", "--max-fetch", "3", "--k-retrieve", "2", "--min-local-sources", "1"]
    monkeypatch.setattr(sys, "argv", argv)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.main()
    out = buf.getvalue()
    assert "CLI_OUT" in out

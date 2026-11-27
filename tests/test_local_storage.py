import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_default_store_keeps_memory_on_disk_with_qdrant_path(tmp_path, monkeypatch):
    from app.main import _default_store

    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "mem"))
    monkeypatch.setenv("EMBEDDER", "hash")

    first = _default_store()
    first.add("asha", "I live in Pune.", key="lives_in")
    first.client.close()  # the folder allows one open client at a time

    second = _default_store()
    assert [m.text for m in second.search("asha", "where do I live")] == ["I live in Pune."]
    assert second.list("ravi") == []
    second.client.close()


def test_qdrant_url_wins_over_qdrant_path(tmp_path, monkeypatch):
    import app.main as main

    seen = {}
    monkeypatch.setenv("QDRANT_URL", "http://example.invalid:6333")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path))
    monkeypatch.setenv("EMBEDDER", "hash")
    monkeypatch.setattr(main, "QdrantClient", lambda **kw: seen.update(kw) or (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        main._default_store()
    assert seen.get("url") == "http://example.invalid:6333" and "path" not in seen


def test_the_mcp_server_keeps_memory_on_disk_across_processes(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ, EMBEDDER="hash", QDRANT_PATH=str(tmp_path / "mem"))
    for name in ("QDRANT_URL", "MCP_API_KEY", "MCP_USER_ID"):
        env.pop(name, None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env, cwd=str(ROOT))

    async def call(tool, args):
        async with stdio_client(params) as (read, write):  # a brand-new server process every time
            async with ClientSession(read, write) as session:
                await session.initialize()
                return json.loads((await session.call_tool(tool, args)).content[0].text)

    async def run():
        await call("remember", {"user_id": "asha", "text": "I live in Pune.", "key": "lives_in"})
        found = await call("recall", {"user_id": "asha", "query": "where do I live"})
        assert [m["text"] for m in found["memories"]] == ["I live in Pune."]

    asyncio.run(run())


def test_importing_app_main_does_not_open_a_store(tmp_path):
    """Importing must have no side effects; `app` (what uvicorn loads) is built on first lookup."""
    import subprocess

    code = (
        "import app.main as m, sys;"
        "assert m._app is None, 'store was opened at import time';"
        "a = m.app;"
        "assert a is m.app and hasattr(a, 'routes')"
    )
    env = dict(os.environ, EMBEDDER="hash", QDRANT_PATH=str(tmp_path / "mem"))
    env.pop("QDRANT_URL", None)
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

"""Tests that need a real Qdrant server. They are skipped unless TEST_QDRANT_URL is set.

    docker run -d -p 6333:6333 qdrant/qdrant:v1.20.0
    TEST_QDRANT_URL=http://127.0.0.1:6333 pytest tests/test_qdrant_server.py

CI starts a Qdrant service container and sets the variable, so these run on every push.
Each test uses its own collection or its own random user ids, so runs do not interfere.
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

URL = os.getenv("TEST_QDRANT_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_QDRANT_URL not set (needs a real Qdrant server)")

ROOT = Path(__file__).resolve().parent.parent


def _store(collection):
    from qdrant_client import QdrantClient

    from app.embeddings import HashEmbedder
    from app.memory import MemoryStore

    return MemoryStore(HashEmbedder(), client=QdrantClient(url=URL), collection=collection)


def test_memories_survive_a_new_store_object_and_stay_per_user():
    collection = f"test_{uuid.uuid4().hex[:12]}"
    first = _store(collection)
    try:
        first.add("asha", "I live in Pune.", key="lives_in")
        first.add("ravi", "I live in Delhi.", key="lives_in")

        second = _store(collection)  # a different object, the same server: nothing is held in process
        assert [m.text for m in second.search("asha", "where do I live")] == ["I live in Pune."]

        _, outcome = second.add("asha", "I live in Mumbai.", key="lives_in")
        assert outcome == "updated"
        assert [m.text for m in first.list("asha")] == ["I live in Mumbai."]
        assert [m.text for m in first.list("ravi")] == ["I live in Delhi."]

        stored = first.list("asha")[0]
        assert second.delete("ravi", stored.id) is False  # another user cannot delete it
        assert first.delete_all("asha") == 1
        assert first.list("asha") == [] and len(first.list("ravi")) == 1
    finally:
        first.client.delete_collection(collection)


def test_the_mcp_server_keeps_memory_in_qdrant_across_processes():
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    user = f"u{uuid.uuid4().hex[:12]}"
    env = dict(os.environ, EMBEDDER="hash", QDRANT_URL=URL)
    for name in ("MCP_API_KEY", "MCP_USER_ID"):
        env.pop(name, None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env, cwd=str(ROOT))

    async def call(tool, args):
        async with stdio_client(params) as (read, write):  # every call starts a brand-new server process
            async with ClientSession(read, write) as session:
                await session.initialize()
                return json.loads((await session.call_tool(tool, args)).content[0].text)

    async def run():
        stored = await call("remember", {"user_id": user, "text": "I live in Pune.", "key": "lives_in"})
        found = await call("recall", {"user_id": user, "query": "where do I live"})
        assert [m["text"] for m in found["memories"]] == ["I live in Pune."]
        assert await call("forget", {"user_id": "someone-else", "memory_id": stored["memory"]["id"]}) == {"deleted": False}
        assert await call("forget", {"user_id": user, "memory_id": stored["memory"]["id"]}) == {"deleted": True}

    asyncio.run(run())

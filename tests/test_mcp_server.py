import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from app.embeddings import HashEmbedder
from app.memory import MemoryStore
from app.mcp_server import forget_memory, recall_memories, remember_memory

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def store():
    from qdrant_client import QdrantClient

    return MemoryStore(HashEmbedder(), client=QdrantClient(":memory:"))


def test_remember_stores_a_fact_and_reports_the_outcome(store):
    result = remember_memory(store, "asha", "I live in Pune.", key="lives_in")
    assert result["outcome"] == "created"
    assert result["memory"]["text"] == "I live in Pune."
    assert result["memory"]["user_id"] == "asha"


def test_remember_replaces_a_fact_with_the_same_key(store):
    remember_memory(store, "asha", "I live in Pune.", key="lives_in")
    result = remember_memory(store, "asha", "I live in Delhi.", key="lives_in")
    assert result["outcome"] == "updated"
    assert result["memory"]["previous_text"] == "I live in Pune."
    assert len(store.list("asha")) == 1


def test_remember_returns_a_validation_error_as_data(store):
    result = remember_memory(store, "has space", "x")
    assert "error" in result and "user_id" in result["error"]


def test_recall_finds_the_fact_for_its_own_user_only(store):
    remember_memory(store, "asha", "I like mangoes.")
    remember_memory(store, "ravi", "I like mangoes too.")
    result = recall_memories(store, "asha", "mangoes")
    texts = [m["text"] for m in result["memories"]]
    assert texts == ["I like mangoes."]


def test_recall_clamps_the_limit(store):
    for i in range(30):
        remember_memory(store, "asha", f"fact number {i} about cats")
    assert len(recall_memories(store, "asha", "cats", limit=1000)["memories"]) == 20
    assert len(recall_memories(store, "asha", "cats", limit=0)["memories"]) == 1


def test_forget_deletes_only_the_users_own_memory(store):
    stored = remember_memory(store, "asha", "I like tea.")["memory"]
    assert forget_memory(store, "ravi", stored["id"]) == {"deleted": False}
    assert len(store.list("asha")) == 1
    assert forget_memory(store, "asha", stored["id"]) == {"deleted": True}
    assert store.list("asha") == []


def test_the_server_registers_the_three_tools():
    pytest.importorskip("mcp")
    from app.mcp_server import create_server

    server = create_server(store=MemoryStore(HashEmbedder()))
    names = sorted(tool.name for tool in server._tool_manager.list_tools())
    assert names == ["forget", "recall", "remember"]


def test_a_client_can_remember_and_recall_over_stdio():
    """Start the real server as a subprocess and call it the way an MCP client does."""
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ, EMBEDDER="hash")
    env.pop("QDRANT_URL", None)  # in-memory store for this test
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env, cwd=str(ROOT))

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                assert tools == {"remember", "recall", "forget"}

                await session.call_tool("remember", {"user_id": "asha", "text": "I live in Pune.", "key": "lives_in"})
                result = await session.call_tool("recall", {"user_id": "asha", "query": "Pune"})
                found = json.loads(result.content[0].text)
                assert any("Pune" in m["text"] for m in found["memories"])

    asyncio.run(run())

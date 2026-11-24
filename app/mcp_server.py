"""Expose the memory store to AI agents over the Model Context Protocol (MCP).

Three tools: `remember`, `recall` and `forget`. They call the same `MemoryStore` as the REST API,
so memories written through one are visible through the other.

Unlike `/chat`, these tools do not call a language model. The calling assistant decides what is
worth remembering and passes the text in, so this server needs no API key and costs nothing to run.

Run it over stdio (what MCP clients such as Claude Desktop launch):

    python -m app.mcp_server

It uses the same environment variables as the web app: QDRANT_URL, QDRANT_API_KEY and EMBEDDER.
Without QDRANT_URL, memory lives in-process and is lost when the server stops.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.main import RECALL_THRESHOLD, _default_store
from app.memory import MemoryStore

MAX_RECALL_RESULTS = 20
DEFAULT_RECALL_RESULTS = 5


def remember_memory(store: MemoryStore, user_id: str, text: str, key: Optional[str] = None) -> dict:
    """Store one fact for a user. Errors are returned as data so the client can show them."""
    try:
        memory, outcome = store.add(user_id, text, key=key)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"outcome": outcome, "memory": _public(memory)}


def recall_memories(store: MemoryStore, user_id: str, query: str, limit: int = DEFAULT_RECALL_RESULTS) -> dict:
    """Find the user's stored facts most similar to the query, best match first."""
    limit = max(1, min(int(limit), MAX_RECALL_RESULTS))
    try:
        found = store.search(user_id, query, limit=limit, threshold=RECALL_THRESHOLD)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"memories": [_public(m) for m in found]}


def forget_memory(store: MemoryStore, user_id: str, memory_id: str) -> dict:
    """Delete one memory. Only that user's memories can be deleted; others return deleted=false."""
    try:
        deleted = store.delete(user_id, memory_id)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"deleted": deleted}


def _public(memory) -> dict:
    data = asdict(memory)
    return {k: v for k, v in data.items() if v is not None}


def create_server(store: Optional[MemoryStore] = None) -> FastMCP:
    """Build the MCP server. Pass a store to test it; otherwise the environment decides."""
    store = store or _default_store()
    server = FastMCP("memory-assistant")

    @server.tool()
    def remember(user_id: str, text: str, key: Optional[str] = None) -> dict:
        """Remember a fact about a user, e.g. text="I live in Pune", key="lives_in".

        Use `key` for a fact that has only one current value (where they live, their name, their job):
        a new value for the same key replaces the old one. Leave `key` empty for facts that can
        accumulate, such as things they like. Store only what the user has said about themselves.
        """
        return remember_memory(store, user_id, text, key)

    @server.tool()
    def recall(user_id: str, query: str, limit: int = DEFAULT_RECALL_RESULTS) -> dict:
        """Search what is remembered about a user for facts relevant to the query."""
        return recall_memories(store, user_id, query, limit)

    @server.tool()
    def forget(user_id: str, memory_id: str) -> dict:
        """Delete one remembered fact by its id (returned by remember and recall)."""
        return forget_memory(store, user_id, memory_id)

    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()

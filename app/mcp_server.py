"""Expose the memory store to AI agents over the Model Context Protocol (MCP).

Three tools: `remember`, `recall` and `forget`. They call the same `MemoryStore` as the REST API,
so memories written through one are visible through the other.

Unlike `/chat`, these tools do not call a language model. The calling assistant decides what is
worth remembering and passes the text in, so this server needs no API key and costs nothing to run.

Run it over stdio (what MCP clients such as Claude Desktop launch):

    python -m app.mcp_server

It uses the same environment variables as the web app: QDRANT_URL, QDRANT_API_KEY and EMBEDDER.
Without QDRANT_URL, memory lives in-process and is lost when the server stops.

Over HTTP (opt-in, for a server other machines can reach):

    MCP_API_KEY=... python -m app.mcp_server --http [--host 127.0.0.1] [--port 8000]

HTTP mode refuses to start without MCP_API_KEY, and every request must carry
`Authorization: Bearer <key>`. It listens on 127.0.0.1 unless you pass --host. Set MCP_USER_ID to
pin every tool call to one user, so a caller cannot read or delete anyone else's memories.
TLS is not handled here: put a reverse proxy in front before exposing it publicly.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
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


class BearerAuth:
    """ASGI middleware: reject HTTP requests that lack `Authorization: Bearer <key>` with 401."""

    def __init__(self, app, key: str):
        if not key:
            raise ValueError("an empty key would let everyone in")
        self.app = app
        self.key = key.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":  # startup and shutdown carry no request to check
            return await self.app(scope, receive, send)
        if scope["type"] != "http":  # anything else (a websocket) is not served at all
            await send({"type": "websocket.close", "code": 1008})
            return
        header = dict(scope["headers"]).get(b"authorization", b"")
        scheme, _, token = header.partition(b" ")
        if scheme.lower() == b"bearer" and secrets.compare_digest(token, self.key):
            return await self.app(scope, receive, send)
        body = json.dumps({"error": "unauthorized"}).encode()
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"www-authenticate", b"Bearer"),
        ]})
        await send({"type": "http.response.body", "body": body})


def create_server(store: Optional[MemoryStore] = None, pinned_user: Optional[str] = None) -> FastMCP:
    """Build the MCP server. Pass a store to test it; otherwise the environment decides.

    With `pinned_user`, every tool acts as that user and the caller's `user_id` is ignored.
    """
    store = store or _default_store()
    server = FastMCP("memory-assistant")

    def who(user_id: str) -> str:
        return pinned_user or user_id

    @server.tool()
    def remember(user_id: str, text: str, key: Optional[str] = None) -> dict:
        """Remember a fact about a user, e.g. text="I live in Pune", key="lives_in".

        Use `key` for a fact that has only one current value (where they live, their name, their job):
        a new value for the same key replaces the old one. Leave `key` empty for facts that can
        accumulate, such as things they like. Store only what the user has said about themselves.
        """
        return remember_memory(store, who(user_id), text, key)

    @server.tool()
    def recall(user_id: str, query: str, limit: int = DEFAULT_RECALL_RESULTS) -> dict:
        """Search what is remembered about a user for facts relevant to the query."""
        return recall_memories(store, who(user_id), query, limit)

    @server.tool()
    def forget(user_id: str, memory_id: str) -> dict:
        """Delete one remembered fact by its id (returned by remember and recall)."""
        return forget_memory(store, who(user_id), memory_id)

    return server


def build_http_app(server: FastMCP, key: str):
    """The server's streamable-HTTP app, wrapped so that every request needs the key."""
    return BearerAuth(server.streamable_http_app(), key)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Memory assistant MCP server (stdio by default).")
    parser.add_argument("--http", action="store_true", help="serve over HTTP; requires MCP_API_KEY")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    key = os.getenv("MCP_API_KEY", "")
    if args.http and not key:
        raise SystemExit("MCP_API_KEY is not set: refusing to serve HTTP without authentication.")
    server = create_server(pinned_user=os.getenv("MCP_USER_ID") or None)
    if not args.http:
        server.run()
        return
    import uvicorn

    uvicorn.run(build_http_app(server, key), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

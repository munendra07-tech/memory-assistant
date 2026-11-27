import asyncio
import json
import socket
import threading
import time

import pytest

pytest.importorskip("mcp")

from app.embeddings import HashEmbedder
from app.memory import MemoryStore
from app.mcp_server import BearerAuth, build_http_app, create_server, main

KEY = "test-key-not-a-real-secret"


def _store():
    from qdrant_client import QdrantClient

    return MemoryStore(HashEmbedder(), client=QdrantClient(":memory:"))


async def _call(app, headers):
    """Send one HTTP request through an ASGI app and return (status, body)."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app({"type": "http", "headers": headers}, receive, send)
    return sent[0]["status"], sent[1]["body"]


async def _inner(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.parametrize("headers", [
    [],                                             # no header
    [(b"authorization", b"Bearer wrong")],          # wrong key
    [(b"authorization", KEY.encode())],             # no scheme
    [(b"authorization", b"Basic " + KEY.encode())], # wrong scheme
    [(b"authorization", b"Bearer ")],               # empty token
])
def test_requests_without_the_right_key_get_401(headers):
    status, body = asyncio.run(_call(BearerAuth(_inner, KEY), headers))
    assert status == 401
    assert json.loads(body) == {"error": "unauthorized"}


def test_the_right_key_passes_through():
    status, body = asyncio.run(_call(BearerAuth(_inner, KEY), [(b"authorization", f"Bearer {KEY}".encode())]))
    assert (status, body) == (200, b"ok")


def test_an_empty_key_is_refused_at_construction():
    with pytest.raises(ValueError):
        BearerAuth(_inner, "")


def test_http_mode_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.delenv("MCP_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDER", "hash")
    with pytest.raises(SystemExit) as exc:
        main(["--http"])
    assert "MCP_API_KEY" in str(exc.value)


def test_a_pinned_user_overrides_the_callers_user_id():
    server = create_server(store=_store(), pinned_user="asha")

    async def run():
        await server.call_tool("remember", {"user_id": "mallory", "text": "I like tea."})
        mine = await server.call_tool("recall", {"user_id": "ravi", "query": "tea"})
        return mine

    result = asyncio.run(run())
    text = json.dumps(result, default=str)
    assert "I like tea." in text and "asha" in text and "mallory" not in text


def test_without_pinning_the_callers_user_id_is_used():
    store = _store()
    server = create_server(store=store)
    asyncio.run(server.call_tool("remember", {"user_id": "ravi", "text": "I like tea."}))
    assert [m.text for m in store.list("ravi")] == ["I like tea."]
    assert store.list("asha") == []


def _serve(app):
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            return server, thread, port
        time.sleep(0.05)
    raise RuntimeError("test server did not start")


def test_a_real_client_needs_the_key_over_http():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    store = _store()
    server, thread, port = _serve(build_http_app(create_server(store=store, pinned_user="asha"), KEY))
    url = f"http://127.0.0.1:{port}/mcp"

    async def with_key():
        async with streamablehttp_client(url, headers={"Authorization": f"Bearer {KEY}"}) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                await session.call_tool("remember", {"user_id": "someone-else", "text": "I live in Pune."})

    async def without_key():
        async with streamablehttp_client(url) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()

    try:
        asyncio.run(with_key())
        assert [m.text for m in store.list("asha")] == ["I live in Pune."]  # pinned, not "someone-else"
        import httpx

        assert httpx.post(url, json={}).status_code == 401
        assert httpx.post(url, json={}, headers={"Authorization": "Bearer nope"}).status_code == 401
        with pytest.raises(BaseException):  # the client cannot open a session without the key
            asyncio.run(without_key())
    finally:
        server.should_exit = True
        thread.join(timeout=5)

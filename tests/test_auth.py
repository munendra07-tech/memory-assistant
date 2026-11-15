import pytest
from fastapi.testclient import TestClient

from app.embeddings import HashEmbedder
from app.llm import EchoLLM
from app.main import create_app
from app.memory import MemoryStore

KEY = "s3cret-key"


def make_client(api_key):
    return TestClient(create_app(store=MemoryStore(HashEmbedder()), llm=EchoLLM(), api_key=api_key))


CHAT = {"user_id": "u1", "message": "I live in Pune."}


def test_health_is_open_even_when_a_key_is_configured():
    assert make_client(KEY).get("/health").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_requests_without_a_valid_key_are_rejected(headers):
    client = make_client(KEY)
    assert client.post("/chat", json=CHAT, headers=headers).status_code == 401
    assert client.get("/memories/u1", headers=headers).status_code == 401
    assert client.delete("/memories/u1/00000000-0000-0000-0000-000000000000", headers=headers).status_code == 401
    assert client.delete("/memories/u1", headers=headers).status_code == 401


def test_a_rejected_request_stores_nothing():
    client = make_client(KEY)
    client.post("/chat", json=CHAT, headers={"X-API-Key": "wrong"})
    assert client.get("/memories/u1", headers={"X-API-Key": KEY}).json() == []


def test_requests_with_the_right_key_work():
    client = make_client(KEY)
    response = client.post("/chat", json=CHAT, headers={"X-API-Key": KEY})
    assert response.status_code == 200
    assert client.get("/memories/u1", headers={"X-API-Key": KEY}).status_code == 200


def test_with_no_key_configured_access_is_open():
    assert make_client(None).post("/chat", json=CHAT).status_code == 200

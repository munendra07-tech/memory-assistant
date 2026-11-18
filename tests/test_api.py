import pytest
from fastapi.testclient import TestClient

from app.embeddings import HashEmbedder
from app.llm import EchoLLM
from app.main import create_app
from app.memory import MemoryStore


@pytest.fixture
def client():
    app = create_app(store=MemoryStore(HashEmbedder()), llm=EchoLLM())
    return TestClient(app)


def test_chat_stores_a_fact_then_recalls_it_next_turn(client):
    first = client.post("/chat", json={"user_id": "u1", "message": "I live in Pune."}).json()
    assert [m["text"] for m in first["memories_stored"]] == ["User lives in Pune"]
    assert first["memories_used"] == []  # nothing known before this message

    second = client.post("/chat", json={"user_id": "u1", "message": "Which city does the user live in?"}).json()
    assert [m["text"] for m in second["memories_used"]] == ["User lives in Pune"]


def test_chat_does_not_leak_between_users(client):
    client.post("/chat", json={"user_id": "u1", "message": "I live in Mumbai."})
    other = client.post("/chat", json={"user_id": "u2", "message": "Which city does the user live in?"}).json()
    assert other["memories_used"] == []


def test_list_and_delete_memories(client):
    client.post("/chat", json={"user_id": "u1", "message": "I love tea."})
    memories = client.get("/memories/u1").json()
    assert [m["text"] for m in memories] == ["User loves tea"]

    assert client.delete(f"/memories/u2/{memories[0]['id']}").status_code == 404  # not the owner
    assert client.delete(f"/memories/u1/{memories[0]['id']}").status_code == 200
    assert client.get("/memories/u1").json() == []


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": "", "message": "hi"},
        {"user_id": "has space", "message": "hi"},
        {"user_id": "u1", "message": ""},
        {"user_id": "u1", "message": "x" * 4001},
        {"user_id": "u1"},
    ],
)
def test_chat_rejects_invalid_input(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_language_model_failure_returns_502_and_stores_nothing():
    class BrokenLLM:
        def complete(self, system, user):
            raise RuntimeError("provider down")

    app = create_app(store=MemoryStore(HashEmbedder()), llm=BrokenLLM())
    client = TestClient(app)
    response = client.post("/chat", json={"user_id": "u1", "message": "I live in Pune."})
    assert response.status_code == 502
    assert client.get("/memories/u1").json() == []


def test_chat_replaces_a_changed_fact_and_reports_the_update(client):
    client.post("/chat", json={"user_id": "u1", "message": "I live in Pune."})
    changed = client.post("/chat", json={"user_id": "u1", "message": "I live in Delhi."}).json()
    assert changed["memories_stored"] == []
    assert [(m["text"], m["previous_text"]) for m in changed["memories_updated"]] == [
        ("User lives in Delhi", "User lives in Pune")
    ]
    assert [m["text"] for m in client.get("/memories/u1").json()] == ["User lives in Delhi"]


def test_repeating_the_same_fact_changes_nothing(client):
    client.post("/chat", json={"user_id": "u1", "message": "I live in Pune."})
    again = client.post("/chat", json={"user_id": "u1", "message": "I live in Pune."}).json()
    assert again["memories_stored"] == [] and again["memories_updated"] == []


def test_delete_all_memories_for_a_user_only(client):
    client.post("/chat", json={"user_id": "u1", "message": "I live in Pune. I love tea."})
    client.post("/chat", json={"user_id": "u2", "message": "I live in Delhi."})
    assert client.delete("/memories/u1").json() == {"deleted": 2}
    assert client.get("/memories/u1").json() == []
    assert len(client.get("/memories/u2").json()) == 1


def test_a_failed_fact_extraction_returns_502_and_stores_nothing():
    class BrokenExtractor:
        def extract(self, message):
            raise RuntimeError("model unavailable")

    app = create_app(store=MemoryStore(HashEmbedder()), llm=EchoLLM(), extractor=BrokenExtractor())
    client = TestClient(app)
    response = client.post("/chat", json={"user_id": "u1", "message": "I live in Pune."})
    assert response.status_code == 502
    assert "extraction failed" in response.json()["detail"]
    assert client.get("/memories/u1").json() == []


def test_llm_extracted_keys_make_a_changed_fact_replace_the_old_one():
    from app.extract import LLMFactExtractor

    class ScriptedModel:
        """Stands in for a language model: returns a keyed fact for each message."""

        def complete(self, system, user):
            city = "Delhi" if "Delhi" in user else "Pune"
            return '[{"text": "User lives in %s", "key": "lives_in"}]' % city

    app = create_app(store=MemoryStore(HashEmbedder()), llm=EchoLLM(), extractor=LLMFactExtractor(ScriptedModel()))
    client = TestClient(app)
    client.post("/chat", json={"user_id": "u1", "message": "I moved to Pune"})
    changed = client.post("/chat", json={"user_id": "u1", "message": "Now I am based in Delhi"}).json()
    assert [(m["text"], m["previous_text"]) for m in changed["memories_updated"]] == [
        ("User lives in Delhi", "User lives in Pune")
    ]
    assert [m["text"] for m in client.get("/memories/u1").json()] == ["User lives in Delhi"]

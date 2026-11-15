import pytest
from fastapi.testclient import TestClient

from app.embeddings import HashEmbedder
from app.llm import EchoLLM
from app.main import create_app
from app.memory import MemoryStore
from app.ratelimit import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_blocks():
    limiter = RateLimiter(3, 60, clock=FakeClock())
    assert [limiter.check("a") for _ in range(3)] == [None, None, None]
    wait = limiter.check("a")
    assert wait is not None and 0 < wait <= 60


def test_clients_are_counted_separately():
    limiter = RateLimiter(1, 60, clock=FakeClock())
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("a") is not None


def test_calls_are_allowed_again_after_the_window():
    clock = FakeClock()
    limiter = RateLimiter(1, 60, clock=clock)
    assert limiter.check("a") is None
    assert limiter.check("a") is not None
    clock.now += 60
    assert limiter.check("a") is None


def test_wait_time_shrinks_as_time_passes():
    clock = FakeClock()
    limiter = RateLimiter(1, 60, clock=clock)
    limiter.check("a")
    first = limiter.check("a")
    clock.now += 20
    assert limiter.check("a") == pytest.approx(first - 20)


def test_zero_disables_limiting():
    limiter = RateLimiter(0, 60, clock=FakeClock())
    assert all(limiter.check("a") is None for _ in range(1000))


def test_idle_clients_are_forgotten():
    clock = FakeClock()
    limiter = RateLimiter(5, 60, clock=clock)
    limiter.check("a")
    clock.now += 61
    limiter.forget_idle()
    assert limiter._calls == {}


@pytest.mark.parametrize("limit, window", [(-1, 60), (1, 0), (1, -5)])
def test_invalid_settings_are_rejected(limit, window):
    with pytest.raises(ValueError):
        RateLimiter(limit, window)


def make_client(limit, api_key=None):
    app = create_app(store=MemoryStore(HashEmbedder()), llm=EchoLLM(), api_key=api_key, rate_limit_per_minute=limit)
    return TestClient(app)


def test_chat_returns_429_with_retry_after_when_over_the_limit():
    client = make_client(2)
    body = {"user_id": "u1", "message": "hello there"}
    assert client.post("/chat", json=body).status_code == 200
    assert client.post("/chat", json=body).status_code == 200
    blocked = client.post("/chat", json=body)
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1


def test_health_and_memory_reads_are_not_rate_limited():
    client = make_client(1)
    client.post("/chat", json={"user_id": "u1", "message": "hello there"})
    for _ in range(5):
        assert client.get("/health").status_code == 200
        assert client.get("/memories/u1").status_code == 200


def test_unauthenticated_calls_do_not_use_up_the_limit():
    client = make_client(1, api_key="k")
    body = {"user_id": "u1", "message": "hello there"}
    for _ in range(5):
        assert client.post("/chat", json=body).status_code == 401  # rejected before counting
    assert client.post("/chat", json=body, headers={"X-API-Key": "k"}).status_code == 200

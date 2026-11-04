import pytest
from fastapi.testclient import TestClient

from src import api, rag
from src.rate_limit import RateLimiter


@pytest.fixture
def client(monkeypatch):
    # Skip the startup warm-up (model download, Cosmos DB) and reset the limiter.
    monkeypatch.setattr(api.app.router, "lifespan_context", _no_lifespan)
    monkeypatch.setattr(api, "ask_limiter", RateLimiter(max_requests=3, window_seconds=60))
    with TestClient(api.app) as c:
        yield c


class _no_lifespan:
    def __init__(self, app):
        pass

    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def fake_answer(question):
    return {"answer": f"echo: {question}", "sources": [{"source": "about.txt", "chunk_id": 0}]}


def test_root_reports_status(client):
    assert client.get("/").json() == {"status": "Ask Bach API is running"}


def test_ask_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(rag, "answer_question", fake_answer)
    response = client.post("/ask", json={"question": "  Who is Bach?  "})
    assert response.status_code == 200
    assert response.json()["answer"] == "echo: Who is Bach?"  # whitespace stripped


@pytest.mark.parametrize("question", ["", "    ", "x" * 501])
def test_ask_rejects_empty_or_oversized_questions(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_ask_returns_503_with_fallback_when_upstream_fails(client, monkeypatch):
    def boom(question):
        raise ConnectionError("Groq is down")

    monkeypatch.setattr(rag, "answer_question", boom)
    response = client.post("/ask", json={"question": "Who is Bach?"})
    assert response.status_code == 503
    body = response.json()
    assert body["answer"] == api.FALLBACK_ANSWER
    assert body["sources"] == []


def test_ask_is_rate_limited_per_client_ip(client, monkeypatch):
    monkeypatch.setattr(rag, "answer_question", fake_answer)
    alice = {"X-Forwarded-For": "203.0.113.1, 10.0.0.1"}
    for _ in range(3):
        assert client.post("/ask", json={"question": "hi"}, headers=alice).status_code == 200
    blocked = client.post("/ask", json={"question": "hi"}, headers=alice)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    # a different visitor is unaffected
    bob = {"X-Forwarded-For": "198.51.100.7"}
    assert client.post("/ask", json={"question": "hi"}, headers=bob).status_code == 200


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = RateLimiter(max_requests=2, window_seconds=60, clock=lambda: now[0])
    assert limiter.hit("ip")[0] and limiter.hit("ip")[0]
    assert limiter.hit("ip") == (False, 60)
    now[0] = 60.0
    assert limiter.hit("ip") == (True, 0)

"""FastAPI app: chat with per-user long-term memory."""

from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.embeddings import embedder_from_name
from app.extract import FactExtractor, RuleFactExtractor
from app.llm import LLM, llm_from_env
from app.memory import CREATED, UPDATED, MemoryStore
from app.ratelimit import RateLimiter

USER_ID_PATTERN = r"^\S+$"
RECALL_LIMIT = 5
RECALL_THRESHOLD = 0.1

SYSTEM_PROMPT = (
    "You are a helpful assistant with long-term memory of the user. "
    "Use the remembered facts below only when they are relevant. "
    "If the facts do not answer the question, say you do not know.\n\nRemembered facts:\n{facts}"
)


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=USER_ID_PATTERN)
    message: str = Field(min_length=1, max_length=4000)


class MemoryOut(BaseModel):
    id: str
    text: str
    created_at: str
    score: Optional[float] = None
    previous_text: Optional[str] = None
    updated_at: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    memories_used: list[MemoryOut]
    memories_stored: list[MemoryOut]
    memories_updated: list[MemoryOut]


def _default_store() -> MemoryStore:
    url = os.getenv("QDRANT_URL")
    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY")) if url else QdrantClient(":memory:")
    return MemoryStore(embedder_from_name(os.getenv("EMBEDDER", "hash")), client=client)


def create_app(
    store: Optional[MemoryStore] = None,
    extractor: Optional[FactExtractor] = None,
    llm: Optional[LLM] = None,
    api_key: Optional[str] = None,
    rate_limit_per_minute: Optional[int] = None,
) -> FastAPI:
    store = store or _default_store()
    extractor = extractor or RuleFactExtractor()
    llm = llm or llm_from_env()
    api_key = api_key if api_key is not None else os.getenv("API_KEY")
    if rate_limit_per_minute is None:
        rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    limiter = RateLimiter(rate_limit_per_minute)

    def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        """If an API key is configured, every memory and chat route needs a matching X-API-Key."""
        if not api_key:
            return  # no key configured: open access, fine for local development only
        if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    def rate_limited(request: Request) -> None:
        """Applied after authentication, so only callers with a valid key (or any caller, if no
        key is configured) are counted. Clients are told apart by network address."""
        client_id = request.client.host if request.client else "unknown"
        wait = limiter.check(client_id)
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail="too many requests",
                headers={"Retry-After": str(int(wait) + 1)},
            )

    app = FastAPI(title="Memory Assistant")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key), Depends(rate_limited)])
    def chat(req: ChatRequest) -> ChatResponse:
        # Recall first, so the answer uses what was known before this message.
        recalled = store.search(req.user_id, req.message, limit=RECALL_LIMIT, threshold=RECALL_THRESHOLD)
        facts = "\n".join(f"- {m.text}" for m in recalled) or "(nothing yet)"
        try:
            reply = llm.complete(SYSTEM_PROMPT.format(facts=facts), req.message)
        except Exception as exc:  # network and provider errors must not look like a server bug
            raise HTTPException(status_code=502, detail=f"language model call failed: {exc}") from exc

        stored, updated = [], []
        for fact in extractor.extract(req.message):
            memory, outcome = store.add(req.user_id, fact.text, key=fact.key)
            if outcome == CREATED:
                stored.append(memory)
            elif outcome == UPDATED:
                updated.append(memory)
        return ChatResponse(
            reply=reply,
            memories_used=[MemoryOut(**m.__dict__) for m in recalled],
            memories_stored=[MemoryOut(**m.__dict__) for m in stored],
            memories_updated=[MemoryOut(**m.__dict__) for m in updated],
        )

    @app.get("/memories/{user_id}", response_model=list[MemoryOut], dependencies=[Depends(require_api_key)])
    def list_memories(user_id: str) -> list[MemoryOut]:
        try:
            return [MemoryOut(**m.__dict__) for m in store.list(user_id)]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/memories/{user_id}", dependencies=[Depends(require_api_key)])
    def delete_all_memories(user_id: str) -> dict:
        """Erase everything remembered about one user."""
        try:
            return {"deleted": store.delete_all(user_id)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/memories/{user_id}/{memory_id}", dependencies=[Depends(require_api_key)])
    def delete_memory(user_id: str, memory_id: str) -> dict:
        try:
            deleted = store.delete(user_id, memory_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="memory not found")
        return {"deleted": memory_id}

    return app


app = create_app()

"""Per-user long-term memory stored in Qdrant.

Every memory is one Qdrant point: the vector is the embedding of the text, and the payload
holds `user_id`, `text`, `created_at`, and optionally `key` and `previous_text`. Every read
and write is filtered by `user_id`, so one user can never see or delete another user's memories.

A memory added with a `key` is single-valued: adding another value for the same key replaces
the old one in place (same id, original `created_at` kept, old text kept as `previous_text`).
"""

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from qdrant_client import QdrantClient, models

from app.embeddings import Embedder

MAX_TEXT_LENGTH = 2000
MAX_KEY_LENGTH = 64

# Outcomes of MemoryStore.add().
CREATED = "created"
UPDATED = "updated"
UNCHANGED = "unchanged"


@dataclass
class Memory:
    id: str
    user_id: str
    text: str
    created_at: str
    score: Optional[float] = None
    previous_text: Optional[str] = None
    updated_at: Optional[str] = None


def _validate_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    user_id = user_id.strip()
    if any(c.isspace() for c in user_id):
        raise ValueError("user_id must not contain whitespace")
    return user_id


def _validate_key(key: Optional[str]) -> Optional[str]:
    if key is None:
        return None
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be a non-empty string or None")
    key = key.strip().lower()
    if len(key) > MAX_KEY_LENGTH:
        raise ValueError(f"key must be at most {MAX_KEY_LENGTH} characters")
    return key


def _field(name: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=name, match=models.MatchValue(value=value))


def _user_filter(user_id: str) -> models.Filter:
    return models.Filter(must=[_field("user_id", user_id)])


class MemoryStore:
    def __init__(
        self,
        embedder: Embedder,
        client: Optional[QdrantClient] = None,
        collection: str = "memories",
        dedupe_threshold: float = 0.97,
    ) -> None:
        self.embedder = embedder
        self.client = client or QdrantClient(":memory:")
        self.collection = collection
        self.dedupe_threshold = dedupe_threshold
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
            )
            # The indexes speed up the per-user and per-key filters on a real Qdrant server.
            # Local (in-memory) mode ignores them and warns, so the warning is silenced here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for field_name in ("user_id", "key"):
                    self.client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )

    @staticmethod
    def _to_memory(point) -> Memory:
        payload = point.payload or {}
        return Memory(
            id=str(point.id),
            user_id=payload.get("user_id", ""),
            text=payload.get("text", ""),
            created_at=payload.get("created_at", ""),
            score=getattr(point, "score", None),
            previous_text=payload.get("previous_text"),
            updated_at=payload.get("updated_at"),
        )

    def _find_by_key(self, user_id: str, key: str):
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(must=[_field("user_id", user_id), _field("key", key)]),
            limit=1,
            with_payload=True,
        )
        return points[0] if points else None

    def add(self, user_id: str, text: str, key: Optional[str] = None) -> tuple[Memory, str]:
        """Store a memory. Returns (memory, outcome) where outcome is CREATED, UPDATED or UNCHANGED.

        With a `key`, an existing memory of this user with the same key is replaced in place.
        Without one, a near-identical existing memory (cosine >= dedupe_threshold) is not stored twice.
        """
        user_id = _validate_user_id(user_id)
        key = _validate_key(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        text = text.strip()
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"text must be at most {MAX_TEXT_LENGTH} characters")

        vector = self.embedder.embed(text)
        now = datetime.now(timezone.utc).isoformat()

        if key is not None:
            existing = self._find_by_key(user_id, key)
            if existing is not None:
                old = existing.payload or {}
                if old.get("text") == text:
                    return self._to_memory(existing), UNCHANGED
                payload = {
                    "user_id": user_id,
                    "key": key,
                    "text": text,
                    "created_at": old.get("created_at", now),
                    "updated_at": now,
                    "previous_text": old.get("text"),
                }
                self.client.upsert(
                    collection_name=self.collection,
                    points=[models.PointStruct(id=existing.id, vector=vector, payload=payload)],
                )
                return Memory(id=str(existing.id), user_id=user_id, text=text, created_at=payload["created_at"],
                              previous_text=payload["previous_text"], updated_at=now), UPDATED
        else:
            nearest = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=_user_filter(user_id),
                limit=1,
                with_payload=True,
            ).points
            if nearest and nearest[0].score >= self.dedupe_threshold:
                return self._to_memory(nearest[0]), UNCHANGED

        memory_id = str(uuid.uuid4())
        payload = {"user_id": user_id, "text": text, "created_at": now}
        if key is not None:
            payload["key"] = key
        self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=memory_id, vector=vector, payload=payload)],
        )
        return Memory(id=memory_id, user_id=user_id, text=text, created_at=now), CREATED

    def search(self, user_id: str, query: str, limit: int = 5, threshold: float = 0.0) -> list[Memory]:
        user_id = _validate_user_id(user_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        points = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.embed(query),
            query_filter=_user_filter(user_id),
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        ).points
        return [self._to_memory(p) for p in points]

    def list(self, user_id: str, limit: int = 100) -> list[Memory]:
        user_id = _validate_user_id(user_id)
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=_user_filter(user_id),
            limit=limit,
            with_payload=True,
        )
        return sorted((self._to_memory(p) for p in points), key=lambda m: m.created_at)

    def delete(self, user_id: str, memory_id: str) -> bool:
        """Delete one memory. Returns False if it does not exist or belongs to someone else."""
        user_id = _validate_user_id(user_id)
        try:
            uuid.UUID(memory_id)
        except (ValueError, AttributeError, TypeError):
            return False
        found = self.client.retrieve(collection_name=self.collection, ids=[memory_id], with_payload=True)
        if not found or (found[0].payload or {}).get("user_id") != user_id:
            return False
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[memory_id]),
        )
        return True

    def delete_all(self, user_id: str) -> int:
        """Delete every memory of one user. Returns how many were removed."""
        user_id = _validate_user_id(user_id)
        flt = _user_filter(user_id)
        count = self.client.count(collection_name=self.collection, count_filter=flt, exact=True).count
        if count:
            self.client.delete(collection_name=self.collection, points_selector=models.FilterSelector(filter=flt))
        return count

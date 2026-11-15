"""Text embedders.

`HashEmbedder` is a deterministic, offline baseline: it hashes words into a fixed-size
vector (the "hashing trick"). It captures word overlap only, not meaning. It exists so the
tests and evals run without a network or API key, and so a real embedding model can later
be compared against it.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "to", "of", "in", "on", "at",
    "for", "and", "or", "do", "does", "did", "what", "which", "who", "where", "when", "how",
    "my", "me", "i", "you", "your", "user", "users", "it", "that", "this", "with", "about",
}


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    """Lowercase words, minus stopwords, with a naive plural strip ("cats" -> "cat")."""
    tokens = []
    for word in re.findall(r"[a-z0-9']+", text.lower()):
        word = word.replace("'s", "").strip("'")
        if not word or word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


class HashEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
            sign = 1.0 if (digest >> 64) & 1 else -1.0  # sign hashing reduces collision bias
            vec[digest % self.dim] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0  # avoid a zero vector, which has no defined cosine similarity
            return vec
        return [v / norm for v in vec]


class FastEmbedEmbedder:
    """Local embedding model via the optional `fastembed` package (ONNX, no API key).

    Needs `pip install -e ".[local]"` and a one-time model download from Hugging Face.
    Not exercised against a real model in this repository's tests, because the build
    environment could not download it. The tests check only this wrapper's logic.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # imported lazily: the dependency is optional

        self._model = TextEmbedding(model_name)
        self.dim = len(self._first(self._model.embed(["dimension probe"])))

    @staticmethod
    def _first(vectors) -> list[float]:
        return [float(x) for x in next(iter(vectors))]

    def embed(self, text: str) -> list[float]:
        return self._first(self._model.embed([text]))


def embedder_from_name(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "fastembed":
        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder {name!r}; choose 'hash' or 'fastembed'")

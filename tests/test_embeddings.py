import sys
import types

import pytest

from app.embeddings import FastEmbedEmbedder, HashEmbedder, embedder_from_name


def test_hash_embedder_is_deterministic_and_unit_length():
    a, b = HashEmbedder().embed("User lives in Pune"), HashEmbedder().embed("User lives in Pune")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_hash_embedder_handles_text_with_no_usable_words():
    vec = HashEmbedder().embed("the is a")  # all stopwords
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-9


def test_shared_words_score_higher_than_unrelated_words():
    e = HashEmbedder()
    q, related, unrelated = e.embed("city lives"), e.embed("User lives in Pune city"), e.embed("User loves tea")
    dot = lambda x, y: sum(a * b for a, b in zip(x, y))
    assert dot(q, related) > dot(q, unrelated)


def test_embedder_from_name_rejects_unknown_names():
    assert isinstance(embedder_from_name("hash"), HashEmbedder)
    with pytest.raises(ValueError):
        embedder_from_name("nope")


def test_fastembed_wrapper_converts_vectors_and_reports_dim(monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            return iter([[0.5, 0.25, 0.125] for _ in texts])  # stands in for numpy arrays

    fake_module = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)

    embedder = FastEmbedEmbedder("any-model")
    assert embedder.dim == 3
    assert embedder.embed("hello") == [0.5, 0.25, 0.125]

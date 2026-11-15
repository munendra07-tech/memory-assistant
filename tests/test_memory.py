import pytest

from app.embeddings import HashEmbedder
from app.memory import CREATED, UNCHANGED, UPDATED, MemoryStore


@pytest.fixture
def store():
    return MemoryStore(HashEmbedder())


def test_add_and_search_returns_the_memory(store):
    store.add("u1", "User lives in Pune")
    results = store.search("u1", "Which city does the user live in?")
    assert [m.text for m in results] == ["User lives in Pune"]
    assert results[0].score > 0


def test_users_are_isolated(store):
    store.add("u1", "User lives in Mumbai")
    store.add("u2", "User lives in Chennai")
    texts = [m.text for m in store.search("u1", "Which city does the user live in?", limit=10)]
    assert texts == ["User lives in Mumbai"]
    assert store.search("u3", "Which city does the user live in?") == []


def test_duplicate_memory_is_not_stored_twice(store):
    first, outcome_first = store.add("u1", "User lives in Pune")
    second, outcome_second = store.add("u1", "User lives in Pune")
    assert outcome_first == CREATED and outcome_second == UNCHANGED
    assert first.id == second.id
    assert len(store.list("u1")) == 1


def test_delete_removes_only_the_owners_memory(store):
    memory, _ = store.add("u1", "User likes tea")
    assert store.delete("u2", memory.id) is False  # another user cannot delete it
    assert len(store.list("u1")) == 1
    assert store.delete("u1", memory.id) is True
    assert store.list("u1") == []


def test_delete_unknown_or_malformed_id_returns_false(store):
    assert store.delete("u1", "not-a-uuid") is False
    assert store.delete("u1", "00000000-0000-0000-0000-000000000000") is False


@pytest.mark.parametrize("bad_user", ["", "   ", "has space", None, 5])
def test_invalid_user_id_is_rejected(store, bad_user):
    with pytest.raises(ValueError):
        store.add(bad_user, "User likes tea")


@pytest.mark.parametrize("bad_text", ["", "   ", None])
def test_empty_text_is_rejected(store, bad_text):
    with pytest.raises(ValueError):
        store.add("u1", bad_text)


def test_overlong_text_is_rejected(store):
    with pytest.raises(ValueError):
        store.add("u1", "x" * 2001)


@pytest.mark.parametrize("bad_limit", [0, -1, True, 1.5])
def test_invalid_search_limit_is_rejected(store, bad_limit):
    with pytest.raises(ValueError):
        store.search("u1", "anything", limit=bad_limit)


def test_same_key_replaces_the_old_value_in_place(store):
    first, _ = store.add("u1", "User lives in Pune", key="lives_in")
    second, outcome = store.add("u1", "User lives in Delhi", key="lives_in")
    assert outcome == UPDATED
    assert second.id == first.id  # same record, not a second one
    assert second.previous_text == "User lives in Pune"
    assert second.created_at == first.created_at  # original creation time is kept
    assert second.updated_at is not None
    assert [m.text for m in store.list("u1")] == ["User lives in Delhi"]


def test_replacement_is_searchable_by_the_new_value_only(store):
    store.add("u1", "User lives in Pune", key="lives_in")
    store.add("u1", "User lives in Delhi", key="lives_in")
    texts = [m.text for m in store.search("u1", "Which city does the user live in?", limit=5)]
    assert texts == ["User lives in Delhi"]


def test_same_key_and_same_text_is_unchanged(store):
    store.add("u1", "User lives in Pune", key="lives_in")
    _, outcome = store.add("u1", "User lives in Pune", key="lives_in")
    assert outcome == UNCHANGED


def test_different_keys_do_not_replace_each_other(store):
    store.add("u1", "User lives in Pune", key="lives_in")
    store.add("u1", "User works as a nurse", key="works_as")
    assert len(store.list("u1")) == 2


def test_keyless_memories_accumulate(store):
    store.add("u1", "User loves tea")
    store.add("u1", "User loves hiking")
    assert len(store.list("u1")) == 2


def test_keys_are_scoped_per_user(store):
    store.add("u1", "User lives in Pune", key="lives_in")
    _, outcome = store.add("u2", "User lives in Delhi", key="lives_in")
    assert outcome == CREATED  # u2's first value, not an update of u1's
    assert [m.text for m in store.list("u1")] == ["User lives in Pune"]


def test_key_is_case_insensitive(store):
    store.add("u1", "User lives in Pune", key="Lives_In")
    _, outcome = store.add("u1", "User lives in Delhi", key="lives_in")
    assert outcome == UPDATED


@pytest.mark.parametrize("bad_key", ["", "   ", 5, "k" * 65])
def test_invalid_key_is_rejected(store, bad_key):
    with pytest.raises(ValueError):
        store.add("u1", "User lives in Pune", key=bad_key)


def test_delete_all_removes_only_that_users_memories(store):
    store.add("u1", "User lives in Pune", key="lives_in")
    store.add("u1", "User loves tea")
    store.add("u2", "User lives in Delhi", key="lives_in")
    assert store.delete_all("u1") == 2
    assert store.list("u1") == []
    assert [m.text for m in store.list("u2")] == ["User lives in Delhi"]


def test_delete_all_on_an_empty_user_returns_zero(store):
    assert store.delete_all("nobody") == 0


def test_delete_all_rejects_a_bad_user_id(store):
    with pytest.raises(ValueError):
        store.delete_all("   ")


def test_memories_can_be_added_again_after_delete_all(store):
    store.add("u1", "User lives in Pune", key="lives_in")
    store.delete_all("u1")
    _, outcome = store.add("u1", "User lives in Pune", key="lives_in")
    assert outcome == CREATED

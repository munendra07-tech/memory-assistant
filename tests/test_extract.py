import pytest

from app.extract import Fact, LLMFactExtractor, RuleFactExtractor


@pytest.mark.parametrize(
    "message, expected",
    [
        ("I live in Pune.", ["User lives in Pune"]),
        ("My name is Asha", ["User's name is Asha"]),
        ("I love spicy food!", ["User loves spicy food"]),
        ("I work as a data engineer.", ["User works as a data engineer"]),
        ("I don't eat mushrooms.", ["User does not eat mushrooms"]),
        ("I live in Delhi. I enjoy cricket.", ["User lives in Delhi", "User enjoys cricket"]),
    ],
)
def test_rule_extractor_rewrites_first_person_facts(message, expected):
    assert [f.text for f in RuleFactExtractor().extract(message)] == expected


def test_rule_extractor_ignores_questions_and_chatter():
    assert RuleFactExtractor().extract("Where do I live?") == []
    assert RuleFactExtractor().extract("Hello there, how are you today") == []


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user):
        return self.reply


def test_llm_extractor_parses_a_json_array_even_with_surrounding_text():
    llm = _FakeLLM('Sure! ["User lives in Pune", "  ", 5, "User likes tea"] Hope that helps.')
    assert LLMFactExtractor(llm).extract("anything") == [Fact("User lives in Pune"), Fact("User likes tea")]


@pytest.mark.parametrize("reply", ["no json here", "[not valid json", ""])
def test_llm_extractor_returns_empty_on_unparseable_output(reply):
    assert LLMFactExtractor(_FakeLLM(reply)).extract("anything") == []


@pytest.mark.parametrize(
    "message, key",
    [
        ("I live in Pune.", "lives_in"),
        ("My name is Asha.", "name"),
        ("I work as a nurse.", "works_as"),
        ("I work at a bank.", "works_at"),
        ("My favourite colour is green.", "attr:favourite_colour"),
    ],
)
def test_single_valued_facts_get_a_key(message, key):
    assert RuleFactExtractor().extract(message)[0].key == key


@pytest.mark.parametrize("message", ["I love tea.", "I have a dog.", "I am vegetarian.", "I don't eat mushrooms."])
def test_multi_valued_facts_get_no_key(message):
    assert RuleFactExtractor().extract(message)[0].key is None

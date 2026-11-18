"""Turn a user's message into short, third-person facts worth remembering.

A `Fact` may carry a `key` naming a single-valued attribute ("lives_in", "name", "works_as").
When a new fact has the same key as a stored one, the store replaces the old value instead of
adding a second, contradictory memory. Facts with no key (likes, possessions) accumulate.

`RuleFactExtractor` is an offline baseline that handles common first-person patterns.
`LLMFactExtractor` asks a language model instead and is the upgrade path. It may assign keys too,
but only from a fixed list, so a model cannot invent a key that overwrites an unrelated memory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from app.llm import LLM, EchoLLM


@dataclass(frozen=True)
class Fact:
    text: str
    key: Optional[str] = None


class FactExtractor(Protocol):
    def extract(self, message: str) -> list[Fact]: ...


def _slug(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower())


_RULES: list[tuple[re.Pattern, Callable[[re.Match], Fact]]] = [
    (re.compile(r"^my name is (.+)$", re.I), lambda m: Fact(f"User's name is {m.group(1)}", "name")),
    (
        re.compile(r"^my (.+?) (is|are) (.+)$", re.I),
        lambda m: Fact(f"User's {m.group(1)} {m.group(2)} {m.group(3)}", f"attr:{_slug(m.group(1))}"),
    ),
    (re.compile(r"^i live in (.+)$", re.I), lambda m: Fact(f"User lives in {m.group(1)}", "lives_in")),
    (
        re.compile(r"^i work (at|for|as) (.+)$", re.I),
        lambda m: Fact(
            f"User works {m.group(1)} {m.group(2)}",
            "works_as" if m.group(1).lower() == "as" else "works_at",
        ),
    ),
    (
        re.compile(r"^i (?:really )?(like|love|enjoy|prefer|hate|dislike) (.+)$", re.I),
        lambda m: Fact(f"User {m.group(1).lower()}s {m.group(2)}"),
    ),
    (
        re.compile(r"^i (?:don't|do not) (like|eat|drink|use) (.+)$", re.I),
        lambda m: Fact(f"User does not {m.group(1)} {m.group(2)}"),
    ),
    (re.compile(r"^i (?:have|own) (.+)$", re.I), lambda m: Fact(f"User has {m.group(1)}")),
    (re.compile(r"^i(?: am|'m) (.+)$", re.I), lambda m: Fact(f"User is {m.group(1)}")),
]


class RuleFactExtractor:
    def extract(self, message: str) -> list[Fact]:
        facts: list[Fact] = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", message.strip()):
            sentence = sentence.strip()
            if not sentence or sentence.endswith("?"):
                continue  # questions are not facts
            sentence = sentence.rstrip(".!").strip()
            for pattern, build in _RULES:
                match = pattern.match(sentence)
                if match:
                    facts.append(build(match))
                    break
        return facts


ALLOWED_KEYS = {"name", "lives_in", "works_as", "works_at"}
_ATTR_KEY = re.compile(r"^attr:[a-z0-9_]{1,40}$")

_EXTRACTION_PROMPT = (
    "Extract durable facts about the user from their message. Return ONLY a JSON array of objects, "
    'each {"text": "<short third-person fact>", "key": "<key or null>"}. '
    'Example: [{"text": "User lives in Pune", "key": "lives_in"}]. '
    "Use a key only for a fact that can have a single value that later replaces the old one: "
    '"name", "lives_in", "works_as" (job title), "works_at" (employer), or "attr:<snake_case_name>" for another '
    'single-valued attribute such as "attr:favourite_colour". Use null for anything that can accumulate '
    "(likes, hobbies, pets, possessions) and whenever you are unsure. "
    "Ignore questions and small talk. Return [] if there is nothing worth remembering."
)


def _clean_key(key) -> Optional[str]:
    if isinstance(key, str) and (key in ALLOWED_KEYS or _ATTR_KEY.match(key)):
        return key
    return None


class LLMFactExtractor:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def extract(self, message: str) -> list[Fact]:
        raw = self.llm.complete(_EXTRACTION_PROMPT, message)
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
        facts: list[Fact] = []
        for item in data:
            if isinstance(item, str):
                text, key = item, None  # plain strings are accepted, with no key
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text, key = item["text"], _clean_key(item.get("key"))
            else:
                continue
            if text.strip():
                facts.append(Fact(text.strip(), key))
        return facts


def extractor_from_name(name: str, llm: Optional[LLM] = None) -> FactExtractor:
    if name == "rules":
        return RuleFactExtractor()
    if name == "llm":
        if llm is None or isinstance(llm, EchoLLM):
            raise RuntimeError("EXTRACTOR=llm needs a real language model; set LLM_PROVIDER and the related variables.")
        return LLMFactExtractor(llm)
    raise ValueError(f"unknown extractor {name!r}; choose 'rules' or 'llm'")

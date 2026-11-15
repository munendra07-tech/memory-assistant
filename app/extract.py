"""Turn a user's message into short, third-person facts worth remembering.

A `Fact` may carry a `key` naming a single-valued attribute ("lives_in", "name", "works_as").
When a new fact has the same key as a stored one, the store replaces the old value instead of
adding a second, contradictory memory. Facts with no key (likes, possessions) accumulate.

`RuleFactExtractor` is an offline baseline that handles common first-person patterns.
`LLMFactExtractor` asks a language model instead and is the upgrade path. It cannot assign
keys yet, so it never triggers replacement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from app.llm import LLM


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


_EXTRACTION_PROMPT = (
    "Extract durable facts about the user from their message. "
    "Return only a JSON array of short third-person strings, for example "
    '["User lives in Pune"]. Return [] if there is nothing worth remembering.'
)


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
        return [Fact(item.strip()) for item in data if isinstance(item, str) and item.strip()]

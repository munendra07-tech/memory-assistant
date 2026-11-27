"""A 30-second, offline tour of what the memory layer does. No API key, no server, no downloads.

    python examples/demo.py                      # hash embedder: instant, word-overlap matching only
    python examples/demo.py --embedder fastembed # real model (pip install -e ".[local]"), understands paraphrase

It uses the same MemoryStore and rule-based extractor as the API and the MCP server, so what you
see here is what the service does. Memory is held in Qdrant's in-memory mode and vanishes on exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings import embedder_from_name  # noqa: E402
from app.extract import RuleFactExtractor  # noqa: E402
from app.memory import MemoryStore  # noqa: E402

QUESTION = "Which city does the user live in?"


def say(store: MemoryStore, extractor: RuleFactExtractor, user: str, message: str, out: list) -> None:
    """Play one chat message into memory, the way POST /chat does after answering."""
    out.append(f'{user:>5} says: "{message}"')
    for fact in extractor.extract(message):
        memory, outcome = store.add(user, fact.text, key=fact.key)
        detail = f"  (was: {memory.previous_text})" if outcome == "updated" else ""
        out.append(f"        -> {outcome}: {memory.text}{detail}")


def recall(store: MemoryStore, user: str, question: str, out: list) -> list:
    found = store.search(user, question, limit=3)
    shown = ", ".join(m.text for m in found) or "nothing"
    out.append(f'{user:>5} asks: "{question}"\n        -> {shown}')
    return found


def run(embedder_name: str = "hash") -> list[str]:
    store = MemoryStore(embedder_from_name(embedder_name))
    extractor = RuleFactExtractor()
    out: list[str] = []

    out.append("1. Facts are extracted and stored per user")
    say(store, extractor, "asha", "My name is Asha.", out)
    say(store, extractor, "asha", "I live in Pune.", out)
    say(store, extractor, "asha", "I like mangoes.", out)
    say(store, extractor, "ravi", "I live in Delhi.", out)

    out.append("\n2. A changed fact replaces the old one; it does not pile up")
    say(store, extractor, "asha", "I live in Mumbai.", out)
    out.append(f"        asha now holds {len(store.list('asha'))} memories, not 4")

    out.append("\n3. Each user only ever sees their own memories")
    recall(store, "asha", QUESTION, out)
    recall(store, "ravi", QUESTION, out)

    out.append("\n4. Users can be forgotten")
    removed = store.delete_all("asha")
    out.append(f"        erased {removed} memories for asha")
    recall(store, "asha", QUESTION, out)
    recall(store, "ravi", QUESTION, out)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embedder", default="hash", choices=["hash", "fastembed"])
    print("\n".join(run(parser.parse_args(argv).embedder)))


if __name__ == "__main__":
    main()

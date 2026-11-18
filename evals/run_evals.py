"""Run the memory evals and print a results table.

Each case plays a short conversation into a fresh memory store, then asks one question and
checks whether the top-K retrieved memories contain the expected text and none of the
forbidden text (forbidden text is another user's data, so a hit means a privacy leak).

    python evals/run_evals.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.embeddings import embedder_from_name  # noqa: E402
from app.extract import FactExtractor, RuleFactExtractor  # noqa: E402
from app.memory import MemoryStore  # noqa: E402

TOP_K = 3
THRESHOLD = 0.0


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_case(case: dict, extractor: FactExtractor, embedder) -> dict:
    store = MemoryStore(embedder)
    for turn in case["turns"]:
        for fact in extractor.extract(turn["message"]):
            store.add(turn["user_id"], fact.text, key=fact.key)

    question = case["question"]
    retrieved = store.search(question["user_id"], question["text"], limit=case.get("k", TOP_K), threshold=THRESHOLD)
    haystack = " ".join(m.text.lower() for m in retrieved)

    missing = [e for e in case["expect"] if e.lower() not in haystack]
    leaked = [f for f in case["forbid"] if f.lower() in haystack]
    too_many = False
    if "max_memories" in case:
        too_many = len(store.list(case["turns"][0]["user_id"])) > case["max_memories"]
    return {
        "id": case["id"],
        "category": case["category"],
        "passed": not missing and not leaked and not too_many,
        "missing": missing,
        "leaked": leaked,
        "duplicated": too_many,
        "retrieved": [m.text for m in retrieved],
    }


def run_all(extractor: FactExtractor | None = None, embedder=None, cases_path: Path | None = None) -> list[dict]:
    extractor = extractor or RuleFactExtractor()
    embedder = embedder or embedder_from_name("hash")
    cases = load_cases(cases_path or ROOT / "evals" / "cases.jsonl")
    return [run_case(c, extractor, embedder) for c in cases]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedder", choices=["hash", "fastembed"], default="hash")
    parser.add_argument("--min-pass", type=int, default=0, help="exit 1 if fewer cases pass (a regression gate for CI)")
    args = parser.parse_args(argv)
    print(f"embedder: {args.embedder}\n")
    results = run_all(embedder=embedder_from_name(args.embedder))
    by_category: dict[str, list[bool]] = defaultdict(list)
    print(f"{'case':<24}{'category':<12}{'result':<8}details")
    for r in results:
        by_category[r["category"]].append(r["passed"])
        details = ""
        if not r["passed"]:
            details = f"missing={r['missing']} leaked={r['leaked']} duplicated={r['duplicated']}"
        print(f"{r['id']:<24}{r['category']:<12}{'PASS' if r['passed'] else 'FAIL':<8}{details}")
    print()
    for category, outcomes in sorted(by_category.items()):
        print(f"{category:<12}{sum(outcomes)}/{len(outcomes)}")
    total = sum(r["passed"] for r in results)
    print(f"{'overall':<12}{total}/{len(results)}")
    if total < args.min_pass:
        print(f"\nFAILED: {total} cases passed, expected at least {args.min_pass}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

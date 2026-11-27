import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from demo import run  # noqa: E402


def test_the_demo_shows_replacement_isolation_and_erasure():
    text = "\n".join(run("hash"))
    assert "updated: User lives in Mumbai  (was: User lives in Pune)" in text
    assert "asha now holds 3 memories, not 4" in text
    assert 'asha asks: "Which city does the user live in?"\n        -> User lives in Mumbai' in text
    assert 'ravi asks: "Which city does the user live in?"\n        -> User lives in Delhi' in text
    assert "erased 3 memories for asha" in text
    assert 'asha asks: "Which city does the user live in?"\n        -> nothing' in text
    assert "Pune" not in text.split("3. Each user")[1]  # the replaced value never comes back

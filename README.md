# Memory Assistant

[![tests](https://github.com/munendra07-tech/memory-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/munendra07-tech/memory-assistant/actions/workflows/tests.yml)

A FastAPI chat service that remembers facts about each user across conversations, with the
memory stored in [Qdrant](https://qdrant.tech/), plus an evaluation suite that measures how
well it remembers.

**Status: work in progress.** The memory layer, API, tests and evals run offline. It has not
been deployed, and the language-model client has not been tested against a real provider.

## How it works

```
POST /chat {user_id, message}
   1. recall   embed the message, search this user's memories in Qdrant        (app/memory.py)
   2. answer   give the recalled facts to the language model as context        (app/llm.py)
   3. remember extract new facts from the message; store new ones, replace changed ones,
               skip repeats                                                    (app/extract.py)
```

- **Per-user isolation:** every Qdrant read and write is filtered by `user_id`. Deleting
  another user's memory returns 404.
- **Updating facts:** single-valued facts carry a key (`lives_in`, `name`, `works_as`, ...). A new
  value for the same key replaces the old one in place: same record, original `created_at` kept,
  old text kept as `previous_text`. Multi-valued facts (likes, possessions) have no key and
  accumulate. The rule-based extractor assigns keys; the LLM extractor does not yet, so it never
  replaces.
- **Duplicate control:** a keyless fact whose embedding is almost identical (cosine >= 0.97) to an
  existing one is not stored twice.
- **Abuse limits:** `/chat` is rate limited per client address (`RATE_LIMIT_PER_MINUTE`, default 30)
  and returns 429 with `Retry-After`. `DELETE /memories/{user_id}` erases one user's data.
- **Input validation:** user ids, message length and search limits are validated and return
  422 on bad input. A language-model failure returns 502 and stores nothing.
- **Pluggable parts:** `Embedder`, `FactExtractor` and `LLM` are small interfaces, so the
  offline baselines can be swapped for real models.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                          # unit and API tests (CI runs them on Python 3.10 and 3.12)
python evals/run_evals.py       # memory evals (offline baseline)
uvicorn app.main:app --reload   # serve at http://127.0.0.1:8000/docs
```

### Try a real embedding model (not yet verified)

```bash
pip install -e ".[local]"                       # adds fastembed (local ONNX model, no API key)
python evals/run_evals.py --embedder fastembed  # downloads BAAI/bge-small-en-v1.5 once
EMBEDDER=fastembed uvicorn app.main:app         # serve with it
```

This path is written but **has not been run against the real model**: the environment this was
built in blocked the model download. Run it, then record the real numbers in the table below.

By default memory is held in Qdrant's in-memory mode and is lost on restart. Set `QDRANT_URL`
to use a real server. To use a language model set `LLM_API_KEY` and `LLM_MODEL` (and
`LLM_BASE_URL` for non-OpenAI servers); without them the app runs in an offline echo mode.

## Docker and deployment

```bash
cp .env.example .env        # set API_KEY
docker compose up --build   # app + Qdrant server, memories persist
```

The API requires an `X-API-Key` header whenever `API_KEY` is set. See [DEPLOY.md](DEPLOY.md) for the
security checklist, deployment options, and exactly which parts have and have not been verified
(the image build and a real Qdrant server have not been run yet).

## Evaluation

`evals/cases.jsonl` holds 21 small conversations. Each plays facts into a fresh store, asks one
question, and passes if the top-3 retrieved memories contain the expected text and none of
another user's text.

Offline baseline (hash embedder + rule-based extractor), measured with `python evals/run_evals.py`:

| Category | Passed |
|---|---|
| direct (word overlap with the question) | 8 / 8 |
| isolation (no leakage between users) | 2 / 2 |
| update (a changed fact replaces the old one) | 6 / 6 |
| paraphrase ("reside" vs "live") | 0 / 4 |
| extraction (fact not stated as "I ...") | 0 / 1 |
| **overall** | **16 / 21** |

The update cases were written first: before the update feature existed, 4 of the 6 failed
(2 of 6 passed, both being guards against over-replacing).

How to read this:
- The baseline only matches shared words, so it fails every paraphrase. That is expected, and it
  is the gap a real embedding model should close.
- The rule-based extractor only understands simple "I live in / I like ..." sentences, so it
  misses facts phrased any other way.
- Twenty-one cases is a small suite. Treat the numbers as a regression check, not a benchmark.

## Roadmap

- [ ] Run the evals with `--embedder fastembed` and record the real numbers (adapter written, model not yet run).
- [ ] Use `LLMFactExtractor` with a real model and re-run the evals.
- [ ] Let the LLM extractor assign keys, so it can trigger replacement too.
- [x] Add a memory-update path (a changed fact replaces the old one).
- [ ] Build the image and run it against a real Qdrant server (written, not yet run).
- [ ] Deploy and add the live URL here.

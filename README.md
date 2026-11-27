# Memory Assistant

[![tests](https://github.com/munendra07-tech/memory-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/munendra07-tech/memory-assistant/actions/workflows/tests.yml)

**Long-term memory for AI assistants, one user at a time.** It remembers what each person tells it,
keeps their memories separate from everyone else's, updates facts when they change, and measures how
well it all works. Use it as a REST service or plug it into Claude Desktop and other agents over
[MCP](https://modelcontextprotocol.io). Built on [Qdrant](https://qdrant.tech/).

**What makes it different is that every claim is measured and every gap is written down.**

| Claim | Evidence |
|---|---|
| Real embeddings beat the offline baseline where it matters | Ranking cases (right fact among 6, must be top-1): **1/6 hash baseline vs 6/6 bge-small-en-v1.5**; 17/27 vs 26/27 overall. [Table below.](#evaluation) |
| One user can never read or delete another's memories | Enforced on every Qdrant call, tested through the REST API, the MCP tools and over HTTP |
| A changed fact replaces the old one | Update cases written first; 4 of 6 failed before the feature existed, all pass now |
| A harder second eval set, written before it was run | 24 cases: **12/24 hash baseline vs 23/24 real model**; isolation 6/6 on both; the one miss is shown ([details](#a-harder-second-set-24-cases)) |
| It works against a real Qdrant server, in a container | Image builds, runs as non-root, memory survives an app restart (2026-11-27, [DEPLOY.md](DEPLOY.md)) |
| Guarded when exposed | `X-API-Key` on the REST API (the compose file refuses to start without it), bearer key on MCP-over-HTTP (the server refuses to start without it), rate limiting on `/chat` |
| 161 tests, checked on every push | CI runs them on Python 3.10 and 3.12, runs the store and MCP server against a real Qdrant server (`qdrant/qdrant:v1.20.0`), builds the Docker image and checks it, runs the real compose stack end to end (a fact survives an app restart), and fails if the eval score drops |

What is **not** verified yet is just as visible: no real Claude API run, no Claude Desktop connection test,
no deployment. See [DEPLOY.md](DEPLOY.md) for the full verified / not-verified table.

## Try it in 30 seconds

```bash
pip install -e ".[dev]" && python examples/demo.py     # offline: no key, no server, no download
```

```text
2. A changed fact replaces the old one; it does not pile up
 asha says: "I live in Mumbai."
        -> updated: User lives in Mumbai  (was: User lives in Pune)
        asha now holds 3 memories, not 4

3. Each user only ever sees their own memories
 asha asks: "Which city does the user live in?"
        -> User lives in Mumbai
 ravi asks: "Which city does the user live in?"
        -> User lives in Delhi
```

(That is real output, and a test fails if it ever changes. `--embedder fastembed` runs the same tour with a real model.)

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

### Try a real embedding model

```bash
pip install -e ".[local]"                       # adds fastembed (local ONNX model, no API key)
python evals/run_evals.py --embedder fastembed  # downloads BAAI/bge-small-en-v1.5 once
EMBEDDER=fastembed uvicorn app.main:app         # serve with it
```

The eval results below include a run with this model. The server path with `EMBEDDER=fastembed`
has not been exercised.

### Use Claude as the language model (optional, costs money)

```bash
pip install -e ".[anthropic]"
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...        # from your Anthropic console; never commit it
export LLM_MODEL=<model id>         # the Claude model you chose
export LLM_EFFORT=low               # or "none" for a model that does not take an effort setting

# optional: a different model per job (each overrides LLM_MODEL for that job only)
export EXTRACTOR_LLM_MODEL=<small model id>   # fact extraction, runs on every message
export REPLY_LLM_MODEL=<better model id>      # the reply the user reads

# measure it: one model call per conversation turn, hard-capped (uses EXTRACTOR_LLM_MODEL if set)
python evals/run_evals.py --extractor llm --embedder fastembed --max-llm-calls 60

# serve with it
EXTRACTOR=llm uvicorn app.main:app
```

The eval prints how many calls were planned and refuses to start if they exceed `--max-llm-calls`;
afterwards it prints the calls and tokens actually used, so you can work out the real cost. Set a monthly
spend limit in the Anthropic console before running anything. The client is tested offline (a fake client
and real SDK response objects); it has **not** been run against the real API yet.

By default memory is held in Qdrant's in-memory mode and is lost on restart. Set `QDRANT_URL`
to use a real server. To use a language model set `LLM_API_KEY` and `LLM_MODEL` (and
`LLM_BASE_URL` for non-OpenAI servers); without them the app runs in an offline echo mode.

## Use it from an AI assistant (MCP)

`app/mcp_server.py` exposes the memory store over the [Model Context Protocol](https://modelcontextprotocol.io)
with three tools: `remember`, `recall` and `forget`. It calls no language model, so it needs no API key.
The assistant decides what is worth remembering and passes it in.

```bash
pip install -e ".[mcp]"
python -m app.mcp_server        # speaks MCP over stdio; clients launch this themselves
```

### Connect Claude Desktop (one command)

```bash
pip install -e ".[mcp]"                    # once, from this folder
python scripts/connect_claude_desktop.py   # check the server, then add it to Claude Desktop
```

The script first starts the server the way Claude Desktop will and calls `remember`, `recall` and `forget`
once; if that fails it writes nothing. Otherwise it adds a `memory-assistant` entry to
`claude_desktop_config.json` (keeping your other servers, after saving a timestamped backup, and refusing to touch
a file that is not valid JSON). It writes no API key. Memory is kept on disk in
`~/.memory-assistant/<embedder>`, so it survives restarting Claude Desktop. Then quit Claude Desktop completely and
reopen it. Useful flags: `--dry-run` (show, change nothing), `--remove`, `--embedder fastembed` (real model,
downloads about 130 MB on first use).

The entry it writes is just:

```json
{ "mcpServers": { "memory-assistant": {
    "command": "/path/to/your/python", "args": ["-m", "app.mcp_server"],
    "env": { "EMBEDDER": "hash", "QDRANT_PATH": "/home/you/.memory-assistant/hash" } } } }
```

(There is no `cwd`: it is not a documented field, so the package is pip-installed instead.) On-disk mode lets
one process use the folder at a time. For a shared or multi-client setup use `QDRANT_URL` with a Qdrant server.
The script and the config it writes are tested end to end here (including launching the server from the written
config in a separate process and recalling a stored fact); **a real Claude Desktop run has not been confirmed yet.**

For a server other machines can reach, there is an opt-in HTTP mode with a required bearer key:

```bash
MCP_API_KEY=<a long random value> python -m app.mcp_server --http    # 127.0.0.1:8000/mcp
```

It refuses to start without the key, and `MCP_USER_ID` pins all calls to one user. It has no TLS and
no per-user keys; see [DEPLOY.md](DEPLOY.md) before exposing it.

With neither `QDRANT_URL` nor `QDRANT_PATH` the memory lives inside the server process and is lost when the client
closes it. The test suite starts the server as a subprocess and calls all three tools over stdio, over HTTP, and
against on-disk and server-backed storage.

## Docker and deployment

```bash
cp .env.example .env        # set API_KEY
docker compose up --build   # app + Qdrant server, memories persist
```

The API requires an `X-API-Key` header whenever `API_KEY` is set. See [DEPLOY.md](DEPLOY.md) for the
security checklist, deployment options, and exactly which parts have and have not been verified
(the image build and a real Qdrant server were run on 2026-11-27 and are now checked by CI on every push; fastembed in Docker and a hosted Qdrant have not been run; `docker compose up` was run on 2026-11-27 via `scripts/compose_smoke.sh`).

## Evaluation

`evals/cases.jsonl` holds 27 small conversations. Each plays facts into a fresh store, asks one
question, and passes if the top-K retrieved memories (K=3, or 1 for the `ranked` cases) contain the
expected text and none of another user's text.

Measured with `python evals/run_evals.py [--embedder fastembed]` on 2026-11-18. The fact extractor
is the rule-based one in both columns, so only the embedder differs.

| Category | Cases | Hash baseline | Real model (bge-small-en-v1.5) |
|---|---|---|---|
| direct (word overlap with the question) | 8 | 8 | 8 |
| isolation (no leakage between users) | 2 | 2 | 2 |
| update (a changed fact replaces the old one) | 6 | 6 | 6 |
| paraphrase, one stored fact | 4 | 0 | 4 |
| ranked, right fact among 6, must be top-1 | 6 | 1 | 6 |
| extraction (fact not stated as "I ...") | 1 | 0 | 0 |
| **overall** | **27** | **17** | **26** |

How to read this:
- The `paraphrase` cases store a single fact, so a real model returns it whatever it understands.
  They were too easy, which is why the `ranked` cases exist: the right fact must beat five
  distractors. The hash baseline gets 1 of 6 there, the real model 6 of 6. That is the meaningful comparison.
- The one remaining failure is the rule-based extractor, which only understands simple "I live in /
  I like ..." sentences. A language-model extractor is the next step (not yet run).
- The update cases were written first: before the update feature existed, 4 of the 6 failed.
- Twenty-seven cases and one model is a small test. Treat the numbers as a regression check and a
  fair comparison, not a benchmark. CI runs the hash baseline and fails below 17 of 27.

### A harder second set (24 cases)

`evals/cases_extra.jsonl` holds 24 harder cases, **written and committed before they were run** so they
could not be tuned to a result: ranking among six facts with no word overlap, near-identical facts across
users, multi-step updates and a few mixed cases. Run with `--cases evals/cases_extra.jsonl`. Measured on
2026-11-27, rule-based extractor in both columns:

| Category | Cases | Hash baseline | Real model (bge-small-en-v1.5) |
|---|---|---|---|
| ranked-hard (right fact among 6, top-1, question shares no words) | 10 | 1 | 9 |
| isolation-hard (near-identical facts across users, no leaks) | 6 | 6 | 6 |
| update-hard (several moves, accumulation, repeats) | 6 | 5 | 6 |
| mixed (negation, two facts needed) | 2 | 0 | 2 |
| **overall** | **24** | **12** | **23** |

- The one real-model miss is `rank2-transport`: for "How does the user get around?" the top result was
  "User works as a teacher", not "User has a bicycle". The extractor stored the bicycle fact correctly, so
  this is a genuine retrieval miss of a small model, and it stays in the table.
- Isolation held in every case with both embedders (6/6): no user's text leaked into another's results.
- The hash baseline's 12/24 is a regression guard in CI (`--min-pass 12`), not an achievement.
- Same caveat as before: 24 more cases is still a small test, not a benchmark.

## Roadmap

- [x] Run the evals with a real embedding model and record the numbers (see Evaluation).
- [ ] Run the evals with `--extractor llm` and record the real numbers (client and tests written; not yet run against the API).
- [x] Let the LLM extractor assign keys, so it can trigger replacement too (keys limited to a fixed list).
- [x] Add a memory-update path (a changed fact replaces the old one).
- [x] Build the image and run it against a real Qdrant server (2026-11-27, including `docker compose up` via `scripts/compose_smoke.sh`).
- [x] MCP server (`remember`, `recall`, `forget`) with an opt-in HTTP mode, bearer auth and user pinning.
- [ ] Confirm the MCP server inside Claude Desktop (steps written; needs a real desktop run).
- [ ] Deploy and add the live URL here.

# Deployment guide

How to run Memory Assistant in a container and put it on the internet safely.

## What has and has not been verified

| Item | Status |
|---|---|
| App, API-key auth, rate limiting, delete-all, validation, tests (160 as of the MCP auth merge) | Run and passing; rate limit and delete-all also checked on a live server |
| Package installs from `pyproject.toml` + `app/` alone, server starts, health check command works | Run in a clean directory, mimicking the image's steps |
| `docker-compose.yml` syntax; refuses to start without `API_KEY` | Run (`docker compose config`) |
| `docker build` of the Dockerfile | Run on 2026-11-27 (Docker 29.9.3, `python:3.12-slim`): builds, container runs as uid 10001, `/health` ok, built-in HEALTHCHECK reports healthy. The build sandbox needed its own CA certificate added for pip (its network proxy re-signs HTTPS); that was done in a temporary copy, not in the Dockerfile, so a normal network is expected to need nothing. |
| App talking to a real Qdrant **server** (`QDRANT_URL`) | Run on 2026-11-27 with `qdrant/qdrant:v1.20.0` and `EMBEDDER=hash`: a stored fact survived replacing the app container, a request without the key got 401, and another user saw nothing. Run with plain `docker run` on one network, **not** `docker compose up`. **Not run:** fastembed inside Docker, a hosted Qdrant |
| `qdrant/qdrant:v1.20.0` image tag exists | Checked 2026-11-27: Docker Hub lists the tag and the image pulled (the first pull hit a rate limit, 429; a retry worked) |
| Real embedding model (fastembed, bge-small-en-v1.5) in the evals | Run; results in the README. **Not run** inside Docker or the server |
| Claude client (`LLM_PROVIDER=anthropic`) | Tested offline with a fake client and real SDK response objects. **Not run against the real API** |
| Any real language model, end to end | **Not run.** No key yet |
| HTTPS, hosting-provider steps below | **Not run.** Written from general practice. Check each provider's current docs. |

| MCP server against a real Qdrant server (`qdrant/qdrant:v1.20.0`, `EMBEDDER=hash`) | Run on 2026-11-27 with a throwaway script (not in the test suite): over stdio, a fact stored by one server process was recalled by a brand-new process; a keyed fact was replaced (`updated`); `forget` as another user returned `deleted: false` and as the owner `deleted: true`. Over HTTP with `MCP_API_KEY` and `MCP_USER_ID=ravi`, a caller claiming `user_id=asha` got only ravi's memory. Qdrant ran as a plain container reached on 127.0.0.1, not inside compose |
| MCP server over HTTP with `MCP_API_KEY` (401 without the key, user pinning, refuses to start without a key) | Run: 11 tests, and a manual check of the CLI with curl (401, 401, 200). **Not run:** behind HTTPS, with a real MCP client other than the SDK's, or against a real Qdrant server |

So the first thing to do on your machine is run the steps in "Run it locally" and fix anything that breaks.

## Read this before putting it on the internet

This service stores things people tell it about themselves. Treat it as holding personal data.

1. **Always set `API_KEY`.** Without it, anyone who finds the URL can read or delete every user's
   memories and use your language-model quota. With the key set, only `/health` is open.
2. **Never publish Qdrant.** The compose file keeps it on a private network. Do not add a `ports:`
   entry for it, and do not open port 6333 in a firewall.
3. **Use HTTPS.** The API key travels in a header. Over plain HTTP it can be read in transit.
4. **Cap spend.** If you set `LLM_API_KEY`, set a monthly spending limit with the provider. `/chat` is
   limited to `RATE_LIMIT_PER_MINUTE` calls (default 30) per client address, but that limit lives in one
   process and is by network address, so behind a proxy or with several instances it is weaker than it
   looks. The provider-side cap is the real protection.
5. **Say what is stored.** A public demo should tell visitors not to enter sensitive information.
6. **Keep secrets out of git.** `.env` is ignored by `.gitignore`. Never paste keys into code,
   issues or screenshots. If one leaks, rotate it.

7. **MCP over HTTP.** Stdio mode (what Claude Desktop uses) has no network port and needs no key.
   `--http` mode needs `MCP_API_KEY`, binds to 127.0.0.1 by default, and has no TLS of its own: use an
   HTTPS reverse proxy. One shared key means anyone holding it can act as any `user_id`; set
   `MCP_USER_ID` for a single-user deployment. There are no per-user keys and no MCP rate limit yet.

Built: `DELETE /memories/{user_id}` erases everything remembered about one user.
Not built yet, worth adding before real users: a shared (multi-instance) rate limiter, request logging
without message text, and automatic memory expiry.

## Run it locally

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste the output as API_KEY in .env
docker compose up --build
```

Check it (replace `YOUR_KEY`):

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/chat \
  -H 'X-API-Key: YOUR_KEY' -H 'content-type: application/json' \
  -d '{"user_id":"asha","message":"I live in Pune."}'
curl -s localhost:8000/memories/asha -H 'X-API-Key: YOUR_KEY'
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/memories/asha   # expect 401 (no key)
```

Restart test: `docker compose restart app`, then list the memories again. They should still be
there, because Qdrant keeps them in the `qdrant_storage` volume. (With no `QDRANT_URL` the app uses
in-memory mode and **loses everything on restart**.)

## Configuration

| Variable | Needed | Meaning |
|---|---|---|
| `API_KEY` | Yes in production | Clients send it as the `X-API-Key` header |
| `QDRANT_URL` | For persistence | e.g. `http://qdrant:6333`. Unset means in-memory (data lost on restart) |
| `QDRANT_API_KEY` | For a hosted Qdrant | Passed to the Qdrant client |
| `LLM_PROVIDER` | Optional | `anthropic` for Claude (needs `ANTHROPIC_API_KEY` and `LLM_MODEL`), or leave empty for an OpenAI-style server |
| `ANTHROPIC_API_KEY`, `LLM_EFFORT`, `LLM_MAX_TOKENS` | With `anthropic` | Key from the Anthropic console, effort level (or `none`), and the reply length cap |
| `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | Optional | An OpenAI-style model. Without any model settings: offline echo mode |
| `EXTRACTOR` | Optional | `rules` (default) or `llm`, which needs a real model |
| `EXTRACTOR_LLM_MODEL`, `REPLY_LLM_MODEL` | Optional | A separate model for fact extraction / chat replies (also `*_LLM_EFFORT`, `*_LLM_MAX_TOKENS`). Fall back to `LLM_MODEL` |
| `EMBEDDER` | Optional | `hash` (default, offline baseline) or `fastembed` (needs the `[local]` extra) |
| `RATE_LIMIT_PER_MINUTE` | Optional | Calls to `/chat` per client address per minute. Default 30, `0` disables |
| `PORT` | Set by most hosts | Port the server listens on. Default 8000 |
| `MCP_API_KEY` | For MCP over HTTP | Required by `python -m app.mcp_server --http`; clients send `Authorization: Bearer <key>`. The server exits without it |
| `MCP_USER_ID` | Optional | Pins every MCP tool call to this one user; the caller's `user_id` is ignored |

## Option A: one small server (VPS) with HTTPS

Good for learning and for a portfolio demo. You manage the machine.

1. Create a small Linux server with Docker installed, and point a domain name at its IP address.
2. Copy the project onto it (`git clone` your repo) and create `.env` with a real `API_KEY`.
3. Add HTTPS with a reverse proxy. One common choice is Caddy, which obtains certificates
   automatically. A `Caddyfile` such as:

   ```
   memory.example.com {
       reverse_proxy app:8000
   }
   ```

   Run Caddy on the same compose network as `app`, publishing ports 80 and 443 for Caddy only, and
   remove the `ports:` entry from `app` so the app is reachable only through Caddy.
4. Open only ports 22 (SSH), 80 and 443 in the firewall.
5. `docker compose up -d --build`, then run the checks above against `https://memory.example.com`.

## Option B: a container platform plus hosted Qdrant

Good if you do not want to manage a server.

1. Create a hosted Qdrant cluster. Note its URL and API key.
2. Create a service on a container hosting platform from your GitHub repo, using the Dockerfile.
3. Set environment variables: `API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and optionally the `LLM_*` ones.
4. Set the health check path to `/health`.
5. Run **one** instance. Memories live in Qdrant, so the app itself is stateless, but only if
   `QDRANT_URL` is set. Without it, every instance has its own empty in-memory store.
6. Check pricing and free-tier limits on the provider's own pages before you rely on them.

## Operating it

- **Logs:** `docker compose logs -f app`. Do not add logging of message text.
- **Update:** `git pull && docker compose up -d --build`.
- **Roll back:** `git checkout <previous-commit> && docker compose up -d --build`.
- **Back up memories:** the `qdrant_storage` volume holds them. Qdrant also offers snapshots. Check its
  documentation for the current procedure before relying on a backup.
- **Delete one user's memories:** `curl -X DELETE localhost:8000/memories/USER_ID -H 'X-API-Key: YOUR_KEY'`.
- **Delete everything:** `docker compose down -v` removes the containers **and all stored memories**.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `docker compose` says `API_KEY` is missing | `.env` is missing or has an empty `API_KEY` |
| 401 on every call | Missing or wrong `X-API-Key` header |
| App container restarts at startup | Qdrant not reachable. Check `QDRANT_URL` and `docker compose logs qdrant` |
| Memories vanish after a restart | `QDRANT_URL` not set, so it is using in-memory mode |
| Qdrant version warning in the logs | Client and server versions differ by more than one minor version. Align the image tag |
| 429 from `/chat` | Rate limit reached. Wait for the `Retry-After` seconds, or raise `RATE_LIMIT_PER_MINUTE` |
| 502 from `/chat` | The language model call failed. Check `LLM_API_KEY`, `LLM_MODEL` and `LLM_BASE_URL` |
| Image pull fails for Qdrant | The pinned tag may not exist. Choose one close to your `qdrant-client` version |

## After deploying

Put the live URL, the date you tested it, and the eval numbers you actually measured in the README.
Only claim what you ran.

# Deployment guide

How to run Memory Assistant in a container and put it on the internet safely.

## What has and has not been verified

| Item | Status |
|---|---|
| App, API-key auth, rate limiting, delete-all, validation, tests (89) | Run and passing; rate limit and delete-all also checked on a live server |
| Package installs from `pyproject.toml` + `app/` alone, server starts, health check command works | Run in a clean directory, mimicking the image's steps |
| `docker-compose.yml` syntax; refuses to start without `API_KEY` | Run (`docker compose config`) |
| `docker build` of the Dockerfile | **Not run.** The build environment had no Docker daemon. |
| App talking to a real Qdrant **server** (`QDRANT_URL`) | **Not run.** Only Qdrant's in-memory mode was tested. |
| `qdrant/qdrant:v1.20.0` image tag exists | **Not checked.** If the pull fails, pick a tag close to `pip show qdrant-client`. |
| Real language model, real embedding model | **Not run.** No keys, and the model download was blocked. |
| HTTPS, hosting-provider steps below | **Not run.** Written from general practice. Check each provider's current docs. |

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
| `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | Optional | A real language model. Without them: offline echo mode |
| `EMBEDDER` | Optional | `hash` (default, offline baseline) or `fastembed` (needs the `[local]` extra) |
| `RATE_LIMIT_PER_MINUTE` | Optional | Calls to `/chat` per client address per minute. Default 30, `0` disables |
| `PORT` | Set by most hosts | Port the server listens on. Default 8000 |

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

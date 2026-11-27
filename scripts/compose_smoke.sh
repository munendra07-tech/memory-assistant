#!/usr/bin/env bash
# Smoke test for docker-compose.yml: app + a real Qdrant server, memories surviving an app restart.
#
#   bash scripts/compose_smoke.sh        # needs Docker with the compose plugin; uses port 8000
#
# It starts the stack with a throwaway API key, stores a fact through /chat, restarts ONLY the app
# container, and checks the fact is still there. It also checks that Qdrant is not published to the
# host and that auth, isolation and the non-root user hold. It tears everything down, volumes included.
set -euo pipefail

export API_KEY="compose-smoke-not-a-secret"
BASE="http://127.0.0.1:8000"
H="X-API-Key: ${API_KEY}"
fail() { echo "FAIL: $*" >&2; docker compose logs --tail=40 >&2 || true; exit 1; }
cleanup() { docker compose down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT

wait_healthy() {
  for _ in $(seq 1 40); do
    curl -sf -m 3 "${BASE}/health" >/dev/null && return 0
    sleep 2
  done
  fail "app did not become healthy"
}

docker compose up -d --build
wait_healthy
echo "ok: stack is up and /health answers"

code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/memories/asha")
[ "$code" = "401" ] || fail "request without the key returned $code, expected 401"
echo "ok: 401 without the key"

curl -sf -X POST "${BASE}/chat" -H "$H" -H 'Content-Type: application/json' \
  -d '{"user_id":"asha","message":"I live in Pune."}' >/dev/null || fail "/chat failed"
curl -sf -X POST "${BASE}/chat" -H "$H" -H 'Content-Type: application/json' \
  -d '{"user_id":"ravi","message":"I live in Delhi."}' >/dev/null || fail "/chat failed"
curl -sf "${BASE}/memories/asha" -H "$H" | grep -q "Pune" || fail "fact not stored"
echo "ok: facts stored"

docker compose restart app >/dev/null
wait_healthy
curl -sf "${BASE}/memories/asha" -H "$H" | grep -q "Pune" || fail "fact lost after the app restarted"
curl -sf "${BASE}/memories/asha" -H "$H" | grep -q "Delhi" && fail "another user's fact leaked"
echo "ok: fact survived an app restart, no leak between users"

# Behavioural check: from the host, Qdrant must be unreachable (it is only on the compose network).
if curl -s -m 3 -o /dev/null "http://127.0.0.1:6333/healthz"; then fail "Qdrant is reachable from the host"; fi
echo "ok: Qdrant is not reachable from the host"

[ "$(docker compose exec -T app id -u)" = "10001" ] || fail "app does not run as uid 10001"
echo "ok: app runs as a non-root user"
echo "compose smoke test passed"

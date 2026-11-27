# Security

This service stores things people tell it about themselves, so please treat reports seriously.

## Reporting a vulnerability

If the repository's **Security** tab offers **Report a vulnerability**, use that. If it does not, open
a public issue titled "security contact request" with **no details**, and a private channel will be
arranged. Please never post exploit details publicly. Include what you did, what you expected and
what happened. This is a one-person project, so replies are best-effort.

## What is in scope

- Reading, changing or deleting another user's memories (per-user isolation).
- Getting past `API_KEY` on the REST API or `MCP_API_KEY` on the MCP HTTP mode.
- Secrets appearing in logs, errors or responses.

## Known limits (already documented, not vulnerabilities)

See [DEPLOY.md](DEPLOY.md). In short: no TLS of its own (use an HTTPS proxy), one shared key per
interface rather than per-user keys, an in-process rate limiter, and no MCP rate limit.

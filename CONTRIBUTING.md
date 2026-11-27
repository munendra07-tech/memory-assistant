# Contributing

Small, well-tested changes are very welcome.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # must pass
python evals/run_evals.py --min-pass 17  # the same gate CI runs
```

CI also runs the store and MCP server against a real Qdrant server and builds the Docker image. To run
the Qdrant tests yourself: `docker run -d -p 6333:6333 qdrant/qdrant:v1.20.0`, then
`TEST_QDRANT_URL=http://127.0.0.1:6333 pytest tests/test_qdrant_server.py` (they skip without it).

Ground rules, which are the ones this project is built on:

1. **Measure before you claim.** If a change affects behaviour, add or change an eval case or test
   first, see it fail, then make it pass. Numbers in the README come from a command you can run.
2. **No secrets in code, issues or commits.** Keys come from environment variables.
3. **Say what you did not verify.** If something was not run, the docs say so (see the table in
   [DEPLOY.md](DEPLOY.md)).
4. **Keep it explainable.** Prefer the simple version you can walk someone through line by line.

Open an issue before a large change. Security problems go through [SECURITY.md](SECURITY.md).

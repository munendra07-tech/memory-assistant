# Memory Assistant image.
# Build:  docker build -t memory-assistant .
# Run:    docker run --rm -p 8000:8000 -e API_KEY=change-me memory-assistant
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install the package and its dependencies (see pyproject.toml).
COPY pyproject.toml ./
COPY app ./app
RUN pip install .

# Do not run as root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Python is already in the image, so no curl is needed for the health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT', '8000'), timeout=3)"

# Hosting platforms set $PORT; default to 8000. `exec` lets uvicorn receive stop signals.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

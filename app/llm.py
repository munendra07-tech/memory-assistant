"""Language-model clients behind one tiny interface.

`EchoLLM` needs no network, so the app and tests run offline. `OpenAICompatibleLLM` talks to
any server that implements the OpenAI chat-completions API. It has NOT been exercised in this
repository's tests, because that needs a real API key.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class EchoLLM:
    def complete(self, system: str, user: str) -> str:
        return f"[offline mode, no language model configured] You said: {user}"


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def llm_from_env() -> LLM:
    """Use a real model only if LLM_API_KEY and LLM_MODEL are set; otherwise run offline."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return EchoLLM()
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_API_KEY is set but LLM_MODEL is not. Set LLM_MODEL to a model name.")
    return OpenAICompatibleLLM(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), api_key, model)

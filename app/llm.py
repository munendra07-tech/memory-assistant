"""Language-model clients behind one tiny interface.

`EchoLLM` needs no network, so the app and tests run offline. `OpenAICompatibleLLM` talks to
any server that implements the OpenAI chat-completions API. It has NOT been exercised in this
repository's tests, because that needs a real API key.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Protocol

import httpx

logger = logging.getLogger(__name__)

VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


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


class AnthropicLLM:
    """Claude through the official `anthropic` SDK (an optional dependency).

    The model id, key and effort come from the caller, so which Claude model is used (and what it
    costs) stays the operator's choice. Not exercised against the real API in this repository's
    tests, which use a fake client and real SDK response objects built offline.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        effort: Optional[str] = "low",
        max_tokens: int = 1024,
        timeout: float = 30.0,
        client=None,
    ) -> None:
        if not model or not isinstance(model, str):
            raise ValueError("model must be a Claude model id such as the one shown in your Anthropic console")
        if effort is not None and effort not in VALID_EFFORTS:
            raise ValueError(f"effort must be one of {sorted(VALID_EFFORTS)} or None, got {effort!r}")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if client is None:
            try:
                import anthropic  # imported lazily: the dependency is optional
            except ImportError as exc:
                raise RuntimeError('The anthropic package is not installed. Run: pip install -e ".[anthropic]"') from exc
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout)  # SDK retries transient errors itself
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str) -> str:
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        response = self.client.messages.create(**request)

        usage = getattr(response, "usage", None)
        with self._lock:
            self.calls += 1
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined this request")
        if response.stop_reason == "max_tokens":
            logger.warning("model output was cut off at max_tokens=%s", self.max_tokens)
        # Reasoning blocks (type "thinking") are not part of the answer; keep only the text.
        return "".join(block.text for block in response.content if block.type == "text")


class BudgetedLLM:
    """Wraps a model and refuses calls beyond a fixed number: a guard against runaway spend."""

    def __init__(self, llm: LLM, max_calls: int) -> None:
        if max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        self.inner = llm
        self.max_calls = max_calls
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"language-model call budget of {self.max_calls} is used up")
        self.calls += 1
        return self.inner.complete(system, user)


def _anthropic_from_env() -> AnthropicLLM:
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_PROVIDER=anthropic needs LLM_MODEL set to the Claude model id you chose.")
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY (or LLM_API_KEY) set.")
    effort = os.getenv("LLM_EFFORT", "low").strip().lower()
    return AnthropicLLM(
        model,
        api_key=api_key,
        effort=None if effort in ("", "none") else effort,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
    )


def llm_from_env() -> LLM:
    """Pick the language model from the environment.

    LLM_PROVIDER=anthropic uses Claude (needs LLM_MODEL and ANTHROPIC_API_KEY). Otherwise a real
    OpenAI-style model is used only if LLM_API_KEY and LLM_MODEL are set; with neither, run offline.
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "anthropic":
        return _anthropic_from_env()
    if provider not in ("", "openai-compatible"):
        raise ValueError(f"unknown LLM_PROVIDER {provider!r}; use 'anthropic' or 'openai-compatible'")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return EchoLLM()
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_API_KEY is set but LLM_MODEL is not. Set LLM_MODEL to a model name.")
    return OpenAICompatibleLLM(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), api_key, model)

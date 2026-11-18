import pytest

from app.llm import AnthropicLLM, BudgetedLLM, EchoLLM, OpenAICompatibleLLM, llm_from_env


class FakeBlock:
    def __init__(self, type, text=""):
        self.type, self.text = type, text


class FakeUsage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class FakeResponse:
    def __init__(self, blocks, stop_reason="end_turn", usage=(10, 4)):
        self.content, self.stop_reason, self.usage = blocks, stop_reason, FakeUsage(*usage)


class FakeClient:
    """Records the request and returns a canned response. Mimics client.messages.create."""

    def __init__(self, response):
        self.requests = []
        self.messages = self
        self._response = response

    def create(self, **request):
        self.requests.append(request)
        return self._response


def make(response=None, **kwargs):
    client = FakeClient(response or FakeResponse([FakeBlock("text", "hello")]))
    return AnthropicLLM("some-model-id", client=client, **kwargs), client


def test_request_has_the_expected_shape():
    llm, client = make()
    assert llm.complete("be brief", "hi there") == "hello"
    request = client.requests[0]
    assert request["model"] == "some-model-id"
    assert request["max_tokens"] == 1024
    assert request["system"] == "be brief"
    assert request["messages"] == [{"role": "user", "content": "hi there"}]
    assert request["output_config"] == {"effort": "low"}


def test_removed_sampling_and_thinking_parameters_are_never_sent():
    llm, client = make()
    llm.complete("s", "u")
    assert not {"temperature", "top_p", "top_k", "thinking", "budget_tokens"} & set(client.requests[0])


def test_effort_can_be_turned_off_for_models_that_do_not_take_it():
    llm, client = make(effort=None)
    llm.complete("s", "u")
    assert "output_config" not in client.requests[0]


def test_only_text_blocks_are_returned_not_reasoning():
    response = FakeResponse([FakeBlock("thinking", "private reasoning"), FakeBlock("text", "ans"), FakeBlock("text", "wer")])
    llm, _ = make(response)
    assert llm.complete("s", "u") == "answer"


def test_usage_is_counted_across_calls():
    llm, _ = make(FakeResponse([FakeBlock("text", "x")], usage=(100, 7)))
    llm.complete("s", "u")
    llm.complete("s", "u")
    assert (llm.calls, llm.input_tokens, llm.output_tokens) == (2, 200, 14)


def test_a_refusal_raises_instead_of_returning_empty_text():
    llm, _ = make(FakeResponse([], stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="declined"):
        llm.complete("s", "u")


@pytest.mark.parametrize("kwargs", [{"effort": "turbo"}, {"max_tokens": 0}, {"max_tokens": True}])
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AnthropicLLM("some-model-id", client=FakeClient(None), **kwargs)


@pytest.mark.parametrize("model", ["", None, 5])
def test_a_model_id_is_required(model):
    with pytest.raises(ValueError):
        AnthropicLLM(model, client=FakeClient(None))


def test_works_with_real_sdk_response_objects():
    anthropic = pytest.importorskip("anthropic")
    from anthropic.types import Message, TextBlock, ThinkingBlock, Usage

    message = Message(
        id="msg_1", type="message", role="assistant", model="some-model-id",
        content=[ThinkingBlock(type="thinking", thinking="hmm", signature="sig"), TextBlock(type="text", text="real answer")],
        stop_reason="end_turn", stop_sequence=None, usage=Usage(input_tokens=11, output_tokens=5),
    )
    llm, _ = make(message)
    assert llm.complete("s", "u") == "real answer"
    assert (llm.input_tokens, llm.output_tokens) == (11, 5)
    assert anthropic.__version__  # the SDK is importable in this environment


def test_budgeted_llm_stops_at_the_limit_without_calling_the_model():
    inner, client = make()
    budgeted = BudgetedLLM(inner, 2)
    budgeted.complete("s", "u")
    budgeted.complete("s", "u")
    with pytest.raises(RuntimeError, match="budget"):
        budgeted.complete("s", "u")
    assert len(client.requests) == 2


def test_budgeted_llm_rejects_a_negative_limit():
    with pytest.raises(ValueError):
        BudgetedLLM(EchoLLM(), -1)


# ---- choosing the model from the environment ----

ENV_VARS = ["LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_EFFORT", "LLM_MAX_TOKENS", "ANTHROPIC_API_KEY"]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_with_no_settings_the_app_runs_offline(clean_env):
    assert isinstance(llm_from_env(), EchoLLM)


def test_anthropic_needs_a_model_and_a_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        llm_from_env()
    clean_env.setenv("LLM_MODEL", "some-model-id")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm_from_env()


def test_anthropic_settings_are_read_from_the_environment(clean_env):
    pytest.importorskip("anthropic")
    clean_env.setenv("LLM_PROVIDER", "Anthropic")  # case-insensitive
    clean_env.setenv("LLM_MODEL", "some-model-id")
    clean_env.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    clean_env.setenv("LLM_EFFORT", "medium")
    clean_env.setenv("LLM_MAX_TOKENS", "256")
    llm = llm_from_env()
    assert isinstance(llm, AnthropicLLM)
    assert (llm.model, llm.effort, llm.max_tokens) == ("some-model-id", "medium", 256)


@pytest.mark.parametrize("value", ["none", "NONE", ""])
def test_effort_none_turns_effort_off(clean_env, value):
    pytest.importorskip("anthropic")
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("LLM_MODEL", "some-model-id")
    clean_env.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    clean_env.setenv("LLM_EFFORT", value)
    assert llm_from_env().effort is None


def test_unknown_provider_is_rejected(clean_env):
    clean_env.setenv("LLM_PROVIDER", "mystery")
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        llm_from_env()


def test_the_openai_style_path_still_works(clean_env):
    clean_env.setenv("LLM_API_KEY", "not-a-real-key")
    clean_env.setenv("LLM_MODEL", "some-model-id")
    assert isinstance(llm_from_env(), OpenAICompatibleLLM)


def test_a_missing_anthropic_package_gives_a_clear_install_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` raise ImportError
    with pytest.raises(RuntimeError, match=r"pip install -e"):
        AnthropicLLM("some-model-id")

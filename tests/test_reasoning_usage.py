from types import SimpleNamespace

from agents.reasoning import ReasoningAgent


def make_agent():
    agent = ReasoningAgent.__new__(ReasoningAgent)
    agent.token_counter = lambda *, text: len(text.split())
    return agent


def test_sglang_top_level_reasoning_tokens_are_not_double_counted():
    usage = SimpleNamespace(
        completion_tokens=144,
        reasoning_tokens=114,
        completion_tokens_details=None,
        model_extra={},
    )

    response_tokens, reasoning_tokens = make_agent()._completion_token_stats(
        usage, "ignored fallback"
    )

    assert response_tokens == 30
    assert reasoning_tokens == 114
    assert response_tokens + reasoning_tokens == usage.completion_tokens


def test_openai_completion_details_reasoning_tokens_are_supported():
    usage = SimpleNamespace(
        completion_tokens=80,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=60),
        model_extra={},
    )

    assert make_agent()._completion_token_stats(usage, "ignored") == (20, 60)


def test_reasoning_text_is_tokenized_when_server_omits_usage_detail():
    usage = SimpleNamespace(
        completion_tokens=10,
        completion_tokens_details=None,
        model_extra={},
    )

    assert make_agent()._completion_token_stats(usage, "one two three") == (7, 3)

from agents.reasoning import ReasoningAgent, build_argparser


def make_agent(include_reasoning_history):
    agent = ReasoningAgent.__new__(ReasoningAgent)
    agent.include_reasoning_history = include_reasoning_history
    agent.history = []
    agent.context_limit = 100
    agent.conversation = True
    agent.allows_system_prompt = True
    agent.llm = "Qwen/Qwen3.5-4B"
    agent.use_vllm_client = True
    agent._tool_system_prompt = "Use submit_command."
    return agent


def set_identity_fields(agent):
    agent.seed = 1
    agent.context_limit = 100
    agent.conversation = True
    agent.act_temp = 0.0
    agent.cot_temp = 0.6
    agent.reasoning_effort = 1024
    agent.tool_actions = True


def test_reasoning_history_flag_defaults_to_disabled():
    args = build_argparser().parse_args(
        ["--reasoning-effort", "1024", "--conversation"]
    )

    assert args.include_reasoning_history is False


def test_reasoning_history_flag_can_be_enabled():
    args = build_argparser().parse_args(
        [
            "--reasoning-effort",
            "1024",
            "--conversation",
            "--include-reasoning-history",
        ]
    )

    assert args.include_reasoning_history is True


def test_default_identity_and_params_remain_unchanged():
    agent = make_agent(False)
    set_identity_fields(agent)

    assert not agent.uid.endswith("_reasoningHistory1")
    assert "include_reasoning_history" not in agent.params


def test_tool_actions_suffix_only_when_enabled():
    agent = make_agent(False)
    set_identity_fields(agent)
    agent.tool_actions = False

    assert agent.uid == (
        "ReasoningAgent_Qwen/Qwen3.5-4B_s1_c100_convTrue_actT0.0_cotT0.6_effort1024"
    )
    assert "tool_actions" not in agent.params

    agent.tool_actions = True
    assert agent.uid.endswith("_effort1024_tool1")
    assert agent.params["tool_actions"] is True


def test_enabled_identity_and_params_are_distinct():
    agent = make_agent(True)
    set_identity_fields(agent)

    assert agent.uid.endswith("_reasoningHistory1")
    assert agent.params["include_reasoning_history"] is True


def test_default_history_shape_and_messages_remain_unchanged():
    agent = make_agent(False)
    agent._append_history("room", "look", "private reasoning")

    assert agent.history == [("room", "look")]
    messages = agent.build_messages("next")
    assert messages[2] == {"role": "assistant", "content": "look"}


def test_free_text_history_uses_structured_reasoning_content():
    agent = make_agent(True)
    agent._append_history("room", "look", "private reasoning")

    messages = agent.build_messages("next")

    assert agent.history == [("room", "look", "private reasoning")]
    assert messages[2] == {
        "role": "assistant",
        "content": "look",
        "reasoning_content": "private reasoning",
    }


def test_tool_history_uses_structured_reasoning_content():
    agent = make_agent(True)
    agent._append_history("room", "look", "private reasoning")

    messages = agent._build_vllm_tool_messages("next")
    assistant = messages[2]

    assert assistant["reasoning_content"] == "private reasoning"
    assert assistant["tool_calls"][0]["function"]["arguments"] == (
        '{"command": "look"}'
    )


def test_legacy_two_tuple_history_is_supported():
    agent = make_agent(True)
    agent.history.append(("room", "look"))

    messages = agent._build_vllm_tool_messages("next")

    assert "reasoning_content" not in messages[2]

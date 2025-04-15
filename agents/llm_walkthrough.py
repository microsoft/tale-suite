import argparse

import gymnasium as gym

from agents.llm import LLMAgent
from tales.agent import register
from tales.utils import merge_messages


# For the LLMWlkThrAgent, the sysprompt is initialized in the __init__ function as we need to change it once we extract the walkthrough from the env
class LLMWalkThroughAgent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sys_prompt = "Not Initialized"

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation is not None}"
            f"Walkthrough Agent"
        )

    def build_messages(self, observation):
        messages = [{"role": "system", "content": self.sys_prompt}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            # Make sure the system prompt is added to the following message.
            messages.pop(0)
            messages[1]["content"] = f"{self.sys_prompt}\n\n{messages[1]['content']}"

        return messages

    def reset(self, obs, info, env_name):
        walkthrough = info.get("extra.walkthrough")
        if walkthrough is None or len(walkthrough) < 1:
            raise ValueError("Walkthrough not initalized: Check the environment")

        # Check if the walkthrough is valid.
        env = gym.make(f"tales/{env_name}-v0", disable_env_checker=True)

        _, _ = env.reset()

        for act in walkthrough:
            _, _, _, info_ = env.step(act)

        if info_["score"] != info_["max_score"]:
            raise ValueError(
                "Provided walkthrough does not successfully complete game."
            )

        numbered_walkthrough = ", ".join(
            f"{i + 1}.){act}" for i, act in enumerate(walkthrough)
        )
        self.sys_prompt = (
            "You are playing a text-based game and your goal is to finish it with the highest score."
            " The following is a walkthrough in the form of a list of actions to beat the game."
            " You should follow this walkthrough as closely as possible to get the maximum score"
            " You must ONLY respond with the action you wish to take with no other special tokens."
            f"Walkthrough: {numbered_walkthrough}"
        )


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        default=10,
        help="Limit context for LLM (in conversation turns). Default: %(default)s",
    )
    group.add_argument(
        "--conversation",
        action="store_true",
        help="Enable conversation mode. Otherwise, use single prompt.",
    )
    group.add_argument(
        "--wlkthr-limit",
        type=int,
        default=10000,
        help="Number of walkthrough actions to provide the LLM. Default: %(default)s",
    )

    return parser


register(
    name="llm-walkthrough",
    desc=(
        "This agent uses the ground-truth walkthrough from the environment to attempt to progress through the game."
    ),
    klass=LLMWalkThroughAgent,
    add_arguments=build_argparser,
)

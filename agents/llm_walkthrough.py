import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import twbench
from agents.llm import LLMAgent
from twbench.agent import register
from twbench.utils import (
    TokenCounter,
    format_messages_to_markdown,
    is_recoverable_error,
    messages2conversation,
)


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

    def reset(self, obs, infos, env=""):
        plain_walkthrough = infos.get("extra.walkthrough")
        numbered_walkthrough = ""

        for i, act in enumerate(plain_walkthrough):
            numbered_walkthrough += str(i + 1) + ".)" + act + ", "

        self.sys_prompt = (
            "You are playing a text-based game and your goal is to finish it with the highest score."
            " The following is a walkthrough in the form of a list of actions to beat the game."
            " You should follow this walkthrough as closely as possible to get the maximum score"
            " You must ONLY respond with the action you wish to take with no other special tokens."
            "Walkthrough: $%^WALKTHROUGH$%^"
        ).replace("$%^WALKTHROUGH$%^", numbered_walkthrough[:-2])

        if "$%^WALKTHROUGH$%^" in self.sys_prompt or len(plain_walkthrough) < 1:
            raise ValueError("Walkthrough not initalized: Check the environment")
        elif not infos["valid_walkthrough"]:
            raise ValueError(
                "Provided walkthrough does not successfully complete game: Terminating run"
            )

        return True


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
        default=51,
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
    name="walkthrough",
    desc=(
        "This agent uses the ground-truth walkthrough from the environment to attempt to progress through the game."
    ),
    klass=LLMWalkThroughAgent,
    add_arguments=build_argparser,
)

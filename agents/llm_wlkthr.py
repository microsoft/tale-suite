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
from twbench.utils import count_tokens, is_recoverable_error


# For the LLMWlkThrAgent, the sysprompt is initialized in the __init__ function as we need to change it once we extract the walkthrough from the env
class LLMWlkThrAgent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sys_prompt = "Not Initialized"
        self.steps = 1

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation is not None}"
            f"Walkthrough Agent"
        )

    def act(self, obs, reward, done, infos):

        # Throw an error if the agent has failed to get the walkthrough from the environment for whatever reason.
        if self.sys_prompt == "Not Initialized":
            raise ValueError("Walkthrough not initalized: Check the environment")

        conversation = self.conversation or self.model.conversation()
        conversation.responses = conversation.responses[-self.context :]
        prompt = f"{obs}\n>" if self.conversation else self.build_prompt(obs)
        response = conversation.prompt(
            prompt=prompt,
            system=self.sys_prompt,
            temperature=self.act_temp,
            seed=self.seed,
            top_p=1,
        )
        action = response.text().strip()
        self.history.append((obs, f"{action}\n"))
        # Compute usage statistics
        messages = conversation.responses[-1]._prompt_json["messages"]
        stats = {
            "prompt": response._prompt_json,
            "response": response.text(),
            "nb_tokens": count_tokens(messages=messages)
            + count_tokens(text=response.text()),
        }

        self.steps += 1

        return action, stats

    # def reset(self, obs, infos):
    #     self.sys_prompt = (
    #         "You are playing a text-based game and your goal is to finish it with the highest score."
    #         " The following is a walkthrough in the form of a list of actions to beat the game."
    #         " You should follow this walkthrough as closely as possible to get the maximum score"
    #         " You must ONLY respond with the action you wish to take with no other special tokens."
    #         "Walkthrough: WALKTHROUGH"
    #     ).replace("WALKTHROUGH", ",".join(infos.get("extra.walkthrough")))
    #     return True

    def reset(self, obs, infos):
        plain_walkthrough = infos.get("extra.walkthrough")
        numbered_walkthrough = ""

        for i, act in enumerate(plain_walkthrough):
            numbered_walkthrough += str(i + 1) + ".)" + act + ", "

        self.sys_prompt = (
            "You are playing a text-based game and your goal is to finish it with the highest score."
            " The following is a walkthrough in the form of a list of actions to beat the game."
            " You should follow this walkthrough as closely as possible to get the maximum score"
            " You must ONLY respond with the action you wish to take with no other special tokens."
            "Walkthrough: WALKTHROUGH"
        ).replace("WALKTHROUGH", numbered_walkthrough[:-2])
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
    name="wlkthr",
    desc=(
        "This agent uses the ground-truth walkthrough from the environment to attempt to progress through the game."
    ),
    klass=LLMWlkThrAgent,
    add_arguments=build_argparser,
)
